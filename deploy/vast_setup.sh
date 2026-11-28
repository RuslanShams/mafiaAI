#!/usr/bin/env bash
# ==============================================================================
# Скрипт автоматической настройки и запуска Мафия AI внутри Vast.ai GPU-сервера
# Рассчитан на 1x RTX 3060 (12 GB VRAM) + 8 CPU + 16 GB RAM и до 4 параллельных сессий
# ==============================================================================
set -Eeuo pipefail
trap 'echo "=== VAST_SETUP_FAILED ==="' ERR

APP_DIR="${APP_DIR:-/workspace/mafiaAI}"
PRIMARY_MODELS="${PRIMARY_MODELS:-gemma3:4b}"
BG_MODELS="${BG_MODELS:-gemma3:12b}"
MAX_SESSIONS="${MAFIA_MAX_SESSIONS:-4}"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

echo "=== [1/5] Проверка системных пакетов ==="
export DEBIAN_FRONTEND=noninteractive
if ! command -v curl >/dev/null 2>&1 || ! python3 -m venv --help >/dev/null 2>&1 || ! command -v zstd >/dev/null 2>&1 || ! command -v aria2c >/dev/null 2>&1; then
  apt-get update -qq || true
  apt-get install -y -qq curl python3-venv python3-pip zstd pciutils lsof iproute2 aria2 || true
fi

echo "=== [2/5] Настройка и запуск Ollama под GPU (параллельно до ${MAX_SESSIONS} сессий) ==="
if [ ! -x /usr/local/bin/ollama ] && ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
OLLAMA_BIN="$(command -v ollama || echo /usr/local/bin/ollama)"

if command -v supervisorctl >/dev/null 2>&1 && [ -d /etc/supervisor/conf.d ]; then
  mkdir -p /opt/supervisor-scripts
  cat >/opt/supervisor-scripts/ollama_svc.sh <<EOF
#!/usr/bin/env bash
set -a
[ -f /etc/environment ] && source /etc/environment || true
set +a
export HOME="/root"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_NUM_PARALLEL="${MAX_SESSIONS}"
export OLLAMA_MAX_LOADED_MODELS="1"
export OLLAMA_FLASH_ATTENTION="1"
export OLLAMA_KEEP_ALIVE="24h"
exec "${OLLAMA_BIN}" serve
EOF
  chmod +x /opt/supervisor-scripts/ollama_svc.sh

  cat >/etc/supervisor/conf.d/ollama.conf <<'EOF'
[program:ollama]
command=/opt/supervisor-scripts/ollama_svc.sh
autostart=true
autorestart=true
stdout_logfile=/var/log/ollama.log
redirect_stderr=true
stdout_logfile_maxbytes=10MB
EOF
  supervisorctl reread >/dev/null
  supervisorctl update >/dev/null
  if ! supervisorctl status ollama 2>/dev/null | grep -q RUNNING; then
    pkill -f "[o]llama serve" || true
    supervisorctl start ollama >/dev/null 2>&1 || true
  fi
else
  if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    export OLLAMA_HOST="127.0.0.1:11434"
    export OLLAMA_NUM_PARALLEL="${MAX_SESSIONS}"
    export OLLAMA_MAX_LOADED_MODELS="1"
    export OLLAMA_FLASH_ATTENTION="1"
    export OLLAMA_KEEP_ALIVE="24h"
    nohup "${OLLAMA_BIN}" serve </dev/null >/var/log/ollama.log 2>&1 &
  fi
fi

echo -n "Ожидание готовности Ollama..."
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo " готово!"
    break
  fi
  echo -n "."
  sleep 1
done

echo "=== Проверка GPU (nvidia-smi) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

echo "=== [3/5] Установка Python-окружения и запуск веб-сервера + HTTP/2 туннеля ==="
cd "${APP_DIR}"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --quiet -r requirements.txt

CHOSEN_INT_PORT=""
CHOSEN_EXT_PORT=""

if [ -f /etc/environment ]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/environment || true
  set +a
fi

if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl stop mafia_ai >/dev/null 2>&1 || true
fi
pkill -f "[p]ython run.py" || true
pkill -f "[u]vicorn backend.app:app" || true
sleep 1

for var in $(compgen -A variable | grep '^VAST_TCP_PORT_' | sort); do
  int_p="${var#VAST_TCP_PORT_}"
  ext_p="${!var}"
  if [ "${int_p}" = "22" ] || [ "${int_p}" = "1111" ] || [ "${int_p}" -gt 65535 ] 2>/dev/null; then
    continue
  fi
  if ! ss -tln | awk '{print $4}' | grep -qE ":${int_p}$"; then
    CHOSEN_INT_PORT="${int_p}"
    CHOSEN_EXT_PORT="${ext_p}"
    break
  fi
done

if [ -z "${CHOSEN_INT_PORT}" ]; then
  CHOSEN_INT_PORT="${MAFIA_PORT:-10100}"
  CHOSEN_EXT_PORT="${VAST_TCP_PORT_10100:-${CHOSEN_INT_PORT}}"
fi

REAL_IP="$(curl -4 -fsS -m 5 https://ifconfig.me 2>/dev/null || echo "${PUBLIC_IPADDR:-<IP_СЕРВЕРА>}")"

if command -v supervisorctl >/dev/null 2>&1 && [ -d /etc/supervisor/conf.d ]; then
  mkdir -p /opt/supervisor-scripts
  cat >/opt/supervisor-scripts/mafia_ai_svc.sh <<EOF
#!/usr/bin/env bash
set -a
[ -f /etc/environment ] && source /etc/environment || true
set +a
export MAFIA_SERVER_MODE="1"
export MAFIA_GPU_MODE="1"
export MAFIA_HOST="0.0.0.0"
export MAFIA_PORT="${CHOSEN_INT_PORT}"
export MAFIA_MAX_SESSIONS="${MAX_SESSIONS}"
export MAFIA_SINGLE_GPU_LOCK="1"
export OLLAMA_MODEL="gemma3:12b"
cd "${APP_DIR}"
exec "${APP_DIR}/.venv/bin/python" run.py
EOF
  chmod +x /opt/supervisor-scripts/mafia_ai_svc.sh

  cat >/etc/supervisor/conf.d/mafia_ai.conf <<'EOF'
[program:mafia_ai]
command=/opt/supervisor-scripts/mafia_ai_svc.sh
autostart=true
autorestart=true
stdout_logfile=/var/log/mafia_ai.log
redirect_stderr=true
stdout_logfile_maxbytes=10MB
EOF

  CLOUDFLARED_BIN="$(command -v cloudflared || echo /opt/instance-tools/bin/cloudflared)"
  TUNNEL_ALREADY_OK="0"
  if supervisorctl status mafia_tunnel 2>/dev/null | grep -q RUNNING && \
     grep -Fq "http://127.0.0.1:${CHOSEN_INT_PORT}" /opt/supervisor-scripts/mafia_tunnel_svc.sh 2>/dev/null && \
     grep -qE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' /var/log/mafia_tunnel.log 2>/dev/null; then
    TUNNEL_ALREADY_OK="1"
  fi

  if [ -x "${CLOUDFLARED_BIN}" ] && [ "${TUNNEL_ALREADY_OK}" = "0" ]; then
    supervisorctl stop mafia_tunnel >/dev/null 2>&1 || true
    rm -f /var/log/mafia_tunnel.log
    cat >/opt/supervisor-scripts/mafia_tunnel_svc.sh <<EOF
#!/usr/bin/env bash
exec "${CLOUDFLARED_BIN}" tunnel --protocol http2 --no-autoupdate --url "http://127.0.0.1:${CHOSEN_INT_PORT}"
EOF
    chmod +x /opt/supervisor-scripts/mafia_tunnel_svc.sh

    cat >/etc/supervisor/conf.d/mafia_tunnel.conf <<'EOF'
[program:mafia_tunnel]
command=/opt/supervisor-scripts/mafia_tunnel_svc.sh
autostart=true
autorestart=true
stdout_logfile=/var/log/mafia_tunnel.log
redirect_stderr=true
stdout_logfile_maxbytes=5MB
EOF
  fi

  supervisorctl reread >/dev/null
  supervisorctl update >/dev/null
  supervisorctl start mafia_ai >/dev/null 2>&1 || true
  if [ "${TUNNEL_ALREADY_OK}" = "0" ] && [ -f /etc/supervisor/conf.d/mafia_tunnel.conf ]; then
    supervisorctl start mafia_tunnel >/dev/null 2>&1 || true
  fi
else
  export MAFIA_SERVER_MODE="1"
  export MAFIA_GPU_MODE="1"
  export MAFIA_HOST="0.0.0.0"
  export MAFIA_PORT="${CHOSEN_INT_PORT}"
  export MAFIA_MAX_SESSIONS="${MAX_SESSIONS}"
  export MAFIA_SINGLE_GPU_LOCK="1"
  export OLLAMA_MODEL="gemma3:12b"
  nohup .venv/bin/python run.py </dev/null >/var/log/mafia_ai.log 2>&1 &
fi

# Генерируем переиспользуемый скрипт докачки GGUF-модели с поддержкой докачки (-c)
mkdir -p /opt/supervisor-scripts
cat >/opt/supervisor-scripts/pull_gguf_model.sh <<'PUEOF'
#!/usr/bin/env bash
set -euo pipefail
mdl="$1"
OLLAMA_BIN="$(command -v ollama || echo /usr/local/bin/ollama)"

if OLLAMA_HOST="127.0.0.1:11434" "${OLLAMA_BIN}" list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "${mdl}"; then
  echo "-> Модель ${mdl} уже установлена, пропускаем скачивание."
  exit 0
fi

case "${mdl}" in
  gemma3:4b)
    hf_repo="ggml-org/gemma-3-4b-it-GGUF"
    hf_file="gemma-3-4b-it-Q4_K_M.gguf"
    dl_dir="/tmp/hf_gguf"
    ;;
  gemma3:12b)
    hf_repo="ggml-org/gemma-3-12b-it-GGUF"
    hf_file="gemma-3-12b-it-Q4_K_M.gguf"
    dl_dir="/tmp/hf_gguf_12b"
    ;;
  *)
    OLLAMA_HOST="127.0.0.1:11434" "${OLLAMA_BIN}" pull "${mdl}"
    exit $?
    ;;
esac

mkdir -p "${dl_dir}"
target_gguf="${dl_dir}/${hf_file}"
url="https://huggingface.co/${hf_repo}/resolve/main/${hf_file}"

echo "-> Загрузка ${mdl} (${hf_file}) с автодокачкой при обрывах..."
attempt=1
until aria2c -c -x 8 -s 8 -k 4M \
  --max-tries=20 --retry-wait=3 --timeout=25 --connect-timeout=15 \
  --summary-interval=10 --console-log-level=warn --file-allocation=none \
  -d "${dl_dir}" -o "${hf_file}" "${url}"; do
  if [ "${attempt}" -ge 30 ]; then
    echo "ОШИБКА: не удалось докачать ${hf_file} после ${attempt} перезапусков." >&2
    exit 1
  fi
  echo "-> Перезапуск докачки ${hf_file} с места обрыва (попытка ${attempt}/30) через 4 сек..."
  attempt=$((attempt + 1))
  sleep 4
done

echo "-> Импорт ${mdl} в Ollama..."
cat >"${dl_dir}/Modelfile" <<MFEOF
FROM ${target_gguf}
PARAMETER stop "<end_of_turn>"
MFEOF
OLLAMA_HOST="127.0.0.1:11434" "${OLLAMA_BIN}" create "${mdl}" -f "${dl_dir}/Modelfile"
rm -rf "${dl_dir}"
echo "-> Модель ${mdl} успешно импортирована в Ollama!"
PUEOF
chmod +x /opt/supervisor-scripts/pull_gguf_model.sh

echo "=== [4/5] Загрузка базовой модели для мгновенного старта (${PRIMARY_MODELS}) ==="
for mdl in ${PRIMARY_MODELS}; do
  /opt/supervisor-scripts/pull_gguf_model.sh "${mdl}"
done

# Запускаем фоновую докачку старшей модели gemma3:12b через supervisor (не блокируя старт игры!)
if [ -n "${BG_MODELS}" ] && command -v supervisorctl >/dev/null 2>&1 && [ -d /etc/supervisor/conf.d ]; then
  cat >/etc/supervisor/conf.d/mafia_pull_bg.conf <<EOF
[program:mafia_pull_bg]
command=/bin/bash -c 'for m in ${BG_MODELS}; do /opt/supervisor-scripts/pull_gguf_model.sh "\$m"; done'
autostart=false
autorestart=false
stdout_logfile=/var/log/mafia_pull_bg.log
redirect_stderr=true
stdout_logfile_maxbytes=10MB
EOF
  supervisorctl reread >/dev/null
  supervisorctl update >/dev/null
  if ! supervisorctl status mafia_pull_bg 2>/dev/null | grep -q RUNNING; then
    supervisorctl start mafia_pull_bg >/dev/null 2>&1 || true
  fi
  echo "-> Фоновая докачка старшей модели (${BG_MODELS}) запущена в supervisor (лог: /var/log/mafia_pull_bg.log)."
fi

echo "=== [5/5] Прогрев GPU, проверка /api/health и вывод публичного адреса ==="
FIRST_READY_MDL="$(OLLAMA_HOST="127.0.0.1:11434" "${OLLAMA_BIN}" list 2>/dev/null | awk 'NR==2 {print $1}')"
if [ -n "${FIRST_READY_MDL}" ]; then
  curl -fsS http://127.0.0.1:11434/api/chat -d "{\"model\":\"${FIRST_READY_MDL}\",\"messages\":[{\"role\":\"system\",\"content\":\"Отвечай строго на русском.\"},{\"role\":\"user\",\"content\":\"Скажи: Готов.\"}],\"stream\":false,\"keep_alive\":\"24h\",\"options\":{\"num_ctx\":3072,\"num_predict\":8}}" >/dev/null 2>&1 || true
fi

for _ in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:${CHOSEN_INT_PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

TUNNEL_URL=""
if [ -f /etc/supervisor/conf.d/mafia_tunnel.conf ]; then
  echo -n "Ожидание публичного HTTPS-туннеля Cloudflare (HTTP/2)..."
  for _ in $(seq 1 30); do
    TUNNEL_URL="$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' /var/log/mafia_tunnel.log 2>/dev/null | tail -n 1 || true)"
    if [ -n "${TUNNEL_URL}" ]; then
      echo " готово!"
      break
    fi
    echo -n "."
    sleep 1
  done
fi

echo ""
echo "=================================================================="
echo " Мафия AI успешно запущена на GPU-сервере!"
echo " Внутренний порт контейнера : ${CHOSEN_INT_PORT}"
echo " Прямой адрес (TCP-порт)    : http://${REAL_IP}:${CHOSEN_EXT_PORT}"
if [ -n "${TUNNEL_URL}" ]; then
  echo " Публичный HTTPS-адрес      : ${TUNNEL_URL}"
fi
echo " Лимит параллельных столов  : ${MAX_SESSIONS}"
echo " Логи игры                  : tail -f /var/log/mafia_ai.log"
echo " Логи докачки 12B           : tail -f /var/log/mafia_pull_bg.log"
echo " Логи Ollama                : tail -f /var/log/ollama.log"
echo "=================================================================="
curl -fsS "http://127.0.0.1:${CHOSEN_INT_PORT}/api/health" && echo ""
echo "=== VAST_SETUP_DONE ==="
