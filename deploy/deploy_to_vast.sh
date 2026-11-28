#!/usr/bin/env bash
# ==============================================================================
# Скрипт отправки проекта с ноутбука на арендованный Vast.ai сервер и запуска
# Использование:
#   bash deploy/deploy_to_vast.sh
# Или с другими портом/хостом:
#   VAST_PORT=37285 VAST_HOST=root@ssh8.vast.ai bash deploy/deploy_to_vast.sh
# ==============================================================================
set -euo pipefail

VAST_PORT="${VAST_PORT:-37285}"
VAST_HOST="${VAST_HOST:-root@ssh8.vast.ai}"
REMOTE_DIR="/workspace/mafiaAI"
SSH_OPTS=(-p "${VAST_PORT}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_ssh_retry() {
  local attempt=1
  until ssh "${SSH_OPTS[@]}" "${VAST_HOST}" "$@"; do
    if [ "${attempt}" -ge 5 ]; then
      return 1
    fi
    echo "SSH-соединение прервалось, повтор (${attempt}/4) через 5 сек..." >&2
    attempt=$((attempt + 1))
    sleep 5
  done
}

echo "=== [1/3] Копирование файлов проекта на ${VAST_HOST} (порт ${VAST_PORT}) ==="
run_ssh_retry "mkdir -p ${REMOTE_DIR}"

upload_attempt=1
until tar -C "${PROJECT_ROOT}" \
  --exclude=".git" \
  --exclude=".venv" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  -czf - . \
  | ssh "${SSH_OPTS[@]}" "${VAST_HOST}" "tar -xzf - -C ${REMOTE_DIR}"; do
  if [ "${upload_attempt}" -ge 4 ]; then
    echo "ОШИБКА: не удалось загрузить архив проекта." >&2
    exit 1
  fi
  echo "Повтор загрузки архива (${upload_attempt}/3)..." >&2
  upload_attempt=$((upload_attempt + 1))
  sleep 4
done

echo "=== [2/3] Запуск автономной установки через supervisor на сервере ==="
run_ssh_retry "rm -f /var/log/vast_setup.log && printf '[program:vast_setup_once]\ncommand=/bin/bash %s/deploy/vast_setup.sh\nenvironment=APP_DIR=\"%s\"\nautostart=false\nautorestart=false\nstdout_logfile=/var/log/vast_setup.log\nredirect_stderr=true\nstdout_logfile_maxbytes=10MB\n' '${REMOTE_DIR}' '${REMOTE_DIR}' >/etc/supervisor/conf.d/vast_setup_once.conf && supervisorctl reread >/dev/null && supervisorctl update >/dev/null && (supervisorctl stop vast_setup_once >/dev/null 2>&1 || true) && supervisorctl start vast_setup_once"

echo "=== [3/3] Ожидание завершения установки и запуска сервера ==="
for _ in $(seq 1 160); do
  status_out="$(ssh "${SSH_OPTS[@]}" "${VAST_HOST}" "supervisorctl status vast_setup_once 2>/dev/null || true; echo '---LOG---'; tail -n 35 /var/log/vast_setup.log 2>/dev/null || true" 2>/dev/null || true)"
  log_part="${status_out#*---LOG---}"
  svc_part="${status_out%%---LOG---*}"
  if echo "${log_part}" | grep -Fq "=== VAST_SETUP_DONE ==="; then
    echo "${log_part}"
    exit 0
  fi
  if echo "${log_part}" | grep -Fq "=== VAST_SETUP_FAILED ==="; then
    echo "${log_part}" >&2
    exit 1
  fi
  if echo "${svc_part}" | grep -Eq "vast_setup_once[[:space:]]+(EXITED|FATAL|BACKOFF)" && ! echo "${log_part}" | grep -Fq "=== VAST_SETUP_DONE ==="; then
    echo "Процесс установки неожиданно завершился:" >&2
    echo "${status_out}" >&2
    exit 1
  fi
  last_line="$(echo "${log_part}" | tr '\r' '\n' | grep -E '\[#|->|===' | tail -n 1 || true)"
  [ -n "${last_line}" ] && echo "[прогресс] ${last_line}"
  sleep 15
done

echo "Таймаут ожидания установки. Последние строки лога:" >&2
run_ssh_retry "tail -n 50 /var/log/vast_setup.log || true"
exit 1
