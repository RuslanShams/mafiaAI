# Мафия AI (Ollama)

Браузерная настольная игра **«Мафия»** (7 игроков: **2 мафии, 1 комиссар, 1 доктор, 3 мирных**; 1 человек против 6 ИИ-соперников) с визуальным нуар-столом, процедурными SVG-персонажами, комикс-облачками с плавной печатью текста и поддержкой до 4 независимых партий на одном сервере.

## Основные возможности
- **Полный ролевой состав на 7 мест:** 2 Мафии, 1 Комиссар, 1 Доктор (не может лечить одного и того же игрока две ночи подряд), 3 Мирных жителя.
- **Классические правила «в закрытую»:**
  - **Ночь 1** проходит без убийства — мафия знакомится, все 7 игроков выходят в День 1.
  - **День:** 2 круга последовательных обсуждений + открытое голосование на выбывание.
  - **Роли выбывших не раскрываются** до окончания партии; у каждого выбывающего есть **Последнее слово**.
  - Если человек выбывает раньше финала, после его последнего слова автоматически открывается **режим наблюдателя** с показом всех ролей за столом.
- **Комфортный темп чтения:** реплики ИИ плавно допечатываются посимвольно (~22 симв/сек) и удерживаются на экране ещё 2.2 сек после последней точки, чтобы игрок успевал всё прочитать даже на быстром GPU.
- **Мультисессионность (до 4 параллельных столов):** каждая вкладка/игрок получает изолированную партию (`Столы: X / 4` в шапке).
- **Выбор любой локальной модели из Ollama:** список автоматически подтягивается из `ollama list` (`gemma3:12b`, `gemma3:4b`, `qwen2.5:7b` и др.).

---

## 1. Локальный запуск (`localhost`)

Требуются Python 3.10+ и установленная [Ollama](https://ollama.com/).

```bash
# 1. Запустите Ollama и скачайте модель (например, gemma3:4b для CPU или gemma3:12b для GPU)
ollama serve
ollama pull gemma3:4b

# 2. Создайте виртуальное окружение и установите зависимости
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Запустите сервер игры
.venv/bin/python run.py
```

Откройте в браузере: **http://127.0.0.1:8000**

---

## 2. Деплой на свой сервер (любой Linux VPS / выделенный сервер)

Подойдёт любой сервер на Ubuntu/Debian (с видеокартой NVIDIA или только на CPU).

### Рекомендуемые характеристики сервера
| Сценарий | Железо | Рекомендуемая модель | Скорость хода ИИ |
| :--- | :--- | :--- | :--- |
| **1–4 игрока одновременно (GPU)** | NVIDIA GPU 12 GB VRAM (RTX 3060 / 4060 Ti / A10 / L4), 4–8 CPU, 16 GB RAM, 30 GB SSD | `gemma3:12b` (или `gemma3:4b`) | ~0.5–2.5 сек |
| **1–2 игрока на GPU 8 GB VRAM** | NVIDIA GPU 8 GB VRAM, 4 CPU, 12–16 GB RAM, 20 GB SSD | `gemma3:4b` | ~0.5–1.5 сек |
| **1 игрок без GPU (только CPU)** | 4–8 vCPU, 8–16 GB RAM, 20 GB SSD | `gemma3:4b` | ~6–15 сек |

### Шаг 1. Установите системные пакеты и Ollama
```bash
sudo apt-get update
sudo apt-get install -y git curl python3-venv python3-pip zstd

# Установка Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### Шаг 2. Настройте Ollama под параллельные сессии и скачайте модели
Создайте override для системного сервиса `ollama` (чтобы удерживать модель в памяти и обрабатывать до 4 параллельных столов):

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KEEP_ALIVE=24h"
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now ollama

# Скачайте нужные модели (для GPU 12 GB рекомендуется обе, для CPU/8 GB достаточно gemma3:4b)
ollama pull gemma3:4b
ollama pull gemma3:12b
```

> **Если на сервере недоступен `registry.ollama.ai`:** скачайте GGUF напрямую с Hugging Face (`ggml-org/gemma-3-4b-it-GGUF` / `ggml-org/gemma-3-12b-it-GGUF`) и импортируйте командой `ollama create gemma3:12b -f Modelfile` (с директивами `FROM /path/to/model.gguf` и `PARAMETER stop "<end_of_turn>"`).

### Шаг 3. Разверните проект и установите зависимости
```bash
sudo mkdir -p /opt/mafiaAI
sudo chown "$USER":"$USER" /opt/mafiaAI
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ> /opt/mafiaAI
cd /opt/mafiaAI

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

### Шаг 4. Запустите Мафия AI как фоновый `systemd`-сервис
> Для сервера **без GPU** поменяйте `MAFIA_GPU_MODE=0`, `OLLAMA_MODEL=gemma3:4b` и при необходимости `MAFIA_MAX_SESSIONS=1`.

```bash
sudo tee /etc/systemd/system/mafia-ai.service >/dev/null <<'EOF'
[Unit]
Description=Mafia AI Web Server (FastAPI + Ollama)
After=network.target ollama.service

[Service]
Type=simple
WorkingDirectory=/opt/mafiaAI
Environment="MAFIA_SERVER_MODE=1"
Environment="MAFIA_GPU_MODE=1"
Environment="MAFIA_HOST=0.0.0.0"
Environment="MAFIA_PORT=8000"
Environment="MAFIA_MAX_SESSIONS=4"
Environment="MAFIA_SINGLE_GPU_LOCK=1"
Environment="OLLAMA_MODEL=gemma3:12b"
ExecStart=/opt/mafiaAI/.venv/bin/python run.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now mafia-ai
sudo systemctl status mafia-ai
```

Проверьте локально, что сервер и Ollama отвечают:
```bash
curl -s http://127.0.0.1:8000/api/health
```

### Шаг 5. Откройте доступ игрокам из интернета (выберите 1 из 3 вариантов)

#### Вариант А. Свой домен + автоматический HTTPS через Caddy *(рекомендуется)*
Установите [Caddy](https://caddyserver.com/docs/install), направьте A-запись домена (например, `mafia.example.com`) на IP сервера и пропишите `/etc/caddy/Caddyfile`:

```caddyfile
mafia.example.com {
    reverse_proxy 127.0.0.1:8000
}
```
```bash
sudo systemctl reload caddy
```
Caddy сам получит TLS-сертификат Let's Encrypt и автоматически проксирует WebSocket (`/ws/game` → `wss://`).

#### Вариант Б. Свой домен через Nginx (обязательно с поддержкой WebSocket)
```nginx
server {
    listen 80;
    server_name mafia.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 3600s;
    }
}
```
После этого выпустите HTTPS-сертификат через `sudo certbot --nginx -d mafia.example.com`.

#### Вариант В. Прямой доступ по IP (без домена) или сервер за NAT
- **Если у сервера есть белый IP:** откройте порт `8000` в файрволе (`sudo ufw allow 8000/tcp`) и раздайте ссылку `http://<IP_СЕРВЕРА>:8000`.
- **Если сервер за NAT / без белого IP:** поднимите бесплатный HTTPS-туннель [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/):
  ```bash
  cloudflared tunnel --protocol http2 --url http://127.0.0.1:8000
  ```
  В консоль выведется готовая ссылка `https://<случайное-имя>.trycloudflare.com` (поддерживает HTTPS и `wss://`).

---

## 3. Быстрый деплой на арендованный GPU-сервер (Vast.ai)

Для посуточной/почасовой аренды на Vast.ai в проекте есть готовые скрипты [deploy/deploy_to_vast.sh](deploy/deploy_to_vast.sh) и [deploy/vast_setup.sh](deploy/vast_setup.sh):

```bash
VAST_PORT=37285 VAST_HOST=root@ssh8.vast.ai bash deploy/deploy_to_vast.sh
```

Скрипт копирует проект на сервер, настраивает `ollama`, `mafia_ai` и HTTPS-туннель `mafia_tunnel` через `supervisor`, скачивает `gemma3:4b` (для быстрого старта) и в фоне `gemma3:12b`, и выводит публичный адрес. Повторный запуск обновляет код без смены активной ссылки туннеля.

> После окончания игры на Vast.ai обязательно удалите инстанс (**Destroy**), чтобы остановить списание средств.

---

## 4. Переменные окружения ([backend/config.py](backend/config.py))

| Переменная | По умолчанию | Назначение |
| :--- | :--- | :--- |
| `MAFIA_SERVER_MODE` | `0` | `1` включает серверный профиль (`0.0.0.0`, `gemma3:12b`, `keep_alive=24h`, лок модели) |
| `MAFIA_GPU_MODE` | `0` (`1` в server mode) | Снимает CPU-ограничение потоков, увеличивает контекст (`3072`) и лимит токенов реплик |
| `MAFIA_HOST` / `MAFIA_PORT` | `127.0.0.1` / `8000` | Адрес и порт FastAPI-сервера |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | URL локального Ollama API |
| `OLLAMA_MODEL` | `gemma3:4b` (`gemma3:12b` в server mode) | Модель по умолчанию |
| `MAFIA_MAX_SESSIONS` | `4` | Максимум одновременно идущих партий на сервере |
| `MAFIA_SINGLE_GPU_LOCK` | `0` (`1` в server mode) | Блокирует смену модели в лобби, пока идёт другая партия (защита от выгрузки VRAM) |
| `MAFIA_SPEECH_CPS` | `22.0` | Скорость расчёта печати текста в облачке (символов в секунду) |
| `MAFIA_SPEECH_HOLD_SEC` | `2.2` | Пауза удержания облачка после завершения печати (в секундах) |
| `MAFIA_ABANDONED_TTL` | `120` | Через сколько секунд освобождается слот после отключения вкладки |
| `MAFIA_IDLE_TTL` | `1800` | Таймаут неактивной сессии (в секундах) |

---

## 5. Запуск автотестов

```bash
.venv/bin/python -m unittest discover -s tests -v
```
