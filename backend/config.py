from __future__ import annotations

import os

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Server / GPU mode (enabled on Vast.ai / dedicated GPU server via MAFIA_SERVER_MODE=1)
SERVER_MODE: bool = os.getenv("MAFIA_SERVER_MODE", "0") == "1"

OLLAMA_MODEL: str = os.getenv(
    "OLLAMA_MODEL",
    "gemma3:12b" if SERVER_MODE else "gemma3:4b",
)

HOST: str = os.getenv("MAFIA_HOST", "0.0.0.0" if SERVER_MODE else "127.0.0.1")
PORT: int = int(os.getenv("MAFIA_PORT", "8000"))

# Multi-session concurrency settings (up to 4 simultaneous active games)
MAX_SESSIONS: int = int(os.getenv("MAFIA_MAX_SESSIONS", "4"))
SINGLE_GPU_MODEL_LOCK: bool = (
    os.getenv("MAFIA_SINGLE_GPU_LOCK", "1" if SERVER_MODE else "0") == "1"
)
SESSION_ABANDONED_TTL_SEC: float = float(os.getenv("MAFIA_ABANDONED_TTL", "120"))
SESSION_IDLE_TTL_SEC: float = float(os.getenv("MAFIA_IDLE_TTL", "1800"))

TOTAL_PLAYERS: int = 7
MAFIA_COUNT: int = 2
DISCUSSION_ROUNDS: int = 2

# Hardware-tuned parameters: GPU server (RTX 3060 12GB + 8-core EPYC) vs local CPU laptop
_DEFAULT_PHYSICAL_CORES = max(2, (os.cpu_count() or 4) // 2)
LLM_USE_GPU_MODE: bool = os.getenv("MAFIA_GPU_MODE", "1" if SERVER_MODE else "0") == "1"
LLM_NUM_THREADS: int = int(os.getenv("MAFIA_NUM_THREADS", str(_DEFAULT_PHYSICAL_CORES)))
LLM_NUM_CTX: int = int(os.getenv("MAFIA_NUM_CTX", "3072" if LLM_USE_GPU_MODE else "2048"))
LLM_NUM_BATCH: int = int(os.getenv("MAFIA_NUM_BATCH", "512"))

# Richer token budgets on GPU (60-90 tok/s -> ~1.5-2.5s per turn!) vs tight CPU budgets
PREDICT_TOKENS_INTRO: int = 65 if LLM_USE_GPU_MODE else 42
PREDICT_TOKENS_DISCUSSION: int = 95 if LLM_USE_GPU_MODE else 52
PREDICT_TOKENS_VOTE: int = 75 if LLM_USE_GPU_MODE else 58
PREDICT_TOKENS_NIGHT: int = 70 if LLM_USE_GPU_MODE else 52
PREDICT_TOKENS_NIGHT_TARGET: int = 20 if LLM_USE_GPU_MODE else 16

LLM_TIMEOUT_SECONDS: float = 60.0 if LLM_USE_GPU_MODE else 45.0
LLM_MAX_RETRIES: int = 1
LLM_KEEP_ALIVE: str = "24h" if SERVER_MODE else "30m"

# Comfortable typewriter & reading pace for comic speech bubbles (~22 chars/sec + 2.2s reading hold)
SPEECH_CHARS_PER_SEC: float = float(os.getenv("MAFIA_SPEECH_CPS", "22.0"))
SPEECH_READ_HOLD_SEC: float = float(os.getenv("MAFIA_SPEECH_HOLD_SEC", "2.2"))
