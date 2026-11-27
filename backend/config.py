from __future__ import annotations

import os

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")

HOST: str = os.getenv("MAFIA_HOST", "127.0.0.1")
PORT: int = int(os.getenv("MAFIA_PORT", "8000"))

TOTAL_PLAYERS: int = 7
MAFIA_COUNT: int = 2
DISCUSSION_ROUNDS: int = 2

# CPU & latency tuning (default physical cores = half of logical threads, e.g. 4 on Ryzen 3 5300U)
_DEFAULT_PHYSICAL_CORES = max(2, (os.cpu_count() or 4) // 2)
LLM_NUM_THREADS: int = int(os.getenv("MAFIA_NUM_THREADS", str(_DEFAULT_PHYSICAL_CORES)))
LLM_NUM_CTX: int = int(os.getenv("MAFIA_NUM_CTX", "2048"))
LLM_NUM_BATCH: int = int(os.getenv("MAFIA_NUM_BATCH", "512"))

# Per-phase generation token caps to guarantee ~8-12s turns on CPU
PREDICT_TOKENS_INTRO: int = 42
PREDICT_TOKENS_DISCUSSION: int = 52
PREDICT_TOKENS_VOTE: int = 58
PREDICT_TOKENS_NIGHT: int = 52

LLM_TIMEOUT_SECONDS: float = 45.0
LLM_MAX_RETRIES: int = 1
LLM_KEEP_ALIVE: str = "30m"
