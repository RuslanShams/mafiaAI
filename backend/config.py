from __future__ import annotations

import os

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

HOST: str = os.getenv("MAFIA_HOST", "127.0.0.1")
PORT: int = int(os.getenv("MAFIA_PORT", "8000"))

TOTAL_PLAYERS: int = 7
MAFIA_COUNT: int = 2
DISCUSSION_ROUNDS: int = 2

LLM_TIMEOUT_SECONDS: float = 90.0
LLM_MAX_RETRIES: int = 2
LLM_KEEP_ALIVE: str = "20m"

