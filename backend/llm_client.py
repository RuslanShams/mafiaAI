from __future__ import annotations

import json
import logging
import random
import re
from typing import Optional

import httpx

from backend.config import (
    LLM_KEEP_ALIVE,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from backend.models import Gender, Player, Role

logger = logging.getLogger("mafia.llm")

# Regex patterns for strict Russian-only & in-character output validation
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]")
LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
META_BANNED_PHRASES = (
    "языковая модель",
    "искусственный интеллект",
    "как ии",
    "нейросет",
    "промпт",
    "ollama",
    "qwen",
    "mistral",
    "gemma",
    "json",
)
MAFIA_CONFESSION_RE = re.compile(
    r"\b(я\s+—?\s*мафи[яейю]|мы\s+с\s+\w+\s+мафи[яей]|признаюсь,?\s+я\s+мафи[яей])\b",
    re.IGNORECASE,
)


def clean_speech_text(text: str) -> str:
    """Normalize whitespace and strip stray outer quotes."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) >= 2 and (
        (cleaned.startswith("«") and cleaned.endswith("»"))
        or (cleaned.startswith('"') and cleaned.endswith('"'))
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def validate_russian_speech(
    speech: str,
    player: Player,
    is_public_day: bool = True,
    allowed_names: Optional[list[str]] = None,
) -> Optional[str]:
    """Return error string if speech violates Russian-only or roleplay constraints, else None."""
    if not speech or len(speech.strip()) < 3:
        return "Реплика пустая или слишком короткая."
    if CJK_RE.search(speech):
        return "В реплике обнаружены китайские/японские иероглифы! Пиши СТРОГО кириллицей на русском языке."
    scrubbed_for_latin = speech
    if allowed_names:
        for name in allowed_names:
            if name:
                scrubbed_for_latin = re.sub(re.escape(name), "", scrubbed_for_latin, flags=re.IGNORECASE)
    if LATIN_WORD_RE.search(scrubbed_for_latin):
        return "В реплике обнаружены слова латиницей! Используй ТОЛЬКО русский язык (кириллицу)."
    lower = speech.lower()
    for phrase in META_BANNED_PHRASES:
        if phrase in lower:
            return f"Запрещено выходить из роли и упоминать «{phrase}»."
    if is_public_day and player.role == Role.MAFIA and MAFIA_CONFESSION_RE.search(speech):
        return "Ты мафия на дневном обсуждении! Нельзя прямо признаваться столу, что ты мафия."
    # Check at least some Cyrillic letters exist
    if not re.search(r"[А-Яа-яЁё]{2,}", speech):
        return "Реплика должна содержать осмысленный текст на русском языке."
    return None


def resolve_target_name(raw_target: str, valid_targets: list[str]) -> Optional[str]:
    """Match LLM target name against valid alive target names."""
    if not raw_target:
        return None
    cleaned = raw_target.strip().strip("«»\"'.,!?:;[]() ")
    lower = cleaned.lower()
    for name in valid_targets:
        if name.lower() == lower:
            return name
    # Substring fallback (e.g. "[2] Виктор" -> "Виктор" or Russian case stem match)
    for name in valid_targets:
        if name.lower() in lower:
            return name
    for name in valid_targets:
        stem = name[: max(3, len(name) - 2)].lower()
        if stem in lower:
            return name
    return None


def _extract_json_dict(raw_content: str) -> Optional[dict]:
    text = raw_content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Try locating outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _format_size_ru(size_bytes: int) -> str:
    if not size_bytes or size_bytes <= 0:
        return ""
    gb = size_bytes / (1024 ** 3)
    if gb >= 1.0:
        return f"{gb:.1f} ГБ"
    mb = size_bytes / (1024 ** 2)
    return f"{round(mb)} МБ"


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None

    def set_model(self, model_name: str) -> None:
        cleaned = model_name.strip()
        if cleaned:
            self.model = cleaned

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def check_health(self) -> dict:
        """Check whether local Ollama is reachable and list all available local models."""
        try:
            resp = await self._get_client().get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            raw_models = data.get("models", [])
            models = [m.get("name", "") for m in raw_models if m.get("name")]
            models_detail = [
                {
                    "name": m.get("name", ""),
                    "size_bytes": int(m.get("size", 0) or 0),
                    "size_label": _format_size_ru(int(m.get("size", 0) or 0)),
                }
                for m in raw_models
                if m.get("name")
            ]

            # If current configured model is missing, fall back to first available model if any
            if models and self.model not in models:
                if OLLAMA_MODEL in models:
                    self.model = OLLAMA_MODEL
                else:
                    self.model = models[0]

            model_ready = self.model in models
            return {
                "ollama_online": True,
                "model_ready": model_ready,
                "configured_model": self.model,
                "available_models": models,
                "models_detail": models_detail,
                "error": None if model_ready else "В Ollama не найдено установленных моделей.",
            }
        except Exception as exc:
            return {
                "ollama_online": False,
                "model_ready": False,
                "configured_model": self.model,
                "available_models": [],
                "models_detail": [],
                "error": f"Не удалось подключиться к Ollama ({self.base_url}): {exc}",
            }

    async def generate_turn(
        self,
        player: Player,
        system_prompt: str,
        user_prompt: str,
        valid_targets: Optional[list[str]] = None,
        is_public_day: bool = True,
        fallback_kind: str = "discussion",
        allowed_names: Optional[list[str]] = None,
    ) -> dict[str, str]:
        """
        Query qwen2.5:7b in JSON mode with Russian/roleplay guardrails and automatic retry.
        Returns {"thought": str, "speech": str, "target": str (if valid_targets provided)}.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error = ""
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "format": "json",
                    "stream": False,
                    "keep_alive": LLM_KEEP_ALIVE,
                    "options": {
                        "temperature": 0.68 if attempt == 0 else 0.45,
                        "top_p": 0.9,
                        "repeat_penalty": 1.15,
                        "num_predict": 320,
                    },
                }
                resp = await self._get_client().post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                raw_content = resp.json().get("message", {}).get("content", "")
                parsed = _extract_json_dict(raw_content)
                if not parsed:
                    last_error = "Ответ не является корректным JSON-объектом."
                else:
                    speech = clean_speech_text(str(parsed.get("speech", "")))
                    thought = clean_speech_text(str(parsed.get("thought", "")))
                    err = validate_russian_speech(
                        speech,
                        player=player,
                        is_public_day=is_public_day,
                        allowed_names=allowed_names,
                    )
                    if err:
                        last_error = err
                    elif valid_targets is not None:
                        raw_target = str(parsed.get("target", ""))
                        matched = resolve_target_name(raw_target, valid_targets)
                        if not matched:
                            # Also check if exactly one valid target was mentioned in speech
                            mentioned = [t for t in valid_targets if t.lower() in speech.lower()]
                            if len(mentioned) == 1:
                                matched = mentioned[0]
                        if not matched:
                            last_error = (
                                f"Поле `target` («{raw_target}») не совпадает ни с одним допустимым именем: "
                                f"{', '.join(valid_targets)}."
                            )
                        else:
                            return {
                                "thought": thought or "Действую по ситуации за столом.",
                                "speech": speech,
                                "target": matched,
                            }
                    else:
                        return {
                            "thought": thought or "Присматриваюсь к реакции игроков.",
                            "speech": speech,
                        }

                # Prepare correction prompt for retry
                logger.warning(
                    "Validation failed for %s (attempt %d): %s", player.name, attempt + 1, last_error
                )
                messages.append({"role": "assistant", "content": raw_content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"ОШИБКА В ОТВЕТЕ: {last_error}\n"
                            "Исправь ответ! Верни валидный JSON строго по схеме, где `speech` написан "
                            "ТОЛЬКО на русском языке (кириллицей, без английских слов и иероглифов)."
                        ),
                    }
                )
            except Exception as exc:
                last_error = str(exc)
                logger.error("Ollama request error for %s (attempt %d): %s", player.name, attempt + 1, exc)

        # Safe Russian in-character fallback so the game never freezes
        return self._build_fallback(player, valid_targets, fallback_kind, last_error)

    def _build_fallback(
        self,
        player: Player,
        valid_targets: Optional[list[str]],
        fallback_kind: str,
        reason: str,
    ) -> dict[str, str]:
        verb_sure = "уверен" if player.gender == Gender.MALE else "уверена"
        verb_ready = "готов" if player.gender == Gender.MALE else "готова"
        target = random.choice(valid_targets) if valid_targets else ""

        if fallback_kind == "intro":
            speech = f"Всем привет. Посмотрим, кто как будет рассуждать — спешить с выводами я пока не {verb_ready}."
        elif fallback_kind == "vote" and target:
            speech = f"Сейчас для меня подозрительнее остальных выглядит {target}, голосую против."
        elif fallback_kind == "night" and target:
            speech = f"Предлагаю этой ночью убрать {target}, чтобы спутать городу карты."
        else:
            speech = f"Пока аргументов мало, я не до конца {verb_sure}, но советую внимательнее следить за тем, кто кого выгораживает."

        result = {
            "thought": f"(Резервный ход из-за сбоя генерации: {reason})",
            "speech": speech,
        }
        if valid_targets is not None:
            result["target"] = target
        return result
