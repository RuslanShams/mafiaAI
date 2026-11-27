from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import Awaitable, Callable, Optional

import httpx

from backend.config import (
    LLM_KEEP_ALIVE,
    LLM_MAX_RETRIES,
    LLM_NUM_BATCH,
    LLM_NUM_CTX,
    LLM_NUM_THREADS,
    LLM_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PREDICT_TOKENS_DISCUSSION,
    PREDICT_TOKENS_INTRO,
    PREDICT_TOKENS_NIGHT,
    PREDICT_TOKENS_VOTE,
)
from backend.models import Gender, Player, Role
from backend.prompts import build_name_mention_regex

logger = logging.getLogger("mafia.llm")

StreamCallback = Callable[[str], Awaitable[None]]

# Regex patterns for strict Russian-only & in-character output validation
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]+")
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
SILENCE_ACCUSATION_RE = re.compile(
    r"\b(?:он\s+сегодня\s+молчит|она\s+сегодня\s+молчит|просто\s+молчит|слишком\s+молчит|почему\s+(?:ты\s+|он\s+|она\s+)?молчи(?:т|шь)|молчалив\w*|отмалчива\w*|мало\s+говори(?:т|шь)|за\s+молчание|в\s+молчании|его\s+молчани\w*|её\s+молчани\w*|ее\s+молчани\w*)\b",
    re.IGNORECASE,
)
TARGET_LINE_RE = re.compile(r"(?:ЦЕЛЬ|ГОЛОС|ЖЕРТВА|TARGET)\s*[:\-—]\s*([^\n|]+)", re.IGNORECASE)
SPEECH_LINE_RE = re.compile(
    r"(?:ПРИЧИНА|РЕЧЬ|КОММЕНТАРИЙ|ФРАЗА|SPEECH)\s*[:\-—]\s*([^\n]+)",
    re.IGNORECASE,
)
EXPLICIT_VOTE_PHRASE_RE = re.compile(
    r"(?:голо[св]у[юем]+\s+(?:против|за)|мой\s+голос\s*(?:[:\-—]|против|за)?|убира[юем]+)\s+([А-ЯЁа-яё]{3,16})",
    re.IGNORECASE,
)


def clean_speech_text(text: str, speaker_name: str = "") -> str:
    """Normalize whitespace, strip speaker prefixes, vote-tag echoes, robotic tags, and self-address."""
    cleaned = text.strip()
    cleaned = re.sub(r"[*_`]+", "", cleaned)
    # Remove echoed "[голос: ...]" or "[Голос против: ...]" tags inside speech
    cleaned = re.sub(r"\[\s*голос[^\]]*\]\s*[—\-:]?\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"^(?:ПРИЧИНА|РЕЧЬ|ОТВЕТ|РЕПЛИКА)\s*[:\-—]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if speaker_name:
        cleaned = re.sub(rf"^{re.escape(speaker_name)}\s*[:\-—]\s*", "", cleaned, flags=re.IGNORECASE)
        # Strip accidental self-address at start like "Светлана, ну я..."
        self_addr_re = re.compile(rf"^{re.escape(speaker_name)}\s*,\s*", re.IGNORECASE)
        if self_addr_re.search(cleaned):
            cleaned = self_addr_re.sub("", cleaned)
            if cleaned:
                cleaned = cleaned[0].upper() + cleaned[1:]
    # Strip any stray "<Name>: " prefix (e.g. "Тетяна: Ну что...")
    cleaned = re.sub(r"^[А-ЯЁа-яё]{3,15}\s*:\s*(?=[А-ЯЁ«\"])", "", cleaned)
    # Convert robotic "Вопрос к Алексею: ..." into natural address
    cleaned = re.sub(
        r"(?:Задаю\s+вопрос|Мой\s+вопрос|Вопрос)\s+(?:к\s+|игроку\s+)?([А-ЯЁ][а-яё]+)\s*[:\-—]\s*[\"«]?",
        r"\1, ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\bСпрашиваю\s*[:\-—]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bголовую\b", "голосую", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bправая\s+насч[её]т\b", "права насчёт", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Приветствую[.!,]\s*", "Привет! ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" —-")
    if len(cleaned) >= 2 and (
        (cleaned.startswith("«") and cleaned.endswith("»"))
        or (cleaned.startswith('"') and cleaned.endswith('"'))
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def trim_incomplete_tail(text: str) -> str:
    """If token limit cut a second/third sentence mid-word, trim cleanly to last punctuation."""
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    # Strip dangling conjunction/preposition right before a period or end of string
    cleaned = re.sub(
        r"[\s,;:\-—]+(?:и|а|но|что|как|или|в|на|к|с|о|об|за|из|от|по|у|для|чтобы|потому|так|это)[.…]*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if not cleaned:
        return text.strip()
    if cleaned[-1] in ".!?…":
        return cleaned
    last_punct = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"), cleaned.rfind("…"))
    if last_punct >= 18:
        return cleaned[: last_punct + 1].strip()
    return cleaned.rstrip(",;:-— ") + "."


def soft_sanitize_speech(
    speech: str,
    speaker_name: str = "",
    allowed_names: Optional[list[str]] = None,
) -> str:
    """Fast 0ms cleanup of minor CJK or stray single Latin glitches so we avoid expensive retries."""
    text = clean_speech_text(speech, speaker_name=speaker_name)
    text = CJK_RE.sub("", text)

    # Check if there are stray Latin words not belonging to player names; if mostly Cyrillic, strip them
    cyr_count = len(re.findall(r"[А-Яа-яЁё]", text))
    lat_words = LATIN_WORD_RE.findall(text)
    allowed_lower = {n.lower() for n in (allowed_names or []) if n}
    bad_lat = [w for w in lat_words if w.lower() not in allowed_lower]

    if bad_lat and cyr_count >= 20 and len("".join(bad_lat)) <= 12:
        for w in bad_lat:
            text = re.sub(rf"\b{re.escape(w)}\b", "", text)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" ,;-")

    return trim_incomplete_tail(text)


def _check_dead_player_misuse(speech: str, dead_names: list[str]) -> Optional[str]:
    """Detect if an AI addresses a dead player or accuses/describes them as still alive."""
    for dname in dead_names:
        if not dname:
            continue
        stem_pat = build_name_mention_regex(dname).pattern
        # 1) Direct vocative address to dead player ("Павел, ты...", "..., Павел?")
        if re.search(rf"(?:^\s*{stem_pat}\s*,|,\s*{stem_pat}\s*[?!.])", speech, re.IGNORECASE):
            return f"Игрок {dname} УЖЕ МЁРТВ! Нельзя обращаться к нему — говори только с живыми."
        # 2) Present-tense suspicion or state ("Павел нервничает / скрывает / темнит / переживает", "подозреваю Павла")
        if re.search(
            rf"(?:{stem_pat}\s+(?:действительно\s+|сейчас\s+|явно\s+|слишком\s+|просто\s+)?(?:нервнича|пережива|скрыва|темнит|молчит|подозрител|пытается\s+скрыть|мафия)"
            rf"|(?:подозреваю|голосую\s+против|спрошу\s+у|вопрос\s+к)\s+{stem_pat}"
            rf"|насч[её]т\s+{stem_pat}\b[^.!?]*(?:скрыть|подозр|нервн))",
            speech,
            re.IGNORECASE,
        ):
            return (
                f"Игрок {dname} УЖЕ МЁРТВ и выбыл из игры! Нельзя его подозревать или обсуждать его текущее поведение — "
                "ищи мафию ТОЛЬКО среди ЖИВЫХ игроков!"
            )
    return None


def _check_unspoken_player_hallucination(speech: str, unspoken_names: list[str]) -> Optional[str]:
    """Reject claims that an alive player whose turn hasn't come yet 'talks a lot', 'is active/nervous', etc."""
    for uname in unspoken_names:
        if not uname:
            continue
        stem_pat = build_name_mention_regex(uname).pattern
        if not re.search(stem_pat, speech, re.IGNORECASE):
            continue
        # Only reject 3rd-person attributions of speech/activity/nervousness to someone who hasn't spoken yet
        if re.search(
            rf"(?:{stem_pat}\s+(?:сегодня\s+|немного\s+|слишком\s+|очень\s+|явно\s+|действительно\s+)?(?:много\s+говорит|активн\w*|пытается\s+привлечь|нервнича\w*|взволнован\w*|неадекват\w*|оправдыва\w*|путает\w*|выгоражива\w*|молчит)"
            rf"|(?:насч[её]т|в\s+поведении)\s+{stem_pat})",
            speech,
            re.IGNORECASE,
        ):
            return (
                f"Ошибка фактов: до игрока {uname} очередь в этом круге ЕЩЁ НЕ ДОШЛА (он ещё не говорил)! "
                "Не оценивай его поведение — прокомментируй тех, кто уже высказался, или просто задай ему прямой вопрос!"
            )
    return None


def validate_russian_speech(
    speech: str,
    player: Player,
    is_public_day: bool = True,
    allowed_names: Optional[list[str]] = None,
    dead_names: Optional[list[str]] = None,
    unspoken_names: Optional[list[str]] = None,
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
    # Block 3rd-person self-reference
    if player.name and build_name_mention_regex(player.name).search(speech):
        return (
            f"Запрещено называть своё собственное имя «{player.name}» в третьем лице! "
            "Перепиши строго от первого лица («я», «мне», «обо мне»)."
        )
    if is_public_day and SILENCE_ACCUSATION_RE.search(speech):
        return (
            "В этой игре все говорят строго по очереди — ЗАПРЕЩЕНО обвинять кого-либо в молчании, "
            "молчаливости или в том, что он мало говорит! Оценивай смысл сказанных слов."
        )
    if is_public_day and dead_names:
        dead_err = _check_dead_player_misuse(speech, dead_names)
        if dead_err:
            return dead_err
    if is_public_day and unspoken_names:
        unspoken_err = _check_unspoken_player_hallucination(speech, unspoken_names)
        if unspoken_err:
            return unspoken_err
    if not re.search(r"[А-Яа-яЁё]{2,}", speech):
        return "Реплика должна содержать осмысленный текст на русском языке."
    return None


def resolve_target_name(raw_target: str, valid_targets: list[str]) -> Optional[str]:
    """Match LLM target name against valid alive target names (including Russian case stems)."""
    if not raw_target:
        return None
    cleaned = raw_target.strip().strip("«»\"'.,!?:;[]()*<> ")
    lower = cleaned.lower()
    for name in valid_targets:
        if name.lower() == lower:
            return name
    for name in valid_targets:
        if name.lower() in lower:
            return name
    for name in valid_targets:
        stem = name[: max(3, len(name) - 2)].lower()
        if stem in lower:
            return name
    return None


def reconcile_target_with_speech(
    matched_target: Optional[str],
    speech: str,
    valid_targets: list[str],
) -> Optional[str]:
    """
    Ensure the vote/kill target ALWAYS matches who the AI actually accused in `speech`.
    Fixes cases like:
    - Speech: 'Я думаю, что голосуем за Дарью.' with ГОЛОС: Светлана -> overrides to Дарья!
    - Speech: 'Светлана права. Николай вызывает вопросы.' with ГОЛОС: Руслан -> overrides to Николай!
    """
    if not speech:
        return matched_target

    # 1. Explicit "голосую против X" / "голосуем за X" inside speech always wins!
    explicit_m = EXPLICIT_VOTE_PHRASE_RE.search(speech)
    if explicit_m:
        explicit_resolved = resolve_target_name(explicit_m.group(1), valid_targets)
        if explicit_resolved:
            return explicit_resolved

    # 2. Check candidates explicitly accused/distrusted in speech (never override to a defended/praised player!)
    scrubbed_positive = re.sub(
        r"[А-ЯЁа-яё]+\s+прав[аы]?\b|(?:соглас(?:ен|на)|поддерживаю|выслушать|верю)\s+(?:с\s+)?[А-ЯЁа-яё]+",
        "",
        speech,
        flags=re.IGNORECASE,
    )
    negative_context = re.search(
        r"(подозр|вопрос|не\s+доверя|странн|темнит|скрывает|мафи|осторож|навязчив|сбить|путает|не\s+верю)",
        scrubbed_positive,
        flags=re.IGNORECASE,
    )
    suspects_in_speech = [
        t for t in valid_targets if build_name_mention_regex(t).search(scrubbed_positive)
    ]

    if suspects_in_speech and (negative_context or matched_target is None):
        if matched_target not in suspects_in_speech:
            return suspects_in_speech[0]

    return matched_target


def parse_target_and_speech(
    raw_text: str,
    valid_targets: list[str],
    speaker_name: str = "",
) -> tuple[Optional[str], str]:
    """Parse `ПРИЧИНА: ...\nГОЛОС: ...` and enforce 100% consistency between speech and target."""
    stripped = raw_text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                t = resolve_target_name(str(obj.get("target", "")), valid_targets)
                s = clean_speech_text(str(obj.get("speech", "")), speaker_name=speaker_name)
                return reconcile_target_with_speech(t, s, valid_targets), s
        except json.JSONDecodeError:
            pass

    target_match = TARGET_LINE_RE.search(stripped)
    speech_match = SPEECH_LINE_RE.search(stripped)

    matched_target: Optional[str] = None
    if target_match:
        matched_target = resolve_target_name(target_match.group(1), valid_targets)

    if speech_match:
        speech_part = speech_match.group(1).strip()
    else:
        lines = [
            ln.strip()
            for ln in stripped.splitlines()
            if ln.strip() and not TARGET_LINE_RE.match(ln.strip())
        ]
        speech_part = " ".join(lines) if lines else stripped

    cleaned_speech = clean_speech_text(speech_part, speaker_name=speaker_name)
    reconciled_target = reconcile_target_with_speech(matched_target, cleaned_speech, valid_targets)

    if not reconciled_target:
        mentioned = [t for t in valid_targets if build_name_mention_regex(t).search(stripped)]
        if mentioned:
            reconciled_target = mentioned[0]

    return reconciled_target, cleaned_speech


def _format_size_ru(size_bytes: int) -> str:
    if not size_bytes or size_bytes <= 0:
        return ""
    gb = size_bytes / (1024**3)
    if gb >= 1.0:
        return f"{gb:.1f} ГБ"
    mb = size_bytes / (1024**2)
    return f"{round(mb)} МБ"


def _predict_budget_for_kind(kind: str) -> int:
    if kind == "intro":
        return PREDICT_TOKENS_INTRO
    if kind == "vote":
        return PREDICT_TOKENS_VOTE
    if kind == "night":
        return PREDICT_TOKENS_NIGHT
    return PREDICT_TOKENS_DISCUSSION


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None
        self._warmup_task: Optional[asyncio.Task] = None

    def set_model(self, model_name: str) -> None:
        cleaned = model_name.strip()
        if cleaned:
            self.model = cleaned

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS)
        return self._client

    async def close(self) -> None:
        if self._warmup_task and not self._warmup_task.done():
            self._warmup_task.cancel()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def schedule_warmup(self) -> None:
        """Fire-and-forget warmup so Ollama loads the chosen model with num_ctx=2048 while in lobby."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._warmup_task and not self._warmup_task.done():
            return
        self._warmup_task = loop.create_task(self._warmup_impl())

    async def _warmup_impl(self) -> None:
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Привет"}],
                "stream": False,
                "keep_alive": LLM_KEEP_ALIVE,
                "options": {
                    "num_ctx": LLM_NUM_CTX,
                    "num_thread": LLM_NUM_THREADS,
                    "num_batch": LLM_NUM_BATCH,
                    "num_predict": 1,
                },
            }
            await self._get_client().post(f"{self.base_url}/api/chat", json=payload, timeout=30.0)
        except Exception:
            pass

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

            if models and self.model not in models:
                if "gemma3:4b" in models:
                    self.model = "gemma3:4b"
                elif OLLAMA_MODEL in models:
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
        dead_names: Optional[list[str]] = None,
        unspoken_names: Optional[list[str]] = None,
        on_stream_text: Optional[StreamCallback] = None,
    ) -> dict[str, str]:
        """
        Stream fast plain-text turns from Ollama (no GBNF JSON overhead, KV-cache friendly).
        """
        t_start = time.monotonic()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        num_predict = _predict_budget_for_kind(fallback_kind)
        can_live_stream = on_stream_text is not None and valid_targets is None

        last_error = ""
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "keep_alive": LLM_KEEP_ALIVE,
                    "options": {
                        "num_ctx": LLM_NUM_CTX,
                        "num_thread": LLM_NUM_THREADS,
                        "num_batch": LLM_NUM_BATCH,
                        "num_predict": num_predict,
                        "temperature": 0.65 if attempt == 0 else 0.4,
                        "top_p": 0.9,
                        "repeat_penalty": 1.12,
                        "stop": ["\n\n", "---", "ЗАДАНИЕ:", "[Ведущий]"],
                    },
                }

                raw_chunks: list[str] = []
                last_emit_ts = 0.0

                async with self._get_client().stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk_data = json.loads(line)
                        piece = chunk_data.get("message", {}).get("content", "")
                        if piece:
                            raw_chunks.append(piece)
                            if can_live_stream and on_stream_text is not None:
                                now = time.monotonic()
                                if now - last_emit_ts >= 0.08:
                                    preview = clean_speech_text("".join(raw_chunks), speaker_name=player.name)
                                    # Hold stream if Mafia confesses or speaker slips into 3rd-person self-name
                                    has_self_3rd = bool(
                                        player.name and build_name_mention_regex(player.name).search(preview)
                                    )
                                    has_dead_err = bool(
                                        is_public_day and dead_names and _check_dead_player_misuse(preview, dead_names)
                                    )
                                    has_unspoken_err = bool(
                                        is_public_day
                                        and unspoken_names
                                        and _check_unspoken_player_hallucination(preview, unspoken_names)
                                    )
                                    has_silence_err = bool(
                                        is_public_day and SILENCE_ACCUSATION_RE.search(preview)
                                    )
                                    if (
                                        preview
                                        and not has_self_3rd
                                        and not has_silence_err
                                        and not has_dead_err
                                        and not has_unspoken_err
                                        and not (
                                            is_public_day
                                            and player.role == Role.MAFIA
                                            and MAFIA_CONFESSION_RE.search(preview)
                                        )
                                    ):
                                        last_emit_ts = now
                                        await on_stream_text(preview)
                        if chunk_data.get("done"):
                            break

                raw_content = "".join(raw_chunks)

                if valid_targets is not None:
                    matched_target, raw_speech = parse_target_and_speech(
                        raw_content, valid_targets, speaker_name=player.name
                    )
                    speech = soft_sanitize_speech(
                        raw_speech, speaker_name=player.name, allowed_names=allowed_names
                    )
                    if not speech or len(speech) < 4:
                        speech = (
                            f"Голосую против {matched_target}."
                            if fallback_kind == "vote" and matched_target
                            else f"Убираем {matched_target}."
                        )
                    err = validate_russian_speech(
                        speech,
                        player=player,
                        is_public_day=is_public_day,
                        allowed_names=allowed_names,
                        dead_names=dead_names,
                        unspoken_names=unspoken_names,
                    )
                    if err:
                        last_error = err
                    elif not matched_target:
                        last_error = f"Цель не распознана из списка: {', '.join(valid_targets)}."
                    else:
                        elapsed = time.monotonic() - t_start
                        logger.info(
                            "%s (%s) completed %s in %.2fs", player.name, self.model, fallback_kind, elapsed
                        )
                        return {
                            "thought": f"Выбор цели: {matched_target} ({elapsed:.1f} с)",
                            "speech": speech,
                            "target": matched_target,
                        }
                else:
                    speech = soft_sanitize_speech(
                        raw_content, speaker_name=player.name, allowed_names=allowed_names
                    )
                    err = validate_russian_speech(
                        speech,
                        player=player,
                        is_public_day=is_public_day,
                        allowed_names=allowed_names,
                        dead_names=dead_names,
                        unspoken_names=unspoken_names,
                    )
                    if not err:
                        elapsed = time.monotonic() - t_start
                        logger.info(
                            "%s (%s) completed %s in %.2fs", player.name, self.model, fallback_kind, elapsed
                        )
                        return {
                            "thought": f"Ход сгенерирован за {elapsed:.1f} с",
                            "speech": speech,
                        }
                    last_error = err

                logger.warning(
                    "Validation retry for %s (attempt %d): %s", player.name, attempt + 1, last_error
                )
                messages.append({"role": "assistant", "content": raw_content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Ошибка: {last_error} Исправь и ответь кратко ТОЛЬКО на русском языке.",
                    }
                )
            except Exception as exc:
                last_error = str(exc)
                logger.error("Ollama error for %s (attempt %d): %s", player.name, attempt + 1, exc)

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
            speech = f"Всем привет, спешить с выводами я пока не {verb_ready}."
        elif fallback_kind == "vote" and target:
            speech = f"Сейчас для меня подозрительнее остальных выглядит {target}, голосую против."
        elif fallback_kind == "night" and target:
            speech = f"Убираем этой ночью {target}, чтобы запутать стол."
        elif fallback_kind == "last_word":
            speech = "Удачи мирному городу за столом — внимательно сверяйте чужие аргументы с тем, кто против кого голосовал."
        else:
            speech = f"Пока я не до конца {verb_sure} в чужих аргументах, но советую следить за тем, кто кого выгораживает."

        result = {
            "thought": f"(Резервный ответ: {reason})",
            "speech": speech,
        }
        if valid_targets is not None:
            result["target"] = target
        return result
