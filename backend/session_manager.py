from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Optional

from fastapi import WebSocket

from backend.config import (
    MAX_SESSIONS,
    SESSION_ABANDONED_TTL_SEC,
    SESSION_IDLE_TTL_SEC,
    SINGLE_GPU_MODEL_LOCK,
)
from backend.engine import GameEngine
from backend.llm_client import OllamaClient
from backend.models import GameState, Phase

logger = logging.getLogger("mafia.sessions")

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{6,64}$")


class SessionSocketGroup:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []
        self.disconnected_since: Optional[float] = None

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        self.disconnected_since = None

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)
        if not self.active and self.disconnected_since is None:
            self.disconnected_since = time.monotonic()

    async def broadcast(self, message: dict) -> None:
        stale: list[WebSocket] = []
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


class GameSession:
    def __init__(self, session_id: str, llm: OllamaClient) -> None:
        self.session_id = session_id
        self.sockets = SessionSocketGroup()
        self.engine = GameEngine(llm=llm, broadcast=self.sockets.broadcast)
        self.engine.state = GameState(phase=Phase.LOBBY, ai_model=llm.model)
        self.last_activity_ts: float = time.monotonic()

    def touch(self) -> None:
        self.last_activity_ts = time.monotonic()

    @property
    def is_playing(self) -> bool:
        return self.engine.state.phase not in (Phase.LOBBY, Phase.GAME_OVER)


class SessionManager:
    """Manages up to `max_sessions` concurrent isolated Mafia tables on one server."""

    def __init__(self, llm: OllamaClient, max_sessions: int = MAX_SESSIONS) -> None:
        self.llm = llm
        self.max_sessions = max(1, max_sessions)
        self.sessions: dict[str, GameSession] = {}

    def normalize_session_id(self, raw_id: Optional[str]) -> str:
        if raw_id and SESSION_ID_RE.match(raw_id.strip()):
            return raw_id.strip()
        return uuid.uuid4().hex[:16]

    async def cleanup_stale_sessions(self) -> None:
        now = time.monotonic()
        to_remove: list[str] = []
        for sid, sess in list(self.sessions.items()):
            no_clients = len(sess.sockets.active) == 0
            disc_age = (
                (now - sess.sockets.disconnected_since)
                if (no_clients and sess.sockets.disconnected_since is not None)
                else 0.0
            )
            idle_age = now - sess.last_activity_ts

            # Lobby/finished sessions without connected tabs can be cleaned after 30s
            if no_clients and not sess.is_playing and disc_age >= 30.0:
                to_remove.append(sid)
            # Abandoned mid-game sessions without any connected browser tab for > SESSION_ABANDONED_TTL_SEC
            elif no_clients and sess.is_playing and disc_age >= SESSION_ABANDONED_TTL_SEC:
                to_remove.append(sid)
            # Very old idle lobby/finished sessions
            elif not sess.is_playing and idle_age >= SESSION_IDLE_TTL_SEC:
                to_remove.append(sid)

        for sid in to_remove:
            sess = self.sessions.pop(sid, None)
            if sess:
                logger.info("Reclaimed stale session %s", sid)
                await sess.engine.stop_current_game()

    def get_or_create(self, raw_session_id: Optional[str]) -> GameSession:
        sid = self.normalize_session_id(raw_session_id)
        sess = self.sessions.get(sid)
        if sess is None:
            sess = GameSession(session_id=sid, llm=self.llm)
            locked = self.active_locked_model()
            if locked:
                sess.engine.state.ai_model = locked
            self.sessions[sid] = sess
        sess.touch()
        return sess

    def active_playing_sessions(self, exclude_sid: Optional[str] = None) -> list[GameSession]:
        return [
            s
            for sid, s in self.sessions.items()
            if sid != exclude_sid and s.is_playing
        ]

    def active_locked_model(self, exclude_sid: Optional[str] = None) -> Optional[str]:
        """When SINGLE_GPU_MODEL_LOCK is active and another game is running, return its GPU-loaded model."""
        if not SINGLE_GPU_MODEL_LOCK:
            return None
        playing = self.active_playing_sessions(exclude_sid=exclude_sid)
        if playing:
            return playing[0].engine.state.ai_model
        return None

    async def can_start_game(self, session: GameSession) -> tuple[bool, Optional[str]]:
        await self.cleanup_stale_sessions()
        others_playing = self.active_playing_sessions(exclude_sid=session.session_id)
        if len(others_playing) >= self.max_sessions:
            return (
                False,
                f"Все {self.max_sessions} игровых стола на сервере сейчас заняты другими игроками. "
                "Подождите окончания одной из партий и попробуйте снова.",
            )
        return True, None

    def slots_summary(self) -> dict:
        active_count = len(self.active_playing_sessions())
        return {
            "max_sessions": self.max_sessions,
            "active_games": active_count,
            "free_slots": max(0, self.max_sessions - active_count),
            "locked_model": self.active_locked_model(),
        }

    async def broadcast_slots_to_all(self) -> None:
        payload = {"type": "slots", "slots": self.slots_summary()}
        for sess in list(self.sessions.values()):
            await sess.sockets.broadcast(payload)

    async def shutdown_all(self) -> None:
        for sess in list(self.sessions.values()):
            await sess.engine.stop_current_game()
        self.sessions.clear()
