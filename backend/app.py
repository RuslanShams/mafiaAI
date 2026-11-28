from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.llm_client import OllamaClient
from backend.models import Phase
from backend.session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mafia.app")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

llm_client = OllamaClient()
session_manager = SessionManager(llm=llm_client)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await session_manager.shutdown_all()
    await llm_client.close()


app = FastAPI(title="Мафия AI (Ollama Multi-Session)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.middleware("http")
async def disable_static_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/health")
async def api_health() -> dict:
    await session_manager.cleanup_stale_sessions()
    health = await llm_client.check_health()
    health["slots"] = session_manager.slots_summary()
    return health


@app.websocket("/ws/game")
async def ws_game(websocket: WebSocket) -> None:
    await session_manager.cleanup_stale_sessions()
    raw_sid = websocket.query_params.get("session_id")
    session = session_manager.get_or_create(raw_sid)
    engine = session.engine

    await session.sockets.connect(websocket)
    try:
        health = await llm_client.check_health()
        if engine.state.phase == Phase.LOBBY and health.get("available_models"):
            locked = session_manager.active_locked_model(exclude_sid=session.session_id)
            if locked and locked in health["available_models"]:
                engine.state.ai_model = locked
            elif engine.state.ai_model not in health["available_models"]:
                engine.state.ai_model = health["configured_model"]

        await websocket.send_json(
            {
                "type": "session",
                "session_id": session.session_id,
                "slots": session_manager.slots_summary(),
            }
        )
        await websocket.send_json({"type": "state", "state": engine.serialize_for_human()})
        await websocket.send_json({"type": "health", "health": health})

        if health.get("ollama_online") and health.get("model_ready") and engine.state.phase == Phase.LOBBY:
            llm_client.schedule_warmup(engine.state.ai_model)

        while True:
            data = await websocket.receive_json()
            session.touch()
            action = data.get("action")

            if action == "check_health":
                health = await llm_client.check_health()
                await websocket.send_json({"type": "health", "health": health})
                await websocket.send_json({"type": "slots", "slots": session_manager.slots_summary()})

            elif action == "select_model":
                if engine.state.phase in (Phase.LOBBY, Phase.GAME_OVER):
                    locked = session_manager.active_locked_model(exclude_sid=session.session_id)
                    chosen = locked or str(data.get("model", "")).strip()
                    if chosen:
                        if not session_manager.active_playing_sessions():
                            llm_client.set_model(chosen)
                        engine.state.ai_model = chosen
                        health = await llm_client.check_health()
                        await session.sockets.broadcast({"type": "health", "health": health})
                        await engine.push_state()
                        if health.get("ollama_online") and health.get("model_ready") and not locked:
                            llm_client.schedule_warmup(chosen)

            elif action == "start_game":
                can_start, err_msg = await session_manager.can_start_game(session)
                if not can_start:
                    await websocket.send_json({"type": "error", "message": err_msg})
                    await websocket.send_json({"type": "slots", "slots": session_manager.slots_summary()})
                    continue

                player_name = str(data.get("name", "Игрок"))
                player_gender = str(data.get("gender", "M"))
                locked = session_manager.active_locked_model(exclude_sid=session.session_id)
                chosen_model = locked or str(data.get("model", "")).strip() or None
                if chosen_model and not locked:
                    llm_client.set_model(chosen_model)

                await engine.start_new_game(
                    human_name=player_name,
                    human_gender=player_gender,
                    ai_model=chosen_model,
                )
                await session_manager.broadcast_slots_to_all()

            elif action == "reset_lobby":
                await engine.reset_to_lobby()
                await session_manager.broadcast_slots_to_all()

            elif action in ("speak", "last_word", "vote", "night_kill", "night_heal", "night_check"):
                ok = engine.submit_human_input(
                    action_type=action,
                    text=str(data.get("text", "")),
                    target=str(data.get("target", "")),
                )
                if not ok:
                    await websocket.send_json(
                        {"type": "error", "message": "Сейчас это действие недоступно или неверно заполнено."}
                    )

    except WebSocketDisconnect:
        session.sockets.disconnect(websocket)
        await session_manager.cleanup_stale_sessions()
        await session_manager.broadcast_slots_to_all()
    except Exception as exc:
        logger.warning("WebSocket connection closed with error: %s", exc)
        session.sockets.disconnect(websocket)
        await session_manager.cleanup_stale_sessions()
        await session_manager.broadcast_slots_to_all()
