from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.engine import GameEngine
from backend.llm_client import OllamaClient
from backend.models import Phase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mafia.app")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict) -> None:
        stale: list[WebSocket] = []
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


ws_manager = ConnectionManager()
llm_client = OllamaClient()
game_engine = GameEngine(llm=llm_client, broadcast=ws_manager.broadcast)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await game_engine.stop_current_game()
    await llm_client.close()


app = FastAPI(title="Мафия AI (qwen2.5:7b)", lifespan=lifespan)
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
    return await llm_client.check_health()


@app.websocket("/ws/game")
async def ws_game(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    try:
        # Send initial state and Ollama status immediately on connect
        await websocket.send_json({"type": "state", "state": game_engine.serialize_for_human()})
        health = await llm_client.check_health()
        await websocket.send_json({"type": "health", "health": health})
        if health.get("ollama_online") and health.get("model_ready") and game_engine.state.phase == Phase.LOBBY:
            llm_client.schedule_warmup()

        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "check_health":
                health = await llm_client.check_health()
                await websocket.send_json({"type": "health", "health": health})

            elif action == "select_model":
                if game_engine.state.phase in (Phase.LOBBY, Phase.GAME_OVER):
                    chosen = str(data.get("model", "")).strip()
                    if chosen:
                        llm_client.set_model(chosen)
                        game_engine.state.ai_model = llm_client.model
                        health = await llm_client.check_health()
                        await ws_manager.broadcast({"type": "health", "health": health})
                        await game_engine.push_state()
                        if health.get("ollama_online") and health.get("model_ready"):
                            llm_client.schedule_warmup()

            elif action == "start_game":
                player_name = str(data.get("name", "Игрок"))
                player_gender = str(data.get("gender", "M"))
                chosen_model = str(data.get("model", "")).strip() or None
                await game_engine.start_new_game(
                    human_name=player_name,
                    human_gender=player_gender,
                    ai_model=chosen_model,
                )
                health = await llm_client.check_health()
                await ws_manager.broadcast({"type": "health", "health": health})

            elif action == "reset_lobby":
                await game_engine.reset_to_lobby()

            elif action in ("speak", "last_word", "vote", "night_kill"):
                ok = game_engine.submit_human_input(
                    action_type=action,
                    text=str(data.get("text", "")),
                    target=str(data.get("target", "")),
                )
                if not ok:
                    await websocket.send_json(
                        {"type": "error", "message": "Сейчас это действие недоступно или неверно заполнено."}
                    )

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as exc:
        logger.warning("WebSocket connection closed with error: %s", exc)
        ws_manager.disconnect(websocket)

