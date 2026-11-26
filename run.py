from __future__ import annotations

import uvicorn

from backend.config import HOST, PORT

if __name__ == "__main__":
    print(f"Запуск Мафия AI на http://{HOST}:{PORT} (только localhost)")
    uvicorn.run("backend.app:app", host=HOST, port=PORT, reload=False)

