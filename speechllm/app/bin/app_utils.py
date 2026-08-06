"""
Shared FastAPI app utilities (static file serving, index page).
"""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


def setup_static(app: FastAPI, static_dir: str):
    """Mount static files and add GET / route serving index.html."""
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def get_index():
        with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
