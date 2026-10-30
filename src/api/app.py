"""Main FastAPI Application Entrypoint for MedAgentX."""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from config.settings import settings
from config.logging import setup_logging
from src.api.routes import router

# Setup structured logging
setup_logging()

app = FastAPI(
    title=settings.APP_NAME,
    description="Adaptive Confidence-Guided Multi-Agent Framework for Evidence-Based Clinical Decision Support",
    version="1.0.0",
)

# Register routes
app.include_router(router)

# Mount static UI assets
ui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ui")
if os.path.exists(ui_dir):
    app.mount("/static", StaticFiles(directory=ui_dir), name="static")

    @app.get("/", tags=["Dashboard UI"])
    async def serve_dashboard():
        """Serve main clinician dashboard UI."""
        return FileResponse(os.path.join(ui_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
