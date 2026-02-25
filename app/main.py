#!/usr/bin/env python3
"""
NFL API - FastAPI Application (Refactored)
Main application with modular router structure
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope
from datetime import datetime
import logging
import os

# Import routers
from api.teams import router as teams_router
from api.schedules import router as schedules_router
from api.players import router as players_router
from api.coaches import router as coaches_router
from api.chat import router as chat_router
from api.pbp import router as pbp_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create DB tables on startup (no-op if they already exist; gracefully skipped
# if the DB is unavailable at startup time — routers will fall back to nflreadpy)
try:
    from database.session import engine
    from database.models import Base
    Base.metadata.create_all(engine)
    logger.info("DB tables verified/created.")
except Exception as _db_startup_err:
    logger.warning("DB unavailable at startup (tables not created): %s", _db_startup_err)

# Initialize FastAPI app
app = FastAPI(
    title="NFL Analytics API",
    description="NFL data and analytics API with modular structure",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(teams_router, prefix="/teams", tags=["teams"])
app.include_router(schedules_router, prefix="/schedules", tags=["schedules"])
app.include_router(players_router, prefix="/players", tags=["players"])
app.include_router(coaches_router, prefix="/coaches", tags=["coaches"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(pbp_router, prefix="/pbp", tags=["pbp"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NFL Analytics API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "teams": "/teams",
            "schedules": "/schedules", 
            "players": "/players",
            "coaches": "/coaches",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Import here to avoid circular imports
    from api.utils import check_grading_systems
    
    systems = check_grading_systems()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "systems": systems
    }

@app.get("/debug/info")
async def debug_info():
    """Debug information about the API"""
    import sys
    import os
    from api.utils import check_grading_systems
    
    return {
        "python_path": sys.path[:3],
        "working_directory": os.getcwd(),
        "available_systems": check_grading_systems(),
        "files": {
            "functions_dir": os.path.exists("functions"),
            "players_dir": os.path.exists("functions/players"),
            "coaching_dir": os.path.exists("functions/coaching"),
            "player_grading_file": os.path.exists("functions/players/grading.py"),
            "coaching_grading_file": os.path.exists("functions/coaching/grading.py")
        }
    }

class SPAStaticFiles(StaticFiles):
    """StaticFiles that falls back to index.html for React Router deep links.

    Paths without a file extension (e.g. /dashboard/teams) are treated as
    client-side routes and served with index.html. Paths that look like
    asset requests but are missing (e.g. /dashboard/assets/gone.js) still
    return 404 so the browser is not silently handed wrong content.
    """

    async def get_response(self, path: str, scope: Scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and "." not in path.rsplit("/", 1)[-1]:
                return await super().get_response("index.html", scope)
            raise


# Mount the React dashboard (only when the build exists)
_dashboard_path = os.path.join(os.path.dirname(__file__), "static", "dashboard")
if os.path.isdir(_dashboard_path):
    app.mount("/dashboard", SPAStaticFiles(directory=_dashboard_path, html=True), name="dashboard")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )