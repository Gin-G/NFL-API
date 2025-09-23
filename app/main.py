#!/usr/bin/env python3
"""
NFL API - FastAPI Application (Refactored)
Main application with modular router structure
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging

# Import routers
from api.teams import router as teams_router
from api.schedules import router as schedules_router
from api.players import router as players_router
from api.coaches import router as coaches_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )