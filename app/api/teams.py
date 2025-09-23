#!/usr/bin/env python3
"""
Teams API Router
Handles all team-related endpoints
"""

from fastapi import APIRouter, HTTPException
import nfl_data_py as nfl
import logging
from .utils import clean_data_for_json

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def get_teams():
    """Get all NFL teams"""
    try:
        teams_data = nfl.import_team_desc()
        cleaned_data = clean_data_for_json(teams_data)
        return {
            "status": "success",
            "total_teams": len(teams_data),
            "data": cleaned_data
        }
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{team_abbr}")
async def get_team_details(team_abbr: str):
    """Get details for a specific team"""
    try:
        teams_data = nfl.import_team_desc()
        team = teams_data[teams_data['team_abbr'] == team_abbr.upper()]
        
        if team.empty:
            raise HTTPException(status_code=404, detail=f"Team {team_abbr} not found")
        
        cleaned_data = clean_data_for_json(team)
        return {
            "status": "success",
            "data": cleaned_data[0] if cleaned_data else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching team {team_abbr}: {e}")
        raise HTTPException(status_code=500, detail=str(e))