#!/usr/bin/env python3
"""
Schedules API Router
Handles all schedule-related endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import nfl_data_py as nfl
import logging
from .utils import clean_data_for_json

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def get_schedules(
    season: Optional[int] = Query(2023, description="Season year"),
    week: Optional[int] = Query(None, description="Specific week"),
    team: Optional[str] = Query(None, description="Team abbreviation")
):
    """Get NFL schedules with optional filters"""
    try:
        schedules = nfl.import_schedules([season])
        
        if week is not None:
            schedules = schedules[schedules['week'] == week]
        
        if team is not None:
            team = team.upper()
            schedules = schedules[
                (schedules['home_team'] == team) | 
                (schedules['away_team'] == team)
            ]
        
        return {
            "status": "success",
            "season": season,
            "total_games": len(schedules),
            "data": clean_data_for_json(schedules)
        }
    except Exception as e:
        logger.error(f"Error fetching schedules: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{season}/week/{week}")
async def get_weekly_schedule(season: int, week: int):
    """Get schedule for a specific week"""
    try:
        schedules = nfl.import_schedules([season])
        week_games = schedules[schedules['week'] == week]
        
        return {
            "status": "success",
            "season": season,
            "week": week,
            "games": clean_data_for_json(week_games)
        }
    except Exception as e:
        logger.error(f"Error fetching week {week} schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))