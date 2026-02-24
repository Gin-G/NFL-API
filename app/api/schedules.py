#!/usr/bin/env python3
"""
Schedules API Router
Handles all schedule-related endpoints.

DB-first: queries the `schedules` table; falls back to nflreadpy when the
table has no data for the requested season.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
from sqlalchemy.orm import Session
import nflreadpy as nfl
import logging
from .utils import clean_data_for_json, get_current_nfl_season, _to_pandas, _orm_to_dict
from database.session import get_db
from database.models import Schedule

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def get_schedules(
    season: Optional[int] = Query(None, description="Season year (defaults to current NFL season)"),
    week: Optional[int] = Query(None, description="Specific week"),
    team: Optional[str] = Query(None, description="Team abbreviation"),
    db: Session = Depends(get_db),
):
    """Get NFL schedules with optional filters."""
    if season is None:
        season = get_current_nfl_season()
    try:
        rows = []
        try:
            q = db.query(Schedule).filter(Schedule.season == season)
            if week is not None:
                q = q.filter(Schedule.week == week)
            if team is not None:
                t = team.upper()
                q = q.filter(
                    (Schedule.home_team == t) | (Schedule.away_team == t)
                )
            rows = q.all()
        except Exception as db_err:
            logger.warning("DB unavailable, falling back to nflreadpy: %s", db_err)
        if rows:
            data = [_orm_to_dict(r) for r in rows]
            return {
                "status": "success",
                "season": season,
                "total_games": len(data),
                "data": data,
            }

        # Fallback: nflreadpy
        schedules = _to_pandas(nfl.load_schedules(seasons=[season]))

        if week is not None:
            schedules = schedules[schedules["week"] == week]

        if team is not None:
            t = team.upper()
            schedules = schedules[
                (schedules["home_team"] == t) | (schedules["away_team"] == t)
            ]

        return {
            "status": "success",
            "season": season,
            "total_games": len(schedules),
            "data": clean_data_for_json(schedules),
        }
    except Exception as e:
        logger.error("Error fetching schedules: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{season}/week/{week}")
async def get_weekly_schedule(season: int, week: int, db: Session = Depends(get_db)):
    """Get schedule for a specific week."""
    try:
        rows = []
        try:
            rows = db.query(Schedule).filter(
                Schedule.season == season,
                Schedule.week == week,
            ).all()
        except Exception as db_err:
            logger.warning("DB unavailable, falling back to nflreadpy: %s", db_err)
        if rows:
            return {
                "status": "success",
                "season": season,
                "week": week,
                "games": [_orm_to_dict(r) for r in rows],
            }

        # Fallback: nflreadpy
        schedules = _to_pandas(nfl.load_schedules(seasons=[season]))
        week_games = schedules[schedules["week"] == week]

        return {
            "status": "success",
            "season": season,
            "week": week,
            "games": clean_data_for_json(week_games),
        }
    except Exception as e:
        logger.error("Error fetching week %d schedule: %s", week, e)
        raise HTTPException(status_code=500, detail=str(e))
