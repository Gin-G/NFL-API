#!/usr/bin/env python3
"""
Teams API Router
Handles all team-related endpoints.

DB-first: queries the `teams` table; falls back to nflreadpy when the table
is empty (e.g. before the initial data load).
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import nflreadpy as nfl
import logging
from .utils import clean_data_for_json, _to_pandas, _orm_to_dict
from database.session import get_db
from database.models import Team as TeamModel

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def get_teams(db: Session = Depends(get_db)):
    """Get all NFL teams."""
    try:
        rows = []
        try:
            rows = db.query(TeamModel).all()
        except Exception as db_err:
            logger.warning("DB unavailable, falling back to nflreadpy: %s", db_err)
        if rows:
            data = [_orm_to_dict(r) for r in rows]
            return {
                "status": "success",
                "total_teams": len(data),
                "data": data,
            }
        # Fallback: nflreadpy
        teams_data = _to_pandas(nfl.load_teams())
        cleaned_data = clean_data_for_json(teams_data)
        return {
            "status": "success",
            "total_teams": len(teams_data),
            "data": cleaned_data,
        }
    except Exception as e:
        logger.error("Error fetching teams: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{team_abbr}")
async def get_team_details(team_abbr: str, db: Session = Depends(get_db)):
    """Get details for a specific team."""
    try:
        row = None
        try:
            row = db.query(TeamModel).filter(
                TeamModel.team_abbr == team_abbr.upper()
            ).first()
        except Exception as db_err:
            logger.warning("DB unavailable, falling back to nflreadpy: %s", db_err)
        if row:
            return {
                "status": "success",
                "data": _orm_to_dict(row),
            }

        # Fallback: nflreadpy
        teams_data = _to_pandas(nfl.load_teams())
        team = teams_data[teams_data["team_abbr"] == team_abbr.upper()]

        if team.empty:
            raise HTTPException(status_code=404, detail=f"Team {team_abbr} not found")

        cleaned_data = clean_data_for_json(team)
        return {
            "status": "success",
            "data": cleaned_data[0] if cleaned_data else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching team %s: %s", team_abbr, e)
        raise HTTPException(status_code=500, detail=str(e))
