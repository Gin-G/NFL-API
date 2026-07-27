#!/usr/bin/env python3
"""
Teams API Router
Handles all team-related endpoints.

DB-first: queries the `teams` table; falls back to nflreadpy when the table
is empty (e.g. before the initial data load).
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session
import nflreadpy as nfl
import logging
from .utils import clean_data_for_json, _to_pandas, _orm_to_dict
from database.session import get_db
from database.models import Team as TeamModel, DepthChart, SnapCount

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


@router.get("/{team_abbr}/depth-chart")
def get_team_depth_chart(
    team_abbr: str,
    season: int,
    week: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Get depth chart for a team in a given season (defaults to latest week)."""
    abbr = team_abbr.upper()
    q = db.query(DepthChart).filter(DepthChart.team == abbr, DepthChart.season == season)
    if week is not None:
        q = q.filter(DepthChart.week == week)
    else:
        max_week = db.query(func.max(DepthChart.week)).filter(
            DepthChart.team == abbr, DepthChart.season == season
        ).scalar() or 1
        q = q.filter(DepthChart.week == max_week)
    rows = q.order_by(DepthChart.position, DepthChart.depth_team).all()
    if not rows:
        return {"status": "no_data", "data": [], "message": "No depth chart data available"}
    return {
        "status": "success",
        "team": abbr,
        "season": season,
        "data": [_orm_to_dict(r) for r in rows],
    }


@router.get("/{team_abbr}/snap-counts")
def get_team_snap_counts(
    team_abbr: str,
    season: int,
    week: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Get snap counts for a team in a given season (optionally filtered by week)."""
    abbr = team_abbr.upper()
    q = db.query(SnapCount).filter(SnapCount.team == abbr, SnapCount.season == season)
    if week is not None:
        q = q.filter(SnapCount.week == week)
    rows = q.order_by(SnapCount.week, SnapCount.offense_snaps.desc()).all()
    if not rows:
        return {"status": "no_data", "data": [], "message": "No snap count data available"}
    return {
        "status": "success",
        "team": abbr,
        "season": season,
        "data": [_orm_to_dict(r) for r in rows],
    }


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
