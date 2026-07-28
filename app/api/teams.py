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
from .utils import (
    clean_data_for_json,
    _to_pandas,
    _orm_to_dict,
    get_current_nfl_season,
    load_schedules_df,
    load_weekly_stats_df,
)
from .fpa import build_opponent_map, compute_fpa_detail
from database.session import get_db
from database.models import Team as TeamModel, DepthChart, SnapCount, Schedule

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


@router.get("/{team_abbr}/results")
async def get_team_results(
    team_abbr: str,
    season: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Completed games for a team in a season, sorted by week.

    Same row shape as /schedules/ plus an `is_home` flag; ports NickKnows'
    per-team `{year}_{team}_schedule.csv`.
    """
    if season is None:
        season = get_current_nfl_season()
    abbr = team_abbr.upper()
    try:
        rows = []
        try:
            rows = (
                db.query(Schedule)
                .filter(
                    Schedule.season == season,
                    (Schedule.home_team == abbr) | (Schedule.away_team == abbr),
                )
                .order_by(Schedule.week)
                .all()
            )
        except Exception as db_err:
            logger.warning("DB unavailable, falling back to nflreadpy: %s", db_err)

        if rows:
            games = []
            for r in rows:
                if r.home_score is None or r.away_score is None:
                    continue  # completed games only
                d = _orm_to_dict(r)
                d["is_home"] = (r.home_team == abbr)
                games.append(d)
        else:
            sched = _to_pandas(nfl.load_schedules(seasons=[season]))
            sched = sched[
                (sched["home_team"] == abbr) | (sched["away_team"] == abbr)
            ]
            sched = sched.dropna(subset=["away_score", "home_score"]).sort_values("week")
            games = clean_data_for_json(sched)
            for g in games:
                g["is_home"] = (g.get("home_team") == abbr)

        return {
            "status": "success" if games else "no_data",
            "season": season,
            "team": abbr,
            "total_games": len(games),
            "data": games,
        }
    except Exception as e:
        logger.error("Error fetching results for %s: %s", team_abbr, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{team_abbr}/fpa")
async def get_team_fpa_detail(
    team_abbr: str,
    season: Optional[int] = None,
    position: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Per opposing player-week fantasy detail against this defense.

    The rows behind /fpa/ averages; optionally filtered to one of QB/RB/WR/TE.
    """
    if season is None:
        season = get_current_nfl_season()
    abbr = team_abbr.upper()
    try:
        sched_df = load_schedules_df(db, season)
        stats_df = load_weekly_stats_df(db, season)
    except Exception as e:
        logger.error("Error loading data for team FPA detail: %s", e)
        return {"status": "no_data", "season": season, "team": abbr,
                "total_records": 0, "data": []}

    opp_map = build_opponent_map(sched_df)
    rows = compute_fpa_detail(stats_df, opp_map, abbr, position=position)
    if not rows:
        return {"status": "no_data", "season": season, "team": abbr,
                "total_records": 0, "data": []}
    return {
        "status": "success",
        "season": season,
        "team": abbr,
        "position": position.upper() if position else None,
        "total_records": len(rows),
        "data": rows,
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
