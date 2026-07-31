#!/usr/bin/env python3
"""
Projections API Router

Serves weekly fantasy projections (mean + floor/median/ceiling) from the
`player_projections` table, which is pre-computed by the
scripts.compute_projections job. Read-only and DB-only — no model training
happens at request time.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database.models import AnalyticsJobStatus, PlayerProjection
from database.session import get_db
from .utils import _orm_to_dict, get_current_nfl_season

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
def get_projections(
    season: Optional[int] = Query(None, description="Season (default: current)"),
    week: Optional[int] = Query(None, description="Week number"),
    position: Optional[str] = Query(None, description="QB/RB/WR/TE"),
    team: Optional[str] = Query(None, description="Team abbreviation"),
    limit: int = Query(500, le=2000),
    db: Session = Depends(get_db),
):
    """Weekly projections, best-projected first. Filter by week/position/team."""
    season = season or get_current_nfl_season()
    q = db.query(PlayerProjection).filter(PlayerProjection.season == season)
    if week is not None:
        q = q.filter(PlayerProjection.week == week)
    if position:
        q = q.filter(PlayerProjection.position == position.upper())
    if team:
        q = q.filter(PlayerProjection.team == team.upper())

    rows = (
        q.order_by(PlayerProjection.projected_points.desc()).limit(limit).all()
    )
    if not rows:
        return {
            "status": "no_data",
            "season": season,
            "week": week,
            "data": [],
            "message": (
                "No projections cached for this query. Run the projections job "
                "(k8s/projections-cronjob.yaml) or check /projections/status."
            ),
        }
    return {
        "status": "success",
        "season": season,
        "week": week,
        "count": len(rows),
        "data": [_orm_to_dict(r) for r in rows],
    }


@router.get("/status")
def get_projections_status(db: Session = Depends(get_db)):
    """Status of the most recent projections pre-computation job."""
    job = (
        db.query(AnalyticsJobStatus)
        .filter(AnalyticsJobStatus.job_type == "projections")
        .order_by(AnalyticsJobStatus.id.desc())
        .first()
    )
    if job is None:
        return {"status": "no_job", "message": "No projections job has been run yet."}
    done = (job.processed_entries or 0) + (job.skipped_entries or 0)
    total = job.total_entries or 0
    pct = round(done / total * 100, 1) if total > 0 else 0.0
    return {
        "status": job.status,
        "job_id": job.id,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "total_entries": total,
        "processed_entries": job.processed_entries or 0,
        "failed_entries": job.failed_entries or 0,
        "pct_complete": pct,
        "current_season": job.current_season,
        "error_message": job.error_message,
    }


@router.get("/player/{player_id}")
def get_player_projections(
    player_id: str,
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """All cached projections for one player (most recent first)."""
    q = db.query(PlayerProjection).filter(PlayerProjection.player_id == player_id)
    if season is not None:
        q = q.filter(PlayerProjection.season == season)
    if week is not None:
        q = q.filter(PlayerProjection.week == week)
    rows = q.order_by(
        PlayerProjection.season.desc(), PlayerProjection.week.desc()
    ).all()
    if not rows:
        return {"status": "no_data", "player_id": player_id, "data": []}
    return {
        "status": "success",
        "player_id": player_id,
        "data": [_orm_to_dict(r) for r in rows],
    }


@router.get("/season/{season}")
def get_season_totals(
    season: int,
    position: Optional[str] = Query(None, description="Filter to QB/RB/WR/TE"),
    limit: int = Query(300, le=2000),
    db: Session = Depends(get_db),
):
    """Season-long projected TOTALS per player: summed fantasy points and every
    component stat (passing/rushing/receiving yards, TDs, receptions, INTs) across all
    projected weeks, plus games and per-game average. Sorted by total points."""
    from sqlalchemy import func

    P = PlayerProjection
    cols = {c: func.sum(getattr(P, c)) for c in (
        "passing_yards", "passing_tds", "passing_interceptions", "rushing_yards",
        "rushing_tds", "receiving_yards", "receptions", "receiving_tds")}
    q = (db.query(
            P.player_id, P.player_name, P.position, P.team,
            func.count(P.week).label("games"),
            func.sum(P.projected_points).label("total_points"),
            func.avg(P.projected_points).label("ppg"),
            func.sum(P.floor).label("floor_total"),
            func.sum(P.ceiling).label("ceiling_total"),
            *[v.label(k) for k, v in cols.items()],
         )
         .filter(P.season == season)
         .group_by(P.player_id, P.player_name, P.position, P.team))
    if position:
        q = q.filter(P.position == position.upper())
    rows = q.order_by(func.sum(P.projected_points).desc()).limit(limit).all()
    if not rows:
        return {"status": "no_data", "season": season, "data": []}

    def r1(v):
        return round(float(v), 1) if v is not None else None

    data = [{
        "player_id": r.player_id, "player_name": r.player_name,
        "position": r.position, "team": r.team, "games": r.games,
        "total_points": r1(r.total_points), "ppg": r1(r.ppg),
        "floor_total": r1(r.floor_total), "ceiling_total": r1(r.ceiling_total),
        "passing_yards": r1(r.passing_yards), "passing_tds": r1(r.passing_tds),
        "interceptions": r1(r.passing_interceptions),
        "rushing_yards": r1(r.rushing_yards), "rushing_tds": r1(r.rushing_tds),
        "receiving_yards": r1(r.receiving_yards), "receptions": r1(r.receptions),
        "receiving_tds": r1(r.receiving_tds),
    } for r in rows]
    return {"status": "success", "season": season, "total": len(data), "data": data}
