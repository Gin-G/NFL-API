#!/usr/bin/env python3
"""
Projections API Router

Serves weekly fantasy projections (mean + floor/median/ceiling) from the
`player_projections` table, which is pre-computed by the
scripts.compute_projections job, plus the prospective accuracy record from
`projection_accuracy` (scripts.score_projections). Read-only and DB-only — no
model training happens at request time.
"""

import logging
import math
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database.models import AnalyticsJobStatus, PlayerProjection, ProjectionAccuracy
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


def _pearson(xs, ys) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def _accuracy_stats(rows) -> dict:
    """MAE / bias / correlation / band coverage for a set of scored rows, plus
    the naive trailing-average baseline they should be judged against."""
    if not rows:
        return {"n": 0}

    errors = [r.abs_error for r in rows if r.abs_error is not None]
    signed = [r.error for r in rows if r.error is not None]
    projected = [r.projected_points for r in rows if r.projected_points is not None
                 and r.actual_points is not None]
    actual = [r.actual_points for r in rows if r.projected_points is not None
              and r.actual_points is not None]
    banded = [r.in_band for r in rows if r.in_band is not None]
    naive = [r.naive_abs_error for r in rows if r.naive_abs_error is not None]

    out = {
        "n": len(rows),
        "mae": round(sum(errors) / len(errors), 3) if errors else None,
        # positive bias = we projected too high
        "bias": round(sum(signed) / len(signed), 3) if signed else None,
        "correlation": None,
        "mean_projected": round(sum(projected) / len(projected), 2) if projected else None,
        "mean_actual": round(sum(actual) / len(actual), 2) if actual else None,
        "band_coverage": round(sum(banded) / len(banded), 3) if banded else None,
        "naive_mae": round(sum(naive) / len(naive), 3) if naive else None,
        "naive_n": len(naive),
    }
    corr = _pearson(projected, actual)
    if corr is not None:
        out["correlation"] = round(corr, 3)
    if out["mae"] is not None and out["naive_mae"] is not None:
        # How much the model beats "just average his last 5 games". Backtests
        # put this near 0.05 — small, and the honest headline number.
        model_mae_on_naive_rows = sum(
            r.abs_error for r in rows if r.naive_abs_error is not None
        ) / len(naive)
        out["skill_over_naive"] = round(out["naive_mae"] - model_mae_on_naive_rows, 3)
    return out


@router.get("/accuracy")
def get_projection_accuracy(
    season: Optional[int] = Query(None, description="Season (default: current)"),
    week: Optional[int] = Query(None, description="Single week (default: all scored)"),
    position: Optional[str] = Query(None, description="QB/RB/WR/TE"),
    min_projected: float = Query(0.0, description="Only rows we projected at least this high"),
    db: Session = Depends(get_db),
):
    """How the projections actually did — scored after the fact, never re-scored.

    Unlike a backtest, every row here was computed before the game was played.
    Returns overall accuracy plus by-position and by-week breakdowns, each with
    the naive trailing-5-game baseline for comparison.
    """
    season = season or get_current_nfl_season()
    q = db.query(ProjectionAccuracy).filter(ProjectionAccuracy.season == season)
    if week is not None:
        q = q.filter(ProjectionAccuracy.week == week)
    if position:
        q = q.filter(ProjectionAccuracy.position == position.upper())
    if min_projected:
        q = q.filter(ProjectionAccuracy.projected_points >= min_projected)
    rows = q.all()

    if not rows:
        return {
            "status": "no_data",
            "season": season,
            "week": week,
            "message": (
                "No scored projections for this query. Projections are scored "
                "once a week's actuals load — run scripts.score_projections "
                "(k8s/projection-accuracy-cronjob.yaml)."
            ),
        }

    by_position, by_week = {}, {}
    for r in rows:
        by_position.setdefault(r.position or "UNK", []).append(r)
        by_week.setdefault(r.week, []).append(r)

    return {
        "status": "success",
        "season": season,
        "week": week,
        "overall": _accuracy_stats(rows),
        "by_position": {pos: _accuracy_stats(rs)
                        for pos, rs in sorted(by_position.items())},
        "by_week": {str(wk): _accuracy_stats(rs)
                    for wk, rs in sorted(by_week.items())},
        "weeks_scored": sorted(by_week),
        "last_scored_at": max(
            (r.scored_at for r in rows if r.scored_at), default=None
        ),
    }


@router.get("/accuracy/misses")
def get_projection_misses(
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(None),
    position: Optional[str] = Query(None),
    direction: str = Query("both", description="over | under | both"),
    limit: int = Query(25, le=200),
    db: Session = Depends(get_db),
):
    """The biggest scored misses — where the model is wrong and in which
    direction. `over` = we projected too high, `under` = too low."""
    season = season or get_current_nfl_season()
    q = db.query(ProjectionAccuracy).filter(ProjectionAccuracy.season == season)
    if week is not None:
        q = q.filter(ProjectionAccuracy.week == week)
    if position:
        q = q.filter(ProjectionAccuracy.position == position.upper())
    if direction == "over":
        q = q.filter(ProjectionAccuracy.error > 0)
    elif direction == "under":
        q = q.filter(ProjectionAccuracy.error < 0)

    rows = q.order_by(ProjectionAccuracy.abs_error.desc()).limit(limit).all()
    if not rows:
        return {"status": "no_data", "season": season, "data": []}
    return {
        "status": "success",
        "season": season,
        "direction": direction,
        "count": len(rows),
        "data": [_orm_to_dict(r) for r in rows],
    }


@router.get("/accuracy/status")
def get_accuracy_job_status(db: Session = Depends(get_db)):
    """Status of the most recent projection-scoring job."""
    job = (
        db.query(AnalyticsJobStatus)
        .filter(AnalyticsJobStatus.job_type == "projection_accuracy")
        .order_by(AnalyticsJobStatus.id.desc())
        .first()
    )
    if job is None:
        return {"status": "no_job",
                "message": "No projection-scoring job has been run yet."}
    return {
        "status": job.status,
        "job_id": job.id,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "season": job.current_season,
        "weeks_processed": job.total_entries or 0,
        "player_weeks_scored": job.processed_entries or 0,
        "already_scored": job.skipped_entries or 0,
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
