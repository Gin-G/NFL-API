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

from database.models import (AnalyticsJobStatus, PlayerProjection,
                             PlayerProjectionVintage, ProjectionAccuracy)
from database.session import get_db
from .utils import _orm_to_dict, get_current_nfl_season

logger = logging.getLogger(__name__)
router = APIRouter()


def _regular_stats(db, season: int) -> dict:
    """``{(week, player_id): actual fanduel points}`` for one regular season.

    Scored the same way the projections are, via ``api.utils.fanduel_points``,
    so a vintage MAE is comparable with every other accuracy number the service
    reports rather than being its own private scale.
    """
    from sqlalchemy import or_

    from database.models import PlayerStat

    from .utils import fanduel_points

    rows = db.query(PlayerStat).filter(
        PlayerStat.season == season,
        or_(PlayerStat.season_type == "REG", PlayerStat.season_type.is_(None)),
    ).all()
    out = {}
    for st in rows:
        vals = {c.name: getattr(st, c.name) for c in st.__table__.columns}
        out[(st.week, st.player_id)] = fanduel_points(vals)
    return out


@router.get("/")
def get_projections(
    season: Optional[int] = Query(None, description="Season (default: current)"),
    week: Optional[int] = Query(None, description="Week number"),
    position: Optional[str] = Query(None, description="QB/RB/WR/TE"),
    team: Optional[str] = Query(None, description="Team abbreviation"),
    as_of: Optional[int] = Query(
        None, description="Historical vintage: the projection as it stood with this "
                          "many completed weeks behind it (0 = preseason). Omit for "
                          "the current projection."),
    limit: int = Query(500, le=2000),
    db: Session = Depends(get_db),
):
    """Weekly projections, best-projected first. Filter by week/position/team.

    By default this serves the CURRENT projection for each week — the most recent
    reprojection. Pass `as_of=N` to read the archive instead and get the
    projection exactly as it stood when N weeks had been played, which is what
    makes "did week 10 look different in August than it did in week 9" answerable.
    """
    season = season or get_current_nfl_season()
    M = PlayerProjection if as_of is None else PlayerProjectionVintage
    q = db.query(M).filter(M.season == season)
    if as_of is not None:
        q = q.filter(M.as_of_week == as_of)
    if week is not None:
        q = q.filter(M.week == week)
    if position:
        q = q.filter(M.position == position.upper())
    if team:
        q = q.filter(M.team == team.upper())

    rows = q.order_by(M.projected_points.desc()).limit(limit).all()
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
        "as_of_week": as_of,
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
        return {"status": "no_data", "season": season,
                "as_of_week": as_of, "data": []}
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
    as_of: Optional[int] = Query(
        None, description="Season outlook as it stood after this many completed "
                          "weeks (0 = preseason). Omit for the current outlook."),
    limit: int = Query(300, le=2000),
    db: Session = Depends(get_db),
):
    """Season-long projected TOTALS per player: summed fantasy points and every
    component stat (passing/rushing/receiving yards, TDs, receptions, INTs) across all
    projected weeks, plus expected games and per-game rate. Sorted by total points.

    `games` is EXPECTED GAMES PLAYED — the availability weights summed, so an
    injury-prone starter reads below the number of weeks on his schedule.
    `ppg` divides by it, making it the rate when he is on the field rather than
    a rate diluted by the weeks he is projected to miss. `scheduled_weeks` is
    the raw count of projected weeks (17 for most: 18 minus the bye).

    Rows written before availability was persisted have no weight stored; those
    players fall back to games = scheduled_weeks, as before.

    `as_of=N` rebuilds the outlook as it stood once N weeks had been played. It
    is deliberately not "rows stamped N": by week 5 the archive holds nothing
    new for weeks 1-4, because played weeks are not reprojected. So each week
    contributes its LATEST vintage at or before N — week 3 keeps whatever was
    last believed about it, weeks 5-18 use the week-5 reprojection — which is
    what "the season as we saw it that morning" actually means.
    """
    from sqlalchemy import and_, func

    P = PlayerProjection
    if as_of is not None:
        V = PlayerProjectionVintage
        latest = (db.query(V.player_id.label("pid"), V.week.label("wk"),
                           func.max(V.as_of_week).label("mx"))
                  .filter(V.season == season, V.as_of_week <= as_of)
                  .group_by(V.player_id, V.week).subquery())
        P = V
    cols = {c: func.sum(getattr(P, c)) for c in (
        "passing_yards", "passing_tds", "passing_interceptions", "rushing_yards",
        "rushing_tds", "receiving_yards", "receptions", "receiving_tds")}
    q = (db.query(
            P.player_id, P.player_name, P.position, P.team,
            func.count(P.week).label("scheduled_weeks"),
            func.sum(P.exp_games).label("exp_games"),
            func.sum(P.projected_points).label("total_points"),
            func.sum(P.floor).label("floor_total"),
            func.sum(P.ceiling).label("ceiling_total"),
            *[v.label(k) for k, v in cols.items()],
         )
         .filter(P.season == season)
         .group_by(P.player_id, P.player_name, P.position, P.team))
    if as_of is not None:
        q = q.join(latest, and_(P.player_id == latest.c.pid,
                                P.week == latest.c.wk,
                                P.as_of_week == latest.c.mx))
    if position:
        q = q.filter(P.position == position.upper())
    rows = q.order_by(func.sum(P.projected_points).desc()).limit(limit).all()
    if not rows:
        return {"status": "no_data", "season": season, "data": []}

    def r1(v):
        return round(float(v), 1) if v is not None else None

    def entry(r):
        # Expected games played, falling back to the week count for rows
        # written before availability was stored.
        games = float((r.exp_games if r.exp_games is not None else r.scheduled_weeks) or 0)
        total = float(r.total_points or 0)
        return {
            "player_id": r.player_id, "player_name": r.player_name,
            "position": r.position, "team": r.team,
            "games": r1(games), "scheduled_weeks": r.scheduled_weeks,
            "total_points": r1(total),
            "ppg": round(total / games, 1) if games > 0 else None,
            "floor_total": r1(r.floor_total), "ceiling_total": r1(r.ceiling_total),
            "passing_yards": r1(r.passing_yards), "passing_tds": r1(r.passing_tds),
            "interceptions": r1(r.passing_interceptions),
            "rushing_yards": r1(r.rushing_yards), "rushing_tds": r1(r.rushing_tds),
            "receiving_yards": r1(r.receiving_yards), "receptions": r1(r.receptions),
            "receiving_tds": r1(r.receiving_tds),
        }

    data = [entry(r) for r in rows]
    return {"status": "success", "season": season, "total": len(data), "data": data}


@router.get("/vintages/accuracy")
def get_vintage_accuracy(
    season: Optional[int] = Query(None, description="Season (default: current)"),
    position: Optional[str] = Query(None, description="Filter to QB/RB/WR/TE"),
    min_points: float = Query(
        5.0, description="Ignore player-weeks projected below this; the tail is "
                         "mostly inactives and swamps the averages."),
    db: Session = Depends(get_db),
):
    """Does a projection get better as the season feeds it?

    For every archived projection whose week has since been played, this scores
    the projection against the actual box score and buckets it by `lead_weeks`
    (week - as_of_week) — how far ahead it was looking. If the weekly
    reprojection is earning its compute, MAE falls as lead time shrinks.

    Buckets are also split by `basis`, and that split is not cosmetic: the rows
    come from three different pipelines and averaging across them would answer
    the wrong question.

      preseason       as_of_week 0 — the ESPN-roster path, with the rookie
                      prior, availability weighting, share model and simulator.
      current_week    the in-season projection for the upcoming week, left
                      exactly as production computes it (no game-environment
                      scaling). This is what the live board bets.
      extrapolated    an in-season projection for a LATER week: the same
                      matchup-neutral mean, scaled by that week's game
                      environment. No availability discount and no share model,
                      because both need the preseason ESPN depth charts.

    So `current_week` vs `extrapolated` at equal lead is a fair comparison of
    information; `preseason` vs either is a comparison of pipelines as much as
    of information, and should be read that way.
    """
    season = season or get_current_nfl_season()
    V = PlayerProjectionVintage
    # Four columns, not whole ORM objects: a finished season holds ~120k
    # vintages (171 week-projections x the player pool) and hydrating all of
    # them to read three floats is the difference between a fast endpoint and
    # a slow one.
    q = db.query(V.week, V.player_id, V.as_of_week, V.projected_points).filter(
        V.season == season, V.projected_points.isnot(None),
        V.projected_points >= min_points)
    if position:
        q = q.filter(V.position == position.upper())
    vintages = q.all()
    if not vintages:
        return {"status": "no_data", "season": season, "data": [],
                "message": "No archived projections yet for this season."}

    stats = _regular_stats(db, season)
    if not stats:
        return {"status": "no_actuals", "season": season, "data": [],
                "message": "No actuals stored yet; nothing to score against."}

    buckets: dict = {}
    for v in vintages:
        actual = stats.get((v.week, v.player_id))
        if actual is None:
            continue                      # week not played, or player didn't
        as_of = v.as_of_week or 0
        lead = v.week - as_of
        if lead < 1:
            continue                      # a projection of an already-played week
        basis = ("preseason" if as_of == 0
                 else "current_week" if lead == 1 else "extrapolated")
        b = buckets.setdefault((basis, lead), {"n": 0, "abs": 0.0, "err": 0.0})
        e = v.projected_points - actual
        b["n"] += 1
        b["abs"] += abs(e)
        b["err"] += e

    if not buckets:
        return {"status": "no_overlap", "season": season, "data": [],
                "message": "Archived projections and actuals do not overlap yet."}
    data = [{"basis": basis,
             "lead_weeks": lead,
             "n": b["n"],
             "mae": round(b["abs"] / b["n"], 2),
             "bias": round(b["err"] / b["n"], 2)}
            for (basis, lead), b in sorted(buckets.items())]
    return {"status": "success", "season": season, "position": position,
            "count": sum(b["n"] for b in buckets.values()), "data": data}


@router.get("/vintages/{player_id}")
def get_player_vintages(
    player_id: str,
    season: Optional[int] = Query(None, description="Season (default: current)"),
    week: Optional[int] = Query(None, description="Single week (default: all)"),
    db: Session = Depends(get_db),
):
    """Every projection ever made for this player, by week and vintage.

    The shape is a week -> list of vintages, oldest first, so a single response
    shows how the model's view of one player-week moved as the season went on.
    """
    season = season or get_current_nfl_season()
    V = PlayerProjectionVintage
    q = db.query(V).filter(V.season == season, V.player_id == player_id)
    if week is not None:
        q = q.filter(V.week == week)
    rows = q.order_by(V.week, V.as_of_week).all()
    if not rows:
        return {"status": "no_data", "season": season, "player_id": player_id,
                "data": {}}
    out: dict = {}
    for r in rows:
        out.setdefault(str(r.week), []).append(_orm_to_dict(r))
    return {"status": "success", "season": season, "player_id": player_id,
            "player_name": rows[-1].player_name, "weeks": len(out), "data": out}
