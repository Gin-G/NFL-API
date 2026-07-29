"""Team power ratings / grades endpoint.

Opponent-adjusted (SRS-style) team offense/defense grades on a 0-100 scale (50 = league
average), computed directly from the Schedule table's scores — no ML dependency. A
preseason prior (last season regressed toward the mean) is blended with in-season results,
the prior fading as real games accrue, so early-season grades aren't overreactive.

Measured in the nfl-data-py research (EXPERIMENTS.md): offense is a stable, predictable
trait (split-half r ~0.49); defense is noisier but opponent adjustment ~doubles its
stability. Grades are for interpretability / matchup context, not a precise score model.
"""
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database.session import get_db
from database.models import Schedule

logger = logging.getLogger(__name__)
router = APIRouter()

_YOY_CARRYOVER = 0.65   # fraction of last year's edge carried into the preseason prior
_PRIOR_GAMES = 6.0      # equivalent games of weight the prior gets when blending
_GRADE_SCALE = 15.0     # 1 SD of rating ~ 15 grade points


def _game_frame(db: Session, season: int, through_week: Optional[int] = None):
    """List of (team, opp, pf, pa) for completed games of a season."""
    q = db.query(Schedule).filter(
        Schedule.season == season,
        Schedule.home_score.isnot(None),
        Schedule.away_score.isnot(None),
    )
    if through_week is not None:
        q = q.filter(Schedule.week <= through_week)
    rows = []
    for g in q.all():
        rows.append((g.home_team, g.away_team, g.home_score, g.away_score))
        rows.append((g.away_team, g.home_team, g.away_score, g.home_score))
    return rows


def _srs(rows, iters: int = 50):
    """Opponent-adjusted offense & defense ratings (points vs league avg), centered 0."""
    if not rows:
        return {}, {}, 22.5
    teams = sorted({r[0] for r in rows})
    lg = float(np.mean([r[2] for r in rows]))
    by_team = {t: [(pf, pa, opp) for (tm, opp, pf, pa) in rows if tm == t] for t in teams}
    off = {t: 0.0 for t in teams}
    dff = {t: 0.0 for t in teams}
    for _ in range(iters):
        off = {t: float(np.mean([pf - lg - dff.get(o, 0.0) for pf, pa, o in by_team[t]]))
               for t in teams}
        dff = {t: float(np.mean([pa - lg - off.get(o, 0.0) for pf, pa, o in by_team[t]]))
               for t in teams}
    return off, dff, lg


def _preseason_prior(db: Session, season: int):
    off, dff, _ = _srs(_game_frame(db, season - 1))
    return ({t: v * _YOY_CARRYOVER for t, v in off.items()},
            {t: v * _YOY_CARRYOVER for t, v in dff.items()})


def _grade(rating: float, spread: float, invert: bool = False) -> float:
    z = rating / spread if spread else 0.0
    if invert:
        z = -z
    return round(min(100.0, max(0.0, 50.0 + _GRADE_SCALE * z)), 1)


def compute_grades(db: Session, season: int, through_week: Optional[int] = None):
    """0-100 offense/defense grades per team, prior-blended. Empty list if no data."""
    prior_off, prior_def = _preseason_prior(db, season)
    rows = _game_frame(db, season, through_week)
    in_off, in_def, _ = _srs(rows)
    games = {}
    for (tm, opp, pf, pa) in rows:
        games[tm] = games.get(tm, 0) + 1

    teams = set(prior_off) | set(in_off)
    if not teams:
        return []
    off_r, def_r = {}, {}
    for t in teams:
        n = games.get(t, 0)
        w = _PRIOR_GAMES
        off_r[t] = (w * prior_off.get(t, 0.0) + n * in_off.get(t, 0.0)) / (w + n)
        def_r[t] = (w * prior_def.get(t, 0.0) + n * in_def.get(t, 0.0)) / (w + n)

    off_sd = float(np.std(list(off_r.values()))) or 1.0
    def_sd = float(np.std(list(def_r.values()))) or 1.0
    out = [{
        "team": t,
        "games": games.get(t, 0),
        "offense_grade": _grade(off_r[t], off_sd),
        "defense_grade": _grade(def_r[t], def_sd, invert=True),  # stingier D -> higher
        "offense_rating": round(off_r[t], 2),
        "defense_rating": round(def_r[t], 2),
    } for t in teams]
    out.sort(key=lambda r: r["offense_grade"], reverse=True)
    return out


@router.get("/{season}")
async def get_ratings(
    season: int,
    through_week: Optional[int] = Query(None, description="Grade as of this week (default: all completed games)"),
    db: Session = Depends(get_db),
):
    """Team offense/defense grades (0-100, 50=avg) for a season."""
    grades = compute_grades(db, season, through_week)
    if not grades:
        raise HTTPException(status_code=404, detail=f"No schedule/score data for {season} or {season-1}")
    return {"status": "success", "season": season, "through_week": through_week,
            "total_teams": len(grades), "data": grades}


@router.get("/{season}/{team}")
async def get_team_rating(
    season: int, team: str,
    through_week: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """Grade for a single team."""
    grades = compute_grades(db, season, through_week)
    match = next((g for g in grades if g["team"].upper() == team.upper()), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No rating for {team} in {season}")
    return {"status": "success", "season": season, "data": match}
