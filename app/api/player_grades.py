"""Player grades endpoint.

Per-player quality grade (0-100, 50 = positional average), computed directly from the
PlayerStat table — no ML dependency. Mirrors the validated nfl_projections research
(EXPERIMENTS.md "player grade"): OPPORTUNITY metrics (WOPR, target share, carries) are the
most stable and most predictive of future fantasy production, so the grade is
opportunity- and production-weighted, efficiency-light. A preseason prior (last season's
grade regressed toward the positional average) is blended with in-season results, fading as
games accrue.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database.session import get_db
from database.models import PlayerStat

logger = logging.getLogger(__name__)
router = APIRouter()

POSITIONS = ["QB", "RB", "WR", "TE"]
_YOY_CARRYOVER = 0.60
_PRIOR_GAMES = 5.0
_GRADE_SCALE = 15.0
_MIN_GAMES = 2
_SPEC = {  # opportunity metric(s) + efficiency metric per position
    "QB": (["attempts", "carries"], "epa_per_play"),
    "RB": (["carries", "wopr"], "yards_per_opp"),
    "WR": (["wopr"], "epa_per_play"),
    "TE": (["wopr"], "epa_per_play"),
}
_W = {"opportunity": 0.45, "production": 0.45, "efficiency": 0.10}


def _fanduel(r) -> float:
    py, pt = r.get("passing_yards", 0) or 0, r.get("passing_tds", 0) or 0
    ints = r.get("interceptions", 0) or 0
    ry, rt = r.get("rushing_yards", 0) or 0, r.get("rushing_tds", 0) or 0
    rec, recy, rect = r.get("receptions", 0) or 0, r.get("receiving_yards", 0) or 0, r.get("receiving_tds", 0) or 0
    return (py * 0.04 + pt * 4 + ints * -1 + (3 if py >= 300 else 0)
            + ry * 0.1 + rt * 6 + (3 if ry >= 100 else 0)
            + rec * 0.5 + recy * 0.1 + rect * 6 + (3 if recy >= 100 else 0))


_COLS = ["player_id", "player_display_name", "position", "week", "passing_yards", "passing_tds",
         "interceptions", "rushing_yards", "rushing_tds", "receptions", "receiving_yards",
         "receiving_tds", "wopr", "target_share", "carries", "attempts", "targets",
         "passing_epa", "rushing_epa", "receiving_epa"]


def _frame(db: Session, season: int, through_week: Optional[int] = None) -> pd.DataFrame:
    q = db.query(*[getattr(PlayerStat, c) for c in _COLS]).filter(
        PlayerStat.season == season, PlayerStat.position.in_(POSITIONS))
    if through_week is not None:
        q = q.filter(PlayerStat.week <= through_week)
    df = pd.DataFrame(q.all(), columns=_COLS)
    return df


def _metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["fp"] = df.apply(_fanduel, axis=1)
    opp_ct = (df["targets"].fillna(0) + df["carries"].fillna(0) + df["attempts"].fillna(0)).replace(0, np.nan)
    epa = df["receiving_epa"].fillna(0) + df["rushing_epa"].fillna(0) + df["passing_epa"].fillna(0)
    df["epa_per_play"] = epa / opp_ct
    df["yards_per_opp"] = (df["receiving_yards"].fillna(0) + df["rushing_yards"].fillna(0)) / opp_ct
    metric_cols = ["fp", "wopr", "target_share", "carries", "attempts", "epa_per_play", "yards_per_opp"]
    g = (df.groupby(["player_id", "position"])
         .agg(games=("week", "count"), player_name=("player_display_name", "first"),
              **{c: (c, "mean") for c in metric_cols})
         .reset_index())
    return g[g["games"] >= _MIN_GAMES]


def _z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0


def _composite(metrics: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pos, g in metrics.groupby("position"):
        opp_cols, eff_col = _SPEC[pos]
        g = g.copy()

        def fam(cols):
            cols = [c for c in cols if c in g.columns and g[c].notna().any()]
            if not cols:
                return pd.Series(0.0, index=g.index)
            return pd.concat([_z(g[c].fillna(g[c].mean())) for c in cols], axis=1).mean(axis=1)

        opp, prod, eff = fam(opp_cols), fam(["fp"]), fam([eff_col])
        comp = _W["opportunity"] * opp + _W["production"] * prod + _W["efficiency"] * eff
        g["grade"] = (50 + _GRADE_SCALE * comp).clip(0, 100).round(1)
        g["opportunity_grade"] = (50 + _GRADE_SCALE * opp).clip(0, 100).round(1)
        g["efficiency_grade"] = (50 + _GRADE_SCALE * eff).clip(0, 100).round(1)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def compute_player_grades(db: Session, season: int, through_week: Optional[int] = None):
    cur = _composite(_metrics(_frame(db, season, through_week)))
    if cur.empty:
        return []
    prior = _composite(_metrics(_frame(db, season - 1)))
    prior_grade = {p: 50 + _YOY_CARRYOVER * (gr - 50)
                   for p, gr in zip(prior["player_id"], prior["grade"])} if not prior.empty else {}
    rows = []
    for _, r in cur.iterrows():
        n = r["games"]
        pg = prior_grade.get(r["player_id"], 50.0)
        grade = round((_PRIOR_GAMES * pg + n * r["grade"]) / (_PRIOR_GAMES + n), 1)
        rows.append({
            "player_id": r["player_id"], "player_name": r["player_name"], "position": r["position"],
            "games": int(n), "grade": grade,
            "opportunity_grade": r["opportunity_grade"], "efficiency_grade": r["efficiency_grade"],
            "fppg": round(r["fp"], 1),
        })
    rows.sort(key=lambda x: x["grade"], reverse=True)
    return rows


@router.get("/{season}")
async def get_player_grades(
    season: int,
    position: Optional[str] = Query(None, description="Filter to QB/RB/WR/TE"),
    through_week: Optional[int] = Query(None, description="Grade as of this week"),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db),
):
    """Player grades (0-100, 50=positional avg) for a season."""
    grades = compute_player_grades(db, season, through_week)
    if not grades:
        raise HTTPException(status_code=404, detail=f"No player-stat data for {season}")
    if position:
        grades = [g for g in grades if g["position"].upper() == position.upper()]
    return {"status": "success", "season": season, "through_week": through_week,
            "total": len(grades), "data": grades[:limit]}


@router.get("/{season}/{player_id}")
async def get_player_grade(season: int, player_id: str,
                           through_week: Optional[int] = Query(None),
                           db: Session = Depends(get_db)):
    """Grade for a single player."""
    grades = compute_player_grades(db, season, through_week)
    match = next((g for g in grades if g["player_id"] == player_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No grade for {player_id} in {season}")
    return {"status": "success", "season": season, "data": match}
