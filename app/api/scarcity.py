"""Positional scarcity: replacement level, usable-week rates, top-N make-up.

The board ranks players by projected points, which answers who scores more and
not who is harder to replace. A quarterback and a tight end at the same
projection are not worth the same, because the next quarterback is nearly as
good and the next tight end is not. These endpoints supply the numbers that
difference is made of:

    /scarcity/replacement   points per game of the last startable player at each
                            position, and of the shelf just past him. Value over
                            replacement is a projection minus that second number.
    /scarcity/usable        how often each position supplies a startable week at
                            a flat bar — the usability idea, measured. A flat bar
                            is generous to quarterbacks and nearly unreachable
                            for tight ends, which is itself the finding.
    /scarcity/composition   positional make-up of the top 25, 50 and 100 scorers
                            in a season: whether a position's good players are
                            clustered at the top or spread through the field.

Scored as FanDuel half-PPR, the same as the board and the optimizer, because
scarcity has a different shape under full PPR and mixing the two would put the
replacement levels slightly wrong for everything downstream. The maths lives in
`nfl_projections.scarcity`; this module supplies the data from player_stats.
"""
import logging
import time
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.utils import fanduel_points
from database.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_SEASONS = 5
_TTL = 6 * 3600
_cache: dict = {}

# Exactly the columns api.utils.fanduel_points reads, and no others. Scarcity
# baselines are compared against projections, so they have to be scored by the
# same function the projections are scored by — including its choice of the
# fumble columns, which is why these are the plain names and not the _lost ones.
_STAT_COLUMNS = [
    "season", "week", "player_id", "player_display_name", "player_name",
    "position", "recent_team", "passing_yards", "passing_tds", "interceptions",
    "rushing_yards", "rushing_tds", "receptions", "receiving_yards",
    "receiving_tds", "rushing_fumbles", "receiving_fumbles", "sack_fumbles",
]


def _scarcity():
    try:
        from nfl_projections import scarcity
        return scarcity
    except ImportError as exc:        # pragma: no cover - packaging failure
        logger.error("nfl_projections.scarcity unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Scarcity baselines need the nfl-projections package in this image.")


def _cached(key, build):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    value = build()
    _cache[key] = (time.time(), value)
    return value


def weekly_points(db: Session, seasons: list[int]) -> pd.DataFrame:
    """One row per player-week with FanDuel points, from player_stats."""
    placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    params = {f"s{i}": int(s) for i, s in enumerate(seasons)}
    cols = ", ".join(_STAT_COLUMNS)
    sql = (f"SELECT {cols} FROM player_stats "
           f"WHERE season IN ({placeholders}) AND season_type = 'REG' "
           f"AND position IN ('QB', 'RB', 'WR', 'TE')")
    rows = db.execute(text(sql), params).mappings().all()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(r) for r in rows])
    frame["points"] = [fanduel_points(r) for _, r in frame.iterrows()]
    name = frame.player_display_name.fillna(frame.player_name)
    return pd.DataFrame({"season": frame.season, "week": frame.week,
                         "player_id": frame.player_id, "player_name": name,
                         "position": frame.position, "team": frame.recent_team,
                         "points": frame.points})


def _seasons(db: Session, count: int, through: Optional[int]) -> list[int]:
    """The most recent `count` seasons with stats, newest last."""
    rows = db.execute(text("SELECT DISTINCT season FROM player_stats "
                           "WHERE season_type = 'REG' ORDER BY season")).all()
    available = [int(r[0]) for r in rows]
    if through is not None:
        available = [s for s in available if s <= through]
    if not available:
        raise HTTPException(status_code=404, detail="No player stats loaded.")
    return available[-count:]


def _frame(db: Session, count: int, through: Optional[int]):
    def build():
        seasons = _seasons(db, count, through)
        weekly = weekly_points(db, seasons)
        if weekly.empty:
            raise HTTPException(status_code=404,
                                detail=f"No stats for seasons {seasons}.")
        return weekly, _scarcity().per_season(weekly)
    return _cached(("frame", count, through), build)


def _records(frame: pd.DataFrame):
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


@router.get("/replacement")
def replacement(seasons: int = Query(DEFAULT_SEASONS, ge=1, le=20),
                through_season: Optional[int] = Query(None, ge=1999),
                db: Session = Depends(get_db)):
    """Starter and replacement points per game, per position.

    `pooled` averages the per-season numbers and is what a projection should be
    valued against: one season's replacement level is a small sample at the exact
    point where the player pool is thinnest.
    """
    scarcity = _scarcity()
    _, frame = _frame(db, seasons, through_season)
    per_season = scarcity.replacement_level(frame)
    if per_season.empty:
        raise HTTPException(status_code=404,
                            detail="Not enough qualified players to set a replacement level.")
    pooled = (per_season.groupby("position")
              .agg(starter_ppg=("starter_ppg", "mean"),
                   replacement_ppg=("replacement_ppg", "mean"),
                   usable_players=("usable_players", "mean")).round(2).reset_index())
    return {
        "scoring": "FanDuel half-PPR",
        "starter_slots": scarcity.STARTERS,
        "usable_bar": scarcity.USABLE_BAR,
        "seasons": sorted(per_season.season.unique().tolist()),
        "pooled": _records(pooled),
        "per_season": _records(per_season),
    }


@router.get("/usable")
def usable(seasons: int = Query(DEFAULT_SEASONS, ge=1, le=20),
           bar: float = Query(None, gt=0, le=60),
           through_season: Optional[int] = Query(None, ge=1999),
           db: Session = Depends(get_db)):
    """Share of player-weeks at each position that clear a startable bar."""
    scarcity = _scarcity()
    weekly, _ = _frame(db, seasons, through_season)
    bar = scarcity.USABLE_BAR if bar is None else bar
    per_season = scarcity.usable_rate(weekly, bar=bar)
    pooled = (per_season.groupby("position")
              .agg(player_weeks=("player_weeks", "sum"),
                   usable_weeks=("usable_weeks", "sum")).reset_index())
    pooled["usable_rate"] = (pooled.usable_weeks / pooled.player_weeks).round(3)
    return {"bar": bar, "scoring": "FanDuel half-PPR",
            "seasons": sorted(per_season.season.unique().tolist()),
            "pooled": _records(pooled), "per_season": _records(per_season)}


@router.get("/composition")
def composition(seasons: int = Query(DEFAULT_SEASONS, ge=1, le=20),
                through_season: Optional[int] = Query(None, ge=1999),
                db: Session = Depends(get_db)):
    """Positional make-up of the top 25, 50 and 100 scorers, per season."""
    scarcity = _scarcity()
    _, frame = _frame(db, seasons, through_season)
    comp = scarcity.top_n_composition(frame)
    pooled = (comp.groupby(["top_n", "position"]).players.mean()
              .round(1).reset_index())
    return {"scoring": "FanDuel half-PPR", "ranked_on": "season total points",
            "seasons": sorted(comp.season.unique().tolist()),
            "pooled": _records(pooled), "per_season": _records(comp)}
