#!/usr/bin/env python3
"""
Stats API Router
Season statistical leaders.

Ports NickKnows' `stat_aggregation_tasks.py` (the six `*_top10_data.csv`
files): filter weekly player stats to the regular season and non-null values
of the requested stat, group by player display name, SUM, sort descending,
take `limit`.

DB-first: computes from the `player_stats` table; falls back to nflreadpy when
the table has no data for the requested season.
"""

import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database.session import get_db
from .utils import get_current_nfl_season, load_weekly_stats_df

logger = logging.getLogger(__name__)
router = APIRouter()

# Stat -> the position most associated with it (used only for enrichment).
VALID_STATS = {
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
}


def _leaders_from_df(df: pd.DataFrame, stat: str, limit: int) -> list[dict]:
    """Compute season leaders for a stat from a weekly-stats DataFrame.

    Mirrors the Celery task semantics: REG only, non-null stat, group by
    player_display_name, sum, sort desc, head(limit). Each row is enriched with
    the player's id / recent team / position taken from their latest week.
    """
    if df.empty or stat not in df.columns or "player_display_name" not in df.columns:
        return []

    work = df.copy()
    if "season_type" in work.columns:
        work = work[work["season_type"] == "REG"]
    work = work[work[stat].notna()]
    if work.empty:
        return []

    totals = (
        work.groupby("player_display_name")[stat]
        .sum()
        .sort_values(ascending=False)
        .head(limit)
    )

    # Metadata (id / team / position) from each player's most recent week.
    if "week" in work.columns:
        latest = work.sort_values("week").drop_duplicates(
            subset=["player_display_name"], keep="last"
        )
    else:
        latest = work.drop_duplicates(subset=["player_display_name"], keep="last")
    latest = latest.set_index("player_display_name")

    leaders = []
    for name, value in totals.items():
        meta = latest.loc[name] if name in latest.index else None
        leaders.append({
            "player_name": name,
            "player_id": (meta.get("player_id") if meta is not None else None),
            "team": (meta.get("recent_team") if meta is not None else None),
            "position": (meta.get("position") if meta is not None else None),
            "value": float(value),
        })
    return leaders


@router.get("/leaders/")
def get_stat_leaders(
    season: Optional[int] = Query(None, description="Season year (default: current)"),
    stat: str = Query(..., description="One of: passing_yards, passing_tds, "
                                       "rushing_yards, rushing_tds, receiving_yards, "
                                       "receiving_tds"),
    limit: int = Query(10, ge=1, le=100, description="Number of leaders to return"),
    db: Session = Depends(get_db),
):
    """Season stat leaders for a single stat, highest total first."""
    season = season or get_current_nfl_season()
    stat = stat.lower()
    if stat not in VALID_STATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stat '{stat}'. Must be one of: {sorted(VALID_STATS)}",
        )

    try:
        df = load_weekly_stats_df(db, season)
    except Exception as e:
        logger.error("Error loading weekly stats for leaders: %s", e)
        return {
            "status": "no_data",
            "season": season,
            "stat": stat,
            "total_players": 0,
            "data": [],
        }

    leaders = _leaders_from_df(df, stat, limit)
    if not leaders:
        return {
            "status": "no_data",
            "season": season,
            "stat": stat,
            "total_players": 0,
            "data": [],
        }

    return {
        "status": "success",
        "season": season,
        "stat": stat,
        "total_players": len(leaders),
        "data": leaders,
    }
