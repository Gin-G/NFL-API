#!/usr/bin/env python3
"""
Fantasy Points Allowed (FPA) API Router and shared logic.

Ports NickKnows' `team_analysis_tasks.py` FPA math (`{year}_FPA.csv` and the
per-team detail behind it):

For each defense, collect every opposing player's `fantasy_points_ppr` from
weekly stats for the games that defense played, group by (week, position), sum
PPR points per position per week, then average those weekly sums across the
weeks played. Positions covered: QB, RB, WR, TE (0 when a position has no data).

The per-team detail endpoint (`/teams/{abbr}/fpa`) lives in the teams router but
reuses the helpers here.

DB-first: computed from `schedules` + `player_stats`; falls back to nflreadpy.
"""

import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database.session import get_db
from .utils import get_current_nfl_season, load_schedules_df, load_weekly_stats_df

logger = logging.getLogger(__name__)
router = APIRouter()

FPA_POSITIONS = ["QB", "RB", "WR", "TE"]

# Weekly stat columns surfaced in the per-player FPA detail rows.
_DETAIL_STAT_COLUMNS = [
    "passing_yards", "passing_tds",
    "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds",
]


def build_opponent_map(sched_df: pd.DataFrame) -> dict:
    """Map (team, week) -> opponent for completed games only.

    A team's opponent in a given week is the defense that team's offense faced,
    so this map keys offensive (team, week) to the defending team.
    """
    opp: dict = {}
    if sched_df is None or sched_df.empty:
        return opp
    for _, g in sched_df.iterrows():
        home, away = g.get("home_team"), g.get("away_team")
        hs, ascore, wk = g.get("home_score"), g.get("away_score"), g.get("week")
        if pd.isna(hs) or pd.isna(ascore):          # completed games only
            continue
        if pd.isna(home) or pd.isna(away) or pd.isna(wk):
            continue
        opp[(home, wk)] = away
        opp[(away, wk)] = home
    return opp


def _with_defense_faced(stats_df: pd.DataFrame, opp_map: dict) -> pd.DataFrame:
    """Return a copy of stats_df with a `defense_faced` column added."""
    df = stats_df.copy()
    if df.empty:
        df["defense_faced"] = None
        return df
    df["defense_faced"] = [
        opp_map.get((team, week))
        for team, week in zip(df.get("recent_team"), df.get("week"))
    ]
    return df


def compute_fpa_table(stats_df: pd.DataFrame, opp_map: dict) -> dict:
    """Return {defense_abbr: {'QB':x, 'RB':x, 'WR':x, 'TE':x}}.

    Mirrors process_team_fpa: sum PPR per (defense, week, position), then take
    the mean across weeks per (defense, position).
    """
    df = _with_defense_faced(stats_df, opp_map)
    df = df[df["defense_faced"].notna() & df["position"].isin(FPA_POSITIONS)]
    if df.empty:
        return {}

    weekly = (
        df.groupby(["defense_faced", "week", "position"])["fantasy_points_ppr"]
        .sum()
        .reset_index()
    )
    avg = (
        weekly.groupby(["defense_faced", "position"])["fantasy_points_ppr"]
        .mean()
        .reset_index()
    )

    table: dict = {}
    for _, r in avg.iterrows():
        table.setdefault(r["defense_faced"], {})[r["position"]] = float(
            r["fantasy_points_ppr"]
        )
    return table


def compute_fpa_detail(
    stats_df: pd.DataFrame,
    opp_map: dict,
    defense: str,
    position: Optional[str] = None,
) -> list[dict]:
    """One row per opposing player-week against `defense`."""
    df = _with_defense_faced(stats_df, opp_map)
    df = df[df["defense_faced"] == defense]
    if position:
        df = df[df["position"] == position.upper()]
    if df.empty:
        return []

    df = df.sort_values(["week", "position"])
    rows = []
    for _, s in df.iterrows():
        row = {
            "week": _int_or_none(s.get("week")),
            "opponent": s.get("recent_team"),
            "player_id": s.get("player_id"),
            "player_name": s.get("player_display_name"),
            "position": s.get("position"),
            "fantasy_points_ppr": _float_or_none(s.get("fantasy_points_ppr")),
            "fantasy_points": _float_or_none(s.get("fantasy_points")),
        }
        for col in _DETAIL_STAT_COLUMNS:
            row[col] = _float_or_none(s.get(col)) if col in df.columns else None
        rows.append(row)
    return rows


def _int_or_none(v):
    return int(v) if pd.notna(v) else None


def _float_or_none(v):
    return float(v) if pd.notna(v) else None


def _team_names(db) -> dict:
    """abbr -> full team name (best-effort; empty on failure)."""
    try:
        from database.models import Team
        return {t.team_abbr: t.team_name for t in db.query(Team).all()}
    except Exception:
        return {}


@router.get("/")
def get_fpa(
    season: Optional[int] = Query(None, description="Season year (default: current)"),
    db: Session = Depends(get_db),
):
    """Fantasy points allowed per defense, one row per team (QB/RB/WR/TE)."""
    season = season or get_current_nfl_season()
    try:
        sched_df = load_schedules_df(db, season)
        stats_df = load_weekly_stats_df(db, season)
    except Exception as e:
        logger.error("Error loading data for FPA: %s", e)
        return {"status": "no_data", "season": season, "total_teams": 0, "data": []}

    opp_map = build_opponent_map(sched_df)
    table = compute_fpa_table(stats_df, opp_map)

    # Every team that played a completed game gets a row.
    defenses = sorted({d for (_t, _w), d in opp_map.items()})
    if not defenses:
        return {"status": "no_data", "season": season, "total_teams": 0, "data": []}

    names = _team_names(db)
    data = []
    for abbr in defenses:
        pos = table.get(abbr, {})
        data.append({
            "team": abbr,
            "team_name": names.get(abbr, abbr),
            "qb": round(pos.get("QB", 0.0), 2),
            "rb": round(pos.get("RB", 0.0), 2),
            "wr": round(pos.get("WR", 0.0), 2),
            "te": round(pos.get("TE", 0.0), 2),
        })

    return {
        "status": "success",
        "season": season,
        "total_teams": len(data),
        "data": data,
    }
