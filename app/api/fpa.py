#!/usr/bin/env python3
"""
Fantasy Points Allowed (FPA) API Router and shared logic.

Two numbers per defence per position, and the second is the one to read.

**Raw** is the NickKnows math this ported (`{year}_FPA.csv`): for each defense,
collect every opposing player's `fantasy_points_ppr`, sum PPR points per
position per week, average those weekly sums. It is kept because it is the
historical series and the site charts it, but on its own it cannot tell two very
different defences apart -- one allowed a lot to tight ends because it faced
four of the best tight ends, the other allowed a lot while facing replacements.
Only the second is a target, and raw FPA ranks them the same.

**Adjusted** measures every opposing player against his own entering baseline,
so the schedule a defence has had nets out. That math lives in
`nfl_projections.fpa` so the research harness and this endpoint cannot drift
apart; this module is the DB adapter and the cache. Measured over 2019-2026 the
two rankings disagree a lot (Spearman 0.60 for receivers, ~0.76 elsewhere) and
the adjusted one predicts next week's over/under-performance better for three
positions of four -- QB 0.094 vs 0.068, WR 0.051 vs 0.026, TE 0.049 vs 0.037,
RB a wash. Those are small correlations and are meant to be read that way: a
tiebreak between comparable players, not a signal to chase.

The adjusted side is scored FanDuel (half-PPR), winsorises each player's weekly
gap at +/-12 so one weather game cannot carry a season, and reports
`{pos}_outlier_week` when dropping a single week would move the rating by a
point or more. 2026 week 8 is the worked example: Tampa Bay completed no passes
against Atlanta, ran for 310 yards and won, which left Atlanta's WR and TE lines
at 0.0 for the week.

Early in a season the adjusted half is empty -- a player needs three prior games
before his baseline is worth subtracting -- and the response says so in
`adjusted` and `adjusted_note` rather than failing. Same when the image is
built without nfl-projections.

The per-team detail endpoint (`/teams/{abbr}/fpa`) lives in the teams router but
reuses the helpers here.

DB-first: computed from `schedules` + `player_stats`; falls back to nflreadpy.
"""

import logging
import time
from typing import Optional

import numpy as np
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

# The adjusted table is a groupby plus a leave-one-week-out pass over ~128
# (defence, position) groups, so it is cheap next to loading the season's stats.
# Cached anyway because the ratings only move once a week. The key carries the
# row counts, so a loader run invalidates it without waiting for the TTL.
_ADJ_TTL = 900
_adj_cache: dict = {}


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


def _fpa_math():
    """`nfl_projections.fpa`, or None with the reason logged.

    Unlike the line ratings in /trenches, a missing package must not take this
    endpoint down: the raw series predates the adjustment and the site charts
    it. The response reports `adjusted: false` instead.
    """
    try:
        from nfl_projections import fpa as fpa_math
        return fpa_math
    except ImportError as exc:        # pragma: no cover - packaging failure
        logger.error("nfl_projections.fpa unavailable: %s", exc)
        return None


def compute_fpa_adjusted(stats_df: pd.DataFrame, opp_map: dict, season: int,
                         through_week: Optional[int] = None):
    """Return ({(defense, POSITION): row}, note).

    An empty mapping with a note is the normal early-season answer, not a
    failure: `nfl_projections.fpa` needs three prior games per player before it
    has a baseline to measure against, so weeks 1-3 produce nothing.
    """
    fpa_math = _fpa_math()
    if fpa_math is None:
        return {}, "This image was built without the nfl-projections package."

    df = _with_defense_faced(stats_df, opp_map)
    df = df[df["defense_faced"].notna()]
    if df.empty:
        return {}, "No completed games yet."

    df = df.rename(columns={"defense_faced": "opponent_team"})
    if "player_name" not in df.columns:
        df["player_name"] = df.get("player_display_name")

    try:
        rated = fpa_math.ratings(df, int(season), through_week)
    except Exception as exc:
        logger.warning("Adjusted FPA failed for %s: %s", season, exc)
        return {}, f"Adjusted ratings could not be built: {exc}"
    if rated.empty:
        return {}, ("Not enough football yet: a player needs three prior games "
                    "before his own baseline can be subtracted.")

    rows = {}
    for r in rated.itertuples():
        rows[(r.defense, r.position)] = {
            "allowed": round(float(r.allowed), 2),
            "faced": round(float(r.expected), 2),
            "adjusted": round(float(r.adjusted), 2),
            "grade": _float_or_none(r.grade),
            "band": r.band,
            "weeks": _int_or_none(r.weeks),
            "outlier_week": _int_or_none(pd.to_numeric(r.worst_week, errors="coerce")),
            "outlier_swing": _float_or_none(r.worst_week_swing),
        }
    return rows, None


def _adjusted_cached(stats_df, opp_map, season, through_week):
    key = (int(season), through_week, len(stats_df), len(opp_map))
    hit = _adj_cache.get(key)
    if hit and time.time() - hit[0] < _ADJ_TTL:
        return hit[1]
    value = compute_fpa_adjusted(stats_df, opp_map, season, through_week)
    _adj_cache[key] = (time.time(), value)
    return value


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
    """int(v), or None for anything that is not a finite number.

    `None` in a DataFrame column becomes NaN, and `if NaN` is true, so the
    finiteness check has to be explicit or `worst_week` raises
    "cannot convert float NaN to integer".
    """
    n = pd.to_numeric(v, errors="coerce")
    return int(n) if n is not None and np.isfinite(n) else None


def _float_or_none(v):
    n = pd.to_numeric(v, errors="coerce")
    return float(n) if n is not None and np.isfinite(n) else None


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
    through_week: Optional[int] = Query(
        None, ge=1, le=22,
        description="Ratings as they stood entering this week (default: the whole season)"),
    db: Session = Depends(get_db),
):
    """Fantasy points allowed per defense, one row per team.

    `qb`/`rb`/`wr`/`te` are the raw PPR averages this endpoint has always
    returned. The `{pos}_grade` columns are the schedule-adjusted ratings:
    0-100, 50 = league average, higher = softer = better to target. The pair
    that makes a grade readable is `{pos}_allowed` against `{pos}_faced` --
    points the defence gave up against what the players it faced were running
    at, both FanDuel-scored. A defence can be top of the raw table and a bad
    target because the second number is just as high.
    """
    season = season or get_current_nfl_season()
    try:
        sched_df = load_schedules_df(db, season)
        stats_df = load_weekly_stats_df(db, season)
    except Exception as e:
        logger.error("Error loading data for FPA: %s", e)
        return {"status": "no_data", "season": season, "total_teams": 0, "data": []}

    if through_week is not None:
        if sched_df is not None and not sched_df.empty and "week" in sched_df:
            sched_df = sched_df[pd.to_numeric(sched_df["week"], errors="coerce")
                                < through_week]
        if stats_df is not None and not stats_df.empty and "week" in stats_df:
            stats_df = stats_df[pd.to_numeric(stats_df["week"], errors="coerce")
                                < through_week]

    opp_map = build_opponent_map(sched_df)
    table = compute_fpa_table(stats_df, opp_map)
    adjusted, note = _adjusted_cached(stats_df, opp_map, season, through_week)

    # Every team that played a completed game gets a row.
    defenses = sorted({d for (_t, _w), d in opp_map.items()})
    if not defenses:
        return {"status": "no_data", "season": season, "total_teams": 0, "data": []}

    names = _team_names(db)
    data = []
    for abbr in defenses:
        pos = table.get(abbr, {})
        row = {
            "team": abbr,
            "team_name": names.get(abbr, abbr),
            "qb": round(pos.get("QB", 0.0), 2),
            "rb": round(pos.get("RB", 0.0), 2),
            "wr": round(pos.get("WR", 0.0), 2),
            "te": round(pos.get("TE", 0.0), 2),
        }
        for position in FPA_POSITIONS:
            adj = adjusted.get((abbr, position), {})
            key = position.lower()
            row[f"{key}_allowed"] = adj.get("allowed")
            row[f"{key}_faced"] = adj.get("faced")
            row[f"{key}_adjusted"] = adj.get("adjusted")
            row[f"{key}_grade"] = adj.get("grade")
            row[f"{key}_band"] = adj.get("band")
            row[f"{key}_weeks"] = adj.get("weeks")
            row[f"{key}_outlier_week"] = adj.get("outlier_week")
            row[f"{key}_outlier_swing"] = adj.get("outlier_swing")
        data.append(row)

    fpa_math = _fpa_math()
    return {
        "status": "success",
        "season": season,
        "through_week": through_week,
        "total_teams": len(data),
        "adjusted": bool(adjusted),
        "adjusted_note": note,
        "scoring": {"qb_rb_wr_te": "PPR, the historical series",
                    "allowed_faced_adjusted": "FanDuel (half-PPR)"},
        "scale": ("{pos}_grade is 0-100, 50 = league average, 15 points per "
                  "standard deviation, higher = softer = better to target"),
        "bands": ([{"from": lo, "to": hi, "label": label}
                   for lo, hi, label in fpa_math.BANDS] if fpa_math else []),
        "data": data,
    }
