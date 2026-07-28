#!/usr/bin/env python3
"""
Opportunities API Router.

Ports NickKnows' `opportunity_tasks.py` (`{year}_opportunity_data.csv` and
`{year}_opportunity_trends.csv`). Opportunity metrics (targets, carries,
touches, situational splits, and shares) are computed from regular-season
play-by-play; trends summarise them per player across weeks.

DB-first: PBP is read from the `play_by_play` table; falls back to nflreadpy.
Roster enrichment is DB-first from `player_rosters`.
"""

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database.session import get_db
from .utils import (
    get_current_nfl_season,
    load_pbp_df,
    load_weekly_rosters_df,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Metrics analysed for trends (must match NickKnows column names exactly).
TREND_METRICS = [
    "targets", "carries", "touches", "target_share", "carry_share",
    "red_zone_targets", "red_zone_carries", "goal_line_touches",
    "deep_targets", "short_targets",
]

# Row keys for a single player-week opportunity record.
_OPP_FIELDS = [
    "targets", "red_zone_targets", "end_zone_targets", "carries",
    "red_zone_carries", "goal_line_carries", "air_yards", "touches",
    "goal_line_touches", "third_down_targets", "deep_targets", "short_targets",
]


def process_week_opportunities(week_data: pd.DataFrame, week, year) -> list[dict]:
    """Compute per-player opportunity records for a single week (from PBP)."""
    def _blank():
        return {
            "player_id": "", "week": week, "season": year,
            "targets": 0, "red_zone_targets": 0, "end_zone_targets": 0,
            "carries": 0, "red_zone_carries": 0, "goal_line_carries": 0,
            "air_yards": 0, "touches": 0, "goal_line_touches": 0,
            "third_down_targets": 0, "deep_targets": 0, "short_targets": 0,
            "team": "",
        }

    opportunities: dict = defaultdict(_blank)
    team_totals: dict = defaultdict(lambda: {"total_targets": 0, "total_carries": 0})

    for _, play in week_data.iterrows():
        play_type = play.get("play_type", "")
        down = play.get("down", 0)
        yardline_100 = play.get("yardline_100", 100)
        air_yards = play.get("air_yards", 0) if pd.notna(play.get("air_yards")) else 0

        if play_type == "pass":
            receiver_id = play.get("receiver_player_id")
            posteam = play.get("posteam")
            if pd.notna(receiver_id):
                opp = opportunities[receiver_id]
                opp["player_id"] = receiver_id
                opp["targets"] += 1
                opp["touches"] += 1
                opp["air_yards"] += air_yards
                opp["team"] = posteam
                if pd.notna(posteam):
                    team_totals[posteam]["total_targets"] += 1
                if yardline_100 <= 20:
                    opp["red_zone_targets"] += 1
                if yardline_100 <= 10:
                    opp["end_zone_targets"] += 1
                    opp["goal_line_touches"] += 1
                if down == 3:
                    opp["third_down_targets"] += 1
                if air_yards >= 20:
                    opp["deep_targets"] += 1
                elif air_yards < 10:
                    opp["short_targets"] += 1

        elif play_type == "run":
            rusher_id = play.get("rusher_player_id")
            posteam = play.get("posteam")
            if pd.notna(rusher_id):
                opp = opportunities[rusher_id]
                opp["player_id"] = rusher_id
                opp["carries"] += 1
                opp["touches"] += 1
                opp["team"] = posteam
                if pd.notna(posteam):
                    team_totals[posteam]["total_carries"] += 1
                if yardline_100 <= 20:
                    opp["red_zone_carries"] += 1
                if yardline_100 <= 5:
                    opp["goal_line_carries"] += 1
                    opp["goal_line_touches"] += 1

    records = []
    for _player_id, stats in opportunities.items():
        team = stats["team"]
        if team and team in team_totals:
            tt = team_totals[team]["total_targets"]
            tc = team_totals[team]["total_carries"]
            stats["target_share"] = (stats["targets"] / tt * 100) if tt > 0 else 0
            stats["carry_share"] = (stats["carries"] / tc * 100) if tc > 0 else 0
        else:
            stats["target_share"] = 0
            stats["carry_share"] = 0
        records.append(stats)
    return records


def compute_opportunities(pbp_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Player-week opportunity records for all weeks in the PBP frame."""
    if pbp_df is None or pbp_df.empty or "week" not in pbp_df.columns:
        return pd.DataFrame()
    records: list[dict] = []
    for week in sorted(pbp_df["week"].dropna().unique()):
        week_data = pbp_df[pbp_df["week"] == week]
        records.extend(process_week_opportunities(week_data, int(week), season))
    return pd.DataFrame(records)


def add_roster_info(opp_df: pd.DataFrame, roster_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich opportunity records with player_name / position / team from rosters.

    Roster team wins over the PBP posteam; position defaults to 'Unknown'.
    """
    if opp_df.empty:
        return opp_df
    if roster_df is None or roster_df.empty or "player_id" not in roster_df.columns:
        opp_df = opp_df.copy()
        opp_df["player_name"] = opp_df["player_id"]
        opp_df["player_display_name"] = opp_df["player_id"]
        opp_df["position"] = "Unknown"
        return opp_df

    agg = {}
    for col in ("player_name", "position", "team"):
        if col in roster_df.columns:
            agg[col] = "first"
    player_info = roster_df.groupby("player_id").agg(agg).reset_index()

    opp_df = opp_df.copy()
    opp_df["opp_team"] = opp_df["team"]
    opp_df = opp_df.merge(player_info, on="player_id", how="left", suffixes=("", "_roster"))

    name_col = "player_name" if "player_name" in opp_df.columns else None
    opp_df["player_display_name"] = (
        opp_df[name_col].fillna(opp_df["player_id"]) if name_col else opp_df["player_id"]
    )
    if "team_roster" in opp_df.columns:
        opp_df["team"] = opp_df["team_roster"].fillna(opp_df["opp_team"])
    else:
        opp_df["team"] = opp_df["opp_team"]
    opp_df.drop(["opp_team", "team_roster"], axis=1, inplace=True, errors="ignore")

    if "position" in opp_df.columns:
        opp_df["position"] = opp_df["position"].fillna("Unknown")
    else:
        opp_df["position"] = "Unknown"
    if "player_name" not in opp_df.columns:
        opp_df["player_name"] = opp_df["player_display_name"]
    return opp_df


def calculate_opportunity_trends(opp_df: pd.DataFrame, min_weeks: int = 2) -> pd.DataFrame:
    """Per-player trend summary across weeks (avg / latest / max / trend / consistency)."""
    trend_records = []
    if opp_df.empty:
        return pd.DataFrame()

    for player_id, player_data in opp_df.groupby("player_id"):
        player_data = player_data.sort_values("week")
        if len(player_data) < min_weeks:
            continue

        if "player_display_name" in player_data.columns and pd.notna(
            player_data["player_display_name"].iloc[0]
        ):
            player_name = player_data["player_display_name"].iloc[0]
        elif "player_name" in player_data.columns and pd.notna(
            player_data["player_name"].iloc[0]
        ):
            player_name = player_data["player_name"].iloc[0]
        else:
            player_name = player_id

        position = (
            player_data["position"].iloc[0]
            if "position" in player_data.columns and pd.notna(player_data["position"].iloc[0])
            else "Unknown"
        )
        team = (
            player_data["team"].iloc[0]
            if "team" in player_data.columns and pd.notna(player_data["team"].iloc[0])
            else "Unknown"
        )

        record = {
            "player_id": player_id,
            "player_name": player_name,
            "position": position,
            "team": team,
            "weeks_played": len(player_data),
            "latest_week": int(player_data["week"].max()),
        }

        for metric in TREND_METRICS:
            if metric not in player_data.columns:
                continue
            values = player_data[metric].values.astype(float)
            record[f"{metric}_avg"] = float(np.mean(values))
            record[f"{metric}_latest"] = float(values[-1]) if len(values) > 0 else 0.0
            record[f"{metric}_max"] = float(np.max(values))

            if len(values) >= 2:
                if len(values) >= 3:
                    recent_avg = np.mean(values[-2:])
                    early_avg = np.mean(values[:-2])
                else:
                    recent_avg = values[-1]
                    early_avg = values[0]
                record[f"{metric}_trend"] = float(
                    ((recent_avg - early_avg) / max(early_avg, 0.1) * 100)
                    if early_avg > 0 else 0.0
                )
            else:
                record[f"{metric}_trend"] = 0.0

            record[f"{metric}_consistency"] = float(
                (np.std(values) / np.mean(values)) * 100 if np.mean(values) > 0 else 0.0
            )

        trend_records.append(record)

    return pd.DataFrame(trend_records)


def _enriched_opportunities(db: Session, season: int, week: Optional[int] = None) -> pd.DataFrame:
    """Load PBP + rosters and return enriched opportunity records."""
    pbp_df = load_pbp_df(db, season, week=week)
    opp_df = compute_opportunities(pbp_df, season)
    if opp_df.empty:
        return opp_df
    roster_df = load_weekly_rosters_df(db, season)
    return add_roster_info(opp_df, roster_df)


def _opp_row(s: pd.Series) -> dict:
    row = {
        "player_id": s.get("player_id"),
        "player_name": s.get("player_display_name") or s.get("player_name") or s.get("player_id"),
        "position": s.get("position") or "Unknown",
        "team": s.get("team"),
        "season": int(s.get("season")) if pd.notna(s.get("season")) else None,
        "week": int(s.get("week")) if pd.notna(s.get("week")) else None,
    }
    for f in _OPP_FIELDS:
        v = s.get(f)
        row[f] = float(v) if pd.notna(v) else 0.0
    for share in ("target_share", "carry_share"):
        v = s.get(share)
        row[share] = float(v) if pd.notna(v) else 0.0
    return row


@router.get("/")
def get_opportunities(
    season: Optional[int] = Query(None, description="Season year (default: current)"),
    week: Optional[int] = Query(None, description="Filter to a single week"),
    team: Optional[str] = Query(None, description="Filter to a team abbreviation"),
    db: Session = Depends(get_db),
):
    """Per-player-week opportunity metrics computed from play-by-play."""
    season = season or get_current_nfl_season()
    try:
        opp_df = _enriched_opportunities(db, season, week=week)
    except Exception as e:
        logger.error("Error computing opportunities: %s", e)
        return {"status": "no_data", "season": season, "total_records": 0, "data": []}

    if opp_df.empty:
        return {"status": "no_data", "season": season, "total_records": 0, "data": []}

    if team:
        opp_df = opp_df[opp_df["team"] == team.upper()]

    data = [_opp_row(s) for _, s in opp_df.iterrows()]
    if not data:
        return {"status": "no_data", "season": season, "total_records": 0, "data": []}
    return {
        "status": "success",
        "season": season,
        "total_records": len(data),
        "data": data,
    }


@router.get("/trends/")
def get_opportunity_trends(
    season: Optional[int] = Query(None, description="Season year (default: current)"),
    team: Optional[str] = Query(None, description="Filter to a team abbreviation"),
    db: Session = Depends(get_db),
):
    """Per-player opportunity trend summary (players with >= 2 weeks played)."""
    season = season or get_current_nfl_season()
    try:
        opp_df = _enriched_opportunities(db, season)
    except Exception as e:
        logger.error("Error computing opportunity trends: %s", e)
        return {"status": "no_data", "season": season, "total_players": 0, "data": []}

    if opp_df.empty:
        return {"status": "no_data", "season": season, "total_players": 0, "data": []}

    trends = calculate_opportunity_trends(opp_df)
    if trends.empty:
        return {"status": "no_data", "season": season, "total_players": 0, "data": []}

    if team:
        trends = trends[trends["team"] == team.upper()]

    from .utils import clean_data_for_json
    data = clean_data_for_json(trends)
    return {
        "status": "success",
        "season": season,
        "total_players": len(data),
        "data": data,
    }
