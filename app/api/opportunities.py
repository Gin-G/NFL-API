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
    "pass_attempts",
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
            "pass_attempts": 0, "team": "",
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
            # A sack is a dropback, not an attempt. Counted this way the
            # attempts match nflverse's box score to within spikes and
            # two-point tries (2026 weeks 1-9: 0.03 a game).
            passer_id = play.get("passer_player_id")
            if pd.notna(passer_id) and play.get("sack") != 1:
                qb = opportunities[passer_id]
                qb["player_id"] = passer_id
                qb["pass_attempts"] += 1
                qb["team"] = posteam
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


# ── Weekly series and a direction call, for charting ─────────────────────────
#
# `calculate_opportunity_trends` above reports a PERCENT change, which is what
# the NickKnows tables were ported with and what makes them hard to read: a
# player going from one target to three is "+200%" and sorts above a starter
# going from six to nine. The chart endpoint below states the change in the
# metric's own unit -- looks a game -- and refuses to call a direction the
# player's own week-to-week spread does not support.

#: The series a chart is drawn for. `opportunities` is targets plus carries,
#: or pass attempts plus carries for a quarterback -- the definition
#: `nfl_projections.opportunity` uses, so the chart and the consistency
#: rankings cannot disagree about who gets the ball.
WEEKLY_METRICS = {
    "opportunities": "opportunities",
    "targets": "targets",
    "carries": "carries",
    "pass_attempts": "pass_attempts",
}

#: Games that count as "lately" -- the same three `nfl_projections.consistency`
#: uses for its usage trend, so the two screens cannot disagree about a player.
RECENT_GAMES = 3

#: Games needed BEFORE those three. One earlier game is not a baseline.
MIN_EARLIER_GAMES = 2

#: The recent games have to sit inside this many calendar weeks, counted back
#: from the last week in the data: three games, a bye and one missed game.
#: Beyond that the "last three" straddle an absence -- two games in September
#: and one last Sunday -- and their average is not a current role.
RECENT_SPAN_WEEKS = 5

#: The smallest change worth a direction, in looks a game. Below this the
#: arrow would be reporting a difference nobody would act on.
MIN_CHANGE = {"opportunities": 2.0, "targets": 1.5, "carries": 1.5,
              "pass_attempts": 3.0}


def _direction(values: list, metric: str, *, recent_span: Optional[int] = None) -> dict:
    """Recent games against the earlier ones, for one player and one metric.

    `values` are his games in week order, weeks he did not appear in left out.
    The call is `up` or `down` only when the change clears BOTH bars: the
    metric's minimum, and the standard error of the difference between two
    means drawn from his own games. Three games against five is a small
    sample, so most real players come back `steady`, which is the honest
    answer and the reason the label means something when it is not.

    `recent_span` is how many calendar weeks back the oldest of the recent
    games sits. Past `RECENT_SPAN_WEEKS` the call is `interrupted`: the numbers
    are still reported, the direction is withheld.
    """
    games = len(values)
    out = {"games": games, "per_game": None, "recent": None, "earlier": None,
           "change": None, "noise": None, "direction": "too_few_games"}
    if not games:
        return out
    out["per_game"] = round(float(np.mean(values)), 2)
    if games < RECENT_GAMES + MIN_EARLIER_GAMES:
        return out

    recent = np.asarray(values[-RECENT_GAMES:], dtype=float)
    earlier = np.asarray(values[:-RECENT_GAMES], dtype=float)
    change = float(recent.mean() - earlier.mean())
    # Spread inside each window, pooled, so a step change between the windows
    # does not inflate the noise it is being judged against.
    within = (((recent - recent.mean()) ** 2).sum()
              + ((earlier - earlier.mean()) ** 2).sum())
    sd = float(np.sqrt(within / (games - 2)))
    noise = sd * float(np.sqrt(1.0 / len(recent) + 1.0 / len(earlier)))
    bar = max(MIN_CHANGE[metric], noise)
    out.update(
        recent=round(float(recent.mean()), 2),
        earlier=round(float(earlier.mean()), 2),
        change=round(change, 2),
        noise=round(noise, 2),
        direction="up" if change >= bar else "down" if change <= -bar else "steady",
    )
    if recent_span is not None and recent_span > RECENT_SPAN_WEEKS:
        out["direction"] = "interrupted"
    return out


def weekly_series(opp_df: pd.DataFrame) -> tuple[list, list]:
    """(weeks, players): every player's week-by-week series with its direction.

    A week the player has no row for is `None`, never 0. The records come from
    play-by-play, so a missing week is a bye, an inactive or a game without a
    look, and those cannot be told apart from here; drawing it as a zero would
    turn every bye into a collapse and every injury into a downtrend.
    """
    if opp_df is None or opp_df.empty:
        return [], []
    weeks = sorted(int(w) for w in opp_df["week"].dropna().unique())
    opp_df = opp_df.copy()
    if "pass_attempts" not in opp_df.columns:
        opp_df["pass_attempts"] = 0.0
    is_qb = opp_df.get("position", pd.Series("", index=opp_df.index)) == "QB"
    opp_df["opportunities"] = np.where(
        is_qb, opp_df["pass_attempts"].fillna(0) + opp_df["carries"].fillna(0),
        opp_df["touches"].fillna(0))
    players = []
    for player_id, rows in opp_df.groupby("player_id"):
        rows = rows.sort_values("week").drop_duplicates("week", keep="last")
        latest = rows.iloc[-1]
        name = None
        for col in ("player_display_name", "player_name"):
            if col in rows.columns and pd.notna(latest.get(col)):
                name = latest.get(col)
                break
        by_week = {int(r["week"]): r for _, r in rows.iterrows()}
        metrics = {}
        played_weeks = [w for w in weeks if w in by_week]
        # Inclusive: a player whose last three games are weeks 7, 8 and 9 of a
        # nine-week record has a span of three.
        span = (weeks[-1] - played_weeks[-RECENT_GAMES] + 1
                if len(played_weeks) >= RECENT_GAMES else None)
        for metric, column in WEEKLY_METRICS.items():
            played = [float(by_week[w][column]) for w in played_weeks]
            metrics[metric] = {
                "values": [float(by_week[w][column]) if w in by_week else None
                           for w in weeks],
                **_direction(played, metric, recent_span=span),
            }
        position = latest.get("position") if "position" in rows.columns else None
        players.append({
            "player_id": player_id,
            "player_name": name or player_id,
            "position": position if pd.notna(position) else "Unknown",
            "team": latest.get("team") if pd.notna(latest.get("team")) else None,
            "games": len(by_week),
            "latest_week": int(latest["week"]),
            "metrics": metrics,
        })
    return weeks, players


#: The weekly series is a full season of play-by-play walked row by row --
#: seconds per request. It only changes when the loader adds plays, so it is
#: kept per season and thrown away when the table's row count or last week
#: moves, or after WEEKLY_CACHE_SECONDS whatever happens.
WEEKLY_CACHE_SECONDS = 6 * 3600
_weekly_cache: dict = {}


def _pbp_fingerprint(db: Session, season: int):
    """(rows, last week) of the season's regular-season plays, or None."""
    from sqlalchemy import text
    try:
        row = db.execute(text(
            "SELECT COUNT(*), MAX(week) FROM play_by_play "
            "WHERE season = :season AND season_type = 'REG'"),
            {"season": season}).first()
        return (int(row[0]), int(row[1]) if row[1] is not None else None)
    except Exception:
        db.rollback()
        return None


def cached_weekly_series(db: Session, season: int) -> tuple[list, list]:
    """`weekly_series` for a season, recomputed only when the plays change."""
    import time
    fingerprint = _pbp_fingerprint(db, season)
    hit = _weekly_cache.get(season)
    now = time.time()
    if (hit and fingerprint is not None and hit["fingerprint"] == fingerprint
            and now - hit["at"] < WEEKLY_CACHE_SECONDS):
        return hit["weeks"], hit["players"]
    weeks, players = weekly_series(_enriched_opportunities(db, season))
    # Nothing is kept without a fingerprint: then there is no way to know when
    # it has gone stale, and a stale chart is worse than a slow one.
    if fingerprint is not None and players:
        _weekly_cache[season] = {"fingerprint": fingerprint, "at": now,
                                 "weeks": weeks, "players": players}
    return weeks, players


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


@router.get("/weekly/")
def get_opportunity_weekly(
    season: Optional[int] = Query(None, description="Season year (default: current)"),
    team: Optional[str] = Query(None, description="Filter to a team abbreviation"),
    position: Optional[str] = Query(None, description="QB, RB, WR or TE"),
    min_per_game: float = Query(
        3.0, ge=0, description="Opportunities a game to be included. A player "
        "seeing one look a week has no trend to draw."),
    db: Session = Depends(get_db),
):
    """Week-by-week opportunities, targets, carries and pass attempts per player.

    Each player carries three series aligned to `weeks`, and for each one a
    direction: his last three games against the games before them. `up` and
    `down` are only called when the change is at least `trend_rule.min_change`
    looks a game AND larger than the noise in his own weekly numbers; everyone
    else is `steady`, a player with fewer than five games is `too_few_games`,
    and one whose last three games straddle a long absence is `interrupted`
    rather than guessed at.

    The direction describes what has happened. It is not a projection and
    nothing here feeds one.
    """
    season = season or get_current_nfl_season()
    empty = {"status": "no_data", "season": season, "weeks": [],
             "total_players": 0, "data": []}
    try:
        weeks, players = cached_weekly_series(db, season)
    except Exception as e:
        logger.error("Error computing weekly opportunities: %s", e)
        return empty

    if team:
        players = [p for p in players if (p["team"] or "").upper() == team.upper()]
    if position:
        players = [p for p in players
                   if (p["position"] or "").upper() == position.upper()]
    players = [p for p in players
               if (p["metrics"]["opportunities"]["per_game"] or 0.0) >= min_per_game]
    if not players:
        return empty
    players.sort(key=lambda p: -(p["metrics"]["opportunities"]["per_game"] or 0.0))

    return {
        "status": "success",
        "season": season,
        "weeks": weeks,
        "metrics": list(WEEKLY_METRICS),
        "total_players": len(players),
        "trend_rule": {
            "recent_games": RECENT_GAMES,
            "min_games": RECENT_GAMES + MIN_EARLIER_GAMES,
            "min_change": MIN_CHANGE,
            "recent_span_weeks": RECENT_SPAN_WEEKS,
            "note": ("change = mean of his last %d games minus the mean of the "
                     "games before them, in looks a game. up/down needs the "
                     "change to clear both min_change and `noise`, the "
                     "standard error of that difference from his own games. "
                     "`interrupted` means his last %d games reach back more "
                     "than %d weeks, across an absence, so no direction is "
                     "called. A "
                     "week with no row is null, not zero: a bye, an inactive "
                     "and a game without a look are indistinguishable in "
                     "play-by-play." % (RECENT_GAMES, RECENT_GAMES,
                                        RECENT_SPAN_WEEKS)),
        },
        "min_per_game": min_per_game,
        "data": players,
    }
