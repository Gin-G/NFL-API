#!/usr/bin/env python3
"""
Data loader: imports nflreadpy data into the database.

Designed to be idempotent — uses SQLAlchemy merge() which does
SELECT + INSERT or UPDATE based on the primary key.
"""

import logging
import pandas as pd
import nflreadpy as nfl
from sqlalchemy.orm import Session

from .models import Team, Schedule, PlayerRoster, PlayerStat

logger = logging.getLogger(__name__)


def _to_pandas(df) -> pd.DataFrame:
    """Convert Polars → Pandas, or return as-is if already Pandas."""
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _safe(value):
    """Convert numpy/pandas scalar to a plain Python value; return None for NaN/inf."""
    if value is None:
        return None
    try:
        import math
        import numpy as np
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, (np.floating, np.integer)):
            v = value.item()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _normalize_roster_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
    if "player_display_name" not in df.columns:
        df = df.copy()
        df["player_display_name"] = df.get("player_name")
    if "height" in df.columns and df["height"].dtype != object:
        df = df.copy()
        df["height"] = df["height"].apply(
            lambda h: f"{int(h)//12}-{int(h)%12}" if pd.notna(h) else None
        )
    return df


def _normalize_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "team": "recent_team",
        "passing_interceptions": "interceptions",
    })


def _col(row, name, default=None):
    """Safely get a column value from a namedtuple row."""
    try:
        v = getattr(row, name, default)
        return _safe(v) if v is not None else default
    except Exception:
        return default


def load_teams(db: Session) -> int:
    """Load/update team data. Returns number of rows processed."""
    logger.info("Loading teams…")
    df = _to_pandas(nfl.load_teams())
    count = 0
    for row in df.itertuples(index=False):
        db.merge(Team(
            team_abbr=_col(row, "team_abbr"),
            team_name=_col(row, "team_name"),
            team_conf=_col(row, "team_conf"),
            team_division=_col(row, "team_division"),
            team_color=_col(row, "team_color"),
            team_color2=_col(row, "team_color2"),
            team_color3=_col(row, "team_color3"),
            team_color4=_col(row, "team_color4"),
            team_logo_espn=_col(row, "team_logo_espn"),
            team_wordmark=_col(row, "team_wordmark"),
            team_conference_logo=_col(row, "team_conference_logo"),
            team_league_logo=_col(row, "team_league_logo"),
        ))
        count += 1
    db.commit()
    logger.info("Teams loaded: %d rows", count)
    return count


def load_schedules(db: Session, season: int) -> int:
    """Load/update schedule data for a season."""
    logger.info("Loading schedules for season %d…", season)
    df = _to_pandas(nfl.load_schedules(seasons=[season]))
    count = 0
    for row in df.itertuples(index=False):
        game_id = _col(row, "game_id")
        if not game_id:
            continue
        db.merge(Schedule(
            game_id=game_id,
            season=_col(row, "season"),
            week=_col(row, "week"),
            game_type=_col(row, "game_type"),
            gameday=_col(row, "gameday"),
            gametime=_col(row, "gametime"),
            home_team=_col(row, "home_team"),
            away_team=_col(row, "away_team"),
            home_score=_col(row, "home_score"),
            away_score=_col(row, "away_score"),
            location=_col(row, "location"),
            result=_col(row, "result"),
            total=_col(row, "total"),
            overtime=_col(row, "overtime"),
            old_game_id=_col(row, "old_game_id"),
            pfr=_col(row, "pfr"),
            pff_game_id=_col(row, "pff_game_id"),
            espn=_col(row, "espn"),
            away_rest=_col(row, "away_rest"),
            home_rest=_col(row, "home_rest"),
            away_moneyline=_col(row, "away_moneyline"),
            home_moneyline=_col(row, "home_moneyline"),
            spread_line=_col(row, "spread_line"),
            away_spread_odds=_col(row, "away_spread_odds"),
            home_spread_odds=_col(row, "home_spread_odds"),
            total_line=_col(row, "total_line"),
            under_odds=_col(row, "under_odds"),
            over_odds=_col(row, "over_odds"),
            div_game=_col(row, "div_game"),
            roof=_col(row, "roof"),
            surface=_col(row, "surface"),
            temp=_col(row, "temp"),
            wind=_col(row, "wind"),
            away_qb_id=_col(row, "away_qb_id"),
            home_qb_id=_col(row, "home_qb_id"),
            away_qb_name=_col(row, "away_qb_name"),
            home_qb_name=_col(row, "home_qb_name"),
            away_coach=_col(row, "away_coach"),
            home_coach=_col(row, "home_coach"),
            referee=_col(row, "referee"),
            stadium_id=_col(row, "stadium_id"),
            stadium=_col(row, "stadium"),
        ))
        count += 1
    db.commit()
    logger.info("Schedules loaded for %d: %d rows", season, count)
    return count


def load_rosters(db: Session, season: int) -> int:
    """Load/update weekly roster data for a season. Available from 2002 onwards."""
    if season < 2002:
        logger.info("Skipping rosters for %d — data only available from 2002", season)
        return 0
    logger.info("Loading rosters for season %d…", season)
    df = _normalize_roster_df(_to_pandas(nfl.load_rosters_weekly(seasons=[season])))
    count = 0
    for row in df.itertuples(index=False):
        player_id = _col(row, "player_id")
        season_val = _col(row, "season")
        week_val = _col(row, "week")
        if not player_id or season_val is None or week_val is None:
            continue
        db.merge(PlayerRoster(
            player_id=player_id,
            season=season_val,
            week=week_val,
            player_name=_col(row, "player_name"),
            player_display_name=_col(row, "player_display_name"),
            position=_col(row, "position"),
            team=_col(row, "team"),
            height=_col(row, "height"),
            weight=_col(row, "weight"),
            college=_col(row, "college"),
            rookie_year=_col(row, "rookie_year"),
            jersey_number=_col(row, "jersey_number"),
            birth_date=_col(row, "birth_date"),
            status=_col(row, "status"),
            status_description_abbr=_col(row, "status_description_abbr"),
            football_name=_col(row, "football_name"),
            esb_id=_col(row, "esb_id"),
            gsis_it_id=_col(row, "gsis_it_id"),
            smart_id=_col(row, "smart_id"),
            entry_year=_col(row, "entry_year"),
            draft_club=_col(row, "draft_club"),
            draft_number=_col(row, "draft_number"),
            years_exp=_col(row, "years_exp"),
            headshot_url=_col(row, "headshot_url"),
        ))
        count += 1
    db.commit()
    logger.info("Rosters loaded for %d: %d rows", season, count)
    return count


def load_player_stats(db: Session, season: int) -> int:
    """Load/update weekly player stats for a season."""
    logger.info("Loading player stats for season %d…", season)
    try:
        df = _normalize_stats_df(_to_pandas(nfl.load_player_stats(seasons=[season])))
    except Exception as exc:
        logger.warning("Skipping player stats for %d: %s", season, exc)
        return 0
    count = 0
    for row in df.itertuples(index=False):
        player_id = _col(row, "player_id")
        season_val = _col(row, "season")
        week_val = _col(row, "week")
        if not player_id or season_val is None or week_val is None:
            continue
        db.merge(PlayerStat(
            player_id=player_id,
            season=season_val,
            week=week_val,
            player_name=_col(row, "player_name"),
            player_display_name=_col(row, "player_display_name"),
            position=_col(row, "position"),
            position_group=_col(row, "position_group"),
            recent_team=_col(row, "recent_team"),
            season_type=_col(row, "season_type"),
            completions=_col(row, "completions"),
            attempts=_col(row, "attempts"),
            passing_yards=_col(row, "passing_yards"),
            passing_tds=_col(row, "passing_tds"),
            interceptions=_col(row, "interceptions"),
            sacks=_col(row, "sacks"),
            sack_yards=_col(row, "sack_yards"),
            sack_fumbles=_col(row, "sack_fumbles"),
            sack_fumbles_lost=_col(row, "sack_fumbles_lost"),
            passing_air_yards=_col(row, "passing_air_yards"),
            passing_yards_after_catch=_col(row, "passing_yards_after_catch"),
            passing_first_downs=_col(row, "passing_first_downs"),
            passing_epa=_col(row, "passing_epa"),
            passing_2pt_conversions=_col(row, "passing_2pt_conversions"),
            pacr=_col(row, "pacr"),
            dakota=_col(row, "dakota"),
            carries=_col(row, "carries"),
            rushing_yards=_col(row, "rushing_yards"),
            rushing_tds=_col(row, "rushing_tds"),
            rushing_fumbles=_col(row, "rushing_fumbles"),
            rushing_fumbles_lost=_col(row, "rushing_fumbles_lost"),
            rushing_first_downs=_col(row, "rushing_first_downs"),
            rushing_epa=_col(row, "rushing_epa"),
            rushing_2pt_conversions=_col(row, "rushing_2pt_conversions"),
            receptions=_col(row, "receptions"),
            targets=_col(row, "targets"),
            receiving_yards=_col(row, "receiving_yards"),
            receiving_tds=_col(row, "receiving_tds"),
            receiving_fumbles=_col(row, "receiving_fumbles"),
            receiving_fumbles_lost=_col(row, "receiving_fumbles_lost"),
            receiving_air_yards=_col(row, "receiving_air_yards"),
            receiving_yards_after_catch=_col(row, "receiving_yards_after_catch"),
            receiving_first_downs=_col(row, "receiving_first_downs"),
            receiving_epa=_col(row, "receiving_epa"),
            receiving_2pt_conversions=_col(row, "receiving_2pt_conversions"),
            racr=_col(row, "racr"),
            target_share=_col(row, "target_share"),
            air_yards_share=_col(row, "air_yards_share"),
            wopr=_col(row, "wopr"),
            special_teams_tds=_col(row, "special_teams_tds"),
            fantasy_points=_col(row, "fantasy_points"),
            fantasy_points_ppr=_col(row, "fantasy_points_ppr"),
        ))
        count += 1
    db.commit()
    logger.info("Player stats loaded for %d: %d rows", season, count)
    return count


def load_all_data(db: Session, seasons: list) -> None:
    """Load all nflreadpy data into the DB. Idempotent (upsert via merge)."""
    load_teams(db)
    for season in seasons:
        logger.info("=== Processing season %d ===", season)
        try:
            load_schedules(db, season)
        except Exception as exc:
            logger.error("Failed to load schedules for %d: %s", season, exc)
        try:
            load_rosters(db, season)
        except Exception as exc:
            logger.error("Failed to load rosters for %d: %s", season, exc)
        try:
            load_player_stats(db, season)
        except Exception as exc:
            logger.error("Failed to load player stats for %d: %s", season, exc)
