#!/usr/bin/env python3
"""
API Utilities
Shared functions and utilities for the NFL API
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def _to_pandas(df):
    """Convert a Polars DataFrame to Pandas, or return as-is if already Pandas.

    nflreadpy returns Polars DataFrames; test fixtures supply Pandas DataFrames
    directly.  This helper lets both paths work without changing either.
    """
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def get_current_nfl_season() -> int:
    """Return the year of the current or most recently completed NFL season.

    The NFL season is named for the year it starts (September).
    January–August refer to the previous year's season (playoffs / off-season).

    Examples
    --------
    Called on 2026-02-23  → 2025  (2025 season playoffs/Super Bowl window)
    Called on 2026-09-10  → 2026  (2026 season has just kicked off)
    Called on 2027-03-01  → 2026  (2026 season is complete, off-season)
    """
    now = datetime.now()
    return now.year if now.month >= 9 else now.year - 1


def clean_data_for_json(data):
    """Clean data to make it JSON serializable - simplified reliable version"""
    try:
        if isinstance(data, dict):
            return {k: clean_data_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [clean_data_for_json(item) for item in data]
        elif isinstance(data, pd.DataFrame):
            logger.debug(f"Processing DataFrame with shape: {data.shape}")
            
            # Use the reliable approach - convert to records first, then clean
            try:
                raw_records = data.to_dict('records')
                cleaned_records = []
                
                for record in raw_records:
                    cleaned_record = {}
                    for key, value in record.items():
                        # Handle different types of problematic values
                        if pd.isna(value):
                            cleaned_record[key] = None
                        elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                            cleaned_record[key] = None
                        elif isinstance(value, (np.integer, np.floating)):
                            # Convert numpy types to Python types
                            if np.isnan(value) or np.isinf(value):
                                cleaned_record[key] = None
                            else:
                                cleaned_record[key] = value.item()
                        elif isinstance(value, np.ndarray):
                            cleaned_record[key] = value.tolist()
                        else:
                            cleaned_record[key] = value
                    cleaned_records.append(cleaned_record)
                
                logger.debug(f"✅ DataFrame cleaned successfully: {len(cleaned_records)} records")
                return cleaned_records
                
            except Exception as df_error:
                logger.error(f"Error processing DataFrame: {df_error}")
                # Final fallback - return empty list rather than crash
                logger.warning("Returning empty list as fallback")
                return []
                
        elif isinstance(data, pd.Series):
            try:
                # Convert series to list, handling NaN values
                result = []
                for value in data:
                    if pd.isna(value) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                        result.append(None)
                    elif isinstance(value, (np.integer, np.floating)):
                        result.append(value.item())
                    else:
                        result.append(value)
                return result
            except Exception:
                return data.tolist()
                
        elif isinstance(data, (np.integer, np.floating)):
            if np.isnan(data) or np.isinf(data):
                return None
            return data.item()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif pd.isna(data) or (isinstance(data, float) and (np.isnan(data) or np.isinf(data))):
            return None
        else:
            return data
            
    except Exception as e:
        logger.error(f"Unexpected error in clean_data_for_json: {e}")
        logger.error(f"Data type: {type(data)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Return a safe fallback rather than crashing
        if isinstance(data, pd.DataFrame):
            return []
        elif isinstance(data, (list, dict)):
            return data
        else:
            return None

def _orm_to_dict(obj) -> dict:
    """Convert a SQLAlchemy ORM row to a plain Python dict.

    Strips SA internal state keys (those starting with '_').
    """
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}


def _normalize_stats_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map nflreadpy load_player_stats column names to the API schema."""
    return df.rename(columns={
        "team": "recent_team",
        "passing_interceptions": "interceptions",
    })


# ---------------------------------------------------------------------------
# DB-first DataFrame loaders
#
# These return pandas DataFrames from the database when a season has been
# loaded, and fall back to nflreadpy otherwise. They back the NickKnows-parity
# endpoints (stat leaders, FPA, opportunities, team results) which port Celery
# pipelines that operated on DataFrames, so a DataFrame is the natural currency.
# ---------------------------------------------------------------------------

def load_weekly_stats_df(db, season: int) -> pd.DataFrame:
    """Weekly player stats for a season as a DataFrame (DB-first)."""
    try:
        from database.models import PlayerStat
        rows = db.query(PlayerStat).filter(PlayerStat.season == season).all()
        if rows:
            return pd.DataFrame([_orm_to_dict(r) for r in rows])
    except Exception as e:
        logger.warning("DB weekly stats unavailable, falling back to nflreadpy: %s", e)
    import nflreadpy as nfl
    return _normalize_stats_columns(_to_pandas(nfl.load_player_stats(seasons=[season])))


def load_schedules_df(db, season: int) -> pd.DataFrame:
    """Schedules for a season as a DataFrame (DB-first)."""
    try:
        from database.models import Schedule
        rows = db.query(Schedule).filter(Schedule.season == season).all()
        if rows:
            return pd.DataFrame([_orm_to_dict(r) for r in rows])
    except Exception as e:
        logger.warning("DB schedules unavailable, falling back to nflreadpy: %s", e)
    import nflreadpy as nfl
    return _to_pandas(nfl.load_schedules(seasons=[season]))


def load_weekly_rosters_df(db, season: int) -> pd.DataFrame:
    """Weekly rosters for a season as a DataFrame (DB-first)."""
    try:
        from database.models import PlayerRoster
        rows = db.query(PlayerRoster).filter(PlayerRoster.season == season).all()
        if rows:
            return pd.DataFrame([_orm_to_dict(r) for r in rows])
    except Exception as e:
        logger.warning("DB rosters unavailable, falling back to nflreadpy: %s", e)
    from nflverse_compat import load_rosters_weekly
    df = _to_pandas(load_rosters_weekly(season))
    return df.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})


# Columns needed to compute opportunities from play-by-play.
_PBP_OPP_COLUMNS = [
    "play_type", "down", "yardline_100", "air_yards",
    "receiver_player_id", "rusher_player_id", "posteam", "week",
]


def load_pbp_df(db, season: int, week: Optional[int] = None) -> pd.DataFrame:
    """Regular-season play-by-play columns needed for opportunity metrics.

    DB-first via raw SQL against the play_by_play table; falls back to
    nflreadpy. Always restricted to season_type == 'REG'.
    """
    from sqlalchemy import text
    try:
        conditions = ["season = :season", "season_type = 'REG'"]
        params: dict = {"season": season}
        if week is not None:
            conditions.append("week = :week")
            params["week"] = week
        where = " AND ".join(conditions)
        cols = ", ".join(_PBP_OPP_COLUMNS)
        sql = f"SELECT {cols} FROM play_by_play WHERE {where}"
        rows = db.execute(text(sql), params).mappings().all()
        if rows:
            return pd.DataFrame([dict(r) for r in rows])
    except Exception as e:
        logger.warning("DB PBP unavailable, falling back to nflreadpy: %s", e)
    import nflreadpy as nfl
    df = _to_pandas(nfl.load_pbp(seasons=[season]))
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    if week is not None:
        df = df[df["week"] == week]
    keep = [c for c in _PBP_OPP_COLUMNS if c in df.columns]
    return df[keep].copy()


def check_grading_systems():
    """Check availability of grading systems"""
    systems = {
        "player_grading": False,
        "coaching_analytics": False
    }
    
    try:
        from functions.players.grading import EnhancedNFLPlayerGrader
        systems["player_grading"] = True
        logger.info("✅ Player grading system available")
    except Exception as e:
        logger.warning(f"⚠️ Player grading system not available: {e}")
    
    try:
        from functions.coaching.grading import RosterAwareCoachingAnalytics
        systems["coaching_analytics"] = True
        logger.info("✅ Coaching analytics system available")
    except Exception as e:
        logger.warning(f"⚠️ Coaching analytics system not available: {e}")
    
    return systems

def get_player_grader(years):
    """Get player grader instance"""
    try:
        from functions.players.grading import EnhancedNFLPlayerGrader
        return EnhancedNFLPlayerGrader(years=years)
    except Exception as e:
        logger.error(f"Failed to initialize player grader: {e}")
        raise

def get_sportradar_coaches_client():
    """Return a SportradarCoachesClient if SPORTRADAR_API_KEY is set, else None."""
    api_key = os.getenv("SPORTRADAR_API_KEY")
    if not api_key:
        return None
    try:
        from functions.data.sportradar_coaches import SportradarCoachesClient
        access_level = os.getenv("SPORTRADAR_ACCESS_LEVEL", "trial")
        return SportradarCoachesClient(api_key, access_level)
    except Exception as e:
        logger.error(f"Failed to initialise SportradarCoachesClient: {e}")
        return None


def get_coaching_analytics(years):
    """Get coaching analytics instance"""
    try:
        from functions.coaching.grading import RosterAwareCoachingAnalytics
        analytics = RosterAwareCoachingAnalytics(years=years)
        analytics.load_data()
        analytics.extract_coaching_info()
        return analytics
    except Exception as e:
        logger.error(f"Failed to initialize coaching analytics: {e}")
        raise