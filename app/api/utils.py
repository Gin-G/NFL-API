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