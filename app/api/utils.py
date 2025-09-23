#!/usr/bin/env python3
"""
API Utilities
Shared functions and utilities for the NFL API
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_data_for_json(data):
    """Clean data to make it JSON serializable"""
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, pd.DataFrame):
        # Replace NaN and infinite values
        df_cleaned = data.replace([np.inf, -np.inf], None).fillna(value=None)
        return df_cleaned.to_dict('records')
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