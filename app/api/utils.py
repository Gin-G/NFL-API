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