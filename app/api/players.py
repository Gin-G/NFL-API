#!/usr/bin/env python3
"""
Players API Router
Handles all player-related endpoints including grading
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import nfl_data_py as nfl
import pandas as pd
import logging
from .utils import clean_data_for_json, get_player_grader, check_grading_systems

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/rosters")
async def get_rosters(
    season: Optional[int] = Query(2023, description="Season year"),
    week: Optional[int] = Query(None, description="Specific week"),
    team: Optional[str] = Query(None, description="Team abbreviation"),
    position: Optional[str] = Query(None, description="Player position")
):
    """Get player rosters with optional filters"""
    try:
        rosters = nfl.import_weekly_rosters([season])
        
        if week is not None:
            rosters = rosters[rosters['week'] == week]
        
        if team is not None:
            rosters = rosters[rosters['team'] == team.upper()]
        
        if position is not None:
            rosters = rosters[rosters['position'] == position.upper()]
        
        return {
            "status": "success",
            "season": season,
            "total_players": len(rosters),
            "data": clean_data_for_json(rosters)
        }
    except Exception as e:
        logger.error(f"Error fetching rosters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_player_stats(
    season: Optional[int] = Query(2023, description="Season year"),
    player_id: Optional[str] = Query(None, description="Specific player ID"),
    position: Optional[str] = Query(None, description="Player position"),
    team: Optional[str] = Query(None, description="Team abbreviation"),
    week: Optional[int] = Query(None, description="Specific week")
):
    """Get player statistics with optional filters"""
    try:
        stats = nfl.import_weekly_data([season])
        
        if player_id is not None:
            stats = stats[stats['player_id'] == player_id]
        
        if position is not None:
            stats = stats[stats['position'] == position.upper()]
        
        if team is not None:
            stats = stats[stats['recent_team'] == team.upper()]
        
        if week is not None:
            stats = stats[stats['week'] == week]
        
        return {
            "status": "success",
            "season": season,
            "total_records": len(stats),
            "data": clean_data_for_json(stats)
        }
    except Exception as e:
        logger.error(f"Error fetching player stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{player_id}")
async def get_player_details(
    player_id: str,
    season: Optional[int] = Query(2023, description="Season year")
):
    """Get detailed information for a specific player"""
    try:
        # Get player stats
        stats = nfl.import_weekly_data([season])
        player_stats = stats[stats['player_id'] == player_id]
        
        # Get roster info
        rosters = nfl.import_weekly_rosters([season])
        player_roster = rosters[rosters['player_id'] == player_id]
        
        if player_stats.empty and player_roster.empty:
            raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
        
        # Combine data
        player_info = {}
        if not player_roster.empty:
            latest_roster = player_roster.iloc[-1]
            player_info = {
                "player_id": player_id,
                "player_name": latest_roster.get('player_name', 'Unknown'),
                "position": latest_roster.get('position', 'Unknown'),
                "team": latest_roster.get('team', 'Unknown'),
                "height": latest_roster.get('height', None),
                "weight": latest_roster.get('weight', None),
                "college": latest_roster.get('college', None),
                "rookie_year": latest_roster.get('rookie_year', None)
            }
        
        return {
            "status": "success",
            "player_info": clean_data_for_json(player_info),
            "season_stats": clean_data_for_json(player_stats) if not player_stats.empty else [],
            "games_played": len(player_stats)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching player {player_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/grades")
async def get_player_grades(
    years: List[int] = Query([2023], description="Years to analyze"),
    min_games: int = Query(3, description="Minimum games played"),
    limit: int = Query(20, description="Maximum results to return")
):
    """Get player performance grades"""
    systems = check_grading_systems()
    if not systems["player_grading"]:
        raise HTTPException(status_code=503, detail="Player grading system not available")
    
    try:
        grader = get_player_grader(years)
        all_grades = grader.calculate_all_grades(min_games=min_games)
        
        # Handle if all_grades is a dict instead of DataFrame
        if isinstance(all_grades, dict):
            if not all_grades or all(df.empty for df in all_grades.values() if hasattr(df, 'empty')):
                return {"status": "success", "message": "No grades calculated", "data": []}
            
            # Combine all DataFrames from the dict
            grade_dfs = []
            for category, df in all_grades.items():
                if hasattr(df, 'empty') and not df.empty:
                    df_copy = df.copy()
                    df_copy['grade_category'] = category
                    grade_dfs.append(df_copy)
            
            if not grade_dfs:
                return {"status": "success", "message": "No grades calculated", "data": []}
            
            combined_grades = pd.concat(grade_dfs, ignore_index=True)
            
            # Get top players from combined data
            if 'numeric_grade' in combined_grades.columns:
                top_players = combined_grades.nlargest(limit, 'numeric_grade')
            else:
                top_players = combined_grades.head(limit)
            
        elif hasattr(all_grades, 'empty'):
            # Original DataFrame logic
            if all_grades.empty:
                return {"status": "success", "message": "No grades calculated", "data": []}
            
            # Get top players
            outliers = grader.identify_performance_outliers(all_grades)
            top_players = grader.get_top_performers(outliers, n=limit)
        else:
            return {"status": "error", "message": "Unexpected data format from grading system"}
        
        return {
            "status": "success",
            "total_players": len(top_players),
            "data": clean_data_for_json(top_players)
        }
    except Exception as e:
        logger.error(f"Error calculating player grades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/grades/{player_name}")
async def get_player_grade_details(
    player_name: str,
    years: List[int] = Query([2023], description="Years to analyze"),
    min_games: int = Query(1, description="Minimum games played")
):
    """Get detailed grade information for a specific player"""
    systems = check_grading_systems()
    if not systems["player_grading"]:
        raise HTTPException(status_code=503, detail="Player grading system not available")
    
    try:
        grader = get_player_grader(years)
        all_grades = grader.calculate_all_grades(min_games=min_games)
        
        # Handle different return types from grading system
        if isinstance(all_grades, dict):
            # Combine all DataFrames
            grade_dfs = []
            for category, df in all_grades.items():
                if hasattr(df, 'empty') and not df.empty:
                    df_copy = df.copy()
                    df_copy['grade_category'] = category
                    grade_dfs.append(df_copy)
            
            if not grade_dfs:
                raise HTTPException(status_code=404, detail=f"No grades found for player '{player_name}'")
            
            combined_grades = pd.concat(grade_dfs, ignore_index=True)
            
            # Filter to specific player
            if 'player_name' in combined_grades.columns:
                player_data = combined_grades[combined_grades['player_name'].str.contains(player_name, case=False)]
            else:
                raise HTTPException(status_code=404, detail=f"Player name column not found")
            
        else:
            outliers = grader.identify_performance_outliers(all_grades)
            player_data = outliers[outliers['player_name'].str.contains(player_name, case=False)]
        
        if player_data.empty:
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")
        
        return {
            "status": "success",
            "player_name": player_name,
            "total_records": len(player_data),
            "data": clean_data_for_json(player_data)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting player grade details: {e}")
        raise HTTPException(status_code=500, detail=str(e))