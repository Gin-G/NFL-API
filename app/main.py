#!/usr/bin/env python3
"""
NFL API - FastAPI Application
Main application file with all endpoints for NFL data and grading
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import uvicorn
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime
import logging

# Import our grading systems
from functions.players.grading import EnhancedNFLPlayerGrader
from functions.coaching.grading import NFLCoachingAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NFL Analytics API",
    description="Comprehensive NFL data and grading API with player and coaching analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
player_grader = None
coaching_analytics = None
cached_data = {}

# Dependency to get player grader
def get_player_grader(years: List[int] = [2023, 2024]):
    global player_grader
    if player_grader is None or getattr(player_grader, 'years', None) != years:
        logger.info(f"Initializing player grader for years: {years}")
        player_grader = EnhancedNFLPlayerGrader(years=years)
    return player_grader

# Dependency to get coaching analytics
def get_coaching_analytics(years: List[int] = [2023, 2024]):
    global coaching_analytics
    if coaching_analytics is None or getattr(coaching_analytics, 'years', None) != years:
        logger.info(f"Initializing coaching analytics for years: {years}")
        coaching_analytics = NFLCoachingAnalytics(years=years)
        coaching_analytics.load_data()
        coaching_analytics.extract_coaching_info()
    return coaching_analytics

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NFL Analytics API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "teams": "/teams",
            "players": "/players",
            "coaches": "/coaches",
            "schedules": "/schedules",
            "grades": "/grades"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# ============================================================================
# TEAM ENDPOINTS
# ============================================================================

@app.get("/teams")
async def get_teams():
    """Get all NFL teams"""
    try:
        teams_data = nfl.import_team_desc()
        return {
            "status": "success",
            "data": teams_data.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/teams/{team_abbr}")
async def get_team_details(team_abbr: str):
    """Get details for a specific team"""
    try:
        teams_data = nfl.import_team_desc()
        team = teams_data[teams_data['team_abbr'] == team_abbr.upper()]
        
        if team.empty:
            raise HTTPException(status_code=404, detail=f"Team {team_abbr} not found")
        
        return {
            "status": "success",
            "data": team.to_dict('records')[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching team {team_abbr}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SCHEDULE ENDPOINTS
# ============================================================================

@app.get("/schedules")
async def get_schedules(
    season: Optional[int] = Query(2023, description="Season year"),
    week: Optional[int] = Query(None, description="Specific week"),
    team: Optional[str] = Query(None, description="Team abbreviation")
):
    """Get NFL schedules with optional filters"""
    try:
        schedules = nfl.import_schedules([season])
        
        if week is not None:
            schedules = schedules[schedules['week'] == week]
        
        if team is not None:
            team = team.upper()
            schedules = schedules[
                (schedules['home_team'] == team) | 
                (schedules['away_team'] == team)
            ]
        
        return {
            "status": "success",
            "season": season,
            "total_games": len(schedules),
            "data": schedules.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Error fetching schedules: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schedules/{season}/week/{week}")
async def get_weekly_schedule(season: int, week: int):
    """Get schedule for a specific week"""
    try:
        schedules = nfl.import_schedules([season])
        week_games = schedules[schedules['week'] == week]
        
        return {
            "status": "success",
            "season": season,
            "week": week,
            "games": week_games.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Error fetching week {week} schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PLAYER ENDPOINTS
# ============================================================================

@app.get("/players/rosters")
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
            "data": rosters.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Error fetching rosters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/players/stats")
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
            "data": stats.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Error fetching player stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/players/{player_id}")
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
            "player_info": player_info,
            "season_stats": player_stats.to_dict('records') if not player_stats.empty else [],
            "games_played": len(player_stats)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching player {player_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PLAYER GRADING ENDPOINTS
# ============================================================================

@app.get("/grades/players")
async def get_player_grades(
    years: List[int] = Query([2023], description="Years to analyze"),
    min_games: int = Query(3, description="Minimum games played"),
    player_type: Optional[str] = Query(None, description="OFFENSE or DEFENSE"),
    position_group: Optional[str] = Query(None, description="Position group filter"),
    limit: int = Query(50, description="Maximum results to return")
):
    """Get player performance grades"""
    try:
        grader = get_player_grader(years)
        
        # Calculate all grades
        all_grades = grader.calculate_all_grades(min_games=min_games)
        
        if all_grades.empty:
            return {
                "status": "success",
                "message": "No grades calculated",
                "data": []
            }
        
        # Identify outliers
        outliers = grader.identify_performance_outliers(all_grades)
        
        # Apply filters
        filtered_data = outliers.copy()
        
        if player_type:
            filtered_data = filtered_data[filtered_data['player_type'] == player_type.upper()]
        
        if position_group:
            filtered_data = filtered_data[filtered_data['position_group'] == position_group.upper()]
        
        # Get aggregated player stats
        player_stats = filtered_data.groupby(['player_name', 'position', 'position_group', 'player_type']).agg({
            'numeric_grade': ['mean', 'std', 'count', 'min', 'max'],
            'performance_type': lambda x: (x == 'Over-Performance').sum(),
            'is_outlier': 'sum'
        }).reset_index()
        
        # Flatten column names
        player_stats.columns = ['player_name', 'position', 'position_group', 'player_type',
                               'avg_grade', 'grade_std', 'games_played', 
                               'over_performances', 'total_outliers']
        
        # Calculate consistency score
        player_stats['consistency'] = 100 - player_stats['grade_std'].fillna(0)
        player_stats['letter_grade'] = player_stats['avg_grade'].apply(grader._numeric_to_letter_grade)
        
        # Sort by average grade and limit results
        player_stats = player_stats.sort_values('avg_grade', ascending=False).head(limit)
        
        return {
            "status": "success",
            "parameters": {
                "years": years,
                "min_games": min_games,
                "player_type": player_type,
                "position_group": position_group
            },
            "total_players": len(player_stats),
            "data": player_stats.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Error calculating player grades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grades/players/{player_name}")
async def get_player_grade_details(
    player_name: str,
    years: List[int] = Query([2023], description="Years to analyze"),
    min_games: int = Query(1, description="Minimum games played")
):
    """Get detailed grade information for a specific player"""
    try:
        grader = get_player_grader(years)
        
        # Calculate all grades
        all_grades = grader.calculate_all_grades(min_games=min_games)
        outliers = grader.identify_performance_outliers(all_grades)
        
        # Filter to specific player
        player_data = outliers[outliers['player_name'].str.contains(player_name, case=False)]
        
        if player_data.empty:
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")
        
        # Calculate player summary stats
        summary = {
            "player_name": player_data['player_name'].iloc[0],
            "position": player_data['position'].iloc[0],
            "position_group": player_data['position_group'].iloc[0],
            "player_type": player_data['player_type'].iloc[0],
            "games_played": len(player_data),
            "avg_grade": float(player_data['numeric_grade'].mean()),
            "best_game": float(player_data['numeric_grade'].max()),
            "worst_game": float(player_data['numeric_grade'].min()),
            "consistency": float(100 - player_data['numeric_grade'].std()),
            "over_performances": int((player_data['performance_type'] == 'Over-Performance').sum()),
            "under_performances": int((player_data['performance_type'] == 'Under-Performance').sum()),
            "letter_grade": grader._numeric_to_letter_grade(player_data['numeric_grade'].mean())
        }
        
        return {
            "status": "success",
            "summary": summary,
            "game_by_game": player_data.to_dict('records')
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting player grade details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grades/players/top/{position_group}")
async def get_top_players_by_position(
    position_group: str,
    years: List[int] = Query([2023], description="Years to analyze"),
    min_games: int = Query(3, description="Minimum games played"),
    limit: int = Query(10, description="Number of top players")
):
    """Get top players by position group"""
    try:
        grader = get_player_grader(years)
        
        all_grades = grader.calculate_all_grades(min_games=min_games)
        outliers = grader.identify_performance_outliers(all_grades)
        
        top_players = grader.get_top_performers(
            outliers, 
            position_group=position_group.upper(), 
            n=limit
        )
        
        if top_players.empty:
            return {
                "status": "success",
                "message": f"No players found for position group: {position_group}",
                "data": []
            }
        
        top_players['letter_grade'] = top_players['avg_grade'].apply(grader._numeric_to_letter_grade)
        
        return {
            "status": "success",
            "position_group": position_group.upper(),
            "total_players": len(top_players),
            "data": top_players.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Error getting top players for {position_group}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# COACHING ENDPOINTS
# ============================================================================

@app.get("/coaches")
async def get_coaches(
    season: Optional[int] = Query(None, description="Filter by season"),
    years: List[int] = Query([2023, 2024], description="Years to load data for")
):
    """Get all available coaches"""
    try:
        analytics = get_coaching_analytics(years)
        coaches = analytics.get_all_coaches(season=season)
        
        # Get additional info for each coach
        coach_info = []
        for coach in coaches:
            info = {"name": coach, "seasons": []}
            
            for (c, s), data in analytics.coaching_data.items():
                if c == coach and (season is None or s == season):
                    teams = list(data['teams'])
                    games = data['games']
                    wins = sum(1 for g in games if g['result'] == 'W')
                    total = len([g for g in games if g['result'] is not None])
                    
                    info["seasons"].append({
                        "season": s,
                        "teams": teams,
                        "record": f"{wins}-{total-wins}",
                        "win_percentage": round((wins/total*100) if total > 0 else 0, 1),
                        "games_coached": len(games)
                    })
            
            coach_info.append(info)
        
        return {
            "status": "success",
            "total_coaches": len(coaches),
            "data": coach_info
        }
    except Exception as e:
        logger.error(f"Error fetching coaches: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/coaches/{coach_name}/analysis")
async def get_coach_analysis(
    coach_name: str,
    season: Optional[int] = Query(None, description="Specific season"),
    years: List[int] = Query([2023, 2024], description="Years to load data for")
):
    """Get comprehensive coaching analysis"""
    try:
        analytics = get_coaching_analytics(years)
        
        # Check if coach exists
        available_coaches = analytics.get_all_coaches(season=season)
        if coach_name not in available_coaches:
            raise HTTPException(status_code=404, detail=f"Coach '{coach_name}' not found")
        
        # Get offensive and defensive analysis
        offensive_analysis = analytics.analyze_offensive_tendencies(
            coach_name=coach_name, season=season
        )
        defensive_analysis = analytics.analyze_defensive_performance(
            coach_name=coach_name, season=season
        )
        
        # Get situational analysis
        situational_analysis = analytics.analyze_situational_performance(
            coach_name=coach_name, season=season
        )
        
        return {
            "status": "success",
            "coach": coach_name,
            "season": season,
            "offensive_analysis": offensive_analysis,
            "defensive_analysis": defensive_analysis,
            "situational_analysis": situational_analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing coach {coach_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/coaches/{coach_name}/grades")
async def get_coach_grades(
    coach_name: str,
    season: Optional[int] = Query(None, description="Specific season"),
    years: List[int] = Query([2023, 2024], description="Years to load data for")
):
    """Get coaching performance grades"""
    try:
        analytics = get_coaching_analytics(years)
        
        # Check if coach exists
        available_coaches = analytics.get_all_coaches(season=season)
        if coach_name not in available_coaches:
            raise HTTPException(status_code=404, detail=f"Coach '{coach_name}' not found")
        
        # Get grades
        grades = analytics.grade_coach_performance(coach_name, season=season)
        
        if not grades:
            return {
                "status": "success",
                "message": "No grading data available",
                "data": None
            }
        
        # Convert to letter grades
        graded_results = {}
        for category, score in grades.items():
            if isinstance(score, dict):
                graded_results[category] = {
                    subcategory: {
                        "score": subscore,
                        "letter_grade": analytics.get_letter_grade(subscore)
                    }
                    for subcategory, subscore in score.items()
                }
            else:
                graded_results[category] = {
                    "score": score,
                    "letter_grade": analytics.get_letter_grade(score)
                }
        
        # Get strengths and weaknesses
        strengths_weaknesses = analytics.get_coach_strengths_weaknesses(coach_name, season)
        
        # Get team record for context
        team_record = analytics._get_team_record(coach_name, season)
        
        return {
            "status": "success",
            "coach": coach_name,
            "season": season,
            "team_record": team_record,
            "specialty": analytics._determine_coach_specialty(coach_name, season),
            "grades": graded_results,
            "strengths_weaknesses": strengths_weaknesses
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error grading coach {coach_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/coaches/compare")
async def compare_coaches(
    coach_names: List[str],
    season: Optional[int] = Query(None, description="Specific season"),
    years: List[int] = Query([2023, 2024], description="Years to load data for")
):
    """Compare multiple coaches side by side"""
    try:
        analytics = get_coaching_analytics(years)
        
        # Validate coaches exist
        available_coaches = analytics.get_all_coaches(season=season)
        invalid_coaches = [coach for coach in coach_names if coach not in available_coaches]
        
        if invalid_coaches:
            raise HTTPException(
                status_code=404, 
                detail=f"Coaches not found: {invalid_coaches}"
            )
        
        # Get grades for each coach
        comparison_data = {}
        for coach in coach_names:
            grades = analytics.grade_coach_performance(coach, season=season)
            team_record = analytics._get_team_record(coach, season)
            
            if grades:
                comparison_data[coach] = {
                    "grades": grades,
                    "team_record": team_record,
                    "specialty": analytics._determine_coach_specialty(coach, season)
                }
        
        # Create comparison matrix
        categories = ['offensive_overall', 'defensive_overall', 'overall']
        comparison_matrix = {}
        
        for category in categories:
            comparison_matrix[category] = {}
            for coach in coach_names:
                if coach in comparison_data and category in comparison_data[coach]['grades']:
                    score = comparison_data[coach]['grades'][category]
                    comparison_matrix[category][coach] = {
                        "score": score,
                        "letter_grade": analytics.get_letter_grade(score)
                    }
                else:
                    comparison_matrix[category][coach] = None
        
        return {
            "status": "success",
            "coaches": coach_names,
            "season": season,
            "comparison_matrix": comparison_matrix,
            "detailed_data": comparison_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing coaches: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PLAY-BY-PLAY DATA ENDPOINTS
# ============================================================================

@app.get("/pbp")
async def get_play_by_play(
    season: int = Query(2023, description="Season year"),
    week: Optional[int] = Query(None, description="Specific week"),
    game_id: Optional[str] = Query(None, description="Specific game ID"),
    team: Optional[str] = Query(None, description="Team abbreviation"),
    limit: int = Query(100, description="Maximum plays to return")
):
    """Get play-by-play data with filters"""
    try:
        pbp_data = nfl.import_pbp_data([season])
        
        if week is not None:
            pbp_data = pbp_data[pbp_data['week'] == week]
        
        if game_id is not None:
            pbp_data = pbp_data[pbp_data['game_id'] == game_id]
        
        if team is not None:
            team = team.upper()
            pbp_data = pbp_data[
                (pbp_data['posteam'] == team) | (pbp_data['defteam'] == team)
            ]
        
        # Limit results
        pbp_data = pbp_data.head(limit)
        
        return {
            "status": "success",
            "season": season,
            "total_plays": len(pbp_data),
            "data": pbp_data.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Error fetching play-by-play data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# STATISTICS ENDPOINTS
# ============================================================================

@app.get("/stats/summary")
async def get_stats_summary(
    season: int = Query(2023, description="Season year"),
    years: List[int] = Query([2023], description="Years for grading analysis")
):
    """Get comprehensive statistics summary"""
    try:
        # Get basic counts
        schedules = nfl.import_schedules([season])
        rosters = nfl.import_weekly_rosters([season])
        
        # Get grading data
        player_grader = get_player_grader(years)
        coaching_analytics = get_coaching_analytics(years)
        
        all_grades = player_grader.calculate_all_grades(min_games=3)
        available_coaches = coaching_analytics.get_all_coaches()
        
        summary = {
            "season": season,
            "total_games": len(schedules),
            "total_weeks": schedules['week'].max() if not schedules.empty else 0,
            "total_players": rosters['player_id'].nunique() if not rosters.empty else 0,
            "total_coaches": len(available_coaches),
            "grading_stats": {
                "players_graded": all_grades['player_id'].nunique() if not all_grades.empty else 0,
                "offensive_players": len(all_grades[all_grades['player_type'] == 'OFFENSE']) if not all_grades.empty else 0,
                "defensive_players": len(all_grades[all_grades['player_type'] == 'DEFENSE']) if not all_grades.empty else 0,
                "total_player_games": len(all_grades) if not all_grades.empty else 0
            },
            "available_positions": rosters['position'].unique().tolist() if not rosters.empty else [],
            "available_teams": schedules['home_team'].unique().tolist() if not schedules.empty else []
        }
        
        return {
            "status": "success",
            "data": summary
        }
    except Exception as e:
        logger.error(f"Error generating stats summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

@app.get("/search/players")
async def search_players(
    query: str = Query(..., description="Search query for player name"),
    season: int = Query(2023, description="Season year"),
    limit: int = Query(20, description="Maximum results")
):
    """Search for players by name"""
    try:
        rosters = nfl.import_weekly_rosters([season])
        
        # Search for players matching query
        mask = rosters['player_name'].str.contains(query, case=False, na=False)
        results = rosters[mask].drop_duplicates('player_id')
        
        # Get latest info for each player
        player_results = []
        for _, player in results.head(limit).iterrows():
            player_results.append({
                "player_id": player['player_id'],
                "player_name": player['player_name'],
                "position": player['position'],
                "team": player['team'],
                "height": player.get('height'),
                "weight": player.get('weight'),
                "college": player.get('college')
            })
        
        return {
            "status": "success",
            "query": query,
            "total_results": len(results),
            "data": player_results
        }
    except Exception as e:
        logger.error(f"Error searching players: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/coaches")
async def search_coaches(
    query: str = Query(..., description="Search query for coach name"),
    years: List[int] = Query([2023, 2024], description="Years to search")
):
    """Search for coaches by name"""
    try:
        analytics = get_coaching_analytics(years)
        available_coaches = analytics.get_all_coaches()
        
        # Filter coaches matching query
        matching_coaches = [
            coach for coach in available_coaches 
            if query.lower() in coach.lower()
        ]
        
        # Get additional info
        coach_results = []
        for coach in matching_coaches:
            coach_info = {"name": coach, "seasons": []}
            
            for (c, s), data in analytics.coaching_data.items():
                if c == coach:
                    teams = list(data['teams'])
                    games = data['games']
                    wins = sum(1 for g in games if g['result'] == 'W')
                    total = len([g for g in games if g['result'] is not None])
                    
                    coach_info["seasons"].append({
                        "season": s,
                        "teams": teams,
                        "record": f"{wins}-{total-wins}",
                        "win_percentage": round((wins/total*100) if total > 0 else 0, 1)
                    })
            
            coach_results.append(coach_info)
        
        return {
            "status": "success",
            "query": query,
            "total_results": len(matching_coaches),
            "data": coach_results
        }
    except Exception as e:
        logger.error(f"Error searching coaches: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )