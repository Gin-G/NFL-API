#!/usr/bin/env python3
"""
Coaches API Router
Handles all coach-related endpoints including grading
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging
from .utils import clean_data_for_json, get_coaching_analytics, check_grading_systems

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def get_coaches(
    season: Optional[int] = Query(None, description="Filter by season"),
    years: List[int] = Query([2023, 2024], description="Years to load data for")
):
    """Get all available coaches"""
    systems = check_grading_systems()
    if not systems["coaching_analytics"]:
        raise HTTPException(status_code=503, detail="Coaching analytics system not available")
    
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
            "data": clean_data_for_json(coach_info)
        }
    except Exception as e:
        logger.error(f"Error fetching coaches: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{coach_name}/analysis")
async def get_coach_analysis(
    coach_name: str,
    season: Optional[int] = Query(None, description="Specific season"),
    years: List[int] = Query([2023, 2024], description="Years to load data for")
):
    """Get comprehensive coaching analysis"""
    systems = check_grading_systems()
    if not systems["coaching_analytics"]:
        raise HTTPException(status_code=503, detail="Coaching analytics system not available")
    
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
        
        # Get situational analysis if available
        situational_analysis = None
        try:
            situational_analysis = analytics.analyze_situational_performance(
                coach_name=coach_name, season=season
            )
        except AttributeError:
            # Method might not exist in this version
            pass
        
        return {
            "status": "success",
            "coach": coach_name,
            "season": season,
            "offensive_analysis": clean_data_for_json(offensive_analysis),
            "defensive_analysis": clean_data_for_json(defensive_analysis),
            "situational_analysis": clean_data_for_json(situational_analysis)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing coach {coach_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{coach_name}/grades")
async def get_coach_grades(
    coach_name: str,
    season: Optional[int] = Query(None, description="Specific season"),
    years: List[int] = Query([2023, 2024], description="Years to load data for")
):
    """Get coaching performance grades"""
    systems = check_grading_systems()
    if not systems["coaching_analytics"]:
        raise HTTPException(status_code=503, detail="Coaching analytics system not available")
    
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
        
        # Get strengths and weaknesses if available
        strengths_weaknesses = None
        try:
            strengths_weaknesses = analytics.get_coach_strengths_weaknesses(coach_name, season)
        except AttributeError:
            # Method might not exist in this version
            pass
        
        # Get team record for context
        team_record = None
        try:
            team_record = analytics._get_team_record(coach_name, season)
        except AttributeError:
            # Method might not exist in this version
            pass
        
        # Get specialty if available
        specialty = None
        try:
            specialty = analytics._determine_coach_specialty(coach_name, season)
        except AttributeError:
            specialty = "unknown"
        
        return {
            "status": "success",
            "coach": coach_name,
            "season": season,
            "team_record": clean_data_for_json(team_record),
            "specialty": specialty,
            "grades": clean_data_for_json(graded_results),
            "strengths_weaknesses": clean_data_for_json(strengths_weaknesses)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error grading coach {coach_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare")
async def compare_coaches(
    coach_names: List[str],
    season: Optional[int] = Query(None, description="Specific season"),
    years: List[int] = Query([2023, 2024], description="Years to load data for")
):
    """Compare multiple coaches side by side"""
    systems = check_grading_systems()
    if not systems["coaching_analytics"]:
        raise HTTPException(status_code=503, detail="Coaching analytics system not available")
    
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
            
            # Get team record if available
            team_record = None
            try:
                team_record = analytics._get_team_record(coach, season)
            except AttributeError:
                pass
            
            # Get specialty if available
            specialty = None
            try:
                specialty = analytics._determine_coach_specialty(coach, season)
            except AttributeError:
                specialty = "unknown"
            
            if grades:
                comparison_data[coach] = {
                    "grades": grades,
                    "team_record": team_record,
                    "specialty": specialty
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
            "comparison_matrix": clean_data_for_json(comparison_matrix),
            "detailed_data": clean_data_for_json(comparison_data)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing coaches: {e}")
        raise HTTPException(status_code=500, detail=str(e))