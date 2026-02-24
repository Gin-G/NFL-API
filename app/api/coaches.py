#!/usr/bin/env python3
"""
Coaches API Router
Handles all coach-related endpoints including grading
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging
from .utils import (
    clean_data_for_json,
    get_coaching_analytics,
    get_sportradar_coaches_client,
    check_grading_systems,
    get_current_nfl_season,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_coach_season_data(analytics, coach_name, season=None):
    """Extract season-by-season records for a coach from coaching_data."""
    results = []
    for (c, s), data in analytics.coaching_data.items():
        if c == coach_name and (season is None or s == season):
            games = data["games"]
            wins = sum(1 for g in games if g["result"] == "W")
            losses = sum(1 for g in games if g["result"] == "L")
            total = wins + losses
            results.append({
                "season": s,
                "teams": list(data["teams"]),
                "record": f"{wins}-{losses}",
                "wins": wins,
                "losses": losses,
                "win_percentage": round((wins / total * 100) if total > 0 else 0, 1),
                "games_coached": len(games),
            })
    return sorted(results, key=lambda x: x["season"])


@router.get("/staff")
async def get_coaching_staff(
    team: Optional[str] = Query(None, description="Filter by team abbreviation (e.g. KC)"),
):
    """Get current HC / OC / DC / STC for every NFL team via SportRadar.

    Returns ``configured: false`` (with an empty data list) when
    ``SPORTRADAR_API_KEY`` is not set, so the API remains functional
    without a key.

    The first call fetches all 32 team profiles from SportRadar and caches
    the result for 24 hours.  Subsequent calls are instant.
    """
    client = get_sportradar_coaches_client()
    if client is None:
        return {
            "status": "success",
            "configured": False,
            "message": (
                "SportRadar API key not configured. "
                "Set the SPORTRADAR_API_KEY environment variable to enable this endpoint."
            ),
            "data": [],
        }

    try:
        staff = client.get_all_team_staff()
        if team:
            staff = [s for s in staff if s["team_abbr"].upper() == team.upper()]
        return {
            "status": "success",
            "configured": True,
            "total_teams": len(staff),
            "data": staff,
        }
    except Exception as e:
        logger.error(f"Error fetching coaching staff: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def get_coaches(
    season: Optional[int] = Query(None, description="Filter by season"),
    years: Optional[List[int]] = Query(None, description="Years to load data for (defaults to current NFL season)"),
):
    """Get all available coaches"""
    if years is None:
        years = [get_current_nfl_season()]
    systems = check_grading_systems()
    if not systems["coaching_analytics"]:
        raise HTTPException(status_code=503, detail="Coaching analytics system not available")

    try:
        analytics = get_coaching_analytics(years)
        coaches = analytics.get_available_coaches(season=season)

        coach_info = []
        for coach in coaches:
            seasons = _get_coach_season_data(analytics, coach, season)
            coach_info.append({"name": coach, "seasons": seasons})

        return {
            "status": "success",
            "total_coaches": len(coaches),
            "data": clean_data_for_json(coach_info),
        }
    except Exception as e:
        logger.error(f"Error fetching coaches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{coach_name}/analysis")
async def get_coach_analysis(
    coach_name: str,
    season: Optional[int] = Query(None, description="Specific season"),
    years: Optional[List[int]] = Query(None, description="Years to load data for (defaults to current NFL season)"),
):
    """Get comprehensive coaching analysis including roster quality breakdown"""
    if years is None:
        years = [get_current_nfl_season()]
    systems = check_grading_systems()
    if not systems["coaching_analytics"]:
        raise HTTPException(status_code=503, detail="Coaching analytics system not available")

    try:
        analytics = get_coaching_analytics(years)
        available_coaches = analytics.get_available_coaches(season=season)
        if coach_name not in available_coaches:
            raise HTTPException(status_code=404, detail=f"Coach '{coach_name}' not found")

        seasons_data = _get_coach_season_data(analytics, coach_name, season)

        offensive_analysis = {}
        defensive_analysis = {}
        for season_entry in seasons_data:
            s = season_entry["season"]
            for team in season_entry["teams"]:
                roster = analytics.analyze_roster_quality(team, s)
                if roster:
                    key = f"{team}_{s}"
                    offensive_analysis[key] = {
                        "team": team,
                        "season": s,
                        "qb_avg_grade": roster.get("qb_avg_grade"),
                        "rb_avg_grade": roster.get("rb_avg_grade"),
                        "wr_te_avg_grade": roster.get("wr_te_avg_grade"),
                    }
                    defensive_analysis[key] = {
                        "team": team,
                        "season": s,
                        "defense_avg_grade": roster.get("defense_avg_grade"),
                        "overall_avg_grade": roster.get("overall_avg_grade"),
                        "roster_tier": roster.get("roster_tier"),
                    }

        return {
            "status": "success",
            "coach": coach_name,
            "season": season,
            "seasons": clean_data_for_json(seasons_data),
            "offensive_analysis": clean_data_for_json(offensive_analysis),
            "defensive_analysis": clean_data_for_json(defensive_analysis),
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
    years: Optional[List[int]] = Query(None, description="Years to load data for (defaults to current NFL season)"),
):
    """Get coaching performance grades derived from win rate and roster quality"""
    if years is None:
        years = [get_current_nfl_season()]
    systems = check_grading_systems()
    if not systems["coaching_analytics"]:
        raise HTTPException(status_code=503, detail="Coaching analytics system not available")

    try:
        analytics = get_coaching_analytics(years)
        available_coaches = analytics.get_available_coaches(season=season)
        if coach_name not in available_coaches:
            raise HTTPException(status_code=404, detail=f"Coach '{coach_name}' not found")

        seasons_data = _get_coach_season_data(analytics, coach_name, season)

        if not seasons_data:
            return {
                "status": "success",
                "coach": coach_name,
                "season": season,
                "grades": None,
                "data": None,
            }

        grade_entries = []
        for season_entry in seasons_data:
            s = season_entry["season"]
            win_pct = season_entry["win_percentage"]
            # Scale win% (0-100) into a 40-95 score range
            win_score = 40 + (win_pct / 100) * 55

            roster_scores = []
            for team in season_entry["teams"]:
                roster = analytics.analyze_roster_quality(team, s)
                if roster and roster.get("overall_avg_grade") is not None:
                    roster_scores.append(roster["overall_avg_grade"])

            entry = {
                "season": s,
                "teams": season_entry["teams"],
                "record": season_entry["record"],
                "win_percentage": win_pct,
                "win_score": round(win_score, 1),
                "win_letter_grade": analytics.get_letter_grade(win_score),
            }
            if roster_scores:
                roster_score = sum(roster_scores) / len(roster_scores)
                entry["roster_quality_score"] = round(roster_score, 1)
                entry["roster_quality_grade"] = analytics.get_letter_grade(roster_score)

            grade_entries.append(entry)

        return {
            "status": "success",
            "coach": coach_name,
            "season": season,
            "grades": clean_data_for_json(grade_entries),
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
    years: Optional[List[int]] = Query(None, description="Years to load data for (defaults to current NFL season)"),
):
    """Compare multiple coaches side by side"""
    if years is None:
        years = [get_current_nfl_season()]
    systems = check_grading_systems()
    if not systems["coaching_analytics"]:
        raise HTTPException(status_code=503, detail="Coaching analytics system not available")

    try:
        analytics = get_coaching_analytics(years)
        available_coaches = analytics.get_available_coaches(season=season)
        invalid_coaches = [c for c in coach_names if c not in available_coaches]

        if invalid_coaches:
            raise HTTPException(status_code=404, detail=f"Coaches not found: {invalid_coaches}")

        comparison_data = {}
        for coach in coach_names:
            seasons_data = _get_coach_season_data(analytics, coach, season)
            total_wins = sum(s["wins"] for s in seasons_data)
            total_losses = sum(s["losses"] for s in seasons_data)
            total_games = total_wins + total_losses
            overall_win_pct = round((total_wins / total_games * 100) if total_games > 0 else 0, 1)
            win_score = round(40 + (overall_win_pct / 100) * 55, 1)
            comparison_data[coach] = {
                "seasons": seasons_data,
                "overall_record": f"{total_wins}-{total_losses}",
                "overall_win_percentage": overall_win_pct,
                "win_score": win_score,
                "win_letter_grade": analytics.get_letter_grade(win_score),
            }

        comparison_matrix = {
            "win_percentage": {
                coach: {
                    "score": comparison_data[coach]["overall_win_percentage"],
                    "letter_grade": comparison_data[coach]["win_letter_grade"],
                }
                for coach in coach_names
            }
        }

        return {
            "status": "success",
            "coaches": coach_names,
            "season": season,
            "comparison_matrix": clean_data_for_json(comparison_matrix),
            "detailed_data": clean_data_for_json(comparison_data),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing coaches: {e}")
        raise HTTPException(status_code=500, detail=str(e))
