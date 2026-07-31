#!/usr/bin/env python3
"""
Players API Router
Handles all player-related endpoints including grading.

DB-first: queries PlayerStat / PlayerRoster tables; falls back to nflreadpy
when the tables have no data for the requested season.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
from sqlalchemy.orm import Session
import nflreadpy as nfl
import pandas as pd
import logging
import json as _json
from .utils import (
    clean_data_for_json,
    get_player_grader,
    check_grading_systems,
    get_current_nfl_season,
    _to_pandas,
    _orm_to_dict,
)
from database.session import get_db
from database.models import PlayerRoster, PlayerStat, PlayerSeasonAnalytics, AnalyticsJobStatus
from nflverse_compat import load_rosters_weekly

logger = logging.getLogger(__name__)
router = APIRouter()


def _normalize_roster_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map nflreadpy load_rosters_weekly column names to the expected API schema."""
    df = df.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
    if "player_display_name" not in df.columns:
        df = df.copy()
        df["player_display_name"] = df.get("player_name")
    if "height" in df.columns and df["height"].dtype != object:
        df = df.copy() if "player_display_name" not in df.columns else df
        df["height"] = df["height"].apply(
            lambda h: f"{int(h)//12}-{int(h)%12}" if pd.notna(h) else None
        )
    return df


def _normalize_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map nflreadpy load_player_stats column names to the expected API schema."""
    return df.rename(columns={
        "team": "recent_team",
        "passing_interceptions": "interceptions",
    })


@router.get("/rosters")
async def get_rosters(
    season: Optional[int] = Query(None, description="Season year (defaults to current NFL season)"),
    week: Optional[int] = Query(None, description="Specific week"),
    team: Optional[str] = Query(None, description="Team abbreviation"),
    position: Optional[str] = Query(None, description="Player position"),
    db: Session = Depends(get_db),
):
    """Get player rosters with optional filters."""
    if season is None:
        season = get_current_nfl_season()
    try:
        rows = []
        try:
            q = db.query(PlayerRoster).filter(PlayerRoster.season == season)
            if week is not None:
                q = q.filter(PlayerRoster.week == week)
            if team is not None:
                q = q.filter(PlayerRoster.team == team.upper())
            if position is not None:
                q = q.filter(PlayerRoster.position == position.upper())
            rows = q.all()
        except Exception as db_err:
            logger.warning("DB unavailable, falling back to nflreadpy: %s", db_err)

        if rows:
            if week is None:
                # Deduplicate: keep only the latest week entry per player
                seen: dict = {}
                for r in rows:
                    existing = seen.get(r.player_id)
                    if existing is None or r.week > existing.week:
                        seen[r.player_id] = r
                rows = list(seen.values())

            data = [_orm_to_dict(r) for r in rows]
            return {
                "status": "success",
                "season": season,
                "total_players": len(data),
                "data": data,
            }

        # Fallback: nflreadpy
        rosters = _normalize_roster_df(_to_pandas(load_rosters_weekly(season)))

        if week is not None:
            rosters = rosters[rosters["week"] == week]
        else:
            rosters = rosters.sort_values("week").drop_duplicates(subset=["player_id"], keep="last")

        if team is not None:
            rosters = rosters[rosters["team"] == team.upper()]

        if position is not None:
            rosters = rosters[rosters["position"] == position.upper()]

        return {
            "status": "success",
            "season": season,
            "total_players": len(rosters),
            "data": clean_data_for_json(rosters),
        }
    except Exception as e:
        logger.error("Error fetching rosters: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_player_stats(
    season: Optional[int] = Query(None, description="Season year (defaults to current NFL season)"),
    player_id: Optional[str] = Query(None, description="Specific player ID"),
    position: Optional[str] = Query(None, description="Player position"),
    team: Optional[str] = Query(None, description="Team abbreviation"),
    week: Optional[int] = Query(None, description="Specific week"),
    db: Session = Depends(get_db),
):
    """Get player statistics with optional filters."""
    if season is None:
        season = get_current_nfl_season()
    try:
        rows = []
        try:
            q = db.query(PlayerStat).filter(PlayerStat.season == season)
            if player_id is not None:
                q = q.filter(PlayerStat.player_id == player_id)
            if position is not None:
                q = q.filter(PlayerStat.position == position.upper())
            if team is not None:
                q = q.filter(PlayerStat.recent_team == team.upper())
            if week is not None:
                q = q.filter(PlayerStat.week == week)
            rows = q.all()
        except Exception as db_err:
            logger.warning("DB unavailable, falling back to nflreadpy: %s", db_err)

        if rows:
            data = [_orm_to_dict(r) for r in rows]
            return {
                "status": "success",
                "season": season,
                "total_records": len(data),
                "data": data,
            }

        # Fallback: nflreadpy
        stats = _normalize_stats_df(_to_pandas(nfl.load_player_stats(seasons=[season])))

        if player_id is not None:
            stats = stats[stats["player_id"] == player_id]
        if position is not None:
            stats = stats[stats["position"] == position.upper()]
        if team is not None:
            stats = stats[stats["recent_team"] == team.upper()]
        if week is not None:
            stats = stats[stats["week"] == week]

        return {
            "status": "success",
            "season": season,
            "total_records": len(stats),
            "data": clean_data_for_json(stats),
        }
    except Exception as e:
        error_str = str(e)
        if "404" in error_str:
            return {
                "status": "no_data",
                "season": season,
                "total_records": 0,
                "data": [],
                "message": (
                    f"Weekly stats are not yet available for the {season} season. "
                    "Try selecting an earlier year."
                ),
            }
        logger.error("Error fetching player stats: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/grades")
async def get_player_grades(
    years: Optional[List[int]] = Query(None, description="Years to analyze (defaults to current NFL season)"),
    min_games: int = Query(3, description="Minimum games played"),
    limit: int = Query(20, description="Maximum results to return"),
):
    """Get player performance grades."""
    if years is None:
        years = [get_current_nfl_season()]
    systems = check_grading_systems()
    if not systems["player_grading"]:
        raise HTTPException(status_code=503, detail="Player grading system not available")

    try:
        grader = get_player_grader(years)
        all_grades = grader.calculate_all_grades(min_games=min_games)

        if isinstance(all_grades, dict):
            if not all_grades or all(
                df.empty for df in all_grades.values() if hasattr(df, "empty")
            ):
                return {"status": "success", "message": "No grades calculated", "data": []}

            grade_dfs = []
            for category, df in all_grades.items():
                if hasattr(df, "empty") and not df.empty:
                    df_copy = df.copy()
                    df_copy["grade_category"] = category
                    grade_dfs.append(df_copy)

            if not grade_dfs:
                return {"status": "success", "message": "No grades calculated", "data": []}

            combined_grades = pd.concat(grade_dfs, ignore_index=True)
            if "numeric_grade" in combined_grades.columns:
                top_players = combined_grades.nlargest(limit, "numeric_grade")
            else:
                top_players = combined_grades.head(limit)

        elif hasattr(all_grades, "empty"):
            if all_grades.empty:
                return {"status": "success", "message": "No grades calculated", "data": []}
            outliers = grader.identify_performance_outliers(all_grades)
            top_players = grader.get_top_performers(outliers, n=limit)
        else:
            return {"status": "error", "message": "Unexpected data format from grading system"}

        return {
            "status": "success",
            "total_players": len(top_players),
            "data": clean_data_for_json(top_players),
        }
    except Exception as e:
        logger.error("Error calculating player grades: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/grades/{player_name}")
async def get_player_grade_details(
    player_name: str,
    years: Optional[List[int]] = Query(None, description="Years to analyze (defaults to current NFL season)"),
    min_games: int = Query(1, description="Minimum games played"),
):
    """Get detailed grade information for a specific player."""
    if years is None:
        years = [get_current_nfl_season()]
    systems = check_grading_systems()
    if not systems["player_grading"]:
        raise HTTPException(status_code=503, detail="Player grading system not available")

    try:
        grader = get_player_grader(years)
        all_grades = grader.calculate_all_grades(min_games=min_games)

        if isinstance(all_grades, dict):
            grade_dfs = []
            for category, df in all_grades.items():
                if hasattr(df, "empty") and not df.empty:
                    df_copy = df.copy()
                    df_copy["grade_category"] = category
                    grade_dfs.append(df_copy)

            if not grade_dfs:
                raise HTTPException(status_code=404, detail=f"No grades found for player '{player_name}'")

            combined_grades = pd.concat(grade_dfs, ignore_index=True)

            if "player_name" in combined_grades.columns:
                player_data = combined_grades[
                    combined_grades["player_name"].str.contains(player_name, case=False)
                ]
            else:
                raise HTTPException(status_code=404, detail="Player name column not found")
        else:
            outliers = grader.identify_performance_outliers(all_grades)
            player_data = outliers[outliers["player_name"].str.contains(player_name, case=False)]

        if player_data.empty:
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

        return {
            "status": "success",
            "player_name": player_name,
            "total_records": len(player_data),
            "data": clean_data_for_json(player_data),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting player grade details: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/status")
async def get_player_analytics_status(db: Session = Depends(get_db)):
    """Get current status of the player analytics pre-computation job."""
    job = (
        db.query(AnalyticsJobStatus)
        .filter(AnalyticsJobStatus.job_type == "player_analytics")
        .order_by(AnalyticsJobStatus.id.desc())
        .first()
    )
    if job is None:
        return {
            "status": "no_job",
            "message": "No player analytics job has been run yet.",
        }
    done = (job.processed_entries or 0) + (job.skipped_entries or 0)
    total = job.total_entries or 0
    pct = round(done / total * 100, 1) if total > 0 else 0.0
    return {
        "status": job.status,
        "job_id": job.id,
        "job_type": job.job_type,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "total_entries": total,
        "processed_entries": job.processed_entries or 0,
        "skipped_entries": job.skipped_entries or 0,
        "failed_entries": job.failed_entries or 0,
        "pct_complete": pct,
        "current_season": job.current_season,
        "error_message": job.error_message,
    }


@router.get("/{player_id}/breakdown")
async def get_player_breakdown(
    player_id: str,
    season: Optional[int] = Query(None, description="Season year (defaults to current NFL season)"),
    db: Session = Depends(get_db),
):
    """Get PBP-derived analytics for a specific player."""
    if season is None:
        season = get_current_nfl_season()

    # Query from cache
    rows = (
        db.query(PlayerSeasonAnalytics)
        .filter(
            PlayerSeasonAnalytics.player_id == player_id,
            PlayerSeasonAnalytics.season == season,
        )
        .all()
    )

    if rows:
        data = []
        for row in rows:
            try:
                blob = _json.loads(row.analytics_json)
            except Exception:
                blob = {}
            blob["season"] = row.season
            blob["team"] = row.team
            blob["position"] = row.position
            blob["player_name"] = row.player_name
            blob["computed_at"] = row.computed_at.isoformat() if row.computed_at else None
            data.append(blob)
        return {
            "status": "success",
            "player_id": player_id,
            "season": season,
            "data": data,
        }

    # Cache miss — check if a player analytics job is currently running
    running_job = (
        db.query(AnalyticsJobStatus)
        .filter(
            AnalyticsJobStatus.status == "running",
            AnalyticsJobStatus.job_type == "player_analytics",
        )
        .order_by(AnalyticsJobStatus.id.desc())
        .first()
    )
    if running_job is not None:
        done = (running_job.processed_entries or 0) + (running_job.skipped_entries or 0)
        total = running_job.total_entries or 0
        pct = round(done / total * 100, 1) if total > 0 else 0.0
        return {
            "status": "computing",
            "message": (
                f"Player analytics pre-computation is {pct}% complete. "
                "Check back shortly."
            ),
            "pct_complete": pct,
            "current_season": running_job.current_season,
            "analytics_status_url": "/players/analytics/status",
        }

    return {
        "status": "no_data",
        "player_id": player_id,
        "season": season,
        "message": (
            "No analytics cached for this player. "
            "Run the player analytics job first: k8s/player-analytics-job.yaml"
        ),
        "data": [],
    }


@router.get("/{player_id}")
async def get_player_details(
    player_id: str,
    season: Optional[int] = Query(None, description="Season year (defaults to current NFL season)"),
    db: Session = Depends(get_db),
):
    """Get detailed information for a specific player."""
    if season is None:
        season = get_current_nfl_season()
    try:
        # Try DB first
        stat_rows, roster_rows = [], []
        try:
            stat_rows = db.query(PlayerStat).filter(
                PlayerStat.player_id == player_id,
                PlayerStat.season == season,
            ).all()
            roster_rows = db.query(PlayerRoster).filter(
                PlayerRoster.player_id == player_id,
                PlayerRoster.season == season,
            ).all()
        except Exception as db_err:
            logger.warning("DB unavailable, falling back to nflreadpy: %s", db_err)

        if stat_rows or roster_rows:
            player_info: dict = {}
            if roster_rows:
                latest = max(roster_rows, key=lambda r: r.week or 0)
                player_info = {
                    "player_id": player_id,
                    "player_name": latest.player_name or "Unknown",
                    "position": latest.position or "Unknown",
                    "team": latest.team or "Unknown",
                    "height": latest.height,
                    "weight": latest.weight,
                    "college": latest.college,
                    "rookie_year": latest.rookie_year,
                }
            return {
                "status": "success",
                "player_info": player_info,
                "season_stats": [_orm_to_dict(r) for r in stat_rows],
                "games_played": len(stat_rows),
            }

        # Fallback: nflreadpy
        stats = _normalize_stats_df(_to_pandas(nfl.load_player_stats(seasons=[season])))
        player_stats = stats[stats["player_id"] == player_id]

        rosters = _normalize_roster_df(_to_pandas(load_rosters_weekly(season)))
        player_roster = rosters[rosters["player_id"] == player_id]

        if player_stats.empty and player_roster.empty:
            raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

        player_info = {}
        if not player_roster.empty:
            latest_roster = player_roster.iloc[-1]
            player_info = {
                "player_id": player_id,
                "player_name": latest_roster.get("player_name", "Unknown"),
                "position": latest_roster.get("position", "Unknown"),
                "team": latest_roster.get("team", "Unknown"),
                "height": latest_roster.get("height", None),
                "weight": latest_roster.get("weight", None),
                "college": latest_roster.get("college", None),
                "rookie_year": latest_roster.get("rookie_year", None),
            }

        return {
            "status": "success",
            "player_info": clean_data_for_json(player_info),
            "season_stats": clean_data_for_json(player_stats) if not player_stats.empty else [],
            "games_played": len(player_stats),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching player %s: %s", player_id, e)
        raise HTTPException(status_code=500, detail=str(e))
