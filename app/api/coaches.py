#!/usr/bin/env python3
"""
Coaches API Router
Handles all coach-related endpoints including grading.

DB-first: queries the schedules table for coach win/loss records.
Falls back to nflreadpy-based analytics when the DB is empty.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
import logging
from sqlalchemy import func, or_, text
from sqlalchemy.orm import Session

from .utils import (
    clean_data_for_json,
    get_coaching_analytics,
    get_sportradar_coaches_client,
    check_grading_systems,
    get_current_nfl_season,
)
from database.session import get_db
from database.models import CoachSeasonAnalytics, Schedule

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _build_coaching_records(schedules):
    """
    Build coach records from a list of Schedule ORM rows.

    Returns: {coach_name: {season: {"wins": int, "losses": int, "teams": set}}}
    Only counts completed games (both scores present).
    """
    records: dict = {}
    for g in schedules:
        if g.home_score is None or g.away_score is None:
            continue
        home_win = g.home_score > g.away_score
        tie = g.home_score == g.away_score
        for coach, team, won in [
            (g.home_coach, g.home_team, home_win),
            (g.away_coach, g.away_team, not home_win),
        ]:
            if not coach:
                continue
            season_map = records.setdefault(coach, {})
            entry = season_map.setdefault(g.season, {"wins": 0, "losses": 0, "teams": set()})
            entry["teams"].add(team)
            if not tie:
                if won:
                    entry["wins"] += 1
                else:
                    entry["losses"] += 1
    return records


def _format_seasons(season_records: dict) -> list:
    """Convert {season: {wins, losses, teams}} → sorted list of season dicts."""
    result = []
    for season, data in sorted(season_records.items()):
        wins = data["wins"]
        losses = data["losses"]
        total = wins + losses
        result.append({
            "season": season,
            "teams": sorted(data["teams"]),
            "record": f"{wins}-{losses}",
            "wins": wins,
            "losses": losses,
            "win_percentage": round((wins / total * 100) if total > 0 else 0, 1),
            "games_coached": total,
        })
    return result


def _coaches_from_db(db: Session, seasons: Optional[List[int]] = None,
                     season_filter: Optional[int] = None) -> Optional[List[dict]]:
    """
    Return coach list from DB, or None if the schedules table is empty.
    """
    q = db.query(Schedule)
    if season_filter is not None:
        q = q.filter(Schedule.season == season_filter)
    elif seasons:
        q = q.filter(Schedule.season.in_(seasons))

    # Quick empty-check
    if not db.query(Schedule).limit(1).count():
        return None

    schedules = q.all()
    records = _build_coaching_records(schedules)
    if not records:
        return None

    max_season = db.query(func.max(Schedule.season)).scalar() or 0

    # Each coach's global latest season — unaffected by any request filter.
    # Two cheap GROUP BY queries (one per side) give the correct max even when
    # the caller has filtered schedules to a specific year window.
    coach_global_max: dict = {}
    for coach, ms in (
        db.query(Schedule.home_coach, func.max(Schedule.season))
        .filter(Schedule.home_coach.isnot(None))
        .group_by(Schedule.home_coach)
        .all()
    ):
        coach_global_max[coach] = max(coach_global_max.get(coach, 0), ms or 0)
    for coach, ms in (
        db.query(Schedule.away_coach, func.max(Schedule.season))
        .filter(Schedule.away_coach.isnot(None))
        .group_by(Schedule.away_coach)
        .all()
    ):
        coach_global_max[coach] = max(coach_global_max.get(coach, 0), ms or 0)

    result = []
    for coach, season_map in records.items():
        result.append({
            "name": coach,
            "is_active": coach_global_max.get(coach, 0) == max_season,
            "seasons": _format_seasons(season_map),
        })
    result.sort(key=lambda c: c["name"])
    return result


# ---------------------------------------------------------------------------
# Shared legacy helper (nflreadpy fallback)
# ---------------------------------------------------------------------------

def _get_coach_season_data_legacy(analytics, coach_name, season=None):
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/staff")
async def get_coaching_staff(
    team: Optional[str] = Query(None, description="Filter by team abbreviation (e.g. KC)"),
):
    """Get current HC / OC / DC / STC for every NFL team via SportRadar."""
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
    years: Optional[List[int]] = Query(None, description="Years to load data for"),
    db: Session = Depends(get_db),
):
    """Get all available coaches with season-by-season records."""
    # --- DB path ---
    coaches = _coaches_from_db(db, seasons=years, season_filter=season)
    if coaches is not None:
        return {
            "status": "success",
            "total_coaches": len(coaches),
            "data": clean_data_for_json(coaches),
        }

    # --- nflreadpy fallback ---
    if years is None:
        years = [get_current_nfl_season()]
    systems = check_grading_systems()
    if not systems["coaching_analytics"]:
        raise HTTPException(status_code=503, detail="Coaching analytics system not available")

    try:
        analytics = get_coaching_analytics(years)
        coach_list = analytics.get_available_coaches(season=season)
        coach_info = [
            {"name": c, "seasons": _get_coach_season_data_legacy(analytics, c, season)}
            for c in coach_list
        ]
        return {
            "status": "success",
            "total_coaches": len(coach_info),
            "data": clean_data_for_json(coach_info),
        }
    except Exception as e:
        logger.error(f"Error fetching coaches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{coach_name}/analysis")
async def get_coach_analysis(
    coach_name: str,
    season: Optional[int] = Query(None),
    years: Optional[List[int]] = Query(None),
    db: Session = Depends(get_db),
):
    """Get coach season records and (when available) roster quality breakdown."""
    # --- DB path (only when the DB has data for the requested seasons) ---
    q = db.query(Schedule)
    if season is not None:
        q = q.filter(Schedule.season == season)
    elif years:
        q = q.filter(Schedule.season.in_(years))
    schedules = q.all()
    if schedules:
        records = _build_coaching_records(schedules)
        if coach_name not in records:
            raise HTTPException(status_code=404, detail=f"Coach '{coach_name}' not found")
        seasons_data = _format_seasons(records[coach_name])
        return {
            "status": "success",
            "coach": coach_name,
            "season": season,
            "seasons": clean_data_for_json(seasons_data),
            "offensive_analysis": {},
            "defensive_analysis": {},
        }

    # --- nflreadpy fallback ---
    if years is None:
        years = [get_current_nfl_season()]
    systems = check_grading_systems()
    if not systems["coaching_analytics"]:
        raise HTTPException(status_code=503, detail="Coaching analytics system not available")

    try:
        analytics = get_coaching_analytics(years)
        if coach_name not in analytics.get_available_coaches(season=season):
            raise HTTPException(status_code=404, detail=f"Coach '{coach_name}' not found")

        seasons_data = _get_coach_season_data_legacy(analytics, coach_name, season)
        offensive_analysis: dict = {}
        defensive_analysis: dict = {}
        for entry in seasons_data:
            s = entry["season"]
            for team in entry["teams"]:
                roster = analytics.analyze_roster_quality(team, s)
                if roster:
                    key = f"{team}_{s}"
                    offensive_analysis[key] = {
                        "team": team, "season": s,
                        "qb_avg_grade": roster.get("qb_avg_grade"),
                        "rb_avg_grade": roster.get("rb_avg_grade"),
                        "wr_te_avg_grade": roster.get("wr_te_avg_grade"),
                    }
                    defensive_analysis[key] = {
                        "team": team, "season": s,
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
    season: Optional[int] = Query(None),
    years: Optional[List[int]] = Query(None),
    db: Session = Depends(get_db),
):
    """Get coaching performance grades derived from win rate."""
    # --- DB path (only when the DB has data for the requested seasons) ---
    q = db.query(Schedule)
    if season is not None:
        q = q.filter(Schedule.season == season)
    elif years:
        q = q.filter(Schedule.season.in_(years))
    schedules = q.all()
    if schedules:
        records = _build_coaching_records(schedules)
        if coach_name not in records:
            raise HTTPException(status_code=404, detail=f"Coach '{coach_name}' not found")

        seasons_data = _format_seasons(records[coach_name])
        grade_entries = []
        for entry in seasons_data:
            win_pct = entry["win_percentage"]
            win_score = round(40 + (win_pct / 100) * 55, 1)
            letter = (
                "A+" if win_score >= 95 else "A" if win_score >= 90 else
                "A-" if win_score >= 85 else "B+" if win_score >= 80 else
                "B" if win_score >= 75 else "B-" if win_score >= 70 else
                "C+" if win_score >= 65 else "C" if win_score >= 60 else
                "C-" if win_score >= 55 else "D+" if win_score >= 50 else
                "D" if win_score >= 45 else "F"
            )
            grade_entries.append({
                "season": entry["season"],
                "teams": entry["teams"],
                "record": entry["record"],
                "win_percentage": win_pct,
                "win_score": win_score,
                "win_letter_grade": letter,
            })

        return {
            "status": "success",
            "coach": coach_name,
            "season": season,
            "grades": clean_data_for_json(grade_entries),
        }

    # --- nflreadpy fallback ---
    if years is None:
        years = [get_current_nfl_season()]
    systems = check_grading_systems()
    if not systems["coaching_analytics"]:
        raise HTTPException(status_code=503, detail="Coaching analytics system not available")

    try:
        analytics = get_coaching_analytics(years)
        if coach_name not in analytics.get_available_coaches(season=season):
            raise HTTPException(status_code=404, detail=f"Coach '{coach_name}' not found")

        seasons_data = _get_coach_season_data_legacy(analytics, coach_name, season)
        if not seasons_data:
            return {"status": "success", "coach": coach_name, "season": season,
                    "grades": None, "data": None}

        grade_entries = []
        for entry in seasons_data:
            win_pct = entry["win_percentage"]
            win_score = 40 + (win_pct / 100) * 55
            roster_scores = []
            for team in entry["teams"]:
                roster = analytics.analyze_roster_quality(team, entry["season"])
                if roster and roster.get("overall_avg_grade") is not None:
                    roster_scores.append(roster["overall_avg_grade"])
            item = {
                "season": entry["season"],
                "teams": entry["teams"],
                "record": entry["record"],
                "win_percentage": win_pct,
                "win_score": round(win_score, 1),
                "win_letter_grade": analytics.get_letter_grade(win_score),
            }
            if roster_scores:
                rs = sum(roster_scores) / len(roster_scores)
                item["roster_quality_score"] = round(rs, 1)
                item["roster_quality_grade"] = analytics.get_letter_grade(rs)
            grade_entries.append(item)

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


@router.get("/{coach_name}/tendencies")
async def get_coach_tendencies(
    coach_name: str,
    season: Optional[int] = Query(None, description="Season year, e.g. 2024"),
    db: Session = Depends(get_db),
):
    """
    Get play-calling tendencies for a coach derived from play-by-play data.

    Requires the play_by_play table to be populated (run the data loader first).
    Returns offensive and defensive aggregations plus sample plays for key situations.
    """
    # Find which (season, team) pairs this coach is associated with via schedules
    q = db.query(Schedule).filter(
        or_(Schedule.home_coach == coach_name, Schedule.away_coach == coach_name)
    )
    if season is not None:
        q = q.filter(Schedule.season == season)
    games = q.all()

    if not games:
        raise HTTPException(status_code=404, detail=f"Coach '{coach_name}' not found")

    # Build set of (season, team) pairs
    season_team_pairs: set = set()
    for g in games:
        if g.home_coach == coach_name:
            season_team_pairs.add((g.season, g.home_team))
        if g.away_coach == coach_name:
            season_team_pairs.add((g.season, g.away_team))

    results = []
    for s, team in sorted(season_team_pairs):
        try:
            offense = _compute_offense_tendencies(db, team, s)
            defense = _compute_defense_tendencies(db, team, s)
            fourth_down_sample = _sample_plays(db, team, s, "fourth_down")
            third_down_sample = _sample_plays(db, team, s, "third_down")
        except Exception as exc:
            logger.warning("PBP query failed for %s %s: %s", team, s, exc)
            return {"status": "no_data", "message": "PBP not yet loaded"}

        results.append({
            "coach": coach_name,
            "season": s,
            "team": team,
            "offense": offense,
            "defense": defense,
            "fourth_down_sample": fourth_down_sample,
            "third_down_sample": third_down_sample,
        })

    return {
        "status": "success",
        "coach": coach_name,
        "season": season,
        "data": clean_data_for_json(results),
    }


def _compute_offense_tendencies(db, team: str, season: int) -> dict:
    """Run offensive aggregation SQL for a team+season."""
    sql = text("""
        SELECT
            COUNT(*) FILTER (WHERE play_type IN ('pass', 'run')) AS total_plays,
            AVG(CASE WHEN play_type = 'pass' THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE play_type IN ('pass', 'run')) AS pass_rate,
            AVG(epa) FILTER (WHERE play_type IN ('pass', 'run')) AS avg_epa_per_play,
            AVG(CASE WHEN play_type = 'pass' THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE yardline_100 <= 20 AND play_type IN ('pass', 'run')) AS red_zone_pass_rate,
            COUNT(*) FILTER (WHERE fourth_down_converted = 1 OR fourth_down_failed = 1) AS fourth_down_attempts,
            AVG(CASE WHEN fourth_down_converted = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE fourth_down_converted = 1 OR fourth_down_failed = 1) AS fourth_down_conv_rate,
            COUNT(*) FILTER (WHERE two_point_attempt = 1) AS two_point_attempts,
            AVG(CASE WHEN two_point_conv_result = 'success' THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE two_point_attempt = 1) AS two_point_success_rate,
            AVG(CASE WHEN third_down_converted = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE third_down_converted = 1 OR third_down_failed = 1) AS third_down_conv_rate
        FROM play_by_play
        WHERE posteam = :team AND season = :season
          AND play_type IN ('pass', 'run', 'no_play')
    """)
    row = db.execute(sql, {"team": team, "season": season}).mappings().first()

    # Pass rate by down
    down_sql = text("""
        SELECT down,
            AVG(CASE WHEN play_type = 'pass' THEN 1.0 ELSE 0.0 END) AS pass_rate
        FROM play_by_play
        WHERE posteam = :team AND season = :season
          AND play_type IN ('pass', 'run')
          AND down IS NOT NULL
        GROUP BY down
        ORDER BY down
    """)
    down_rows = db.execute(down_sql, {"team": team, "season": season}).mappings().all()
    pass_rate_by_down = {
        str(int(r["down"])): round(float(r["pass_rate"]), 3) if r["pass_rate"] is not None else None
        for r in down_rows
    }

    def _pct(v):
        return round(float(v), 3) if v is not None else None

    def _int(v):
        return int(v) if v is not None else 0

    return {
        "total_plays": _int(row["total_plays"]) if row else 0,
        "pass_rate": _pct(row["pass_rate"]) if row else None,
        "pass_rate_by_down": pass_rate_by_down,
        "avg_epa_per_play": _pct(row["avg_epa_per_play"]) if row else None,
        "red_zone_pass_rate": _pct(row["red_zone_pass_rate"]) if row else None,
        "fourth_down_attempts": _int(row["fourth_down_attempts"]) if row else 0,
        "fourth_down_conversion_rate": _pct(row["fourth_down_conv_rate"]) if row else None,
        "two_point_attempts": _int(row["two_point_attempts"]) if row else 0,
        "two_point_success_rate": _pct(row["two_point_success_rate"]) if row else None,
        "third_down_conversion_rate": _pct(row["third_down_conv_rate"]) if row else None,
    }


def _compute_defense_tendencies(db, team: str, season: int) -> dict:
    """Run defensive aggregation SQL for a team+season."""
    sql = text("""
        SELECT
            COUNT(*) FILTER (WHERE play_type IN ('pass', 'run')) AS total_plays,
            AVG(epa) FILTER (WHERE play_type IN ('pass', 'run')) AS avg_epa_allowed_per_play,
            AVG(CASE WHEN third_down_failed = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE third_down_converted = 1 OR third_down_failed = 1) AS third_down_stop_rate,
            AVG(CASE WHEN touchdown = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE yardline_100 <= 20 AND play_type IN ('pass', 'run')) AS red_zone_td_rate_allowed,
            AVG(CASE WHEN sack = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE play_type IN ('pass', 'run')) AS sack_rate
        FROM play_by_play
        WHERE defteam = :team AND season = :season
          AND play_type IN ('pass', 'run', 'no_play')
    """)
    row = db.execute(sql, {"team": team, "season": season}).mappings().first()

    def _pct(v):
        return round(float(v), 3) if v is not None else None

    def _int(v):
        return int(v) if v is not None else 0

    return {
        "total_plays": _int(row["total_plays"]) if row else 0,
        "avg_epa_allowed_per_play": _pct(row["avg_epa_allowed_per_play"]) if row else None,
        "third_down_stop_rate": _pct(row["third_down_stop_rate"]) if row else None,
        "red_zone_td_rate_allowed": _pct(row["red_zone_td_rate_allowed"]) if row else None,
        "sack_rate": _pct(row["sack_rate"]) if row else None,
    }


def _sample_plays(db, team: str, season: int, situation: str) -> list:
    """Return up to 5 plays sorted by |epa| for a given situation."""
    if situation == "fourth_down":
        where_extra = "AND (fourth_down_converted = 1 OR fourth_down_failed = 1)"
    else:  # third_down
        where_extra = "AND (third_down_converted = 1 OR third_down_failed = 1)"

    sql = text(f"""
        SELECT desc, yards_gained, epa, week, game_id
        FROM play_by_play
        WHERE posteam = :team AND season = :season
          AND play_type IN ('pass', 'run')
          {where_extra}
        ORDER BY ABS(epa) DESC
        LIMIT 5
    """)
    rows = db.execute(sql, {"team": team, "season": season}).mappings().all()
    return [
        {
            "desc": r["desc"],
            "yards_gained": r["yards_gained"],
            "epa": round(float(r["epa"]), 3) if r["epa"] is not None else None,
            "week": r["week"],
            "game_id": r["game_id"],
        }
        for r in rows
    ]


@router.get("/{coach_name}/breakdown")
async def get_coach_breakdown(
    coach_name: str,
    season: Optional[int] = Query(None, description="Season year, e.g. 2024"),
    db: Session = Depends(get_db),
):
    """
    Full coaching breakdown from PBP data.

    Returns formation tendencies (shotgun %, no-huddle, play-action),
    personnel groupings (11/12/21 personnel frequency + EPA),
    run scheme (inside vs outside, direction breakdown),
    pass detail (air yards, depth zones, direction, scramble rate),
    defensive scheme (personnel groupings, blitz rate, box density),
    and derived strengths, weaknesses, and tendency labels.

    Requires the play_by_play table to be populated (run the data loader first).
    """
    from .coaching_pbp import (
        compute_formation_breakdown,
        compute_personnel_breakdown,
        compute_run_scheme,
        compute_pass_detail,
        compute_defense_scheme,
        compute_strengths_weaknesses,
    )

    q = db.query(Schedule).filter(
        or_(Schedule.home_coach == coach_name, Schedule.away_coach == coach_name)
    )
    if season is not None:
        q = q.filter(Schedule.season == season)
    games = q.all()

    if not games:
        raise HTTPException(status_code=404, detail=f"Coach '{coach_name}' not found")

    season_team_pairs: set = set()
    for g in games:
        if g.home_coach == coach_name:
            season_team_pairs.add((g.season, g.home_team))
        if g.away_coach == coach_name:
            season_team_pairs.add((g.season, g.away_team))

    import json

    results = []
    for s, team in sorted(season_team_pairs):
        # Cache lookup
        cached = db.get(CoachSeasonAnalytics, (coach_name, s, team))
        if cached is not None:
            results.append(json.loads(cached.breakdown_json))
            continue

        try:
            offense_summary = _compute_offense_tendencies(db, team, s)
            defense_summary = _compute_defense_tendencies(db, team, s)
            formation = compute_formation_breakdown(db, team, s)
            personnel = compute_personnel_breakdown(db, team, s)
            run_scheme = compute_run_scheme(db, team, s)
            pass_detail = compute_pass_detail(db, team, s)
            def_scheme = compute_defense_scheme(db, team, s)
            fourth_sample = _sample_plays(db, team, s, "fourth_down")
            third_sample = _sample_plays(db, team, s, "third_down")
            insights = compute_strengths_weaknesses(
                offense_summary, defense_summary,
                formation, run_scheme, pass_detail, def_scheme,
            )
        except Exception as exc:
            import traceback
            logger.warning("Breakdown failed for %s %s %s: %s\n%s",
                           coach_name, team, s, exc, traceback.format_exc())
            return {"status": "no_data", "message": f"Computation failed: {type(exc).__name__}: {exc}"}

        result_entry = {
            "season": s,
            "team": team,
            "offense": {
                **offense_summary,
                "formation": formation,
                "personnel": personnel,
                "run_scheme": run_scheme,
                "passing": pass_detail,
                "fourth_down_sample": fourth_sample,
                "third_down_sample": third_sample,
            },
            "defense": {
                **defense_summary,
                "scheme": def_scheme,
            },
            "strengths": insights["strengths"],
            "weaknesses": insights["weaknesses"],
            "tendencies": insights["tendencies"],
        }

        # Write-through to cache
        try:
            db.merge(CoachSeasonAnalytics(
                coach_name=coach_name,
                season=s,
                team=team,
                breakdown_json=json.dumps(result_entry, default=str),
            ))
            db.commit()
        except Exception:
            db.rollback()

        results.append(result_entry)

    return {
        "status": "success",
        "coach": coach_name,
        "season": season,
        "data": clean_data_for_json(results),
    }


@router.post("/compare")
async def compare_coaches(
    coach_names: List[str],
    season: Optional[int] = Query(None),
    years: Optional[List[int]] = Query(None),
    db: Session = Depends(get_db),
):
    """Compare multiple coaches side by side."""
    # --- DB path (only when the DB has data for the requested seasons) ---
    q = db.query(Schedule)
    if season is not None:
        q = q.filter(Schedule.season == season)
    elif years:
        q = q.filter(Schedule.season.in_(years))
    schedules = q.all()
    if schedules:
        records = _build_coaching_records(schedules)
        missing = [c for c in coach_names if c not in records]
        if missing:
            raise HTTPException(status_code=404, detail=f"Coaches not found: {missing}")

        comparison_data = {}
        for coach in coach_names:
            seasons_data = _format_seasons(records[coach])
            total_wins = sum(s["wins"] for s in seasons_data)
            total_losses = sum(s["losses"] for s in seasons_data)
            total = total_wins + total_losses
            overall_win_pct = round((total_wins / total * 100) if total else 0, 1)
            win_score = round(40 + (overall_win_pct / 100) * 55, 1)
            comparison_data[coach] = {
                "seasons": seasons_data,
                "overall_record": f"{total_wins}-{total_losses}",
                "overall_win_percentage": overall_win_pct,
                "win_score": win_score,
            }

        return {
            "status": "success",
            "coaches": coach_names,
            "season": season,
            "comparison_matrix": clean_data_for_json({
                "win_percentage": {
                    c: {"score": comparison_data[c]["overall_win_percentage"],
                        "win_score": comparison_data[c]["win_score"]}
                    for c in coach_names
                }
            }),
            "detailed_data": clean_data_for_json(comparison_data),
        }

    # --- nflreadpy fallback ---
    if years is None:
        years = [get_current_nfl_season()]
    systems = check_grading_systems()
    if not systems["coaching_analytics"]:
        raise HTTPException(status_code=503, detail="Coaching analytics system not available")

    try:
        analytics = get_coaching_analytics(years)
        available = analytics.get_available_coaches(season=season)
        invalid = [c for c in coach_names if c not in available]
        if invalid:
            raise HTTPException(status_code=404, detail=f"Coaches not found: {invalid}")

        comparison_data = {}
        for coach in coach_names:
            seasons_data = _get_coach_season_data_legacy(analytics, coach, season)
            total_wins = sum(s["wins"] for s in seasons_data)
            total_losses = sum(s["losses"] for s in seasons_data)
            total = total_wins + total_losses
            overall_win_pct = round((total_wins / total * 100) if total else 0, 1)
            win_score = round(40 + (overall_win_pct / 100) * 55, 1)
            comparison_data[coach] = {
                "seasons": seasons_data,
                "overall_record": f"{total_wins}-{total_losses}",
                "overall_win_percentage": overall_win_pct,
                "win_score": win_score,
                "win_letter_grade": analytics.get_letter_grade(win_score),
            }

        return {
            "status": "success",
            "coaches": coach_names,
            "season": season,
            "comparison_matrix": clean_data_for_json({
                "win_percentage": {
                    c: {"score": comparison_data[c]["overall_win_percentage"],
                        "letter_grade": comparison_data[c]["win_letter_grade"]}
                    for c in coach_names
                }
            }),
            "detailed_data": clean_data_for_json(comparison_data),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing coaches: {e}")
        raise HTTPException(status_code=500, detail=str(e))
