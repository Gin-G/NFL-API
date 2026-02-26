#!/usr/bin/env python3
"""
Standalone analytics pre-computation job.

Computes coaching breakdown analytics for all coach+team+season combinations
and caches results in the coach_season_analytics table. Processes newest
seasons first so recent data is available quickly.

Progress is tracked in the analytics_job_status table and exposed via
GET /coaches/analytics/status.

Usage (from the app/ directory):
    python -m scripts.compute_analytics
    python -m scripts.compute_analytics --start-season 2025 --end-season 2020
    python -m scripts.compute_analytics --force   # recompute even if cached

Environment variable:
    DATABASE_URL  — SQLAlchemy connection string.
                    Falls back to sqlite:///./nfl_dev.db if unset.
"""

import sys
import os
import logging
import argparse
import json
from datetime import datetime

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _current_nfl_season() -> int:
    now = datetime.now()
    return now.year if now.month >= 9 else now.year - 1


def _collect_pairs(db, seasons: list) -> list[tuple]:
    """Return sorted (season, coach_name, team) triples, newest season first."""
    from database.models import Schedule
    pairs: set = set()
    for season in seasons:
        games = db.query(Schedule).filter(Schedule.season == season).all()
        for g in games:
            if g.home_coach:
                pairs.add((season, g.home_coach, g.home_team))
            if g.away_coach:
                pairs.add((season, g.away_coach, g.away_team))
    # Newest season first; within season, alphabetical by coach
    return sorted(pairs, key=lambda x: (-x[0], x[1]))


def _pbp_loaded(db, season: int) -> bool:
    from database.session import engine
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            n = conn.execute(
                text("SELECT COUNT(*) FROM play_by_play WHERE season = :s"),
                {"s": season},
            ).scalar()
            return bool(n and n > 0)
    except Exception:
        return False


def _compute_one(db, coach_name: str, team: str, season: int) -> dict:
    """Run all analytics functions for a single coach+team+season."""
    from api.coaches import (
        _compute_offense_tendencies,
        _compute_defense_tendencies,
        _sample_plays,
    )
    from api.coaching_pbp import (
        compute_formation_breakdown,
        compute_personnel_breakdown,
        compute_run_scheme,
        compute_pass_detail,
        compute_defense_scheme,
        compute_strengths_weaknesses,
    )

    off = _compute_offense_tendencies(db, team, season)
    defe = _compute_defense_tendencies(db, team, season)
    formation = compute_formation_breakdown(db, team, season)
    personnel = compute_personnel_breakdown(db, team, season)
    run_scheme = compute_run_scheme(db, team, season)
    pass_detail = compute_pass_detail(db, team, season)
    def_scheme = compute_defense_scheme(db, team, season)
    fourth_sample = _sample_plays(db, team, season, "fourth_down")
    third_sample = _sample_plays(db, team, season, "third_down")
    insights = compute_strengths_weaknesses(
        off, defe, formation, run_scheme, pass_detail, def_scheme,
    )
    return {
        "season": season,
        "team": team,
        "offense": {
            **off,
            "formation": formation,
            "personnel": personnel,
            "run_scheme": run_scheme,
            "passing": pass_detail,
            "fourth_down_sample": fourth_sample,
            "third_down_sample": third_sample,
        },
        "defense": {**defe, "scheme": def_scheme},
        "strengths": insights["strengths"],
        "weaknesses": insights["weaknesses"],
        "tendencies": insights["tendencies"],
    }


def _update_job(db, job, **kwargs):
    """Update job status fields and commit."""
    for k, v in kwargs.items():
        setattr(job, k, v)
    job.updated_at = datetime.utcnow()
    db.merge(job)
    db.commit()


def run(db, pairs: list, force: bool, job) -> None:
    from database.models import CoachSeasonAnalytics

    total = len(pairs)
    _update_job(db, job, total_entries=total, status="running")
    logger.info("Starting analytics computation: %d entries (newest season first)", total)

    # Track which seasons have PBP data (cache the check)
    pbp_seasons: dict[int, bool] = {}

    for idx, (season, coach_name, team) in enumerate(pairs, start=1):
        # Update current position in job record
        _update_job(db, job, current_season=season, current_coach=coach_name)

        # Check PBP availability (cached per season)
        if season not in pbp_seasons:
            pbp_seasons[season] = _pbp_loaded(db, season)
        if not pbp_seasons[season]:
            logger.info("  [%d/%d] Skipping %s %s %d — PBP not loaded",
                        idx, total, coach_name, team, season)
            _update_job(db, job, skipped_entries=job.skipped_entries + 1)
            continue

        # Skip if already cached (unless force)
        if not force:
            existing = db.get(CoachSeasonAnalytics, (coach_name, season, team))
            if existing is not None:
                _update_job(db, job, skipped_entries=job.skipped_entries + 1)
                if idx % 50 == 0:
                    done = job.processed_entries + job.skipped_entries
                    pct = round(done / total * 100, 1) if total else 0
                    logger.info("  [%d/%d] %.1f%% — skipped (cached)", idx, total, pct)
                continue

        # Compute
        try:
            blob = _compute_one(db, coach_name, team, season)
            db.merge(CoachSeasonAnalytics(
                coach_name=coach_name,
                season=season,
                team=team,
                breakdown_json=json.dumps(blob, default=str),
            ))
            db.commit()
            new_processed = job.processed_entries + 1
            done = new_processed + job.skipped_entries
            pct = round(done / total * 100, 1) if total else 0
            _update_job(db, job, processed_entries=new_processed)
            logger.info("  [%d/%d] %.1f%% — computed %s %s %d",
                        idx, total, pct, coach_name, team, season)
        except Exception as exc:
            db.rollback()
            _update_job(db, job, failed_entries=job.failed_entries + 1)
            logger.warning("  [%d/%d] FAILED %s %s %d: %s",
                           idx, total, coach_name, team, season, exc)

    _update_job(
        db, job,
        status="completed",
        current_coach=None,
        current_season=None,
    )
    done = job.processed_entries + job.skipped_entries
    logger.info(
        "Done. %d computed, %d skipped (cached), %d failed. Total: %d/%d",
        job.processed_entries, job.skipped_entries, job.failed_entries,
        done, total,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute coaching analytics and cache in DB (newest seasons first)"
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=None,
        help="Newest season to process (default: current NFL season)",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=1999,
        help="Oldest season to process (default: 1999)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if already cached",
    )
    args = parser.parse_args()

    start = args.start_season or _current_nfl_season()
    end = args.end_season
    # Build descending list (newest first)
    seasons = list(range(start, end - 1, -1))
    logger.info("Analytics job: seasons %d → %d (%d total)", start, end, len(seasons))

    from database.session import engine, SessionLocal
    from database.models import Base, AnalyticsJobStatus
    from database.loader import ensure_pbp_indexes

    logger.info("Ensuring DB tables and PBP indexes exist…")
    Base.metadata.create_all(engine)
    ensure_pbp_indexes()

    db = SessionLocal()
    try:
        # Mark any stuck running jobs as interrupted
        stuck = db.query(AnalyticsJobStatus).filter(
            AnalyticsJobStatus.status == "running",
            AnalyticsJobStatus.job_type == "coach_analytics",
        ).all()
        for s in stuck:
            s.status = "interrupted"
        if stuck:
            db.commit()
            logger.info("Marked %d interrupted job(s)", len(stuck))

        # Create a new job record
        job = AnalyticsJobStatus(
            job_type="coach_analytics",
            status="starting",
            started_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            total_entries=0,
            processed_entries=0,
            skipped_entries=0,
            failed_entries=0,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        logger.info("Created analytics job id=%d", job.id)

        # Collect all (season, coach, team) pairs
        logger.info("Collecting coach+team+season pairs…")
        pairs = _collect_pairs(db, seasons)
        logger.info("Found %d pairs across %d seasons", len(pairs), len(seasons))

        run(db, pairs, args.force, job)

    except Exception as exc:
        logger.error("Analytics job failed: %s", exc, exc_info=True)
        try:
            job.status = "failed"
            job.error_message = str(exc)
            job.updated_at = datetime.utcnow()
            db.merge(job)
            db.commit()
        except Exception:
            pass
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
