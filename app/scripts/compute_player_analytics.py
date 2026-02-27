#!/usr/bin/env python3
"""
Standalone player analytics pre-computation job.

Computes PBP-derived analytics for all players in each season and caches
results in the player_season_analytics table. Processes newest seasons first
so current-season data is available quickly.

Progress is tracked in the analytics_job_status table (job_type="player_analytics")
and exposed via GET /players/analytics/status.

Usage (from the app/ directory):
    python -m scripts.compute_player_analytics
    python -m scripts.compute_player_analytics --start-season 2025 --end-season 2020
    python -m scripts.compute_player_analytics --force   # recompute even if cached

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


def _count_expected_players(db, seasons: list) -> int:
    """Estimate total (player_id, season) combos from player_stats."""
    from sqlalchemy import text
    if not seasons:
        return 0
    placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    try:
        n = db.execute(
            text(f"""
                SELECT COUNT(DISTINCT player_id)
                FROM player_stats
                WHERE season IN ({placeholders}) AND player_id IS NOT NULL
            """),
            params,
        ).scalar()
        # Multiply by number of seasons as rough estimate (players appear in many seasons)
        return int(n or 0) * max(1, len(seasons))
    except Exception:
        db.rollback()
        return len(seasons) * 800  # fallback rough estimate


def _compute_season(db, season: int) -> dict:
    """
    Run all batch queries for a season.
    Returns {player_id: blob} where blob contains all available metric sections.
    """
    from api.player_pbp import (
        compute_qb_metrics,
        compute_rush_metrics,
        compute_receiver_metrics,
        compute_defensive_metrics,
        get_player_positions,
    )

    logger.info("  Running batch queries for season %d…", season)
    qb_data = compute_qb_metrics(db, season)
    rush_data = compute_rush_metrics(db, season)
    recv_data = compute_receiver_metrics(db, season)
    def_data = compute_defensive_metrics(db, season)
    pos_map = get_player_positions(db, season)

    logger.info(
        "  Season %d: %d QBs, %d rushers, %d receivers, %d defenders",
        season, len(qb_data), len(rush_data), len(recv_data), len(def_data),
    )

    all_pids = set(qb_data) | set(rush_data) | set(recv_data) | set(def_data)
    blobs: dict = {}

    for pid in all_pids:
        blob: dict = {}
        if pid in qb_data:
            blob["passing"] = qb_data[pid]
        if pid in rush_data:
            blob["rushing"] = rush_data[pid]
        if pid in recv_data:
            blob["receiving"] = recv_data[pid]
        if pid in def_data:
            blob["defense"] = def_data[pid]

        # Determine name/position/team — prefer player_stats position map
        if pid in pos_map:
            name, pos, team = pos_map[pid]
        else:
            name, pos, team = "", "", ""
            for key in ("passing", "rushing", "receiving", "defense"):
                if key in blob:
                    name = blob[key].get("player_name", "")
                    team = blob[key].get("team", "")
                    break

        blob["player_name"] = name
        blob["position"] = pos
        blob["team"] = team
        blobs[pid] = blob

    return blobs


def _update_job(db, job, **kwargs):
    """Update job status fields and commit."""
    for k, v in kwargs.items():
        setattr(job, k, v)
    job.updated_at = datetime.utcnow()
    db.merge(job)
    db.commit()


def run(db, seasons: list, force: bool, job) -> None:
    from database.models import PlayerSeasonAnalytics

    pbp_seasons: dict[int, bool] = {}
    total_processed = 0
    total_skipped = 0
    total_failed = 0

    for season_idx, season in enumerate(seasons, start=1):
        _update_job(
            db, job,
            current_season=season,
            current_coach=f"season {season} ({season_idx}/{len(seasons)})",
        )

        if season not in pbp_seasons:
            pbp_seasons[season] = _pbp_loaded(db, season)
        if not pbp_seasons[season]:
            logger.info("  Skipping season %d — PBP not loaded", season)
            continue

        try:
            blobs = _compute_season(db, season)
        except Exception as exc:
            logger.warning("  Season %d batch failed: %s", season, exc)
            db.rollback()
            continue

        season_processed = 0
        season_skipped = 0
        season_failed = 0

        for pid, blob in blobs.items():
            team = blob.get("team", "")
            pos = blob.get("position", "")
            name = blob.get("player_name", "")

            if not force:
                existing = db.get(PlayerSeasonAnalytics, (pid, season, team))
                if existing is not None:
                    season_skipped += 1
                    continue

            try:
                db.merge(PlayerSeasonAnalytics(
                    player_id=pid,
                    season=season,
                    team=team,
                    player_name=name,
                    position=pos,
                    analytics_json=json.dumps(blob, default=str),
                ))
                db.commit()
                season_processed += 1
            except Exception as exc:
                db.rollback()
                logger.warning("  Player %s season %d failed: %s", pid, season, exc)
                season_failed += 1

        total_processed += season_processed
        total_skipped += season_skipped
        total_failed += season_failed
        total_done = total_processed + total_skipped + total_failed
        pct = round(season_idx / len(seasons) * 100, 1)

        _update_job(
            db, job,
            processed_entries=total_processed,
            skipped_entries=total_skipped,
            failed_entries=total_failed,
            total_entries=total_done,
        )
        logger.info(
            "  Season %d: %d computed, %d skipped, %d failed — %.1f%% of seasons done",
            season, season_processed, season_skipped, season_failed, pct,
        )

    _update_job(
        db, job,
        status="completed",
        current_season=None,
        current_coach=None,
    )
    logger.info(
        "Done. %d computed, %d skipped, %d failed across %d seasons",
        total_processed, total_skipped, total_failed, len(seasons),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute player PBP analytics and cache in DB (newest seasons first)"
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
    logger.info(
        "Player analytics job: seasons %d → %d (%d total)",
        start, end, len(seasons),
    )

    from database.session import engine, SessionLocal
    from database.models import Base, AnalyticsJobStatus
    from database.loader import ensure_pbp_indexes

    logger.info("Ensuring DB tables and PBP indexes exist…")
    Base.metadata.create_all(engine)
    ensure_pbp_indexes()

    db = SessionLocal()
    try:
        # Mark any stuck running player analytics jobs as interrupted
        stuck = db.query(AnalyticsJobStatus).filter(
            AnalyticsJobStatus.status == "running",
            AnalyticsJobStatus.job_type == "player_analytics",
        ).all()
        for s in stuck:
            s.status = "interrupted"
        if stuck:
            db.commit()
            logger.info("Marked %d interrupted job(s)", len(stuck))

        # Create a new job record
        job = AnalyticsJobStatus(
            job_type="player_analytics",
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
        logger.info("Created player analytics job id=%d", job.id)

        run(db, seasons, args.force, job)

    except Exception as exc:
        logger.error("Player analytics job failed: %s", exc, exc_info=True)
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
