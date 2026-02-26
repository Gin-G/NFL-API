#!/usr/bin/env python3
"""
CLI entry point for loading NFL data into the database.

Usage (from the app/ directory):
    python -m scripts.load_data

Environment variable:
    DATABASE_URL  — SQLAlchemy connection string.
                    Falls back to sqlite:///./nfl_dev.db if unset.
"""

import sys
import os
import logging
import argparse

# Ensure app/ is on sys.path when run as a module from inside app/
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _current_nfl_season() -> int:
    from datetime import datetime
    now = datetime.now()
    return now.year if now.month >= 9 else now.year - 1


def _is_already_loaded(db, current_season: int) -> bool:
    """Return True if teams, PBP for current + prior seasons, AND coach analytics cache are in DB."""
    from database.models import Team, PlayerStat, CoachSeasonAnalytics
    from database.session import engine
    from sqlalchemy import text
    try:
        team_count = db.query(Team).limit(1).count()
        if team_count == 0:
            return False
        stat_count = db.query(PlayerStat).filter(
            PlayerStat.season == current_season
        ).limit(1).count()
        if stat_count == 0:
            return False
        # PBP table is created dynamically — use a fresh engine connection so a
        # missing-table error doesn't abort the session's PostgreSQL transaction.
        # Check both current and prior season so a partial load doesn't skip reloading.
        try:
            with engine.connect() as conn:
                for season_to_check in [current_season, current_season - 1]:
                    pbp_count = conn.execute(
                        text("SELECT COUNT(*) FROM play_by_play WHERE season = :s"),
                        {"s": season_to_check},
                    ).scalar()
                    if not (pbp_count and pbp_count > 0):
                        return False
        except Exception:
            return False
        # Coach analytics cache must be populated for recent seasons, otherwise
        # breakdown queries fall through to expensive real-time PBP computation.
        try:
            analytics_count = db.query(CoachSeasonAnalytics).filter(
                CoachSeasonAnalytics.season >= current_season - 1
            ).limit(1).count()
            if analytics_count == 0:
                return False
        except Exception:
            return False
        return True
    except Exception:
        db.rollback()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Load NFL data into PostgreSQL/SQLite")
    parser.add_argument(
        "--start-season",
        type=int,
        default=1999,
        help="First season to load (default: 1999)",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help="Last season to load (default: current NFL season)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Load data even if the DB already appears populated",
    )
    args = parser.parse_args()

    end = args.end_season or _current_nfl_season()
    seasons = list(range(args.start_season, end + 1))

    from database.session import engine, SessionLocal
    from database.models import Base
    from database.loader import load_all_data, ensure_pbp_indexes

    logger.info("Creating tables (no-op if they already exist)…")
    Base.metadata.create_all(engine)

    # Always ensure PBP indexes exist — fast no-op if already present, critical for
    # coaching analytics queries. Must run before the early-exit check so indexes
    # are created even when the data load is skipped.
    ensure_pbp_indexes()

    db = SessionLocal()
    try:
        if not args.force and _is_already_loaded(db, end):
            logger.info(
                "DB already contains teams and season %d stats — nothing to do. "
                "Run with --force to reload.",
                end,
            )
            return

        logger.info("Loading seasons %d–%d (%d total)", seasons[0], seasons[-1], len(seasons))
        load_all_data(db, seasons, force=args.force)
    finally:
        db.close()

    logger.info("Done.")


if __name__ == "__main__":
    main()
