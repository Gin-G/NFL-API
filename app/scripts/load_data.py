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
    """Return True if teams, current-season stats, AND current-season PBP are in the DB."""
    from database.models import Team, PlayerStat
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
        # PBP table is created dynamically — missing table means not loaded
        pbp_count = db.execute(
            text("SELECT COUNT(*) FROM play_by_play WHERE season = :s"),
            {"s": current_season},
        ).scalar()
        return bool(pbp_count and pbp_count > 0)
    except Exception:
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
    from database.loader import load_all_data

    logger.info("Creating tables (no-op if they already exist)…")
    Base.metadata.create_all(engine)

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
