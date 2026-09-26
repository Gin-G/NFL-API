#!/usr/bin/env python3
"""Sync current NFL rosters into the `current_roster` table.

Once a season is under way nflverse publishes weekly rosters — team, position,
status, and every id worth joining on — so that is the source: official, already
in the image, and no scraping. Before the season it has nothing for the new year,
which is why this started life reading ESPN's live API, kept here as the
fallback for exactly that stretch (and for the day nflverse is late).

Depth rank comes from nflverse's depth charts, which since 2025 are dated
league-wide snapshots rather than weekly rows; the newest snapshot is the chart
in force. It is the one field here that says whether a player will be on the
field rather than merely employed.

Each run replaces the table, so cuts, trades and signings are reflected.

Usage (from app/):
    python -m scripts.sync_rosters                  # nflverse, ESPN if it is empty
    python -m scripts.sync_rosters --source espn    # force the fallback
"""
import argparse
import logging
import os
import sys
from datetime import date, datetime

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("sync_rosters")

OFFENSIVE_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}

# nflverse status code -> the vocabulary this table has always used.
# INA is a GAMEDAY designation, not roster standing: a player inactive for last
# week's game is still on the active roster, and treating him otherwise is what
# briefly dropped Tua, Bowers and Kamara off the week 2 projection board.
NFLVERSE_STATUS = {
    "ACT": "active",
    "INA": "active",
    "RES": "injured_reserve",
    "PUP": "injured_reserve",
    "NON": "injured_reserve",
    "DEV": "practice_squad",
    "SUS": "suspended",
    "EXE": "exempt",
    "CUT": "cut",
    "TRC": "cut",
    "RET": "retired",
}


def _to_pandas(frame):
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame


def _int(value):
    try:
        if value is None or value != value:      # None / NaN
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _str(value):
    if value is None or value != value:
        return None
    text = str(value).strip()
    return text or None


def _age(birth_date, today=None):
    """Age in years from a date-ish value, or None."""
    raw = _str(birth_date)
    if not raw:
        return None
    try:
        born = datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    today = today or date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def latest_depth_ranks(season: int) -> dict:
    """``{gsis_id: pos_rank}`` from the newest depth-chart snapshot."""
    import nflreadpy as nfl

    try:
        charts = _to_pandas(nfl.load_depth_charts(seasons=[season]))
    except Exception as exc:
        logger.warning("no depth charts for %d (%s); ranks will be null", season, exc)
        return {}
    if charts.empty:
        return {}
    if "dt" in charts.columns:
        charts = charts[charts["dt"] == charts["dt"].max()]
    elif "week" in charts.columns:
        charts = charts[charts["week"] == charts["week"].max()]
    ranks = {}
    for row in charts.itertuples(index=False):
        gsis = _str(getattr(row, "gsis_id", None))
        rank = _int(getattr(row, "pos_rank", None))
        if gsis and rank is not None:
            # A player listed at several slots keeps his best rank.
            ranks[gsis] = min(rank, ranks.get(gsis, rank))
    logger.info("depth chart: %d players ranked", len(ranks))
    return ranks


def rows_from_nflverse(season: int):
    """Roster rows for the latest published week of `season`, or [] if there are
    none yet (the preseason case the ESPN fallback exists for)."""
    import nflreadpy as nfl

    from database.models import CurrentRoster

    try:
        rosters = _to_pandas(nfl.load_rosters_weekly(seasons=[season]))
    except Exception as exc:
        logger.warning("nflverse has no %d rosters (%s)", season, exc)
        return []
    if rosters.empty or "week" not in rosters.columns:
        return []

    week = int(rosters["week"].max())
    current = rosters[rosters["week"] == week]
    ranks = latest_depth_ranks(season)
    logger.info("nflverse rosters: %d rows at week %d", len(current), week)

    rows, now = [], datetime.utcnow()
    for r in current.itertuples(index=False):
        gsis = _str(getattr(r, "gsis_id", None))
        raw = (_str(getattr(r, "status", None)) or "").upper()
        rows.append(CurrentRoster(
            gsis_id=gsis,
            espn_id=_str(getattr(r, "espn_id", None)),
            full_name=_str(getattr(r, "full_name", None)),
            position=_str(getattr(r, "position", None)),
            team=_str(getattr(r, "team", None)),
            status=NFLVERSE_STATUS.get(raw, raw.lower() or None),
            raw_status=raw or None,
            jersey=_str(_int(getattr(r, "jersey_number", None))),
            age=_age(getattr(r, "birth_date", None)),
            experience=_int(getattr(r, "years_exp", None)),
            depth_rank=ranks.get(gsis),
            source="nflverse",
            week=week,
            updated_at=now,
        ))
    return rows


def rows_from_espn():
    """Roster rows scraped from ESPN — the preseason source, before nflverse
    publishes a new season."""
    from database.models import CurrentRoster
    from .espn_rosters import fetch_espn_rosters

    now = datetime.utcnow()
    return [CurrentRoster(source="espn", updated_at=now, **fields)
            for fields in fetch_espn_rosters()]


def sync(db, season: int = None, source: str = "auto") -> int:
    from api.utils import get_current_nfl_season
    from database.models import CurrentRoster

    season = season or get_current_nfl_season()
    rows = []
    if source in ("auto", "nflverse"):
        rows = rows_from_nflverse(season)
    if not rows and source in ("auto", "espn"):
        logger.info("falling back to ESPN")
        rows = rows_from_espn()
    if not rows:
        raise RuntimeError(f"no roster rows from any source for {season}")

    # Full-snapshot replace so transactions (cuts, trades, signings) land.
    db.query(CurrentRoster).delete()
    db.bulk_save_objects(rows)
    db.commit()

    ranked = sum(1 for r in rows if r.depth_rank is not None)
    mapped = sum(1 for r in rows if r.gsis_id)
    logger.info("Synced %d rows from %s (%d with a depth rank, %d with a gsis id)",
                len(rows), rows[0].source, ranked, mapped)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Sync current NFL rosters")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--source", choices=["auto", "nflverse", "espn"], default="auto",
                        help="auto (default): nflverse, ESPN only if it has nothing yet")
    args = parser.parse_args()

    from database.models import AnalyticsJobStatus, Base, apply_light_migrations
    from database.session import SessionLocal, engine

    Base.metadata.create_all(engine)
    apply_light_migrations(engine)
    db = SessionLocal()
    job = AnalyticsJobStatus(job_type="roster_sync", status="running",
                             started_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.add(job)
    db.commit()
    ok = True
    try:
        n = sync(db, season=args.season, source=args.source)
        job.status, job.processed_entries = "completed", n
    except Exception as exc:
        logger.error("roster sync failed: %s", exc, exc_info=True)
        job.status, job.error_message = "failed", str(exc)[:500]
        db.rollback()
        ok = False
    finally:
        job.updated_at = datetime.utcnow()
        db.merge(job)
        db.commit()
        db.close()
    logger.info("Done.")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
