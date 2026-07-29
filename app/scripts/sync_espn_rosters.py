#!/usr/bin/env python3
"""Sync current NFL rosters from ESPN into the `espn_roster` table.

nflreadpy doesn't publish weekly rosters until in-season, so ESPN's live API is our
source of truth for current team assignments and transactions (offseason and in-season).
Run nightly (see k8s/roster-sync-cronjob.yaml). Each run is a full snapshot: the table is
replaced, so cuts/trades/signings are reflected. ESPN athlete ids are mapped to gsis_id
(via nflreadpy's player crosswalk) so these rows join to historical stats and projections.

Usage (from app/):  python -m scripts.sync_espn_rosters
"""
import json
import logging
import os
import sys
import urllib.request
from datetime import datetime

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("sync_espn_rosters")

_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
# ESPN abbreviation -> nflverse abbreviation (for joins to schedules/grades/stats)
_TEAM_FIX = {"WSH": "WAS", "LAR": "LA"}
_STATUS = {
    "offense": "active", "defense": "active", "specialTeam": "active",
    "injuredReserveOrOut": "injured_reserve", "practiceSquad": "practice_squad",
    "suspended": "suspended",
}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _crosswalk():
    """espn_id -> gsis_id from nflreadpy's player table."""
    import nflreadpy as nfl
    pl = nfl.load_players().to_pandas()
    return {str(int(e)): g for e, g in zip(pl["espn_id"], pl["gsis_id"])
            if e == e and g == g}


def _iter_players(roster_json):
    for grp in roster_json.get("athletes", []):
        status = _STATUS.get(grp.get("position"), "active")
        for p in grp.get("items", []):
            yield p, status


def sync(db) -> int:
    from database.models import EspnRoster
    esp2gsis = _crosswalk()
    logger.info("crosswalk: %d espn->gsis entries", len(esp2gsis))

    teams = _get(f"{_BASE}/teams")["sports"][0]["leagues"][0]["teams"]
    rows = []
    for t in teams:
        tid = t["team"]["id"]
        abbr = _TEAM_FIX.get(t["team"]["abbreviation"], t["team"]["abbreviation"])
        try:
            roster = _get(f"{_BASE}/teams/{tid}/roster")
        except Exception as exc:
            logger.warning("roster fetch failed for %s: %s", abbr, exc)
            continue
        n = 0
        for p, status in _iter_players(roster):
            espn_id = str(p.get("id"))
            exp = p.get("experience") or {}
            rows.append(EspnRoster(
                espn_id=espn_id,
                gsis_id=esp2gsis.get(espn_id),
                full_name=p.get("fullName"),
                position=(p.get("position") or {}).get("abbreviation"),
                team=abbr,
                status=status,
                jersey=str(p.get("jersey")) if p.get("jersey") is not None else None,
                age=p.get("age"),
                experience=exp.get("years"),
                updated_at=datetime.utcnow(),
            ))
            n += 1
        logger.info("%s: %d players", abbr, n)

    # Full-snapshot replace so transactions (cuts/trades) are reflected.
    db.query(EspnRoster).delete()
    db.bulk_save_objects(rows)
    db.commit()
    mapped = sum(1 for r in rows if r.gsis_id)
    logger.info("Synced %d roster rows (%d mapped to gsis, %.0f%%)",
                len(rows), mapped, 100 * mapped / max(len(rows), 1))
    return len(rows)


def main():
    from database.session import engine, SessionLocal
    from database.models import Base, AnalyticsJobStatus
    Base.metadata.create_all(engine)
    db = SessionLocal()
    job = AnalyticsJobStatus(job_type="roster_sync", status="running",
                             started_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.add(job); db.commit()
    try:
        n = sync(db)
        job.status = "completed"; job.processed_entries = n
    except Exception as exc:
        logger.error("roster sync failed: %s", exc)
        job.status = "failed"; job.error_message = str(exc)[:500]
        db.rollback()
    finally:
        job.updated_at = datetime.utcnow()
        db.merge(job); db.commit()
        db.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
