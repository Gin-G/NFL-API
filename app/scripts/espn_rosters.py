#!/usr/bin/env python3
"""Scrape current rosters from ESPN — the preseason roster source.

nflverse publishes weekly rosters only once a season is under way, so through the
summer this is the only read on who is on which team. scripts.sync_rosters uses
it as a fallback and nflverse the rest of the year; this module is the fetching
half, kept separate so it has no database imports and can be exercised alone.

ESPN athlete ids map to gsis_id through nflreadpy's player crosswalk, so the rows
join to stats and projections like any other.
"""
import json
import logging
import os
import re
import sys
import urllib.request
from datetime import datetime

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("espn_rosters")

_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
# ESPN abbreviation -> nflverse abbreviation (for joins to schedules/grades/stats)
_TEAM_FIX = {"WSH": "WAS", "LAR": "LA"}
_STATUS = {
    "offense": "active", "defense": "active", "specialTeam": "active",
    "injuredReserveOrOut": "injured_reserve", "practiceSquad": "practice_squad",
    "suspended": "suspended",
}


_OFF_POS = {"QB", "RB", "WR", "TE", "FB"}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _get_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def _espn_id(athlete):
    """espn athlete id from a depth-chart cell's href (.../id/<n>/...) or uid (a:<n>)."""
    for key in ("href", "uid"):
        m = re.search(r"(?:id/|a:)(\d+)", athlete.get(key, ""))
        if m:
            return m.group(1)
    return None


def _depth_ranks(abbr_lower, slug):
    """{espn_id: pos_rank} from a team's ESPN depth page. The page embeds
    window['__espnfitt__']; the offensive rows are [POS, starter, 2nd, 3rd, ...] so a
    cell's column index is its depth rank. Best (lowest) rank wins across formations."""
    url = f"https://www.espn.com/nfl/team/depth/_/name/{abbr_lower}/{slug}"
    html = _get_text(url)
    m = re.search(r"window\['__espnfitt__'\]\s*=\s*(\{.*?\});</script>", html, re.S)
    if not m:
        return {}
    depth = json.loads(m.group(1))["page"]["content"]["depth"]
    ranks = {}
    for grp in depth.get("dethTeamGroups", []):
        for row in grp.get("rows", []):
            if not row or not isinstance(row[0], str) or row[0].strip().upper() not in _OFF_POS:
                continue
            for rank, cell in enumerate(row[1:], start=1):
                if not isinstance(cell, dict):
                    continue
                eid = _espn_id(cell)
                if eid and (eid not in ranks or rank < ranks[eid]):
                    ranks[eid] = rank
    return ranks


def _crosswalk():
    """espn_id -> gsis_id from nflreadpy's player table."""
    import nflreadpy as nfl
    import pandas as pd
    pl = nfl.load_players().to_pandas()
    xwalk = {}
    for e, g in zip(pl["espn_id"], pl["gsis_id"]):
        if pd.isna(e) or pd.isna(g):
            continue
        try:
            xwalk[str(int(float(e)))] = str(g)
        except (TypeError, ValueError):
            continue
    return xwalk


def _iter_players(roster_json):
    for grp in roster_json.get("athletes", []):
        status = _STATUS.get(grp.get("position"), "active")
        for p in grp.get("items", []):
            yield p, status


def fetch_espn_rosters():
    """Roster rows as plain dicts, ready for CurrentRoster(**fields).

    Depth rank comes from each team's depth page, where a cell's column index IS
    the rank. A team whose page fails to parse still contributes its players,
    without ranks, rather than taking the whole sync down.
    """
    esp2gsis = _crosswalk()
    logger.info("crosswalk: %d espn->gsis entries", len(esp2gsis))

    teams = _get(f"{_BASE}/teams")["sports"][0]["leagues"][0]["teams"]
    out = []
    for t in teams:
        tid = t["team"]["id"]
        abbr = _TEAM_FIX.get(t["team"]["abbreviation"], t["team"]["abbreviation"])
        try:
            roster = _get(f"{_BASE}/teams/{tid}/roster")
        except Exception as exc:
            logger.warning("roster fetch failed for %s: %s", abbr, exc)
            continue
        try:
            ranks = _depth_ranks(t["team"]["abbreviation"].lower(), t["team"]["slug"])
        except Exception as exc:
            logger.warning("depth-chart fetch failed for %s: %s", abbr, exc)
            ranks = {}
        n = 0
        for p, status in _iter_players(roster):
            espn_id = str(p.get("id"))
            exp = p.get("experience") or {}
            out.append({
                "espn_id": espn_id,
                "gsis_id": esp2gsis.get(espn_id),
                "full_name": p.get("fullName"),
                "position": (p.get("position") or {}).get("abbreviation"),
                "team": abbr,
                "status": status,
                "raw_status": status,
                "jersey": str(p.get("jersey")) if p.get("jersey") is not None else None,
                "age": p.get("age"),
                "experience": exp.get("years"),
                "depth_rank": ranks.get(espn_id),
                "week": None,
            })
            n += 1
        logger.info("%s: %d players (%d with depth rank)", abbr, n, len(ranks))
    return out
