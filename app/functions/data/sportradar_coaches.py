#!/usr/bin/env python3
"""
Sportradar NFL Coaching Staff Client

Fetches current head coach, offensive coordinator, defensive coordinator,
and special-teams coordinator for every NFL team via the SportRadar v7 API.

Results are cached in memory for 24 hours so the first call per process
pays the full round-trip cost (~50 s on a trial key due to rate limits)
and all subsequent calls are instant.

Environment variables
---------------------
SPORTRADAR_API_KEY       – required; trial or production API key
SPORTRADAR_ACCESS_LEVEL  – optional; "trial" (default) or "production"
"""

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SportRadar position-code → response field mapping
# ---------------------------------------------------------------------------
_POSITION_MAP: dict[str, str] = {
    "HC":  "head_coach",
    "OC":  "offensive_coordinator",
    "DC":  "defensive_coordinator",
    "STC": "special_teams_coordinator",
}

_CACHE_TTL_SECONDS = 86_400  # 24 hours

# Module-level cache: key → (unix_timestamp, data)
_cache: dict[str, tuple[float, object]] = {}


def _from_cache(key: str):
    """Return cached value if still fresh, else None."""
    entry = _cache.get(key)
    if entry and time.time() - entry[0] < _CACHE_TTL_SECONDS:
        return entry[1]
    return None


def _to_cache(key: str, data) -> None:
    _cache[key] = (time.time(), data)


def clear_cache() -> None:
    """Force-expire the in-memory cache (useful for testing / admin)."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class SportradarCoachesClient:
    """
    Thin client for the SportRadar NFL v7 API focused on coaching staff.

    Parameters
    ----------
    api_key : str
    access_level : str
        "trial" for the developer trial key, "production" for a paid key.
        The trial key is rate-limited to 1 request per second.
    """

    def __init__(self, api_key: str, access_level: str = "trial") -> None:
        self.api_key = api_key
        self.base_url = (
            f"https://api.sportradar.com/nfl/official/{access_level}/v7/en"
        )
        self.headers = {
            "accept": "application/json",
            "x-api-key": api_key,
        }
        # Trial allows 1 req/s; use 1.1 s to be safe.
        # Production keys have much higher limits — override via env var.
        self._delay = float(os.getenv("SPORTRADAR_REQUEST_DELAY", "1.1"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict | None:
        url = f"{self.base_url}/{path}"
        try:
            time.sleep(self._delay)
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            logger.warning("SportRadar HTTP error for %s: %s", path, exc)
        except Exception as exc:
            logger.warning("SportRadar request failed for %s: %s", path, exc)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_team_staff(self) -> list[dict]:
        """
        Return a list of dicts, one per NFL team, with HC/OC/DC/STC fields.

        The result is cached for 24 hours.  The first call may take ~50 s
        on a trial key (33 HTTP requests × 1.1 s delay).
        """
        cached = _from_cache("all_staff")
        if cached is not None:
            logger.debug("Returning coaching staff from cache")
            return cached  # type: ignore[return-value]

        logger.info("Fetching NFL coaching staff from SportRadar …")

        teams = self._fetch_teams()
        if not teams:
            logger.error("Could not load league hierarchy from SportRadar")
            return []

        all_staff: list[dict] = []
        for team in teams:
            staff = self._fetch_team_staff(team)
            if staff:
                all_staff.append(staff)

        logger.info("Fetched staff for %d teams", len(all_staff))
        _to_cache("all_staff", all_staff)
        return all_staff

    # ------------------------------------------------------------------
    # Private fetch helpers
    # ------------------------------------------------------------------

    def _fetch_teams(self) -> list[dict]:
        """Return [{id, alias, name}, …] for every NFL team."""
        data = self._get("league/hierarchy.json")
        if not data:
            return []

        teams: list[dict] = []
        # The hierarchy may nest conferences at the top level or under a
        # "league" wrapper — handle both.
        conferences = data.get("conferences") or data.get("league", {}).get(
            "conferences", []
        )
        for conf in conferences:
            for div in conf.get("divisions", []):
                for team in div.get("teams", []):
                    teams.append(
                        {
                            "id": team.get("id", ""),
                            "alias": team.get("alias", ""),
                            "name": team.get("market", "") + " " + team.get("name", ""),
                        }
                    )
        return teams

    def _fetch_team_staff(self, team: dict) -> dict | None:
        """Fetch one team's profile and return a normalised staff dict."""
        team_id = team["id"]
        if not team_id:
            return None

        profile = self._get(f"teams/{team_id}/profile.json")
        if not profile:
            return None

        # Some responses wrap team data in a "team" key
        team_data = profile if "coaches" in profile else profile.get("team", profile)

        staff: dict = {
            "team_abbr": team["alias"],
            "team_name": team["name"].strip(),
            "head_coach": None,
            "offensive_coordinator": None,
            "defensive_coordinator": None,
            "special_teams_coordinator": None,
        }

        for coach in team_data.get("coaches", []):
            pos = coach.get("position", "")
            field = _POSITION_MAP.get(pos)
            if field:
                staff[field] = coach.get("full_name") or coach.get("name")

        return staff
