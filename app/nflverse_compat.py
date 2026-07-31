"""Workarounds for upstream nflreadpy bugs.

Kept in one place so the reasoning lives with the workaround rather than being
duplicated across routers, the loader and the grading functions.
"""
import logging

logger = logging.getLogger(__name__)


def load_rosters_weekly(seasons):
    """Weekly rosters for a season (or list of seasons), bypassing nflreadpy's
    wrong season cap.

    nflreadpy's ``load_rosters_weekly`` validates the requested season against
    ``get_current_season()``, which only rolls over to the new year on the
    Thursday after Labor Day. Rosters, though, roll over with the new league
    year in March — nflreadpy's own ``get_current_season(roster=True)`` exists
    for exactly this, and its ``load_rosters()`` uses it. ``load_rosters_weekly``
    does not (still true as of 0.1.5), so between roughly March and September it
    raises ``ValueError: Season must be between 2002 and <last year>`` for the
    current roster year even though nflverse has already published that
    season's ``weekly_rosters/roster_weekly_<season>.parquet``.

    When the requested season is inside the roster-year window that nflverse
    actually publishes, fetch it through nflreadpy's downloader — the same call
    ``load_rosters_weekly`` makes once its own validation passes, so caching and
    configuration still apply. Genuinely out-of-range seasons re-raise.

    Returns a Polars DataFrame, matching ``nflreadpy.load_rosters_weekly``.
    """
    import nflreadpy as nfl

    season_list = [seasons] if isinstance(seasons, int) else list(seasons)
    try:
        return nfl.load_rosters_weekly(seasons=season_list)
    except ValueError:
        import polars as pl
        from nflreadpy.utils_date import get_current_season

        roster_year = get_current_season(roster=True)
        if not all(2002 <= s <= roster_year for s in season_list):
            raise
        logger.info(
            "nflreadpy rejected seasons %s for weekly rosters; fetching from "
            "nflverse directly (upstream roster-year validation bug)", season_list
        )
        from nflreadpy.downloader import get_downloader

        downloader = get_downloader()
        frames = [
            downloader.download(
                "nflverse-data", f"weekly_rosters/roster_weekly_{s}", season=s
            )
            for s in season_list
        ]
        return frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
