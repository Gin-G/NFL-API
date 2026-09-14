#!/usr/bin/env python3
"""Tests for database/loader.py's season orchestration.

The per-table loaders skip a season that already has rows. That is right for a
finished season and wrong for the one in progress, which gains a week every
week: before this, the first in-season run loaded week 1 and nothing after.
"""

import pytest


@pytest.fixture
def recorded(monkeypatch):
    from api import utils
    from database import loader

    calls = []

    def recorder(name):
        def fn(db, season, force=False):
            calls.append((name, season, force))
            return 0
        return fn

    for name in ("load_schedules", "load_rosters", "load_player_stats",
                 "load_depth_charts", "load_snap_counts", "load_pbp"):
        monkeypatch.setattr(loader, name, recorder(name))
    monkeypatch.setattr(loader, "ensure_pbp_indexes", lambda: None)
    monkeypatch.setattr(loader, "load_teams", lambda db: 0)
    monkeypatch.setattr(utils, "get_current_nfl_season", lambda: 2026)
    return calls


def _forced(calls, season):
    return {name: force for name, s, force in calls if s == season}


def test_current_season_tables_are_reloaded(recorded):
    from database import loader

    loader.load_all_data(db=None, seasons=[2026, 2025])

    now = _forced(recorded, 2026)
    for name in ("load_schedules", "load_rosters", "load_player_stats",
                 "load_depth_charts", "load_snap_counts"):
        assert now[name] is True, f"{name} would stop at the first week loaded"
    # PBP appends rather than merging, so forcing it would duplicate plays.
    assert now["load_pbp"] is False


def test_finished_seasons_keep_the_fast_skip(recorded):
    from database import loader

    loader.load_all_data(db=None, seasons=[2026, 2025])

    assert not any(_forced(recorded, 2025).values())


def test_explicit_force_still_forces_everything(recorded):
    from database import loader

    loader.load_all_data(db=None, seasons=[2025], force=True)

    assert all(_forced(recorded, 2025).values())
