#!/usr/bin/env python3
"""Tests for load_pbp's replace-the-season semantics.

This loader appended for most of its life, which is why load_all_data left it
out of the in-season refresh: forcing it would have doubled the season. The
price was that the current season never updated -- 2026 held week 1 and nothing
else for six weeks, and the log said "already in DB, skipping", which was true
and useless. These pin the behaviour that let it join the refresh.
"""

import pandas as pd
import pytest
from sqlalchemy import create_engine, text


def _plays(season, game, n, yards=1.0):
    """n plays of one game, with a column we can tell runs apart by."""
    return pd.DataFrame([
        {"game_id": game, "play_id": float(i), "season": season,
         "week": 1, "posteam": "KC", "defteam": "SF", "yards_gained": yards}
        for i in range(1, n + 1)
    ])


@pytest.fixture
def pbp_db(tmp_path, monkeypatch):
    """A real sqlite file, so to_sql, DELETE and transactions all behave."""
    from database import loader, session

    engine = create_engine(f"sqlite:///{tmp_path}/pbp.db")
    monkeypatch.setattr(session, "engine", engine)

    def upstream(frame):
        """Point nflreadpy at a frame of our choosing."""
        monkeypatch.setattr(loader.nfl, "load_pbp", lambda seasons: frame)

    def rows(season=None):
        sql = "SELECT COUNT(*) FROM play_by_play"
        params = {}
        if season is not None:
            sql += " WHERE season = :s"
            params = {"s": season}
        with engine.connect() as conn:
            return conn.execute(text(sql), params).scalar()

    engine.upstream, engine.rows = upstream, rows
    return engine


# ---------------------------------------------------------------------------
# The bug
# ---------------------------------------------------------------------------

def test_reloading_a_season_replaces_it_instead_of_duplicating(pbp_db):
    """The whole reason PBP could not be refreshed. 3 plays reloaded as 5 is 5."""
    from database import loader

    pbp_db.upstream(_plays(2026, "2026_01_KC_SF", 3))
    assert loader.load_pbp(db=None, season=2026) == 3
    assert pbp_db.rows(2026) == 3

    pbp_db.upstream(_plays(2026, "2026_01_KC_SF", 5))
    assert loader.load_pbp(db=None, season=2026, force=True) == 5
    assert pbp_db.rows(2026) == 5, "appending would leave 8"


def test_a_replaced_season_is_the_new_data_not_the_old(pbp_db):
    """Not just the right count -- the rows are the ones upstream just sent."""
    from database import loader

    pbp_db.upstream(_plays(2026, "2026_01_KC_SF", 2, yards=1.0))
    loader.load_pbp(db=None, season=2026)
    pbp_db.upstream(_plays(2026, "2026_01_KC_SF", 2, yards=99.0))
    loader.load_pbp(db=None, season=2026, force=True)

    with pbp_db.connect() as conn:
        assert conn.execute(text(
            "SELECT DISTINCT yards_gained FROM play_by_play")).scalars().all() == [99.0]


def test_replacing_one_season_leaves_the_others_alone(pbp_db):
    """The DELETE is scoped to the season. 2002-2025 must not move."""
    from database import loader

    pbp_db.upstream(_plays(2025, "2025_01_KC_SF", 4))
    loader.load_pbp(db=None, season=2025)
    pbp_db.upstream(_plays(2026, "2026_01_KC_SF", 3))
    loader.load_pbp(db=None, season=2026)

    pbp_db.upstream(_plays(2026, "2026_01_KC_SF", 7))
    loader.load_pbp(db=None, season=2026, force=True)

    assert pbp_db.rows(2025) == 4
    assert pbp_db.rows(2026) == 7


# ---------------------------------------------------------------------------
# Not losing what we have
# ---------------------------------------------------------------------------

def test_a_failed_insert_leaves_the_season_as_it_was(pbp_db, monkeypatch):
    """Delete-then-insert is only safe if a failure rolls the delete back too.

    Otherwise an upstream change could empty a season on the way to failing.
    """
    from database import loader

    pbp_db.upstream(_plays(2026, "2026_01_KC_SF", 6))
    loader.load_pbp(db=None, season=2026)
    assert pbp_db.rows(2026) == 6

    broken = _plays(2026, "2026_01_KC_SF", 3)
    original = pd.DataFrame.to_sql

    def explode(self, *args, **kwargs):
        raise RuntimeError("driver blew up mid-insert")

    monkeypatch.setattr(pd.DataFrame, "to_sql", explode)
    pbp_db.upstream(broken)
    with pytest.raises(RuntimeError):
        loader.load_pbp(db=None, season=2026, force=True)
    monkeypatch.setattr(pd.DataFrame, "to_sql", original)

    assert pbp_db.rows(2026) == 6, "the delete committed without its insert"


def test_an_empty_upstream_does_not_wipe_the_season(pbp_db):
    """An nflverse hiccup is not an instruction to delete six weeks of plays."""
    from database import loader

    pbp_db.upstream(_plays(2026, "2026_01_KC_SF", 5))
    loader.load_pbp(db=None, season=2026)

    pbp_db.upstream(_plays(2026, "2026_01_KC_SF", 0))
    assert loader.load_pbp(db=None, season=2026, force=True) == 0
    assert pbp_db.rows(2026) == 5


def test_an_unknown_upstream_column_is_dropped_not_fatal(pbp_db):
    """The table has no ORM model, so it is whatever the first load created. A
    column added upstream later must not take the whole season down with it."""
    from database import loader

    pbp_db.upstream(_plays(2026, "2026_01_KC_SF", 2))
    loader.load_pbp(db=None, season=2026)

    widened = _plays(2026, "2026_01_KC_SF", 4)
    widened["brand_new_nflverse_metric"] = 1.23
    pbp_db.upstream(widened)

    assert loader.load_pbp(db=None, season=2026, force=True) == 4
    assert pbp_db.rows(2026) == 4


# ---------------------------------------------------------------------------
# The parts that already worked, kept working
# ---------------------------------------------------------------------------

def test_a_first_load_creates_the_table(pbp_db):
    """There is nothing to DELETE from before the table exists."""
    from database import loader

    pbp_db.upstream(_plays(2026, "2026_01_KC_SF", 3))
    assert loader.load_pbp(db=None, season=2026) == 3
    assert pbp_db.rows() == 3


def test_a_loaded_season_is_skipped_without_force(pbp_db):
    """Finished seasons keep the fast skip -- that is what makes a daily run cheap."""
    from database import loader

    pbp_db.upstream(_plays(2025, "2025_01_KC_SF", 3))
    loader.load_pbp(db=None, season=2025)

    pbp_db.upstream(_plays(2025, "2025_01_KC_SF", 9))
    assert loader.load_pbp(db=None, season=2025) == 0
    assert pbp_db.rows(2025) == 3


def test_seasons_before_2002_are_not_loaded(pbp_db):
    from database import loader

    pbp_db.upstream(_plays(2001, "2001_01_KC_SF", 3))
    assert loader.load_pbp(db=None, season=2001) == 0
