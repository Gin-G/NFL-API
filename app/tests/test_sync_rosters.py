#!/usr/bin/env python3
"""Tests for the roster sync's source selection and status mapping.

The sync went from ESPN-only to nflverse-first once nflverse published 2026: the
rules worth pinning are which week it takes, how statuses map, and that ESPN is
still there for the preseason stretch when nflverse has nothing.
"""

from datetime import date

import pandas as pd
import pytest

from database.models import CurrentRoster


@pytest.fixture(autouse=True)
def _clean(db_session):
    db_session.query(CurrentRoster).delete()
    db_session.commit()
    yield


def _rosters():
    """Two weeks of rosters; a player who changed teams between them."""
    return pd.DataFrame([
        {"week": 2, "gsis_id": "00-1", "espn_id": "8439", "full_name": "Traded Guy",
         "position": "WR", "team": "NYJ", "status": "ACT", "jersey_number": 11.0,
         "years_exp": 4, "birth_date": "1999-03-01"},
        {"week": 3, "gsis_id": "00-1", "espn_id": "8439", "full_name": "Traded Guy",
         "position": "WR", "team": "WAS", "status": "ACT", "jersey_number": 11.0,
         "years_exp": 4, "birth_date": "1999-03-01"},
        {"week": 3, "gsis_id": "00-2", "espn_id": None, "full_name": "Sat Last Week",
         "position": "TE", "team": "LV", "status": "INA", "jersey_number": 85.0,
         "years_exp": 2, "birth_date": None},
        {"week": 3, "gsis_id": "00-3", "espn_id": "1", "full_name": "On IR",
         "position": "RB", "team": "LV", "status": "RES", "jersey_number": 22.0,
         "years_exp": 6, "birth_date": None},
        {"week": 3, "gsis_id": "00-4", "espn_id": "2", "full_name": "Squadder",
         "position": "QB", "team": "LV", "status": "DEV", "jersey_number": 9.0,
         "years_exp": 0, "birth_date": None},
    ])


@pytest.fixture
def nflverse(monkeypatch):
    """nflverse with rosters through week 3 and one depth-chart snapshot."""
    import sys
    import types

    charts = pd.DataFrame([
        {"dt": "2026-09-25T12:44:44Z", "gsis_id": "00-1", "pos_abb": "WR", "pos_rank": 2},
        {"dt": "2026-09-25T12:44:44Z", "gsis_id": "00-1", "pos_abb": "WR", "pos_rank": 1},
        {"dt": "2026-03-01T12:00:00Z", "gsis_id": "00-2", "pos_abb": "TE", "pos_rank": 1},
    ])
    fake = types.ModuleType("nflreadpy")
    fake.load_rosters_weekly = lambda seasons: _rosters()
    fake.load_depth_charts = lambda seasons: charts
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)
    return fake


class TestNflverseRows:
    def test_takes_the_latest_week_only(self, nflverse):
        from scripts import sync_rosters

        rows = sync_rosters.rows_from_nflverse(2026)
        traded = [r for r in rows if r.gsis_id == "00-1"]
        assert len(traded) == 1
        assert traded[0].team == "WAS"          # not last week's team
        assert traded[0].week == 3

    def test_status_mapping(self, nflverse):
        from scripts import sync_rosters

        by_id = {r.gsis_id: r for r in sync_rosters.rows_from_nflverse(2026)}
        # INA is a gameday designation, not roster standing.
        assert (by_id["00-2"].status, by_id["00-2"].raw_status) == ("active", "INA")
        assert by_id["00-3"].status == "injured_reserve"
        assert by_id["00-4"].status == "practice_squad"

    def test_carries_ids_jersey_and_experience(self, nflverse):
        from scripts import sync_rosters

        row = next(r for r in sync_rosters.rows_from_nflverse(2026) if r.gsis_id == "00-1")
        assert (row.espn_id, row.jersey, row.experience) == ("8439", "11", 4)
        assert row.source == "nflverse"

    def test_missing_espn_id_is_fine(self, nflverse):
        from scripts import sync_rosters

        row = next(r for r in sync_rosters.rows_from_nflverse(2026) if r.gsis_id == "00-2")
        assert row.espn_id is None

    def test_no_rosters_yet_is_empty_not_an_error(self, monkeypatch):
        import sys
        import types

        from scripts import sync_rosters

        fake = types.ModuleType("nflreadpy")
        fake.load_rosters_weekly = lambda seasons: pd.DataFrame()
        fake.load_depth_charts = lambda seasons: pd.DataFrame()
        monkeypatch.setitem(sys.modules, "nflreadpy", fake)
        assert sync_rosters.rows_from_nflverse(2027) == []


class TestDepthRanks:
    def test_newest_snapshot_and_best_slot(self, nflverse):
        from scripts import sync_rosters

        ranks = sync_rosters.latest_depth_ranks(2026)
        # Best (lowest) rank within the newest snapshot; March's row is ignored.
        assert ranks == {"00-1": 1}


class TestSourceSelection:
    def test_espn_only_when_nflverse_is_empty(self, monkeypatch, db_session):
        import sys
        import types

        from scripts import sync_rosters

        fake = types.ModuleType("nflreadpy")
        fake.load_rosters_weekly = lambda seasons: pd.DataFrame()
        fake.load_depth_charts = lambda seasons: pd.DataFrame()
        monkeypatch.setitem(sys.modules, "nflreadpy", fake)
        monkeypatch.setattr(sync_rosters, "rows_from_espn",
                            lambda: [CurrentRoster(gsis_id="00-9", full_name="Preseason Guy",
                                                   team="GB", position="WR", status="active",
                                                   source="espn")])

        assert sync_rosters.sync(db_session, season=2027) == 1
        assert db_session.query(CurrentRoster).one().source == "espn"

    def test_nflverse_wins_when_it_has_rows(self, nflverse, db_session, monkeypatch):
        from scripts import sync_rosters

        def no(*a, **k):
            raise AssertionError("ESPN should not be touched when nflverse has rosters")

        monkeypatch.setattr(sync_rosters, "rows_from_espn", no)
        assert sync_rosters.sync(db_session, season=2026) == 4
        assert {r.source for r in db_session.query(CurrentRoster).all()} == {"nflverse"}

    def test_each_run_replaces_the_table(self, nflverse, db_session):
        from scripts import sync_rosters

        db_session.add(CurrentRoster(gsis_id="00-old", full_name="Cut Last Week",
                                     team="LV", position="RB", status="active",
                                     source="nflverse"))
        db_session.commit()
        sync_rosters.sync(db_session, season=2026)
        assert "00-old" not in {r.gsis_id for r in db_session.query(CurrentRoster).all()}

    def test_no_source_has_anything_is_an_error(self, monkeypatch, db_session):
        import sys
        import types

        from scripts import sync_rosters

        fake = types.ModuleType("nflreadpy")
        fake.load_rosters_weekly = lambda seasons: pd.DataFrame()
        fake.load_depth_charts = lambda seasons: pd.DataFrame()
        monkeypatch.setitem(sys.modules, "nflreadpy", fake)
        monkeypatch.setattr(sync_rosters, "rows_from_espn", lambda: [])
        with pytest.raises(RuntimeError):
            sync_rosters.sync(db_session, season=2027)


class TestAge:
    def test_from_birth_date(self):
        from scripts import sync_rosters

        assert sync_rosters._age("1999-03-01", today=date(2026, 9, 25)) == 27
        assert sync_rosters._age("1999-12-01", today=date(2026, 9, 25)) == 26

    def test_missing_or_junk_is_none(self):
        from scripts import sync_rosters

        assert sync_rosters._age(None) is None
        assert sync_rosters._age("not a date") is None
