#!/usr/bin/env python3
"""The shadow model: a challenger trained on a shallower window, graded against
the live one on football neither had seen.

EXPERIMENTS #26/#27 could only ask "would less history be better" of finished
seasons, and liked the shallow window by a margin too small to ship on. These
tests pin the parts that make the prospective answer trustworthy: the challenger
is never published, it is scored on exactly the player-weeks the live model was
scored on, and a challenger that falls over does not take the real projection
with it.
"""

import json
import sys
import types
from datetime import datetime

import pandas as pd
import pytest

from database.models import (AnalyticsJobStatus, PlayerProjection,
                             ProjectionAccuracy, ShadowProjection)


@pytest.fixture(autouse=True)
def _clean(db_session):
    db_session.query(ShadowProjection).delete()
    db_session.query(ProjectionAccuracy).delete()
    db_session.query(PlayerProjection).delete()
    db_session.query(AnalyticsJobStatus).delete()
    db_session.commit()
    yield


def _frame():
    return pd.DataFrame({
        "player_id": ["00-0000001", "00-0000002"],
        "player_name": ["RB One", "WR Two"],
        "position": ["RB", "WR"],
        "team": ["DAL", "PHI"],
        "fanduel_fantasy_points": [15.0, 12.0],
        "floor": [6.0, 4.0], "ceiling": [25.0, 20.0],
        "prediction_type": ["veteran_ml"] * 2,
    })


@pytest.fixture
def fake_package(monkeypatch):
    """A stand-in nfl_projections that records how the service was built."""
    built = []

    class FakeService:
        def __init__(self, **kw):
            built.append(kw)

        def project(self, season, week, as_frame=True, **kw):
            return _frame()

    mod = types.ModuleType("nfl_projections")
    mod.__version__ = "9.9.9"
    mod.ProjectionService = FakeService
    mod.built = built
    monkeypatch.setitem(sys.modules, "nfl_projections", mod)
    return mod


class TestShadowRun:
    def _call(self, db, **kw):
        from scripts import compute_projections as cp

        args = dict(min_season=2025, seeds=5, epochs=60, proj_kwargs={},
                    env_all=None, cache_dir=None, as_of=4, started=())
        args.update(kw)
        return cp._run_shadow(db, pd.DataFrame({"season": [2026]}), 2026, 5, "1.5.0", **args)

    def test_records_rows_under_a_named_variant(self, db_session, fake_package):
        assert self._call(db_session) == 2

        rows = db_session.query(ShadowProjection).all()
        assert {r.variant for r in rows} == {"min_season_2025"}
        assert json.loads(rows[0].config_json)["min_season"] == 2025
        assert rows[0].model_version == "1.5.0"

    def test_trains_on_the_shallow_window(self, db_session, fake_package):
        self._call(db_session)
        (built,) = fake_package.built
        assert built["min_season"] == 2025
        # Quantiles and the simulator are skipped: the comparison is of the
        # point projection, and they are most of a run's remaining time.
        assert built["quantiles"] is False

    def test_publishes_nothing(self, db_session, fake_package):
        self._call(db_session)
        assert db_session.query(PlayerProjection).count() == 0

    def test_teams_already_playing_are_left_out(self, db_session, fake_package):
        assert self._call(db_session, started={"PHI"}) == 1
        assert {r.team for r in db_session.query(ShadowProjection).all()} == {"DAL"}

    def test_a_rerun_replaces_its_own_rows(self, db_session, fake_package):
        self._call(db_session)
        self._call(db_session)
        assert db_session.query(ShadowProjection).count() == 2

    def test_a_broken_challenger_does_not_raise(self, db_session, monkeypatch):
        mod = types.ModuleType("nfl_projections")
        mod.__version__ = "9.9.9"

        class Boom:
            def __init__(self, **kw):
                raise RuntimeError("out of memory")

        mod.ProjectionService = Boom
        monkeypatch.setitem(sys.modules, "nfl_projections", mod)

        assert self._call(db_session) == 0        # logged, not raised
        assert db_session.query(ShadowProjection).count() == 0


class TestCacheIsPerWindow:
    def test_the_window_is_part_of_the_key(self):
        from scripts import compute_projections as cp

        live = cp._cache_key(2026, 4, 5, 60, min_season=2020)
        shadow = cp._cache_key(2026, 4, 5, 60, min_season=2025)
        assert live != shadow, "a shadow run would have reused the live model"


class TestShadowAccuracy:
    """Graded on the live model's own scored rows, so both see one population."""

    @staticmethod
    def _scored(db, week, player_id, live_proj, actual):
        db.add(ProjectionAccuracy(
            season=2026, week=week, player_id=player_id, player_name=player_id,
            position="RB", team="DAL", projected_points=live_proj,
            actual_points=actual, error=live_proj - actual,
            abs_error=abs(live_proj - actual), scored_at=datetime.utcnow()))

    @staticmethod
    def _shadow(db, week, player_id, proj, variant="min_season_2025"):
        db.add(ShadowProjection(
            season=2026, week=week, player_id=player_id, variant=variant,
            player_name=player_id, position="RB", team="DAL",
            projected_points=proj, model_version="1.5.0",
            config_json=json.dumps({"min_season": 2025}),
            computed_at=datetime.utcnow()))

    def test_challenger_closer_reads_negative(self, client, db_session):
        # Actual 14: live missed by 4, challenger by 1.
        self._scored(db_session, 5, "00-1", live_proj=10.0, actual=14.0)
        self._shadow(db_session, 5, "00-1", proj=13.0)
        db_session.commit()

        body = client.get("/projections/shadow/accuracy?season=2026").json()
        (entry,) = body["data"]
        assert entry["variant"] == "min_season_2025"
        assert entry["overall"] == {"n": 1, "shadow_mae": 1.0, "live_mae": 4.0,
                                    "delta": -3.0, "shadow_bias": -1.0, "live_bias": -4.0}

    def test_only_player_weeks_the_live_model_was_scored_on_count(self, client, db_session):
        self._scored(db_session, 5, "00-1", live_proj=10.0, actual=14.0)
        self._shadow(db_session, 5, "00-1", proj=13.0)
        # A player who did not play is not in projection_accuracy, so the
        # challenger does not get credit for him either.
        self._shadow(db_session, 5, "00-2", proj=99.0)
        db_session.commit()

        body = client.get("/projections/shadow/accuracy?season=2026").json()
        assert body["data"][0]["overall"]["n"] == 1

    def test_reports_each_week(self, client, db_session):
        self._scored(db_session, 5, "00-1", live_proj=10.0, actual=14.0)
        self._shadow(db_session, 5, "00-1", proj=13.0)
        self._scored(db_session, 6, "00-1", live_proj=12.0, actual=12.0)
        self._shadow(db_session, 6, "00-1", proj=8.0)
        db_session.commit()

        entry = client.get("/projections/shadow/accuracy?season=2026").json()["data"][0]
        assert entry["weeks_scored"] == [5, 6]
        assert entry["by_week"]["5"]["delta"] == -3.0
        assert entry["by_week"]["6"]["delta"] == +4.0
        assert entry["overall"]["n"] == 2

    def test_several_challengers_at_once(self, client, db_session):
        self._scored(db_session, 5, "00-1", live_proj=10.0, actual=14.0)
        self._shadow(db_session, 5, "00-1", proj=13.0, variant="min_season_2025")
        self._shadow(db_session, 5, "00-1", proj=11.0, variant="min_season_2024")
        db_session.commit()

        body = client.get("/projections/shadow/accuracy?season=2026").json()
        assert {e["variant"] for e in body["data"]} == {"min_season_2025", "min_season_2024"}
        one = client.get("/projections/shadow/accuracy?season=2026"
                         "&variant=min_season_2024").json()
        assert len(one["data"]) == 1

    def test_nothing_scored_yet_says_so(self, client, db_session):
        self._shadow(db_session, 5, "00-1", proj=13.0)
        db_session.commit()
        assert client.get("/projections/shadow/accuracy?season=2026").json()["status"] == (
            "no_actuals")

    def test_no_shadow_rows_says_so(self, client, db_session):
        self._scored(db_session, 5, "00-1", live_proj=10.0, actual=14.0)
        db_session.commit()
        assert client.get("/projections/shadow/accuracy?season=2026").json()["status"] == (
            "no_data")
