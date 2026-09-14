#!/usr/bin/env python3
"""Tests for the projection vintage archive.

`player_projections` answers "what do we think about week 10"; the vintage
archive answers "what did we think about week 10 in August, and did the extra
football help". The tests that matter here are the ones that prove the second
table is not quietly the first: that reprojecting a week REPLACES the current
row while ADDING a vintage, and that nothing already archived moves.
"""

from datetime import datetime

import pytest

from database.models import (PlayerProjection, PlayerProjectionVintage,
                             PlayerStat)


@pytest.fixture(autouse=True)
def _clean(db_session):
    db_session.query(PlayerProjection).delete()
    db_session.query(PlayerProjectionVintage).delete()
    db_session.query(PlayerStat).delete()
    db_session.commit()
    yield


def _vintage(db, **kw):
    defaults = dict(
        season=2025, week=10, player_id="00-0000001", as_of_week=0,
        player_name="Test RB", position="RB", team="DAL",
        projected_points=15.0, floor=6.0, median=14.0, ceiling=25.0,
        rushing_yards=70.0, prediction_type="veteran_ml",
        model_version="1.2.1", computed_at=datetime.utcnow(),
    )
    defaults.update(kw)
    db.add(PlayerProjectionVintage(**defaults))
    db.commit()


class TestVintageKey:
    def test_same_week_holds_multiple_vintages(self, db_session):
        """The whole point: week 10 keeps its August row when week 9 reprojects it."""
        _vintage(db_session, as_of_week=0, projected_points=15.0)
        _vintage(db_session, as_of_week=9, projected_points=21.0)
        rows = db_session.query(PlayerProjectionVintage).filter(
            PlayerProjectionVintage.week == 10).all()
        assert len(rows) == 2
        assert sorted(r.projected_points for r in rows) == [15.0, 21.0]

    def test_same_vintage_merges_rather_than_duplicating(self, db_session):
        """Re-running one vintage corrects it in place; it does not stack."""
        _vintage(db_session, as_of_week=3, projected_points=15.0)
        db_session.merge(PlayerProjectionVintage(
            season=2025, week=10, player_id="00-0000001", as_of_week=3,
            projected_points=17.0))
        db_session.commit()
        rows = db_session.query(PlayerProjectionVintage).filter(
            PlayerProjectionVintage.as_of_week == 3).all()
        assert len(rows) == 1
        assert rows[0].projected_points == 17.0


class TestAsOfQuery:
    def test_as_of_reads_the_archive_not_the_live_table(self, client, db_session):
        """`as_of` must not silently fall through to the current projection."""
        db_session.add(PlayerProjection(
            season=2025, week=10, player_id="00-0000001", player_name="Test RB",
            position="RB", team="DAL", projected_points=99.0))
        _vintage(db_session, as_of_week=0, projected_points=15.0)
        db_session.commit()

        live = client.get("/projections/?season=2025&week=10").json()
        assert live["data"][0]["projected_points"] == 99.0
        assert live["as_of_week"] is None

        old = client.get("/projections/?season=2025&week=10&as_of=0").json()
        assert old["as_of_week"] == 0
        assert old["data"][0]["projected_points"] == 15.0

    def test_unknown_vintage_is_no_data(self, client, db_session):
        _vintage(db_session, as_of_week=0)
        body = client.get("/projections/?season=2025&week=10&as_of=7").json()
        assert body["status"] == "no_data"


class TestSeasonAsOf:
    def test_season_total_uses_latest_vintage_at_or_before(self, client, db_session):
        """Weeks already played are never reprojected, so an `as_of=5` season
        outlook has to carry week 2's last-known row forward rather than
        dropping the week entirely."""
        # Week 2 was last projected before the season; week 10 got refreshed.
        _vintage(db_session, week=2, as_of_week=0, projected_points=10.0)
        _vintage(db_session, week=10, as_of_week=0, projected_points=10.0)
        _vintage(db_session, week=10, as_of_week=5, projected_points=30.0)

        body = client.get("/projections/season/2025?as_of=5").json()
        assert body["status"] == "success"
        row = body["data"][0]
        # week 2 @ as_of 0 (10) + week 10 @ as_of 5 (30) — NOT week 10 @ 0.
        assert row["total_points"] == pytest.approx(40.0)

    def test_season_total_ignores_vintages_from_the_future(self, client, db_session):
        _vintage(db_session, week=10, as_of_week=0, projected_points=10.0)
        _vintage(db_session, week=10, as_of_week=9, projected_points=99.0)
        body = client.get("/projections/season/2025?as_of=5").json()
        assert body["data"][0]["total_points"] == pytest.approx(10.0)


class TestTrajectory:
    def test_groups_by_week_oldest_first(self, client, db_session):
        _vintage(db_session, week=10, as_of_week=9, projected_points=21.0)
        _vintage(db_session, week=10, as_of_week=0, projected_points=15.0)
        body = client.get("/projections/vintages/00-0000001?season=2025").json()
        assert body["status"] == "success"
        assert [v["as_of_week"] for v in body["data"]["10"]] == [0, 9]

    def test_accuracy_route_is_not_read_as_a_player_id(self, client):
        """/vintages/accuracy is declared first; if that ordering is ever lost
        this returns a player trajectory for a player named 'accuracy'."""
        body = client.get("/projections/vintages/accuracy?season=2025").json()
        assert "lead_weeks" not in str(body.get("player_id", ""))
        assert body["status"] in ("no_data", "no_actuals", "no_overlap", "success")
        assert "player_id" not in body


class TestVintageAccuracy:
    def test_buckets_by_basis_and_lead(self, client, db_session):
        db_session.add(PlayerStat(
            season=2025, week=10, player_id="00-0000001", season_type="REG",
            rushing_yards=100.0, rushing_tds=1.0))
        db_session.commit()
        # Preseason (lead 10) and an in-season extrapolation (lead 5).
        _vintage(db_session, week=10, as_of_week=0, projected_points=15.0)
        _vintage(db_session, week=10, as_of_week=5, projected_points=17.0)

        body = client.get("/projections/vintages/accuracy?season=2025").json()
        assert body["status"] == "success"
        got = {(r["basis"], r["lead_weeks"]) for r in body["data"]}
        assert ("preseason", 10) in got
        assert ("extrapolated", 5) in got

    def test_skips_weeks_with_no_actuals(self, client, db_session):
        _vintage(db_session, week=10, as_of_week=0, projected_points=15.0)
        body = client.get("/projections/vintages/accuracy?season=2025").json()
        assert body["status"] in ("no_actuals", "no_overlap")
        assert body["data"] == []


class TestComputeRunWritesBothTables:
    """Execute `run()` end-to-end against stubs.

    The API tests above all pass if the compute script never writes a vintage at
    all — they only prove the read path works once rows exist. This is the test
    that fails if `_write_week` forgets the archive, if the in-season path still
    writes one week, or if `as_of_week` is stamped wrong. It runs the real
    function; only the model and the network are replaced.
    """

    @pytest.fixture
    def stub_nfl_projections(self, monkeypatch):
        import sys
        import types

        import pandas as pd

        base = pd.DataFrame({
            "player_id": ["00-0000001", "00-0000002"],
            "player_name": ["RB One", "WR Two"],
            "position": ["RB", "WR"],
            "team": ["DAL", "PHI"],
            "fanduel_fantasy_points": [15.0, 12.0],
            "floor": [6.0, 4.0], "projection_median": [14.0, 11.0],
            "ceiling": [25.0, 20.0],
            "rushing_yards": [70.0, 0.0], "receiving_yards": [10.0, 80.0],
            "receptions": [2.0, 6.0], "prediction_type": ["veteran_ml"] * 2,
        })

        calls = []

        class FakeService:
            def __init__(self, **kw):
                pass

            def project(self, season, week, as_frame=True, **kw):
                calls.append(kw)
                return base.copy()

        mod = types.ModuleType("nfl_projections")
        mod.__version__ = "9.9.9"
        mod.ProjectionService = FakeService
        mod.calls = calls           # every project() call's keyword arguments
        ds = types.ModuleType("nfl_projections.dataset")
        # The dataset must REACH the target season, or `use_espn` flips true and
        # run() takes the preseason ESPN-roster path instead of the in-season one.
        ds.build_dataset = lambda output_path=None: pd.DataFrame(
            {"season": [2024, 2025], "week": [1, 2]})
        mod.dataset = ds
        monkeypatch.setitem(sys.modules, "nfl_projections", mod)
        monkeypatch.setitem(sys.modules, "nfl_projections.dataset", ds)
        return base

    def test_projects_remaining_weeks_into_both_tables(
            self, db_session, stub_nfl_projections, monkeypatch):
        from database.models import AnalyticsJobStatus
        from scripts import compute_projections as cp

        # Week 5 is next, four weeks are in the books, season ends at week 7.
        monkeypatch.setattr(cp, "_completed_weeks", lambda season, when=None: 4)
        monkeypatch.setattr(cp, "_final_week", lambda season: 7)

        job = AnalyticsJobStatus(job_type="projections", status="pending")
        db_session.add(job)
        db_session.commit()

        cp.run(db_session, 2025, 5, epochs=1, job=job)

        live = db_session.query(PlayerProjection).filter(
            PlayerProjection.season == 2025).all()
        arch = db_session.query(PlayerProjectionVintage).filter(
            PlayerProjectionVintage.season == 2025).all()

        # The archive gets weeks 5, 6 and 7 — the whole remaining season.
        assert sorted({r.week for r in arch}) == [5, 6, 7]
        assert len(arch) == 6                  # two players x three weeks
        # But only the upcoming week is PUBLISHED. Weeks 6-7 are built without
        # the availability and share models, so publishing them would strip the
        # availability weighting out of the season-totals endpoint.
        assert sorted({r.week for r in live}) == [5]
        assert len(live) == 2
        # Every archived row is stamped with the football behind it.
        assert {r.as_of_week for r in arch} == {4}

    def test_reprojection_adds_a_vintage_without_destroying_the_old_one(
            self, db_session, stub_nfl_projections, monkeypatch):
        from database.models import AnalyticsJobStatus
        from scripts import compute_projections as cp

        monkeypatch.setattr(cp, "_final_week", lambda season: 6)
        job = AnalyticsJobStatus(job_type="projections", status="pending")
        db_session.add(job)
        db_session.commit()

        # Preseason run, then a run with four weeks played.
        monkeypatch.setattr(cp, "_completed_weeks", lambda season, when=None: 0)
        cp.run(db_session, 2025, 5, epochs=1, job=job)
        monkeypatch.setattr(cp, "_completed_weeks", lambda season, when=None: 4)
        cp.run(db_session, 2025, 5, epochs=1, job=job)

        wk6 = db_session.query(PlayerProjectionVintage).filter(
            PlayerProjectionVintage.season == 2025,
            PlayerProjectionVintage.week == 6,
            PlayerProjectionVintage.player_id == "00-0000001").all()
        assert sorted(v.as_of_week for v in wk6) == [0, 4]
        # Week 6 was never the upcoming week in either run, so nothing was
        # published for it — the archive grew, the live table did not.
        live = db_session.query(PlayerProjection).filter(
            PlayerProjection.season == 2025, PlayerProjection.week == 6).all()
        assert live == []

    def test_backfill_archives_preexisting_rows_before_they_are_overwritten(
            self, db_session, stub_nfl_projections, monkeypatch):
        from database.models import AnalyticsJobStatus
        from scripts import compute_projections as cp

        # An August season-long row that predates the archive entirely.
        db_session.add(PlayerProjection(
            season=2025, week=6, player_id="00-0000001", player_name="RB One",
            position="RB", team="DAL", projected_points=11.0,
            computed_at=datetime(2025, 8, 11, 3, 0)))
        db_session.commit()

        # Faithful stub: the backfill dates each existing row from its own
        # computed_at, and a stub that ignores `when` would file the August row
        # under the CURRENT vintage, where the fresh write then lands on top of
        # it — hiding exactly the data loss this test exists to catch.
        monkeypatch.setattr(cp, "_completed_weeks",
                            lambda season, when=None: 0 if when is not None
                            and when < datetime(2025, 9, 1) else 4)
        monkeypatch.setattr(cp, "_final_week", lambda season: 6)
        job = AnalyticsJobStatus(job_type="projections", status="pending")
        db_session.add(job)
        db_session.commit()

        cp.run(db_session, 2025, 5, epochs=1, job=job)

        wk6 = db_session.query(PlayerProjectionVintage).filter(
            PlayerProjectionVintage.season == 2025,
            PlayerProjectionVintage.week == 6,
            PlayerProjectionVintage.player_id == "00-0000001").all()
        by_asof = {v.as_of_week: v.projected_points for v in wk6}
        # The August projection was captured as its own vintage before the run
        # touched anything.
        assert by_asof[0] == 11.0, "the pre-existing August row was lost"
        assert by_asof[4] == 15.0, "the fresh run should be its own vintage"
        # And because week 6 is not the upcoming week, the August row is still
        # the one being SERVED — this job only added a record.
        still_live = db_session.query(PlayerProjection).filter(
            PlayerProjection.season == 2025, PlayerProjection.week == 6,
            PlayerProjection.player_id == "00-0000001").one()
        assert still_live.projected_points == 11.0

    def test_in_season_run_keeps_the_rookie_prior_and_injuries(
            self, db_session, stub_nfl_projections, monkeypatch):
        import sys

        from database.models import AnalyticsJobStatus
        from scripts import compute_projections as cp

        monkeypatch.setattr(cp, "_completed_weeks", lambda season, when=None: 1)
        monkeypatch.setattr(cp, "_final_week", lambda season: 2)
        job = AnalyticsJobStatus(job_type="projections", status="pending")
        db_session.add(job)
        db_session.commit()

        cp.run(db_session, 2025, 2, epochs=1, job=job)

        (kw,) = sys.modules["nfl_projections"].calls
        # Rookies have one game at most in week 2; the network needs two, so
        # without the prior every first-year player vanishes from the board.
        assert kw.get("rookie_fallback") is True
        # In season the injury report is real information: nothing turns it off,
        # and nflverse rosters, not the preseason ESPN frames, are used.
        assert kw.get("use_injuries", True) is True
        assert "rosters" not in kw and "depth_charts" not in kw

    def test_upcoming_week_gets_its_environment_and_simulator(
            self, db_session, stub_nfl_projections, monkeypatch):
        import pandas as pd

        from database.models import AnalyticsJobStatus
        from scripts import compute_projections as cp

        env = pd.DataFrame({"week": [5, 5, 6, 6], "team": ["DAL", "PHI", "DAL", "PHI"],
                            "env_mult": [1.10, 0.90, 1.0, 1.0]})
        monkeypatch.setattr(cp, "_game_environments", lambda season: env)
        simulated = []
        monkeypatch.setattr(cp, "_apply_simulator",
                            lambda frame, history, week: simulated.append(week))
        monkeypatch.setattr(cp, "_completed_weeks", lambda season, when=None: 4)
        monkeypatch.setattr(cp, "_final_week", lambda season: 6)
        job = AnalyticsJobStatus(job_type="projections", status="pending")
        db_session.add(job)
        db_session.commit()

        cp.run(db_session, 2025, 5, epochs=1, job=job)

        live = {r.player_id: r.projected_points for r in db_session.query(
            PlayerProjection).filter(PlayerProjection.season == 2025,
                                     PlayerProjection.week == 5)}
        # The published week used to go out matchup-neutral: 15.0 and 12.0.
        assert live["00-0000001"] == pytest.approx(16.5)
        assert live["00-0000002"] == pytest.approx(10.8)
        assert simulated == [5, 6]
