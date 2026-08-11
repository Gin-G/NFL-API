#!/usr/bin/env python3
"""
Tests for /projections endpoints (DB-first, read-only).
"""

from datetime import datetime

import pytest

from database.models import AnalyticsJobStatus, PlayerProjection


@pytest.fixture(autouse=True)
def _clean_projection_tables(db_session):
    """The test engine is session-scoped and committed rows persist across
    tests, so clear these tables before each test to avoid PK collisions."""
    db_session.query(PlayerProjection).delete()
    db_session.query(AnalyticsJobStatus).delete()
    db_session.commit()
    yield


def _add(db, **kw):
    defaults = dict(
        season=2025, week=3, player_id="00-0000001", player_name="Test RB",
        position="RB", team="DAL", projected_points=15.0, floor=6.0,
        median=14.0, ceiling=25.0, prediction_type="veteran_ml",
        model_version="1.2.1", computed_at=datetime.utcnow(),
    )
    defaults.update(kw)
    db.add(PlayerProjection(**defaults))
    db.commit()


class TestGetProjections:
    def test_no_data_message(self, client):
        body = client.get("/projections/?season=2025&week=3").json()
        assert body["status"] == "no_data"
        assert body["data"] == []

    def test_returns_projections_sorted(self, client, db_session):
        _add(db_session, player_id="p1", player_name="Star RB", projected_points=20.0)
        _add(db_session, player_id="p2", player_name="Mid WR", position="WR",
             projected_points=12.0)
        body = client.get("/projections/?season=2025&week=3").json()
        assert body["status"] == "success"
        assert body["count"] == 2
        assert body["data"][0]["player_name"] == "Star RB"  # highest projection first
        assert {"floor", "median", "ceiling"} <= set(body["data"][0].keys())

    def test_position_filter(self, client, db_session):
        _add(db_session, player_id="p1", position="RB")
        _add(db_session, player_id="p2", position="WR")
        body = client.get("/projections/?season=2025&week=3&position=wr").json()
        assert body["count"] == 1
        assert body["data"][0]["position"] == "WR"

    def test_week_filter(self, client, db_session):
        _add(db_session, player_id="p1", week=3)
        _add(db_session, player_id="p1", week=4)
        body = client.get("/projections/?season=2025&week=4").json()
        assert body["count"] == 1
        assert body["data"][0]["week"] == 4


class TestPlayerProjections:
    def test_player_history(self, client, db_session):
        _add(db_session, player_id="pX", week=3)
        _add(db_session, player_id="pX", week=4)
        body = client.get("/projections/player/pX").json()
        assert body["status"] == "success"
        assert len(body["data"]) == 2

    def test_player_no_data(self, client):
        body = client.get("/projections/player/nobody").json()
        assert body["status"] == "no_data"


class TestProjectionsStatus:
    def test_no_job(self, client):
        assert client.get("/projections/status").json()["status"] == "no_job"

    def test_reports_job(self, client, db_session):
        db_session.add(AnalyticsJobStatus(
            job_type="projections", status="completed",
            total_entries=100, processed_entries=100,
            started_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        ))
        db_session.commit()
        body = client.get("/projections/status").json()
        assert body["status"] == "completed"
        assert body["pct_complete"] == 100.0


class TestSeasonTotalsAvailability:
    """/projections/season/{season} must report EXPECTED GAMES PLAYED, not the
    number of weeks on the schedule. Availability is already multiplied into
    projected_points, so dividing by the week count understated the per-game
    rate for anyone projected to miss time."""

    def _weeks(self, db, player_id="p1", n=17, points=10.0, exp_games=None,
               position="QB"):
        for wk in range(1, n + 1):
            _add(db, player_id=player_id, week=wk, position=position,
                 projected_points=points, exp_games=exp_games,
                 player_name=f"Player {player_id}")

    def test_durable_player_reads_full_schedule(self, client, db_session):
        self._weeks(db_session, exp_games=1.0)
        row = client.get("/projections/season/2025").json()["data"][0]
        assert row["games"] == pytest.approx(17.0)
        assert row["scheduled_weeks"] == 17
        assert row["ppg"] == pytest.approx(10.0)

    def test_injury_prone_player_reads_fewer_games_and_a_higher_rate(
            self, client, db_session):
        # 17 weeks at a 0.7 availability weight: the points are already
        # discounted, so the on-field rate is points / 11.9, not points / 17
        self._weeks(db_session, points=7.0, exp_games=0.7)
        row = client.get("/projections/season/2025").json()["data"][0]
        assert row["games"] == pytest.approx(11.9)
        assert row["scheduled_weeks"] == 17
        assert row["total_points"] == pytest.approx(119.0)
        assert row["ppg"] == pytest.approx(10.0)   # not 7.0

    def test_rows_written_before_the_column_fall_back_to_week_count(
            self, client, db_session):
        self._weeks(db_session, exp_games=None)
        row = client.get("/projections/season/2025").json()["data"][0]
        assert row["games"] == pytest.approx(17.0)
        assert row["ppg"] == pytest.approx(10.0)

    def test_bye_week_does_not_inflate_expected_games(self, client, db_session):
        self._weeks(db_session, n=16, exp_games=1.0)   # a week is missing
        row = client.get("/projections/season/2025").json()["data"][0]
        assert row["games"] == pytest.approx(16.0)
        assert row["scheduled_weeks"] == 16
