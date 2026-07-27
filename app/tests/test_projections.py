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
