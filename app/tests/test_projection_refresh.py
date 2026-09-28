#!/usr/bin/env python3
"""Tests for POST /projections/refresh — the dashboard's recompute button.

The endpoint is open (no auth), so the rate limit IS the protection: one run at
a time, and a quiet period after each. These tests pin that behaviour, and that
a crashed run cannot wedge the button shut forever.
"""

from datetime import datetime, timedelta

import pytest

from database.models import AnalyticsJobStatus, PlayerProjection


@pytest.fixture(autouse=True)
def _clean(db_session):
    db_session.query(PlayerProjection).delete()
    db_session.query(AnalyticsJobStatus).delete()
    db_session.commit()
    yield


@pytest.fixture(autouse=True)
def no_cluster(monkeypatch):
    """No Kubernetes in tests: nothing is running unless a test says so."""
    from api import k8s_jobs

    monkeypatch.setattr(k8s_jobs, "active_projection_jobs", lambda *a, **k: [])


@pytest.fixture
def started(monkeypatch):
    """Record job starts instead of talking to Kubernetes."""
    from api import k8s_jobs

    calls = []
    monkeypatch.setattr(k8s_jobs, "start_projection_job",
                        lambda *a, **k: calls.append(1) or "nfl-api-projections-manual-1")
    return calls


def _job(db, status="completed", minutes_ago=0):
    when = datetime.utcnow() - timedelta(minutes=minutes_ago)
    db.add(AnalyticsJobStatus(job_type="projections", status=status,
                              started_at=when, updated_at=when,
                              total_entries=17, processed_entries=17))
    db.commit()


class TestRefreshState:
    def test_first_ever_refresh_is_allowed(self, client):
        body = client.get("/projections/refresh").json()
        assert body["can_refresh"] is True
        assert body["running"] is False

    def test_recent_run_is_in_cooldown(self, client, db_session):
        _job(db_session, minutes_ago=5)
        body = client.get("/projections/refresh").json()
        assert body["can_refresh"] is False
        assert 0 < body["retry_after_seconds"] <= 15 * 60
        assert "minutes" in body["reason"]

    def test_old_run_is_refreshable(self, client, db_session):
        _job(db_session, minutes_ago=90)
        assert client.get("/projections/refresh").json()["can_refresh"] is True

    def test_live_run_blocks(self, client, db_session):
        _job(db_session, status="running", minutes_ago=3)
        body = client.get("/projections/refresh").json()
        assert (body["can_refresh"], body["running"]) == (False, True)

    def test_dead_run_does_not_block_forever(self, client, db_session):
        # A pod that was OOM-killed leaves "running" behind; without the
        # heartbeat timeout the button would never come back.
        _job(db_session, status="running", minutes_ago=60 * 5)
        assert client.get("/projections/refresh").json()["can_refresh"] is True

    def test_failed_run_still_serves_its_cooldown(self, client, db_session):
        # A failure that repeats instantly is a retry loop; make it wait.
        _job(db_session, status="failed", minutes_ago=1)
        assert client.get("/projections/refresh").json()["can_refresh"] is False


class TestRefreshTrigger:
    def test_starts_a_job(self, client, started):
        body = client.post("/projections/refresh").json()
        assert body["status"] == "started"
        assert body["job_name"].startswith("nfl-api-projections-manual")
        assert len(started) == 1

    def test_cooldown_answers_429_with_retry_after(self, client, db_session, started):
        _job(db_session, minutes_ago=2)
        resp = client.post("/projections/refresh")
        assert resp.status_code == 429
        assert int(resp.headers["Retry-After"]) > 0
        assert started == []

    def test_running_answers_409(self, client, db_session, started):
        _job(db_session, status="running", minutes_ago=1)
        assert client.post("/projections/refresh").status_code == 409
        assert started == []

    def test_kubernetes_failure_is_a_503(self, client, monkeypatch):
        from api import k8s_jobs

        def boom(*a, **k):
            raise k8s_jobs.JobStartError("not running in a cluster")

        monkeypatch.setattr(k8s_jobs, "start_projection_job", boom)
        resp = client.post("/projections/refresh")
        assert resp.status_code == 503
        assert "not running in a cluster" in resp.json()["detail"]


class TestStatusPercent:
    def test_percent_is_capped(self, client, db_session):
        # Older rows counted archived ROWS against a published-row total and
        # reported 1500% complete.
        db_session.add(AnalyticsJobStatus(
            job_type="projections", status="completed", started_at=datetime.utcnow(),
            updated_at=datetime.utcnow(), total_entries=592, processed_entries=8880))
        db_session.commit()
        assert client.get("/projections/status").json()["pct_complete"] == 100.0


class TestStartRace:
    """A started job writes its status row minutes later — the window that let a
    second press start a second run in production."""

    def test_the_placeholder_row_refuses_a_second_press(self, client, started):
        client.post("/projections/refresh")
        # The pod exists but has written nothing yet; only the placeholder knows.
        assert client.post("/projections/refresh").status_code == 409
        assert len(started) == 1

    def test_kubernetes_alone_refuses_a_second_press(
            self, client, db_session, started, monkeypatch):
        from api import k8s_jobs

        # Belt and braces: even with no placeholder — a cron run, say, which the
        # API never queued — an active Job is enough to refuse.
        monkeypatch.setattr(
            k8s_jobs, "active_projection_jobs",
            lambda *a, **k: [{"name": "nfl-api-projections-29836980", "ready": True}])
        assert client.post("/projections/refresh").status_code == 409
        assert started == []

    def test_the_placeholder_says_which_job_it_stands_for(self, client, db_session, started):
        client.post("/projections/refresh")
        row = db_session.query(AnalyticsJobStatus).order_by(
            AnalyticsJobStatus.id.desc()).first()
        assert row.status == "running"
        assert "nfl-api-projections-manual-1" in row.current_coach

    def test_a_job_with_nowhere_to_run_says_queued(self, client, monkeypatch, started):
        from api import k8s_jobs

        # 2026-09-27: a refresh sat Pending for 37 hours while the cluster had no
        # room, and the dashboard showed "Recomputing... 0%" the whole time.
        monkeypatch.setattr(
            k8s_jobs, "active_projection_jobs",
            lambda *a, **k: [{"name": "nfl-api-projections-manual-1", "ready": False}])
        body = client.get("/projections/refresh").json()
        assert (body["running"], body["queued"]) == (True, True)
        assert "waiting for room" in body["reason"]
