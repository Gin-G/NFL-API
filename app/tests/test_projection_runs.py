#!/usr/bin/env python3
"""
Tests for `projection_runs` — the record of what each board could see.

The thing being protected: an empty injury report and a healthy league produce
identical projections. The cron runs Tuesday and Wednesday nights, clubs file
their first injury report on Wednesday, so the Tuesday board is systematically
the blind one — and nothing in the rows says which kind you are reading.
"""

from datetime import datetime

import pytest

from database.models import PlayerProjection, ProjectionRun
from scripts import compute_projections


@pytest.fixture(autouse=True)
def _clean(db_session):
    db_session.query(ProjectionRun).delete()
    db_session.query(PlayerProjection).delete()
    db_session.commit()
    yield


def _run(db, **kw):
    defaults = dict(
        season=2026, week=9, as_of_week=8, computed_at=datetime(2026, 11, 3, 20, 0),
        job_id=343, model_version="1.6.0", model_reused=True,
        training_min_season=2026, seeds=5, epochs=100,
        injury_source="nflverse", injury_report_available=False,
        injury_report_rows=0, injury_report_latest_week=8, injury_report_players=0,
        ir_players=24, players_zeroed=24, backups_elevated=3, questionable=0,
        roster_source="nflverse", roster_rows=1744,
        depth_chart_as_of=datetime(2026, 11, 3, 14, 59),
        projections_published=500, weeks_archived=10, teams_frozen=None,
    )
    defaults.update(kw)
    row = ProjectionRun(**defaults)
    db.add(row)
    db.commit()
    return row


def _projection(db, **kw):
    defaults = dict(
        season=2026, week=9, player_id="00-0000001", player_name="Tyler Huntley",
        position="QB", team="BAL", projected_points=8.3, floor=3.0, median=8.0,
        ceiling=15.0, prediction_type="veteran_ml", model_version="1.6.0",
        computed_at=datetime.utcnow(),
    )
    defaults.update(kw)
    db.add(PlayerProjection(**defaults))
    db.commit()


class TestRunsEndpoint:
    def test_no_runs_recorded(self, client):
        body = client.get("/projections/runs?season=2026").json()
        assert body["status"] == "no_data"
        assert body["data"] == []

    def test_a_tuesday_run_says_it_was_blind(self, client, db_session):
        _run(db_session)
        body = client.get("/projections/runs?season=2026").json()
        assert body["count"] == 1
        run = body["data"][0]
        assert run["injuries"]["report_available"] is False
        assert run["injuries"]["report_latest_week"] == 8   # the feed is not empty
        assert run["injuries"]["ir_players"] == 24
        note = next(n for n in run["notes"] if "injury report" in n)
        assert "week 9" in note and "IR/PUP/Reserve" in note

    def test_a_wednesday_run_carries_no_warning(self, client, db_session):
        _run(db_session, injury_report_available=True, injury_report_rows=312,
             injury_report_latest_week=9, injury_report_players=180,
             model_reused=False)
        run = client.get("/projections/runs?season=2026").json()["data"][0]
        assert run["injuries"]["report_available"] is True
        assert not any("injury report" in n for n in run["notes"])

    def test_both_runs_of_a_week_are_kept(self, client, db_session):
        """Append-only: re-projecting a week records a second run, newest first."""
        _run(db_session)                                      # Tuesday
        _run(db_session, injury_report_available=True, injury_report_rows=312,
             computed_at=datetime(2026, 11, 4, 20, 0))        # Wednesday
        body = client.get("/projections/runs?season=2026&week=9").json()
        assert body["count"] == 2
        assert body["data"][0]["injuries"]["report_available"] is True
        assert body["data"][1]["injuries"]["report_available"] is False

    def test_week_filter(self, client, db_session):
        _run(db_session, week=9)
        _run(db_session, week=10, as_of_week=9)
        assert client.get("/projections/runs?season=2026&week=10"
                          ).json()["count"] == 1

    def test_a_reused_model_says_so(self, client, db_session):
        _run(db_session, model_reused=True)
        run = client.get("/projections/runs?season=2026").json()["data"][0]
        assert run["model"]["reused"] is True
        assert any("Reused the model" in n for n in run["notes"])

    def test_frozen_teams_are_named(self, client, db_session):
        _run(db_session, teams_frozen="GB,LA")
        run = client.get("/projections/runs?season=2026").json()["data"][0]
        assert any("GB,LA" in n for n in run["notes"])

    def test_injuries_disabled_is_its_own_note(self, client, db_session):
        _run(db_session, injury_source="disabled", ir_players=0, players_zeroed=0,
             model_reused=False)
        run = client.get("/projections/runs?season=2026").json()["data"][0]
        assert run["notes"] == ["Injuries were not applied to this run."]


class TestProvenanceOnTheBoard:
    def test_a_week_carries_the_run_that_made_it(self, client, db_session):
        _run(db_session)
        _projection(db_session)
        body = client.get("/projections/?season=2026&week=9").json()
        assert body["provenance"]["run_id"] is not None
        assert body["provenance"]["injuries"]["report_available"] is False
        assert body["provenance"]["inputs"]["depth_chart_as_of"].startswith("2026-11-03")

    def test_the_newest_run_for_the_week_wins(self, client, db_session):
        _run(db_session)
        _run(db_session, injury_report_available=True, injury_report_rows=312)
        _projection(db_session)
        body = client.get("/projections/?season=2026&week=9").json()
        assert body["provenance"]["injuries"]["report_available"] is True

    def test_a_whole_season_has_no_single_provenance(self, client, db_session):
        _run(db_session)
        _projection(db_session)
        body = client.get("/projections/?season=2026").json()
        assert body["provenance"] is None     # many runs; none of them is it

    def test_a_vintage_matches_on_as_of_not_week(self, client, db_session):
        """Week 12's archived row came from the run that had 8 weeks behind it,
        and that run published week 9."""
        from database.models import PlayerProjectionVintage

        db_session.query(PlayerProjectionVintage).delete()
        db_session.commit()
        _run(db_session, week=9, as_of_week=8)
        db_session.add(PlayerProjectionVintage(
            season=2026, week=12, player_id="p1", as_of_week=8, player_name="X",
            position="QB", team="BAL", projected_points=10.0))
        db_session.commit()
        body = client.get("/projections/?season=2026&week=12&as_of=8").json()
        assert body["count"] == 1
        assert body["provenance"]["week"] == 9        # the run that published
        assert body["provenance"]["as_of_week"] == 8

    def test_a_week_projected_before_runs_were_recorded(self, client, db_session):
        _projection(db_session)
        body = client.get("/projections/?season=2026&week=9").json()
        assert body["status"] == "success"
        assert body["provenance"] is None


class TestRecordRun:
    """The writer: what compute_projections stores at publish time."""

    PROVENANCE = {
        "season": 2026, "week": 9, "roster_rows": 1744,
        "depth_chart_as_of": "2026-11-03T14:59:00+00:00",
        "depth_chart_rows": 7008,
        "injuries": {
            "source": "nflverse", "season": 2026, "week": 9,
            "report_weeks": [5, 6, 7, 8], "report_latest_week": 8,
            "report_rows_for_week": 0, "report_available": False,
            "report_players": 0, "ir_players": 24, "out": 24,
            "backups_elevated": 3, "questionable": 0,
        },
    }

    def _record(self, db, provenance, **kw):
        job = type("Job", (), {"id": 343})()
        args = dict(job=job, season=2026, week=9, as_of=8, model_version="1.6.0",
                    model_reused=True, min_season=2026, seeds=5, epochs=100,
                    provenance=provenance, roster_source="nflverse",
                    published=500, weeks_archived=10, started=())
        args.update(kw)
        compute_projections._record_run(db, **args)
        return db.query(ProjectionRun).order_by(ProjectionRun.id.desc()).first()

    def test_an_empty_report_is_recorded_as_empty_not_healthy(self, db_session):
        row = self._record(db_session, self.PROVENANCE)
        assert row.injury_report_available is False
        assert row.injury_report_rows == 0
        assert row.injury_report_players == 0
        # 24 players out, none of them because anybody filed a report.
        assert row.ir_players == 24 and row.players_zeroed == 24
        assert row.injury_report_latest_week == 8

    def test_the_depth_chart_stamp_lands_as_naive_utc(self, db_session):
        row = self._record(db_session, self.PROVENANCE)
        assert row.depth_chart_as_of == datetime(2026, 11, 3, 14, 59)

    def test_frozen_teams_are_sorted_into_a_string(self, db_session):
        row = self._record(db_session, self.PROVENANCE, started={"LA", "GB"})
        assert row.teams_frozen == "GB,LA"

    def test_no_frozen_teams_is_null(self, db_session):
        assert self._record(db_session, self.PROVENANCE, started=()).teams_frozen is None

    def test_an_older_package_with_no_provenance_still_records_the_run(self, db_session):
        """`last_provenance` arrives as None from a package predating it. The
        run is a fact worth keeping even when its inputs cannot be described."""
        row = self._record(db_session, None)
        assert row.season == 2026 and row.week == 9 and row.job_id == 343
        assert row.injury_source is None
        assert row.ir_players == 0
        assert row.projections_published == 500

    def test_a_filed_report_is_recorded_as_filed(self, db_session):
        prov = {**self.PROVENANCE, "injuries": {
            **self.PROVENANCE["injuries"], "report_available": True,
            "report_rows_for_week": 312, "report_latest_week": 9,
            "report_players": 180, "out": 31}}
        row = self._record(db_session, prov)
        assert row.injury_report_available is True
        assert row.injury_report_players == 180
        assert row.players_zeroed == 31

    def test_a_write_failure_never_fails_the_job(self, db_session, monkeypatch):
        """A board that cannot write its own audit trail is still a board."""
        def boom(*a, **k):
            raise RuntimeError("table is gone")
        monkeypatch.setattr(db_session, "add", boom)
        compute_projections._record_run(
            db_session, job=None, season=2026, week=9, as_of=8,
            model_version="1.6.0", model_reused=False, min_season=2026, seeds=5,
            epochs=100, provenance=self.PROVENANCE, roster_source="nflverse",
            published=500, weeks_archived=10, started=())      # does not raise


class TestStampParsing:
    def test_a_z_suffix(self):
        assert compute_projections._as_naive_utc("2026-11-03T14:59:00Z") == \
            datetime(2026, 11, 3, 14, 59)

    def test_an_offset_is_converted_not_dropped(self):
        assert compute_projections._as_naive_utc("2026-11-03T09:59:00-05:00") == \
            datetime(2026, 11, 3, 14, 59)

    def test_naive_passes_through(self):
        assert compute_projections._as_naive_utc("2026-11-03T14:59:00") == \
            datetime(2026, 11, 3, 14, 59)

    def test_nonsense_is_none_not_an_exception(self):
        assert compute_projections._as_naive_utc("not a date") is None
        assert compute_projections._as_naive_utc(None) is None
        assert compute_projections._as_naive_utc("") is None
