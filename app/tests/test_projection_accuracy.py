#!/usr/bin/env python3
"""
Tests for prospective projection scoring: scripts.score_projections and the
/projections/accuracy endpoints.
"""

from datetime import datetime, timedelta

import pytest

from database.models import (
    AnalyticsJobStatus,
    PlayerProjection,
    PlayerStat,
    ProjectionAccuracy,
    Schedule,
)
from scripts import score_projections as scorer


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in (ProjectionAccuracy, PlayerProjection, PlayerStat,
                  AnalyticsJobStatus, Schedule):
        db_session.query(model).delete()
    db_session.commit()
    yield


def _game(db, season=2025, week=3, gameday="2025-09-21", home="DAL", away="NYG"):
    db.add(Schedule(game_id=f"{season}_{week}_{away}_{home}", season=season, week=week,
                    game_type="REG", gameday=gameday, home_team=home, away_team=away))
    db.commit()


def _project(db, **kw):
    defaults = dict(
        season=2025, week=3, player_id="p1", player_name="Test RB", position="RB",
        team="DAL", projected_points=15.0, floor=6.0, median=14.0, ceiling=25.0,
        prediction_type="veteran_ml", model_version="1.3.0",
        computed_at=datetime(2025, 9, 17, 9, 0),
    )
    defaults.update(kw)
    db.add(PlayerProjection(**defaults))
    db.commit()


def _stat(db, **kw):
    """A stat line worth 20.0 FanDuel points by default:
    100 rushing yards (10) + 100-yard bonus (3) + 4 rec (2) + 50 rec yds (5)."""
    defaults = dict(
        season=2025, week=3, player_id="p1", player_name="Test RB",
        player_display_name="Test RB", position="RB", recent_team="DAL",
        season_type="REG", rushing_yards=100.0, rushing_tds=0.0, receptions=4.0,
        receiving_yards=50.0, receiving_tds=0.0, passing_yards=0.0, passing_tds=0.0,
        interceptions=0.0,
    )
    defaults.update(kw)
    db.add(PlayerStat(**defaults))
    db.commit()


class TestFanduelScoring:
    def test_matches_the_projection_packages_scale(self):
        from api.utils import fanduel_points

        # 100 rush yds + bonus, 4 rec, 50 rec yds
        assert fanduel_points({
            "rushing_yards": 100, "receptions": 4, "receiving_yards": 50,
        }) == pytest.approx(20.0)

    def test_fumbles_cost_two_each(self):
        from api.utils import fanduel_points

        assert fanduel_points({"rushing_yards": 50, "rushing_fumbles": 1}) == pytest.approx(3.0)

    def test_accepts_either_interception_column_name(self):
        from api.utils import fanduel_points

        assert fanduel_points({"interceptions": 2}) == pytest.approx(-2.0)
        assert fanduel_points({"passing_interceptions": 2}) == pytest.approx(-2.0)


class TestScoreWeek:
    def test_scores_a_player_against_actuals(self, db_session):
        _project(db_session, projected_points=15.0)
        _stat(db_session)

        summary = scorer.score_week(db_session, 2025, 3)
        assert summary["scored"] == 1

        row = db_session.query(ProjectionAccuracy).one()
        assert row.actual_points == pytest.approx(20.0)
        assert row.projected_points == pytest.approx(15.0)
        assert row.error == pytest.approx(-5.0)      # projected under
        assert row.abs_error == pytest.approx(5.0)
        assert row.in_band == 1                       # 6 <= 20 <= 25
        assert row.projected_at == datetime(2025, 9, 17, 9, 0)

    def test_out_of_band_actual_is_flagged(self, db_session):
        _project(db_session, floor=6.0, ceiling=18.0)
        _stat(db_session)   # 20.0 actual, above the ceiling

        scorer.score_week(db_session, 2025, 3)
        assert db_session.query(ProjectionAccuracy).one().in_band == 0

    def test_player_who_did_not_play_is_not_scored_as_zero(self, db_session):
        _project(db_session, player_id="played")
        _project(db_session, player_id="inactive")
        _stat(db_session, player_id="played")

        summary = scorer.score_week(db_session, 2025, 3)
        assert summary["scored"] == 1
        assert summary["projected_but_did_not_play"] == 1
        assert {r.player_id for r in db_session.query(ProjectionAccuracy).all()} == {"played"}

    def test_scored_rows_are_frozen_against_later_projection_runs(self, db_session):
        _project(db_session, projected_points=15.0)
        _stat(db_session)
        scorer.score_week(db_session, 2025, 3)

        # A later projections run overwrites player_projections for the same week
        db_session.query(PlayerProjection).delete()
        db_session.commit()
        _project(db_session, projected_points=20.0,
                 computed_at=datetime(2025, 9, 25, 9, 0))

        summary = scorer.score_week(db_session, 2025, 3)
        assert summary["scored"] == 0
        assert summary["already_scored"] == 1

        row = db_session.query(ProjectionAccuracy).one()
        assert row.projected_points == pytest.approx(15.0)   # history intact
        assert row.projected_at == datetime(2025, 9, 17, 9, 0)

    def test_rescore_overwrites_deliberately(self, db_session):
        _project(db_session, projected_points=15.0)
        _stat(db_session)
        scorer.score_week(db_session, 2025, 3)

        db_session.query(PlayerProjection).delete()
        db_session.commit()
        _project(db_session, projected_points=22.0)

        summary = scorer.score_week(db_session, 2025, 3, rescore=True)
        assert summary["scored"] == 1
        row = db_session.query(ProjectionAccuracy).one()
        assert row.projected_points == pytest.approx(22.0)
        assert row.error == pytest.approx(2.0)

    def test_naive_baseline_uses_the_trailing_five_games(self, db_session):
        _project(db_session, week=8)
        # six prior games; only the last five (weeks 3-7) should count
        for week, rush in [(2, 0.0), (3, 100.0), (4, 100.0), (5, 100.0),
                           (6, 100.0), (7, 100.0)]:
            _stat(db_session, week=week, rushing_yards=rush, receptions=0.0,
                  receiving_yards=0.0)
        _stat(db_session, week=8)   # the game being scored

        scorer.score_week(db_session, 2025, 8)
        row = db_session.query(ProjectionAccuracy).one()
        # weeks 3-7 are 13.0 each (100 yds + bonus); week 2's 0.0 is out of window
        assert row.naive_points == pytest.approx(13.0)
        assert row.naive_abs_error == pytest.approx(7.0)   # |13 - 20|

    def test_naive_falls_back_into_the_prior_season(self, db_session):
        _project(db_session, week=1)
        _stat(db_session, season=2024, week=17, rushing_yards=100.0,
              receptions=0.0, receiving_yards=0.0)
        _stat(db_session, week=1)

        scorer.score_week(db_session, 2025, 1)
        assert db_session.query(ProjectionAccuracy).one().naive_points == pytest.approx(13.0)

    def test_projection_computed_after_kickoff_is_left_out(self, db_session):
        _game(db_session, gameday="2025-09-21")
        _project(db_session, computed_at=datetime(2025, 9, 23, 9, 0))  # two days late
        _stat(db_session)

        summary = scorer.score_week(db_session, 2025, 3)
        assert summary["scored"] == 0
        assert summary["computed_after_kickoff"] == 1
        assert db_session.query(ProjectionAccuracy).count() == 0

    def test_projection_made_on_gameday_still_counts(self, db_session):
        _game(db_session, gameday="2025-09-21")
        _project(db_session, computed_at=datetime(2025, 9, 21, 9, 0))
        _stat(db_session)

        summary = scorer.score_week(db_session, 2025, 3)
        assert summary["scored"] == 1
        assert summary["computed_after_kickoff"] == 0

    def test_no_actuals_yet_scores_nothing(self, db_session):
        _project(db_session)
        summary = scorer.score_week(db_session, 2025, 3)
        assert summary["status"] == "no_actuals"
        assert db_session.query(ProjectionAccuracy).count() == 0

    def test_postseason_rows_are_ignored(self, db_session):
        _project(db_session)
        _stat(db_session, season_type="POST")
        assert scorer.score_week(db_session, 2025, 3)["status"] == "no_actuals"


class TestRun:
    def test_scores_every_week_with_both_sides(self, db_session):
        for week in (1, 2):
            _project(db_session, week=week)
            _stat(db_session, week=week)
        _project(db_session, week=3)          # projected, not played yet

        summaries = scorer.run(db_session, 2025)
        assert [s["week"] for s in summaries] == [1, 2]
        assert db_session.query(ProjectionAccuracy).count() == 2

        job = db_session.query(AnalyticsJobStatus).one()
        assert job.job_type == "projection_accuracy"
        assert job.status == "completed"
        assert job.processed_entries == 2


class TestAccuracyEndpoints:
    def _scored(self, db, **kw):
        defaults = dict(
            season=2025, week=3, player_id="p1", player_name="Test RB",
            position="RB", team="DAL", projected_points=15.0, floor=6.0,
            median=14.0, ceiling=25.0, actual_points=20.0, error=-5.0,
            abs_error=5.0, in_band=1, naive_points=13.0, naive_abs_error=7.0,
            prediction_type="veteran_ml", model_version="1.3.0",
            projected_at=datetime(2025, 9, 17, 9, 0), scored_at=datetime.utcnow(),
        )
        defaults.update(kw)
        db.add(ProjectionAccuracy(**defaults))
        db.commit()

    def test_no_data_message(self, client):
        body = client.get("/projections/accuracy?season=2025").json()
        assert body["status"] == "no_data"
        assert "score_projections" in body["message"]

    def test_overall_and_breakdowns(self, client, db_session):
        self._scored(db_session, player_id="p1", position="RB", abs_error=5.0, error=-5.0)
        self._scored(db_session, player_id="p2", position="WR", abs_error=3.0, error=3.0,
                     projected_points=12.0, actual_points=9.0, in_band=0,
                     naive_points=10.0, naive_abs_error=1.0)

        body = client.get("/projections/accuracy?season=2025").json()
        assert body["status"] == "success"
        assert body["overall"]["n"] == 2
        assert body["overall"]["mae"] == pytest.approx(4.0)
        assert body["overall"]["bias"] == pytest.approx(-1.0)
        assert body["overall"]["band_coverage"] == pytest.approx(0.5)
        assert body["overall"]["naive_mae"] == pytest.approx(4.0)
        assert body["overall"]["skill_over_naive"] == pytest.approx(0.0)
        assert set(body["by_position"]) == {"RB", "WR"}
        assert body["by_position"]["RB"]["mae"] == pytest.approx(5.0)
        assert body["weeks_scored"] == [3]

    def test_skill_over_naive_is_positive_when_the_model_wins(self, client, db_session):
        self._scored(db_session, player_id="p1", abs_error=2.0, naive_abs_error=6.0)
        body = client.get("/projections/accuracy?season=2025").json()
        assert body["overall"]["skill_over_naive"] == pytest.approx(4.0)

    def test_filters_by_week_and_position(self, client, db_session):
        self._scored(db_session, player_id="p1", week=3, position="RB")
        self._scored(db_session, player_id="p2", week=4, position="WR")

        assert client.get("/projections/accuracy?season=2025&week=4").json()["overall"]["n"] == 1
        by_pos = client.get("/projections/accuracy?season=2025&position=rb").json()
        assert by_pos["overall"]["n"] == 1
        assert set(by_pos["by_position"]) == {"RB"}

    def test_min_projected_filter(self, client, db_session):
        self._scored(db_session, player_id="p1", projected_points=15.0)
        self._scored(db_session, player_id="p2", projected_points=3.0)
        body = client.get("/projections/accuracy?season=2025&min_projected=10").json()
        assert body["overall"]["n"] == 1

    def test_misses_ranked_by_size_and_direction(self, client, db_session):
        self._scored(db_session, player_id="small", abs_error=2.0, error=2.0)
        self._scored(db_session, player_id="big_under", abs_error=18.0, error=-18.0)
        self._scored(db_session, player_id="big_over", abs_error=11.0, error=11.0)

        data = client.get("/projections/accuracy/misses?season=2025").json()["data"]
        assert [r["player_id"] for r in data] == ["big_under", "big_over", "small"]

        over = client.get("/projections/accuracy/misses?season=2025&direction=over").json()
        assert [r["player_id"] for r in over["data"]] == ["big_over", "small"]

    def test_status_before_and_after_a_job(self, client, db_session):
        assert client.get("/projections/accuracy/status").json()["status"] == "no_job"

        now = datetime.utcnow()
        db_session.add(AnalyticsJobStatus(
            job_type="projection_accuracy", status="completed",
            started_at=now - timedelta(minutes=1), updated_at=now,
            total_entries=2, processed_entries=350, skipped_entries=10,
            failed_entries=0, current_season=2025,
        ))
        db_session.commit()

        body = client.get("/projections/accuracy/status").json()
        assert body["status"] == "completed"
        assert body["player_weeks_scored"] == 350
        assert body["season"] == 2025
