"""Matchup signals frozen before kickoff, and the verdict read back."""
from datetime import datetime

import pytest

from database.models import MatchupSignal, PlayerProjection

pytest.importorskip("nfl_projections.signals",
                    reason="signals comes from nfl_projections")


@pytest.fixture(autouse=True)
def _clean(db_session):
    """Empty the two tables these tests write.

    The suite's engine is session-scoped and `db_session` only rolls back, so
    a committed row outlives its test. Most tests never notice; these assert
    on counts and on an empty record, so they have to start empty.
    """
    for model in (MatchupSignal, PlayerProjection):
        db_session.query(model).delete()
    db_session.commit()
    yield
    for model in (MatchupSignal, PlayerProjection):
        db_session.query(model).delete()
    db_session.commit()


def _board(db, season=2026, week=8):
    for i, (name, pos, team, points) in enumerate([
            ("Back A", "RB", "CIN", 14.0), ("Back B", "RB", "TEN", 11.0),
            ("Bench C", "RB", "CIN", 2.0), ("Receiver D", "WR", "TEN", 12.0)]):
        db.add(PlayerProjection(
            season=season, week=week, player_id=f"00-{i:04d}", player_name=name,
            position=pos, team=team, projected_points=points,
            prediction_type="veteran_ml", model_version="1.6.0",
            computed_at=datetime.utcnow()))
    db.commit()


def _signal(db, **kw):
    base = dict(season=2026, week=8, player_id="x", player_name="n", position="RB",
                team="A", opponent="B", projected=12.0, run_matchup=65.0,
                pass_matchup=50.0, opp_cover_grade=40.0, run_lean="lean up",
                rosterable=1, actual=None, recorded_at=datetime.utcnow())
    base.update(kw)
    db.add(MatchupSignal(**base))
    db.commit()


class TestRecord:
    def test_it_freezes_the_board_against_the_signals(self, db_session, monkeypatch):
        import scripts.record_signals as rs
        import pandas as pd

        monkeypatch.setattr(rs, "_trenches", lambda s, w: pd.DataFrame(
            {"team": ["CIN", "TEN"], "run_matchup": [66.1, 46.0],
             "pass_matchup": [47.6, 50.6], "opponent": ["TEN", "CIN"]}))
        monkeypatch.setattr(rs, "_coverage", lambda s, w: pd.DataFrame(
            {"team": ["CIN", "TEN"], "opp_cover_grade": [31.0, 55.0]}))
        _board(db_session)

        assert rs.record(db_session, 2026, 8) == 4
        rows = {r.player_name: r for r in db_session.query(MatchupSignal).all()}
        assert rows["Back A"].run_matchup == pytest.approx(66.1)
        assert rows["Back A"].run_lean == "lean up"
        assert rows["Back A"].opp_cover_grade == pytest.approx(31.0)
        assert rows["Bench C"].rosterable == 0       # projected 2.0

    def test_it_never_changes_a_projection(self, db_session, monkeypatch):
        import scripts.record_signals as rs
        import pandas as pd

        monkeypatch.setattr(rs, "_trenches", lambda s, w: pd.DataFrame())
        monkeypatch.setattr(rs, "_coverage", lambda s, w: pd.DataFrame())
        _board(db_session)
        rs.record(db_session, 2026, 8)
        stored = {r.player_name: r.projected for r in db_session.query(MatchupSignal)}
        assert stored["Back A"] == pytest.approx(14.0)

    def test_missing_signals_still_record_the_week(self, db_session, monkeypatch):
        # A signal that could not be computed is a fact about the week.
        import scripts.record_signals as rs
        import pandas as pd

        monkeypatch.setattr(rs, "_trenches", lambda s, w: pd.DataFrame())
        monkeypatch.setattr(rs, "_coverage", lambda s, w: pd.DataFrame())
        _board(db_session)
        assert rs.record(db_session, 2026, 8) == 4
        assert all(r.run_matchup is None for r in db_session.query(MatchupSignal))

    def test_no_board_records_nothing(self, db_session, monkeypatch):
        import scripts.record_signals as rs
        import pandas as pd

        monkeypatch.setattr(rs, "_trenches", lambda s, w: pd.DataFrame())
        monkeypatch.setattr(rs, "_coverage", lambda s, w: pd.DataFrame())
        assert rs.record(db_session, 2026, 8) == 0

    def test_a_rerun_refreshes_an_unscored_week(self, db_session, monkeypatch):
        import scripts.record_signals as rs
        import pandas as pd

        monkeypatch.setattr(rs, "_trenches", lambda s, w: pd.DataFrame())
        monkeypatch.setattr(rs, "_coverage", lambda s, w: pd.DataFrame())
        _board(db_session)
        rs.record(db_session, 2026, 8)
        assert rs.record(db_session, 2026, 8) == 4
        assert db_session.query(MatchupSignal).count() == 4

    def test_a_scored_week_is_left_alone(self, db_session, monkeypatch):
        # Once scored it is evidence, not a draft, and must not be rewritten.
        import scripts.record_signals as rs
        import pandas as pd

        monkeypatch.setattr(rs, "_trenches", lambda s, w: pd.DataFrame())
        monkeypatch.setattr(rs, "_coverage", lambda s, w: pd.DataFrame())
        _board(db_session)
        _signal(db_session, player_id="00-0000", projected=99.0,
                actual=5.0, scored_at=datetime.utcnow())
        rs.record(db_session, 2026, 8)
        kept = db_session.query(MatchupSignal).filter_by(player_id="00-0000").one()
        assert kept.projected == pytest.approx(99.0)


class TestVerdictEndpoint:
    def test_an_empty_record_says_so(self, client, db_session):
        body = client.get("/trenches/signals?season=2026").json()
        assert body["status"] == "no_data"
        assert "record_signals" in body["message"]

    def test_it_reports_the_weeks_that_pointed_the_right_way(self, client, db_session):
        for week, gap in ((4, 5.0), (5, 3.0), (6, -2.0)):
            _signal(db_session, week=week, player_id=f"up{week}", run_matchup=65.0,
                    projected=12.0, actual=12.0 + gap, scored_at=datetime.utcnow())
            _signal(db_session, week=week, player_id=f"dn{week}", run_matchup=45.0,
                    run_lean="neutral", projected=12.0, actual=12.0,
                    scored_at=datetime.utcnow())
        body = client.get("/trenches/signals?season=2026").json()
        assert body["status"] == "success"
        assert body["verdict"]["weeks"] == 3
        assert body["verdict"]["weeks_right"] == 2

    def test_per_week_detail_is_exposed(self, client, db_session):
        # So one big week cannot hide inside the pooled gap.
        _signal(db_session, week=5, player_id="u", projected=10.0, actual=24.0,
                run_matchup=65.0, scored_at=datetime.utcnow())
        _signal(db_session, week=5, player_id="d", projected=10.0, actual=10.0,
                run_matchup=45.0, run_lean="neutral", scored_at=datetime.utcnow())
        body = client.get("/trenches/signals?season=2026").json()
        assert body["verdict"]["per_week"][0]["gap"] == pytest.approx(14.0)

    def test_coverage_can_be_asked_about_too(self, client, db_session):
        _signal(db_session, position="WR", player_id="w", projected=10.0,
                actual=12.0, opp_cover_grade=30.0, scored_at=datetime.utcnow())
        body = client.get("/trenches/signals?season=2026"
                          "&position=WR&signal=opp_cover_grade&threshold=40").json()
        assert body["signal"] == "opp_cover_grade"
        assert body["status"] == "success"
