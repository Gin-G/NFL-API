#!/usr/bin/env python3
"""Tests for the coverage report endpoints.

What these exercise that the research tests cannot: the receiver join. The
grade math is covered in nfl-data-py's `tests/test_coverage.py`; here the
question is whether a secondary's grade reaches the right offense's receivers
with the right advisory label.

PFR charting arrives over the network rather than from the DB, so it is
injected. Without that this would reach out to nflverse during a test run,
which is exactly the behaviour a test must not have.
"""
import pandas as pd
import pytest

from database.models import PlayerProjection, Schedule

pytest.importorskip("nfl_projections.coverage",
                    reason="coverage grades come from nfl_projections (installed --no-deps)")

SEASON, WEEK = 2024, 8
# Two secondaries. SOFT has a weak corner and a good one -- same average as
# EVEN, but an exploitable side. EVEN has two identical average corners.
CORNERS = {
    "beaten":  ("SOFT", 0.85),
    "sticky":  ("SOFT", 0.45),
    "even_a":  ("EVEN", 0.65),
    "even_b":  ("EVEN", 0.65),
    "benched": ("SOFT", 0.95),     # last season only: must never be named
}


def _charting(weeks=7, season=SEASON):
    rows = []
    for pid, (team, rate) in CORNERS.items():
        if pid == "benched":
            continue
        for w in range(1, weeks + 1):
            rows.append({
                "season": season, "week": w, "game_type": "REG", "team": team,
                "opponent": "OPP", "pfr_player_id": pid,
                "pfr_player_name": pid, "def_targets": 10.0,
                "def_completions_allowed": 10.0 * rate, "def_adot": 9.0,
                "def_yards_allowed": 100.0 * rate,
            })
    for w in range(1, 17):          # the scratch, graded off last season alone
        rows.append({
            "season": season - 1, "week": w, "game_type": "REG", "team": "SOFT",
            "opponent": "OPP", "pfr_player_id": "benched",
            "pfr_player_name": "benched", "def_targets": 10.0,
            "def_completions_allowed": 9.5, "def_adot": 9.0,
            "def_yards_allowed": 130.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _charted(monkeypatch, db_session):
    """Synthetic charting, a schedule, and receivers to hang off it."""
    from nfl_projections import coverage as research
    from api import coverage

    coverage._cache.clear()
    monkeypatch.setattr(research, "load_charting", lambda seasons: _charting())
    monkeypatch.setattr(research, "_positions", lambda: pd.DataFrame(
        [{"pfr_player_id": p, "position": "CB", "player_name": p} for p in CORNERS]))

    # Only this module's own rows get cleared. `db_session` is per-test but the
    # engine behind it is session-scoped, so a committed table-wide DELETE here
    # reaches other modules' seeded fixtures -- which is exactly what it did.
    def _mine():
        db_session.query(Schedule).filter(Schedule.season == SEASON,
                                          Schedule.week == WEEK).delete()
        db_session.query(PlayerProjection).filter(
            PlayerProjection.season == SEASON,
            PlayerProjection.week == WEEK).delete()

    _mine()
    # OFFA plays the soft secondary, OFFB the even one.
    db_session.add_all([
        Schedule(game_id=f"{SEASON}_{WEEK}_SOFT_OFFA", season=SEASON, week=WEEK,
                 home_team="SOFT", away_team="OFFA"),
        Schedule(game_id=f"{SEASON}_{WEEK}_EVEN_OFFB", season=SEASON, week=WEEK,
                 home_team="EVEN", away_team="OFFB"),
    ])
    db_session.add_all([
        PlayerProjection(season=SEASON, week=WEEK, player_id="wr1",
                         player_name="Alpha Receiver", position="WR",
                         team="OFFA", projected_points=14.0, ceiling=28.0),
        PlayerProjection(season=SEASON, week=WEEK, player_id="te1",
                         player_name="Beta End", position="TE",
                         team="OFFA", projected_points=8.0, ceiling=17.0),
        PlayerProjection(season=SEASON, week=WEEK, player_id="rb1",
                         player_name="Gamma Back", position="RB",
                         team="OFFA", projected_points=11.0, ceiling=22.0),
        PlayerProjection(season=SEASON, week=WEEK, player_id="wr2",
                         player_name="Delta Receiver", position="WR",
                         team="OFFB", projected_points=12.0, ceiling=25.0),
    ])
    db_session.commit()
    yield
    _mine()
    db_session.commit()
    coverage._cache.clear()


class TestCorners:
    def test_grades_every_charted_corner_within_the_scale(self, client):
        body = client.get(f"/coverage/corners?season={SEASON}&min_targets=0").json()
        graded = {c["player_name"] for c in body["corners"]}
        assert {"beaten", "sticky", "even_a", "even_b"} <= graded
        for c in body["corners"]:
            assert 0 <= c["cover_grade"] <= 100

    def test_worst_coverage_comes_first(self, client):
        body = client.get(f"/coverage/corners?season={SEASON}&min_targets=0").json()
        grades = [c["cover_grade"] for c in body["corners"]]
        assert grades == sorted(grades)
        assert body["corners"][0]["player_name"] in ("benched", "beaten")

    def test_allowing_more_completions_grades_lower(self, client):
        by = {c["player_name"]: c for c in
              client.get(f"/coverage/corners?season={SEASON}").json()["corners"]}
        assert by["beaten"]["cover_grade"] < by["sticky"]["cover_grade"]

    def test_min_targets_hides_a_corner_nobody_has_tested(self, client):
        """The scratch has a grade from last season and no targets this one."""
        loose = client.get(f"/coverage/corners?season={SEASON}&min_targets=0").json()
        tight = client.get(f"/coverage/corners?season={SEASON}&min_targets=10").json()
        assert "benched" in {c["player_name"] for c in loose["corners"]}
        assert "benched" not in {c["player_name"] for c in tight["corners"]}

    def test_the_response_says_yards_is_not_a_trait(self, client):
        body = client.get(f"/coverage/corners?season={SEASON}").json()
        assert "0.188" in body["not_a_trait"]
        assert all("yards_per_target" in c for c in body["corners"])

    def test_unknown_season_is_a_404(self, client, monkeypatch):
        from nfl_projections import coverage as research
        monkeypatch.setattr(research, "load_charting", lambda seasons: pd.DataFrame())
        from api import coverage
        coverage._cache.clear()
        assert client.get("/coverage/corners?season=2019").status_code == 404


class TestSecondaries:
    def test_names_the_weakest_regular_corner_not_the_scratch(self, client):
        rows = {s["team"]: s for s in
                client.get(f"/coverage/secondaries?season={SEASON}").json()["secondaries"]}
        assert rows["SOFT"]["weakest_corner"] == "beaten"

    def test_spread_sees_the_exploitable_side_the_average_cannot(self, client):
        rows = {s["team"]: s for s in
                client.get(f"/coverage/secondaries?season={SEASON}").json()["secondaries"]}
        assert rows["SOFT"]["cover_spread"] > rows["EVEN"]["cover_spread"]
        assert rows["EVEN"]["cover_spread"] == pytest.approx(0.0, abs=1e-9)


class TestMatchups:
    def test_receivers_reach_the_secondary_they_face(self, client):
        body = client.get(f"/coverage/matchups?season={SEASON}&week={WEEK}").json()
        rows = {m["team"]: m for m in body["matchups"]}
        assert rows["OFFA"]["opponent"] == "SOFT"
        assert rows["OFFA"]["opp_weakest_corner"] == "beaten"
        names = [r["player_name"] for r in rows["OFFA"]["receivers"]]
        assert names == ["Alpha Receiver", "Gamma Back", "Beta End"]  # by projection

    def test_the_wideout_lean_is_always_context_only(self, client):
        """Measured harmful, so the page must never suggest an adjustment."""
        body = client.get(f"/coverage/matchups?season={SEASON}&week={WEEK}").json()
        leans = {r["player_name"]: r["coverage_lean"]
                 for m in body["matchups"] for r in m["receivers"]}
        assert leans["Alpha Receiver"] == "context only"
        assert leans["Delta Receiver"] == "context only"

    def test_projected_points_are_not_adjusted(self, client):
        body = client.get(f"/coverage/matchups?season={SEASON}&week={WEEK}").json()
        rows = {m["team"]: m for m in body["matchups"]}
        proj = {r["player_name"]: r["projected_points"]
                for r in rows["OFFA"]["receivers"]}
        assert proj["Alpha Receiver"] == pytest.approx(14.0)
        assert proj["Gamma Back"] == pytest.approx(11.0)
        assert body["lean_is_advisory"] is True

    def test_the_opposing_corners_come_along(self, client):
        body = client.get(f"/coverage/matchups?season={SEASON}&week={WEEK}").json()
        rows = {m["team"]: m for m in body["matchups"]}
        names = {c["player_name"] for c in rows["OFFA"]["opp_corners"]}
        assert "beaten" in names

    def test_a_week_with_no_schedule_is_a_404(self, client):
        assert client.get(
            f"/coverage/matchups?season={SEASON}&week=15").status_code == 404
