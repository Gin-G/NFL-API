#!/usr/bin/env python3
"""Tests for the trench report endpoints (line ratings and weekly matchups).

The DB path is what these exercise: `play_by_play` is loaded by the data
pipeline rather than declared as an ORM model, so the table is created here by
hand. Without it the endpoint would silently fall back to downloading a season
from nflverse, which is exactly the behaviour a test must not reach.
"""
import numpy as np
import pytest
from sqlalchemy import text

from database.models import Schedule

pytest.importorskip("nfl_projections.trenches",
                    reason="line ratings come from nfl_projections (installed --no-deps)")

SEASON = 2024
# Two teams that block and two that defend, so a matchup has a direction.
TEAMS = {"BLKA": (1.4, 0.0), "BLKZ": (-1.4, 0.0),
         "STOP": (0.0, -1.2), "SOFT": (0.0, 1.2)}

_DDL = """
CREATE TABLE IF NOT EXISTS play_by_play (
    season INTEGER, week INTEGER, season_type TEXT,
    posteam TEXT, defteam TEXT, yards_gained REAL,
    rush_attempt REAL, qb_dropback REAL, qb_scramble REAL, qb_kneel REAL,
    two_point_attempt REAL, sack REAL, qb_hit REAL
)
"""


def _plays(seasons=(SEASON,), weeks=16, seed=3):
    rng = np.random.default_rng(seed)
    names = list(TEAMS)
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            offset = 1 + (week - 1) % (len(names) - 1)
            for i, off in enumerate(names):
                dfn = names[(i + offset) % len(names)]
                mean = 4.0 + TEAMS[off][0] + TEAMS[dfn][1]
                for _ in range(24):
                    rows.append((season, week, "REG", off, dfn,
                                 float(rng.normal(mean, 3.0)), 1.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0))
                p = float(np.clip(0.22 - 0.08 * TEAMS[off][0], 0.02, 0.6))
                for hit in rng.random(28) < p:
                    rows.append((season, week, "REG", off, dfn, 6.0, 0.0, 1.0,
                                 0.0, 0.0, 0.0, 0.0, 1.0 if hit else 0.0))
    return rows


@pytest.fixture(autouse=True)
def _pbp(db_session):
    """A synthetic league in the play_by_play table, cleared between tests."""
    from api import trenches

    trenches._cache.clear()
    db_session.execute(text(_DDL))
    db_session.execute(text("DELETE FROM play_by_play"))
    db_session.query(Schedule).delete()
    db_session.execute(
        text("INSERT INTO play_by_play VALUES "
             "(:season, :week, :season_type, :posteam, :defteam, :yards_gained, "
             ":rush_attempt, :qb_dropback, :qb_scramble, :qb_kneel, "
             ":two_point_attempt, :sack, :qb_hit)"),
        [dict(zip(["season", "week", "season_type", "posteam", "defteam",
                   "yards_gained", "rush_attempt", "qb_dropback", "qb_scramble",
                   "qb_kneel", "two_point_attempt", "sack", "qb_hit"], row))
         for row in _plays()])
    db_session.commit()
    yield
    db_session.execute(text("DELETE FROM play_by_play"))
    db_session.query(Schedule).delete()
    db_session.commit()
    trenches._cache.clear()


def _schedule(db_session, week=17):
    db_session.add_all([
        Schedule(game_id=f"{SEASON}_{week}_BLKA_SOFT", season=SEASON, week=week,
                 home_team="SOFT", away_team="BLKA"),
        Schedule(game_id=f"{SEASON}_{week}_BLKZ_STOP", season=SEASON, week=week,
                 home_team="STOP", away_team="BLKZ"),
    ])
    db_session.commit()


class TestRatings:
    def test_returns_a_grade_per_unit_per_team(self, client):
        r = client.get(f"/trenches/ratings?season={SEASON}")
        assert r.status_code == 200
        body = r.json()
        assert {u["team"] for u in body["units"]} == set(TEAMS)
        for unit in body["units"]:
            for key in ("run_block", "run_stop", "pass_block", "pass_rush"):
                assert 0 <= unit[key] <= 100

    def test_grades_order_the_lines_it_was_given(self, client):
        units = {u["team"]: u for u in
                 client.get(f"/trenches/ratings?season={SEASON}").json()["units"]}
        assert units["BLKA"]["run_block"] > units["BLKZ"]["run_block"]
        assert units["STOP"]["run_stop"] > units["SOFT"]["run_stop"]
        assert units["BLKA"]["pass_block"] > units["BLKZ"]["pass_block"]

    def test_through_week_uses_only_earlier_football(self, client):
        early = client.get(f"/trenches/ratings?season={SEASON}&through_week=6").json()
        late = client.get(f"/trenches/ratings?season={SEASON}&through_week=16").json()
        assert early["through_week"] == 6 and late["through_week"] == 16
        # Same league, more evidence: the spread between best and worst widens
        # as shrinkage toward the league mean lets go.
        def spread(body):
            vals = [u["run_block"] for u in body["units"]]
            return max(vals) - min(vals)
        assert spread(late) > spread(early)

    def test_unknown_season_is_a_404(self, client):
        assert client.get("/trenches/ratings?season=1999").status_code == 404


class TestMatchups:
    def test_both_sides_of_every_game(self, client, db_session):
        _schedule(db_session)
        body = client.get(f"/trenches/matchups?season={SEASON}&week=17").json()
        assert len(body["matchups"]) == 4
        pairs = {(m["team"], m["opponent"]) for m in body["matchups"]}
        assert pairs == {("BLKA", "SOFT"), ("SOFT", "BLKA"),
                         ("BLKZ", "STOP"), ("STOP", "BLKZ")}

    def test_a_good_line_against_a_soft_front_rates_high(self, client, db_session):
        _schedule(db_session)
        rows = {m["team"]: m for m in
                client.get(f"/trenches/matchups?season={SEASON}&week=17").json()["matchups"]}
        assert rows["BLKA"]["run_matchup"] > 60
        assert rows["BLKZ"]["run_matchup"] < 40
        # And the rating is an edge, not a restatement of the offence's grade.
        assert rows["BLKA"]["run_matchup"] != rows["BLKA"]["run_block"]

    def test_matchups_are_sorted_by_the_run_number(self, client, db_session):
        _schedule(db_session)
        ratings = [m["run_matchup"] for m in
                   client.get(f"/trenches/matchups?season={SEASON}&week=17").json()["matchups"]]
        assert ratings == sorted(ratings, reverse=True)

    def test_bands_are_published_with_the_numbers(self, client, db_session):
        _schedule(db_session)
        body = client.get(f"/trenches/matchups?season={SEASON}&week=17").json()
        labels = [b["label"] for b in body["bands"]]
        assert labels == ["fade", "lean down", "neutral", "lean up", "smash"]

    def test_week_with_no_schedule_is_a_404(self, client, db_session):
        assert client.get(f"/trenches/matchups?season={SEASON}&week=21").status_code == 404


class TestBacktest:
    def test_grades_both_ladders(self, client):
        body = client.get(f"/trenches/backtest?seasons=1&first_week=6").json()
        assert body["team_games"] > 0
        assert body["matchup_bands"] and body["opponent_bands"]
        for row in body["matchup_bands"]:
            assert row["team_games"] > 0

    def test_reports_the_own_versus_opponent_split(self, client):
        """The split is the honest part: without it a dashboard can show the
        combined rating's flattering ladder and nothing else."""
        corr = client.get("/trenches/backtest?seasons=1").json()["correlations"]
        assert set(corr["own_vs_opp"]) == {"own_run_block", "opp_run_stop"}
        assert "ypc_vs_own" in corr["own_vs_opp"]["opp_run_stop"]

    def test_a_softer_front_gives_up_more(self, client):
        """The fixture builds a real matchup effect, so the ladder must show it."""
        bands = client.get("/trenches/backtest?seasons=1").json()["opponent_bands"]
        by_label = {b["band"]: b for b in bands}
        soft = by_label.get("very soft") or by_label.get("soft")
        stout = by_label.get("very stout") or by_label.get("stout")
        assert soft and stout and soft["ypc"] > stout["ypc"]
