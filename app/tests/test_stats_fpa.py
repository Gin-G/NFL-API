#!/usr/bin/env python3
"""
Tests for the NickKnows-parity endpoints backed by the DB:
  - /stats/leaders/
  - /fpa/
  - /teams/{abbr}/fpa
  - /teams/{abbr}/results

These seed the in-memory SQLite DB directly so the DB-first code path runs
(no nflreadpy involved). Rows are flushed (not committed) so db_session's
rollback isolates each test.
"""

import pandas as pd
import pytest

from api import fpa as fpa_api
from database.models import PlayerStat, Schedule, Team


@pytest.fixture(autouse=True)
def _clear_fpa_cache():
    """The adjusted ratings are cached at module scope for 15 minutes."""
    fpa_api._adj_cache.clear()
    yield
    fpa_api._adj_cache.clear()


# ── seeding helpers ───────────────────────────────────────────────────────────

def _stat(**kw):
    base = dict(season=2024, season_type="REG")
    base.update(kw)
    return PlayerStat(**base)


def _seed_stats(db):
    """A small, hand-checkable slate for 2024."""
    db.add_all([
        # KC faces DET (wk1) then hosts BUF (wk2)
        _stat(player_id="P_MAHOMES", week=1, player_display_name="Patrick Mahomes",
              position="QB", recent_team="KC", passing_yards=300, passing_tds=3,
              fantasy_points_ppr=25.0, fantasy_points=25.0),
        _stat(player_id="P_MAHOMES", week=2, player_display_name="Patrick Mahomes",
              position="QB", recent_team="KC", passing_yards=250, passing_tds=2,
              fantasy_points_ppr=35.0, fantasy_points=35.0),
        # DET RB faces KC in wk1
        _stat(player_id="D_MONTGOMERY", week=1, player_display_name="David Montgomery",
              position="RB", recent_team="DET", rushing_yards=90, rushing_tds=1,
              fantasy_points_ppr=10.0, fantasy_points=8.0),
        # BUF WR faces KC in wk2
        _stat(player_id="BUF_WR", week=2, player_display_name="Khalil Shakir",
              position="WR", recent_team="BUF", receiving_yards=80, receiving_tds=1,
              fantasy_points_ppr=20.0, fantasy_points=14.0),
        # A postseason row that must be excluded from REG-only leaders
        _stat(player_id="P_MAHOMES", week=20, player_display_name="Patrick Mahomes",
              position="QB", recent_team="KC", passing_yards=999, passing_tds=9,
              season_type="POST", fantasy_points_ppr=50.0, fantasy_points=50.0),
    ])
    db.flush()


def _seed_schedule(db):
    db.add_all([
        Schedule(game_id="2024_01_KC_DET", season=2024, week=1,
                 home_team="DET", away_team="KC", home_score=20, away_score=27),
        Schedule(game_id="2024_02_BUF_KC", season=2024, week=2,
                 home_team="KC", away_team="BUF", home_score=30, away_score=24),
        # An unplayed KC game that must be excluded from results (no scores)
        Schedule(game_id="2024_03_KC_LV", season=2024, week=3,
                 home_team="LV", away_team="KC", home_score=None, away_score=None),
    ])
    db.flush()


def _seed_teams(db):
    db.add_all([
        Team(team_abbr="KC", team_name="Kansas City Chiefs"),
        Team(team_abbr="DET", team_name="Detroit Lions"),
        Team(team_abbr="BUF", team_name="Buffalo Bills"),
    ])
    db.flush()


# ── /stats/leaders/ ───────────────────────────────────────────────────────────

class TestStatLeaders:
    def test_passing_yards_leader_sum_and_order(self, client, db_session):
        _seed_stats(db_session)
        body = client.get("/stats/leaders/?season=2024&stat=passing_yards").json()
        assert body["status"] == "success"
        # 300 + 250 = 550 (POST week excluded)
        top = body["data"][0]
        assert top["player_name"] == "Patrick Mahomes"
        assert top["value"] == 550.0
        assert top["team"] == "KC"
        assert top["position"] == "QB"
        assert top["player_id"] == "P_MAHOMES"

    def test_limit_respected(self, client, db_session):
        _seed_stats(db_session)
        body = client.get("/stats/leaders/?season=2024&stat=rushing_yards&limit=1").json()
        assert len(body["data"]) == 1
        assert body["data"][0]["player_name"] == "David Montgomery"

    def test_invalid_stat_returns_400(self, client, db_session):
        resp = client.get("/stats/leaders/?season=2024&stat=bogus")
        assert resp.status_code == 400

    def test_receiving_tds_leader(self, client, db_session):
        _seed_stats(db_session)
        body = client.get("/stats/leaders/?season=2024&stat=receiving_tds").json()
        assert body["data"][0]["player_name"] == "Khalil Shakir"
        assert body["data"][0]["value"] == 1.0


# ── /fpa/ ─────────────────────────────────────────────────────────────────────

class TestFPA:
    def test_fpa_per_defense(self, client, db_session):
        _seed_teams(db_session)
        _seed_schedule(db_session)
        _seed_stats(db_session)
        body = client.get("/fpa/?season=2024").json()
        assert body["status"] == "success"
        rows = {r["team"]: r for r in body["data"]}
        # DET's defense faced KC (Mahomes 25 PPR) in wk1
        assert rows["DET"]["qb"] == 25.0
        assert rows["DET"]["team_name"] == "Detroit Lions"
        # BUF's defense faced KC (Mahomes 35 PPR) in wk2
        assert rows["BUF"]["qb"] == 35.0
        # KC's defense faced DET RB (10) in wk1 and BUF WR (20) in wk2
        assert rows["KC"]["rb"] == 10.0
        assert rows["KC"]["wr"] == 20.0
        assert rows["KC"]["qb"] == 0.0



def _seed_adjusted_slate(db):
    """Two defences that allow the same points against very different players.

    SF's star receiver and NYJ's replacement both put up exactly 9.0 FanDuel
    points in weeks 4-6, so raw FPA has SEA and WAS tied. But the star was
    running at 20.5 a week and the replacement at 1.5, so SEA held a good
    player well under his level and WAS let a bad one beat his -- which is the
    distinction the raw table cannot make.

    Weeks 1-3 build the two baselines against filler defences; those filler
    teams never earn a grade, because a delta needs three prior games first.
    """
    rows, games = [], []
    for team, player, position, fill, early, late_ppr in (
        ("SF", "STAR_WR", "WR", ["DAL", "NYG", "PHI"],
         dict(receiving_yards=95, receptions=10, receiving_tds=1), 9.0),
        ("NYJ", "SCRUB_WR", "WR", ["NE", "MIA", "BUF"],
         dict(receiving_yards=10, receptions=1, receiving_tds=0), 9.0),
    ):
        defence = "SEA" if team == "SF" else "WAS"
        for week, opponent in enumerate(fill, start=1):
            rows.append(_stat(player_id=player, week=week, position=position,
                              player_display_name=player, recent_team=team,
                              fantasy_points_ppr=20.0, **early))
            games.append(Schedule(game_id=f"2024_{week:02d}_{team}_{opponent}",
                                  season=2024, week=week, home_team=opponent,
                                  away_team=team, home_score=17, away_score=20))
        for week in (4, 5, 6):
            rows.append(_stat(player_id=player, week=week, position=position,
                              player_display_name=player, recent_team=team,
                              receiving_yards=60, receptions=6, receiving_tds=0,
                              fantasy_points_ppr=late_ppr))
            games.append(Schedule(game_id=f"2024_{week:02d}_{team}_{defence}",
                                  season=2024, week=week, home_team=defence,
                                  away_team=team, home_score=17, away_score=20))
    db.add_all(rows + games)
    db.add_all([Team(team_abbr="SEA", team_name="Seattle Seahawks"),
                Team(team_abbr="WAS", team_name="Washington Commanders")])
    db.flush()


class TestFPAAdjusted:
    @staticmethod
    def _rows(client):
        body = client.get("/fpa/?season=2024").json()
        assert body["adjusted"] is True
        assert body["adjusted_note"] is None
        return {r["team"]: r for r in body["data"]}

    def test_raw_ties_while_the_adjustment_splits_them(self, client, db_session):
        _seed_adjusted_slate(db_session)
        rows = self._rows(client)
        assert rows["SEA"]["wr"] == rows["WAS"]["wr"]            # raw, PPR
        assert rows["WAS"]["wr_grade"] > rows["SEA"]["wr_grade"]
        assert rows["SEA"]["wr_band"] == "tough"
        assert rows["WAS"]["wr_band"] == "soft"

    def test_allowed_and_faced_are_the_pair_that_explains_a_grade(self, client, db_session):
        _seed_adjusted_slate(db_session)
        rows = self._rows(client)
        # Identical points allowed (FanDuel this time: 6 yards + 3 receptions)
        assert rows["SEA"]["wr_allowed"] == rows["WAS"]["wr_allowed"] == 9.0
        # The schedule each one had is what differs, and by a long way.
        assert rows["SEA"]["wr_faced"] > 15.0
        assert rows["WAS"]["wr_faced"] < 5.0
        assert rows["SEA"]["wr_adjusted"] < 0 < rows["WAS"]["wr_adjusted"]
        assert rows["SEA"]["wr_weeks"] == 3

    def test_the_most_distorting_week_is_named(self, client, db_session):
        _seed_adjusted_slate(db_session)
        rows = self._rows(client)
        # SEA's three weekly gaps are -11.5, -8.6 and -6.9: dropping week 4
        # moves the rating +1.25, more than dropping either of the others.
        assert rows["SEA"]["wr_outlier_week"] == 4
        assert rows["SEA"]["wr_outlier_swing"] == pytest.approx(1.25, abs=0.01)
        # WAS has no week worth a point, so it reports none rather than a zero.
        assert rows["WAS"]["wr_outlier_week"] is None

    def test_a_defence_with_no_rated_weeks_is_null_not_zero(self, client, db_session):
        _seed_adjusted_slate(db_session)
        rows = self._rows(client)
        # DAL only appears in week 1, before anyone has a baseline. A missing
        # grade must not read as 0.0, which the bands would call "very tough".
        assert rows["DAL"]["wr_grade"] is None
        assert rows["DAL"]["wr_band"] is None
        assert rows["DAL"]["wr"] == 20.0          # the raw number is still there

    def test_through_week_rates_only_the_earlier_weeks(self, client, db_session):
        _seed_adjusted_slate(db_session)
        body = client.get("/fpa/?season=2024&through_week=5").json()
        rows = {r["team"]: r for r in body["data"]}
        assert body["through_week"] == 5
        assert rows["SEA"]["wr_weeks"] == 1       # week 4 only
        assert rows["WAS"]["wr_grade"] > rows["SEA"]["wr_grade"]

    def test_early_season_explains_itself_instead_of_failing(self, client, db_session):
        _seed_teams(db_session)
        _seed_schedule(db_session)
        _seed_stats(db_session)                   # two weeks: no baselines yet
        body = client.get("/fpa/?season=2024").json()
        assert body["status"] == "success"
        assert body["adjusted"] is False
        assert "three prior games" in body["adjusted_note"]
        # The raw series the site charts is unaffected by that.
        rows = {r["team"]: r for r in body["data"]}
        assert rows["DET"]["qb"] == 25.0
        assert rows["DET"]["qb_grade"] is None

    def test_the_response_carries_the_scale_and_the_bands(self, client, db_session):
        _seed_adjusted_slate(db_session)
        body = client.get("/fpa/?season=2024").json()
        assert "higher = softer" in body["scale"]
        assert [b["label"] for b in body["bands"]][0] == "very tough"
        assert body["scoring"]["qb_rb_wr_te"].startswith("PPR")

# ── /teams/{abbr}/fpa ─────────────────────────────────────────────────────────

class TestTeamFPADetail:
    def test_detail_rows(self, client, db_session):
        _seed_schedule(db_session)
        _seed_stats(db_session)
        body = client.get("/teams/KC/fpa?season=2024").json()
        assert body["status"] == "success"
        # KC faced DET RB (wk1) and BUF WR (wk2)
        assert body["total_records"] == 2
        opponents = {r["opponent"] for r in body["data"]}
        assert opponents == {"DET", "BUF"}

    def test_position_filter(self, client, db_session):
        _seed_schedule(db_session)
        _seed_stats(db_session)
        body = client.get("/teams/KC/fpa?season=2024&position=RB").json()
        assert body["total_records"] == 1
        assert body["data"][0]["player_name"] == "David Montgomery"
        assert body["data"][0]["fantasy_points_ppr"] == 10.0


# ── /teams/{abbr}/results ─────────────────────────────────────────────────────

class TestTeamResults:
    def test_completed_games_only_sorted(self, client, db_session):
        _seed_schedule(db_session)
        body = client.get("/teams/KC/results?season=2024").json()
        assert body["status"] == "success"
        assert body["total_games"] == 2  # wk3 unplayed excluded
        weeks = [g["week"] for g in body["data"]]
        assert weeks == [1, 2]

    def test_is_home_flag(self, client, db_session):
        _seed_schedule(db_session)
        body = client.get("/teams/KC/results?season=2024").json()
        by_week = {g["week"]: g for g in body["data"]}
        assert by_week[1]["is_home"] is False   # KC away @ DET
        assert by_week[2]["is_home"] is True     # KC home vs BUF
