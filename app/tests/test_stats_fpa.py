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

from database.models import PlayerStat, Schedule, Team


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
