#!/usr/bin/env python3
"""
End-to-end tests for the coaching breakdown system.

Uses a dedicated in-memory SQLite engine that has BOTH:
  - ORM tables (via Base.metadata.create_all)
  - play_by_play table (created via pandas to_sql with realistic sample data)

Tests cover:
  - Unit: each compute_* helper function
  - Integration: GET /coaches/{name}/breakdown endpoint
  - Graceful no_data when PBP table absent
"""

import sys
import os
import pytest
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Sample PBP data ───────────────────────────────────────────────────────────

SAMPLE_PBP = [
    # ── KC offense (posteam=KC, defteam=DET) ─────────────────────────────────
    {
        "game_id": "2023_01_KC_DET", "play_id": 1, "season": 2023, "week": 1,
        "posteam": "KC", "defteam": "DET", "play_type": "pass", "down": 1,
        "ydstogo": 10, "yardline_100": 65, "yards_gained": 12, "epa": 0.8, "success": 1,
        "offense_formation": "SHOTGUN", "offense_personnel": "1 RB, 1 TE, 3 WR",
        "defense_personnel": "4 DL, 2 LB, 5 DB",
        "shotgun": 1, "no_huddle": 0, "play_action": 0,
        "qb_dropback": 1, "qb_scramble": 0, "qb_hit": 0,
        "pass_location": "right", "air_yards": 8.0, "yards_after_catch": 4.0,
        "run_location": None, "run_gap": None,
        "fourth_down_converted": 0, "fourth_down_failed": 0,
        "third_down_converted": 0, "third_down_failed": 0,
        "sack": 0, "touchdown": 0, "two_point_attempt": 0, "two_point_conv_result": None,
        "number_of_pass_rushers": 4, "defenders_in_box": 6,
        "desc": "P.Mahomes pass short right to T.Kelce for 12 yards",
    },
    {
        "game_id": "2023_01_KC_DET", "play_id": 2, "season": 2023, "week": 1,
        "posteam": "KC", "defteam": "DET", "play_type": "run", "down": 2,
        "ydstogo": 5, "yardline_100": 53, "yards_gained": 5, "epa": 0.1, "success": 1,
        "offense_formation": "SINGLEBACK", "offense_personnel": "2 RB, 1 TE, 2 WR",
        "defense_personnel": "4 DL, 3 LB, 4 DB",
        "shotgun": 0, "no_huddle": 0, "play_action": 0,
        "qb_dropback": 0, "qb_scramble": 0, "qb_hit": 0,
        "pass_location": None, "air_yards": None, "yards_after_catch": None,
        "run_location": "middle", "run_gap": "guard",
        "fourth_down_converted": 0, "fourth_down_failed": 0,
        "third_down_converted": 0, "third_down_failed": 0,
        "sack": 0, "touchdown": 0, "two_point_attempt": 0, "two_point_conv_result": None,
        "number_of_pass_rushers": None, "defenders_in_box": 7,
        "desc": "I.Pacheco up the middle for 5 yards",
    },
    {
        "game_id": "2023_01_KC_DET", "play_id": 3, "season": 2023, "week": 1,
        "posteam": "KC", "defteam": "DET", "play_type": "pass", "down": 3,
        "ydstogo": 7, "yardline_100": 48, "yards_gained": 9, "epa": 1.2, "success": 1,
        "offense_formation": "SHOTGUN", "offense_personnel": "1 RB, 1 TE, 3 WR",
        "defense_personnel": "4 DL, 2 LB, 5 DB",
        "shotgun": 1, "no_huddle": 0, "play_action": 0,
        "qb_dropback": 1, "qb_scramble": 0, "qb_hit": 0,
        "pass_location": "middle", "air_yards": 9.0, "yards_after_catch": 0.0,
        "run_location": None, "run_gap": None,
        "fourth_down_converted": 0, "fourth_down_failed": 0,
        "third_down_converted": 1, "third_down_failed": 0,
        "sack": 0, "touchdown": 0, "two_point_attempt": 0, "two_point_conv_result": None,
        "number_of_pass_rushers": 3, "defenders_in_box": 5,
        "desc": "P.Mahomes pass short middle to T.Hill for 9 yards, 1st down",
    },
    {
        "game_id": "2023_01_KC_DET", "play_id": 4, "season": 2023, "week": 1,
        "posteam": "KC", "defteam": "DET", "play_type": "pass", "down": 1,
        "ydstogo": 10, "yardline_100": 40, "yards_gained": 22, "epa": 2.1, "success": 1,
        "offense_formation": "SHOTGUN", "offense_personnel": "1 RB, 2 TE, 2 WR",
        "defense_personnel": "4 DL, 2 LB, 5 DB",
        "shotgun": 1, "no_huddle": 0, "play_action": 1,
        "qb_dropback": 1, "qb_scramble": 0, "qb_hit": 0,
        "pass_location": "left", "air_yards": 20.0, "yards_after_catch": 2.0,
        "run_location": None, "run_gap": None,
        "fourth_down_converted": 0, "fourth_down_failed": 0,
        "third_down_converted": 0, "third_down_failed": 0,
        "sack": 0, "touchdown": 0, "two_point_attempt": 0, "two_point_conv_result": None,
        "number_of_pass_rushers": 4, "defenders_in_box": 6,
        "desc": "P.Mahomes play action pass deep left to M.Hardman for 22 yards",
    },
    {
        "game_id": "2023_01_KC_DET", "play_id": 5, "season": 2023, "week": 1,
        "posteam": "KC", "defteam": "DET", "play_type": "run", "down": 1,
        "ydstogo": 10, "yardline_100": 70, "yards_gained": -1, "epa": -0.5, "success": 0,
        "offense_formation": "I_FORM", "offense_personnel": "2 RB, 1 TE, 2 WR",
        "defense_personnel": "4 DL, 3 LB, 4 DB",
        "shotgun": 0, "no_huddle": 0, "play_action": 0,
        "qb_dropback": 0, "qb_scramble": 0, "qb_hit": 0,
        "pass_location": None, "air_yards": None, "yards_after_catch": None,
        "run_location": "left", "run_gap": "end",
        "fourth_down_converted": 0, "fourth_down_failed": 0,
        "third_down_converted": 0, "third_down_failed": 0,
        "sack": 0, "touchdown": 0, "two_point_attempt": 0, "two_point_conv_result": None,
        "number_of_pass_rushers": None, "defenders_in_box": 8,
        "desc": "I.Pacheco left end for -1 yards",
    },
    {
        "game_id": "2023_01_KC_DET", "play_id": 6, "season": 2023, "week": 1,
        "posteam": "KC", "defteam": "DET", "play_type": "pass", "down": 1,
        "ydstogo": 10, "yardline_100": 55, "yards_gained": 2, "epa": -0.3, "success": 0,
        "offense_formation": "SHOTGUN", "offense_personnel": "1 RB, 1 TE, 3 WR",
        "defense_personnel": "4 DL, 2 LB, 5 DB",
        "shotgun": 1, "no_huddle": 0, "play_action": 0,
        "qb_dropback": 1, "qb_scramble": 0, "qb_hit": 0,
        "pass_location": "right", "air_yards": -2.0, "yards_after_catch": 4.0,
        "run_location": None, "run_gap": None,
        "fourth_down_converted": 0, "fourth_down_failed": 0,
        "third_down_converted": 0, "third_down_failed": 0,
        "sack": 0, "touchdown": 0, "two_point_attempt": 0, "two_point_conv_result": None,
        "number_of_pass_rushers": 4, "defenders_in_box": 6,
        "desc": "P.Mahomes pass short right to T.Kelce (screen) for 2 yards",
    },
    {
        "game_id": "2023_01_KC_DET", "play_id": 7, "season": 2023, "week": 1,
        "posteam": "KC", "defteam": "DET", "play_type": "pass", "down": 4,
        "ydstogo": 1, "yardline_100": 35, "yards_gained": 18, "epa": 1.8, "success": 1,
        "offense_formation": "SHOTGUN", "offense_personnel": "1 RB, 1 TE, 3 WR",
        "defense_personnel": "4 DL, 2 LB, 5 DB",
        "shotgun": 1, "no_huddle": 0, "play_action": 0,
        "qb_dropback": 1, "qb_scramble": 0, "qb_hit": 0,
        "pass_location": "right", "air_yards": 16.0, "yards_after_catch": 2.0,
        "run_location": None, "run_gap": None,
        "fourth_down_converted": 1, "fourth_down_failed": 0,
        "third_down_converted": 0, "third_down_failed": 0,
        "sack": 0, "touchdown": 0, "two_point_attempt": 0, "two_point_conv_result": None,
        "number_of_pass_rushers": 5, "defenders_in_box": 6,
        "desc": "P.Mahomes pass deep right to J.Watson for 18 yards, FIRST DOWN",
    },
    {
        "game_id": "2023_01_KC_DET", "play_id": 8, "season": 2023, "week": 1,
        "posteam": "KC", "defteam": "DET", "play_type": "run", "down": 2,
        "ydstogo": 3, "yardline_100": 45, "yards_gained": 7, "epa": 0.3, "success": 1,
        "offense_formation": "SINGLEBACK", "offense_personnel": "2 RB, 1 TE, 2 WR",
        "defense_personnel": "4 DL, 3 LB, 4 DB",
        "shotgun": 0, "no_huddle": 0, "play_action": 0,
        "qb_dropback": 0, "qb_scramble": 0, "qb_hit": 0,
        "pass_location": None, "air_yards": None, "yards_after_catch": None,
        "run_location": "right", "run_gap": "tackle",
        "fourth_down_converted": 0, "fourth_down_failed": 0,
        "third_down_converted": 0, "third_down_failed": 0,
        "sack": 0, "touchdown": 0, "two_point_attempt": 0, "two_point_conv_result": None,
        "number_of_pass_rushers": None, "defenders_in_box": 7,
        "desc": "I.Pacheco right tackle for 7 yards",
    },
    {
        "game_id": "2023_01_KC_DET", "play_id": 9, "season": 2023, "week": 1,
        "posteam": "KC", "defteam": "DET", "play_type": "pass", "down": 3,
        "ydstogo": 8, "yardline_100": 42, "yards_gained": 0, "epa": -0.9, "success": 0,
        "offense_formation": "SHOTGUN", "offense_personnel": "1 RB, 1 TE, 3 WR",
        "defense_personnel": "4 DL, 2 LB, 5 DB",
        "shotgun": 1, "no_huddle": 0, "play_action": 0,
        "qb_dropback": 1, "qb_scramble": 0, "qb_hit": 1,
        "pass_location": "middle", "air_yards": 6.0, "yards_after_catch": None,
        "run_location": None, "run_gap": None,
        "fourth_down_converted": 0, "fourth_down_failed": 0,
        "third_down_converted": 0, "third_down_failed": 1,
        "sack": 0, "touchdown": 0, "two_point_attempt": 0, "two_point_conv_result": None,
        "number_of_pass_rushers": 6, "defenders_in_box": 5,
        "desc": "P.Mahomes incomplete pass short middle intended for T.Hill",
    },
    {
        "game_id": "2023_01_KC_DET", "play_id": 10, "season": 2023, "week": 1,
        "posteam": "KC", "defteam": "DET", "play_type": "pass", "down": 1,
        "ydstogo": 10, "yardline_100": 30, "yards_gained": 15, "epa": 1.0, "success": 1,
        "offense_formation": "SHOTGUN", "offense_personnel": "1 RB, 0 TE, 4 WR",
        "defense_personnel": "4 DL, 1 LB, 6 DB",
        "shotgun": 1, "no_huddle": 1, "play_action": 0,
        "qb_dropback": 1, "qb_scramble": 0, "qb_hit": 0,
        "pass_location": "left", "air_yards": 12.0, "yards_after_catch": 3.0,
        "run_location": None, "run_gap": None,
        "fourth_down_converted": 0, "fourth_down_failed": 0,
        "third_down_converted": 0, "third_down_failed": 0,
        "sack": 0, "touchdown": 0, "two_point_attempt": 0, "two_point_conv_result": None,
        "number_of_pass_rushers": 4, "defenders_in_box": 4,
        "desc": "P.Mahomes no-huddle pass short left to M.Hardman for 15 yards",
    },
    # ── DET offense vs KC defense (posteam=DET, defteam=KC) ──────────────────
    {
        "game_id": "2023_01_KC_DET", "play_id": 20, "season": 2023, "week": 1,
        "posteam": "DET", "defteam": "KC", "play_type": "pass", "down": 1,
        "ydstogo": 10, "yardline_100": 50, "yards_gained": 8, "epa": 0.3, "success": 1,
        "offense_formation": "SHOTGUN", "offense_personnel": "1 RB, 1 TE, 3 WR",
        "defense_personnel": "4 DL, 2 LB, 5 DB",
        "shotgun": 1, "no_huddle": 0, "play_action": 0,
        "qb_dropback": 1, "qb_scramble": 0, "qb_hit": 0,
        "pass_location": "right", "air_yards": 6.0, "yards_after_catch": 2.0,
        "run_location": None, "run_gap": None,
        "fourth_down_converted": 0, "fourth_down_failed": 0,
        "third_down_converted": 0, "third_down_failed": 0,
        "sack": 0, "touchdown": 0, "two_point_attempt": 0, "two_point_conv_result": None,
        "number_of_pass_rushers": 4, "defenders_in_box": 6,
        "desc": "J.Goff pass short right to A.St.Brown for 8 yards",
    },
    {
        "game_id": "2023_01_KC_DET", "play_id": 21, "season": 2023, "week": 1,
        "posteam": "DET", "defteam": "KC", "play_type": "run", "down": 2,
        "ydstogo": 2, "yardline_100": 42, "yards_gained": 1, "epa": -0.2, "success": 0,
        "offense_formation": "SINGLEBACK", "offense_personnel": "2 RB, 1 TE, 2 WR",
        "defense_personnel": "4 DL, 3 LB, 4 DB",
        "shotgun": 0, "no_huddle": 0, "play_action": 0,
        "qb_dropback": 0, "qb_scramble": 0, "qb_hit": 0,
        "pass_location": None, "air_yards": None, "yards_after_catch": None,
        "run_location": "middle", "run_gap": "guard",
        "fourth_down_converted": 0, "fourth_down_failed": 0,
        "third_down_converted": 0, "third_down_failed": 0,
        "sack": 0, "touchdown": 0, "two_point_attempt": 0, "two_point_conv_result": None,
        "number_of_pass_rushers": None, "defenders_in_box": 7,
        "desc": "D.Swift up the middle for 1 yard",
    },
    {
        "game_id": "2023_01_KC_DET", "play_id": 22, "season": 2023, "week": 1,
        "posteam": "DET", "defteam": "KC", "play_type": "pass", "down": 3,
        "ydstogo": 7, "yardline_100": 38, "yards_gained": -5, "epa": -1.5, "success": 0,
        "offense_formation": "SHOTGUN", "offense_personnel": "1 RB, 1 TE, 3 WR",
        "defense_personnel": "4 DL, 2 LB, 5 DB",
        "shotgun": 1, "no_huddle": 0, "play_action": 0,
        "qb_dropback": 1, "qb_scramble": 0, "qb_hit": 1,
        "pass_location": "middle", "air_yards": None, "yards_after_catch": None,
        "run_location": None, "run_gap": None,
        "fourth_down_converted": 0, "fourth_down_failed": 0,
        "third_down_converted": 0, "third_down_failed": 1,
        "sack": 1, "touchdown": 0, "two_point_attempt": 0, "two_point_conv_result": None,
        "number_of_pass_rushers": 5, "defenders_in_box": 5,
        "desc": "J.Goff sacked by C.Jones for -5 yards",
    },
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def _pbp_engine():
    """Separate in-memory engine with ORM tables + play_by_play table populated."""
    from database.models import Base
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    # Create play_by_play table via pandas
    df = pd.DataFrame(SAMPLE_PBP)
    with engine.connect() as conn:
        df.to_sql("play_by_play", con=conn, if_exists="replace", index=False)
        conn.commit()

    return engine


@pytest.fixture(scope="module")
def pbp_session(_pbp_engine):
    """Module-scoped session with PBP data + a Schedule row for Andy Reid/KC."""
    from database.models import Schedule
    Session = sessionmaker(bind=_pbp_engine)
    session = Session()

    # Insert schedule row so coach lookup works
    game = Schedule(
        game_id="2023_01_KC_DET",
        season=2023, week=1,
        home_team="KC", away_team="DET",
        home_score=21, away_score=20,
        home_coach="Andy Reid",
        away_coach="Dan Campbell",
    )
    session.merge(game)
    session.commit()
    yield session
    session.close()


@pytest.fixture(scope="module")
def breakdown_client(_pbp_engine, pbp_session):
    """Test client wired to the PBP-populated engine."""
    from main import app
    from database.session import get_db

    def _override():
        yield pbp_session

    app.dependency_overrides[get_db] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── Unit tests: compute_* helpers ─────────────────────────────────────────────

class TestComputeFormationBreakdown:
    def test_returns_dict(self, pbp_session):
        from api.coaching_pbp import compute_formation_breakdown
        result = compute_formation_breakdown(pbp_session, "KC", 2023)
        assert isinstance(result, dict)

    def test_shotgun_rate_present(self, pbp_session):
        from api.coaching_pbp import compute_formation_breakdown
        result = compute_formation_breakdown(pbp_session, "KC", 2023)
        assert result["shotgun_rate"] is not None
        # 7 of 10 KC plays are shotgun (play_ids 1,3,4,6,7,9,10)
        assert 0.5 < result["shotgun_rate"] < 1.0

    def test_no_huddle_rate_present(self, pbp_session):
        from api.coaching_pbp import compute_formation_breakdown
        result = compute_formation_breakdown(pbp_session, "KC", 2023)
        assert result["no_huddle_rate"] is not None
        # Only play_id 10 is no-huddle (1 of 10)
        assert result["no_huddle_rate"] == pytest.approx(0.1, abs=0.01)

    def test_play_action_rate_present(self, pbp_session):
        from api.coaching_pbp import compute_formation_breakdown
        result = compute_formation_breakdown(pbp_session, "KC", 2023)
        assert result["play_action_rate"] is not None
        # Only play_id 4 has play_action=1 (1 of 10 plays)
        assert result["play_action_rate"] == pytest.approx(0.1, abs=0.01)

    def test_formation_breakdown_has_shotgun(self, pbp_session):
        from api.coaching_pbp import compute_formation_breakdown
        result = compute_formation_breakdown(pbp_session, "KC", 2023)
        assert "SHOTGUN" in result["formation_breakdown"]

    def test_formation_breakdown_has_singleback(self, pbp_session):
        from api.coaching_pbp import compute_formation_breakdown
        result = compute_formation_breakdown(pbp_session, "KC", 2023)
        assert "SINGLEBACK" in result["formation_breakdown"]

    def test_formation_breakdown_stats(self, pbp_session):
        from api.coaching_pbp import compute_formation_breakdown
        result = compute_formation_breakdown(pbp_session, "KC", 2023)
        shotgun = result["formation_breakdown"]["SHOTGUN"]
        assert shotgun["plays"] > 0
        assert shotgun["pass_rate"] is not None
        assert "avg_epa" in shotgun
        assert "pct_of_plays" in shotgun

    def test_unknown_team_returns_none_rates(self, pbp_session):
        from api.coaching_pbp import compute_formation_breakdown
        result = compute_formation_breakdown(pbp_session, "XYZ", 2023)
        # No plays for unknown team — rates will be None
        assert result["shotgun_rate"] is None or result["shotgun_rate"] == 0.0


class TestComputePersonnelBreakdown:
    def test_returns_dict(self, pbp_session):
        from api.coaching_pbp import compute_personnel_breakdown
        result = compute_personnel_breakdown(pbp_session, "KC", 2023)
        assert isinstance(result, dict)

    def test_11_personnel_present(self, pbp_session):
        from api.coaching_pbp import compute_personnel_breakdown
        result = compute_personnel_breakdown(pbp_session, "KC", 2023)
        # "1 RB, 1 TE, 3 WR" → "11 (1RB 1TE 3WR)"
        assert any("11" in k for k in result)

    def test_21_personnel_present(self, pbp_session):
        from api.coaching_pbp import compute_personnel_breakdown
        result = compute_personnel_breakdown(pbp_session, "KC", 2023)
        # "2 RB, 1 TE, 2 WR" → "21 (2RB 1TE 2WR)"
        assert any("21" in k for k in result)

    def test_personnel_entry_has_required_fields(self, pbp_session):
        from api.coaching_pbp import compute_personnel_breakdown
        result = compute_personnel_breakdown(pbp_session, "KC", 2023)
        for entry in result.values():
            assert "plays" in entry
            assert "pct_of_plays" in entry
            assert "pass_rate" in entry
            assert "avg_epa" in entry

    def test_pct_sums_to_one(self, pbp_session):
        from api.coaching_pbp import compute_personnel_breakdown
        result = compute_personnel_breakdown(pbp_session, "KC", 2023)
        total_pct = sum(v["pct_of_plays"] for v in result.values())
        assert total_pct == pytest.approx(1.0, abs=0.01)


class TestComputeRunScheme:
    def test_returns_dict(self, pbp_session):
        from api.coaching_pbp import compute_run_scheme
        result = compute_run_scheme(pbp_session, "KC", 2023)
        assert isinstance(result, dict)

    def test_has_required_keys(self, pbp_session):
        from api.coaching_pbp import compute_run_scheme
        result = compute_run_scheme(pbp_session, "KC", 2023)
        assert "total_runs" in result
        assert "inside_rate" in result
        assert "outside_rate" in result
        assert "by_location" in result

    def test_total_runs_correct(self, pbp_session):
        from api.coaching_pbp import compute_run_scheme
        result = compute_run_scheme(pbp_session, "KC", 2023)
        # KC has 3 run plays (play_ids 2, 5, 8) all with run_location set
        assert result["total_runs"] == 3

    def test_by_location_has_middle(self, pbp_session):
        from api.coaching_pbp import compute_run_scheme
        result = compute_run_scheme(pbp_session, "KC", 2023)
        assert "middle" in result["by_location"]

    def test_location_stats(self, pbp_session):
        from api.coaching_pbp import compute_run_scheme
        result = compute_run_scheme(pbp_session, "KC", 2023)
        for loc_data in result["by_location"].values():
            assert "plays" in loc_data
            assert "avg_yards" in loc_data
            assert "avg_epa" in loc_data
            assert "success_rate" in loc_data

    def test_inside_outside_rates_sum_to_one(self, pbp_session):
        from api.coaching_pbp import compute_run_scheme
        result = compute_run_scheme(pbp_session, "KC", 2023)
        if result["inside_rate"] is not None and result["outside_rate"] is not None:
            assert (result["inside_rate"] + result["outside_rate"]) == pytest.approx(1.0, abs=0.01)


class TestComputePassDetail:
    def test_returns_dict(self, pbp_session):
        from api.coaching_pbp import compute_pass_detail
        result = compute_pass_detail(pbp_session, "KC", 2023)
        assert isinstance(result, dict)

    def test_has_required_keys(self, pbp_session):
        from api.coaching_pbp import compute_pass_detail
        result = compute_pass_detail(pbp_session, "KC", 2023)
        for key in ("avg_air_yards", "deep_pass_rate", "screen_rate",
                    "intermediate_rate", "scramble_rate", "by_direction"):
            assert key in result

    def test_avg_air_yards_positive(self, pbp_session):
        from api.coaching_pbp import compute_pass_detail
        result = compute_pass_detail(pbp_session, "KC", 2023)
        # KC has passes with air_yards: 8, 9, 20, -2, 16, 6, 12 (7 passes with air_yards set)
        assert result["avg_air_yards"] is not None

    def test_deep_pass_rate_between_0_and_1(self, pbp_session):
        from api.coaching_pbp import compute_pass_detail
        result = compute_pass_detail(pbp_session, "KC", 2023)
        if result["deep_pass_rate"] is not None:
            assert 0 <= result["deep_pass_rate"] <= 1.0

    def test_screen_rate_detected(self, pbp_session):
        from api.coaching_pbp import compute_pass_detail
        result = compute_pass_detail(pbp_session, "KC", 2023)
        # play_id 6 has air_yards=-2 (screen)
        assert result["screen_rate"] is not None and result["screen_rate"] > 0

    def test_by_direction_has_entries(self, pbp_session):
        from api.coaching_pbp import compute_pass_detail
        result = compute_pass_detail(pbp_session, "KC", 2023)
        assert len(result["by_direction"]) > 0

    def test_direction_entry_has_stats(self, pbp_session):
        from api.coaching_pbp import compute_pass_detail
        result = compute_pass_detail(pbp_session, "KC", 2023)
        for entry in result["by_direction"].values():
            assert "plays" in entry
            assert "avg_epa" in entry
            assert "success_rate" in entry


class TestComputeDefenseScheme:
    def test_returns_dict(self, pbp_session):
        from api.coaching_pbp import compute_defense_scheme
        result = compute_defense_scheme(pbp_session, "KC", 2023)
        assert isinstance(result, dict)

    def test_has_required_keys(self, pbp_session):
        from api.coaching_pbp import compute_defense_scheme
        result = compute_defense_scheme(pbp_session, "KC", 2023)
        for key in ("blitz_rate", "avg_defenders_in_box", "sack_rate",
                    "qb_hit_rate", "personnel_breakdown"):
            assert key in result

    def test_sack_rate_detected(self, pbp_session):
        from api.coaching_pbp import compute_defense_scheme
        result = compute_defense_scheme(pbp_session, "KC", 2023)
        # play_id 22: DET vs KC defense, sack=1 (1 sack out of 3 plays)
        assert result["sack_rate"] is not None and result["sack_rate"] > 0

    def test_personnel_breakdown_populated(self, pbp_session):
        from api.coaching_pbp import compute_defense_scheme
        result = compute_defense_scheme(pbp_session, "KC", 2023)
        assert len(result["personnel_breakdown"]) > 0

    def test_blitz_rate_between_0_and_1(self, pbp_session):
        from api.coaching_pbp import compute_defense_scheme
        result = compute_defense_scheme(pbp_session, "KC", 2023)
        if result["blitz_rate"] is not None:
            assert 0 <= result["blitz_rate"] <= 1.0


class TestComputeStrengthsWeaknesses:
    def test_returns_dict_with_three_keys(self, pbp_session):
        from api.coaching_pbp import (
            compute_formation_breakdown, compute_personnel_breakdown,
            compute_run_scheme, compute_pass_detail,
            compute_defense_scheme, compute_strengths_weaknesses,
        )
        from api.coaches import _compute_offense_tendencies, _compute_defense_tendencies
        off = _compute_offense_tendencies(pbp_session, "KC", 2023)
        dfn = _compute_defense_tendencies(pbp_session, "KC", 2023)
        form = compute_formation_breakdown(pbp_session, "KC", 2023)
        runs = compute_run_scheme(pbp_session, "KC", 2023)
        passes = compute_pass_detail(pbp_session, "KC", 2023)
        def_sch = compute_defense_scheme(pbp_session, "KC", 2023)
        result = compute_strengths_weaknesses(off, dfn, form, runs, passes, def_sch)
        assert "strengths" in result
        assert "weaknesses" in result
        assert "tendencies" in result

    def test_all_lists(self, pbp_session):
        from api.coaching_pbp import (
            compute_formation_breakdown, compute_run_scheme, compute_pass_detail,
            compute_defense_scheme, compute_strengths_weaknesses,
        )
        from api.coaches import _compute_offense_tendencies, _compute_defense_tendencies
        off = _compute_offense_tendencies(pbp_session, "KC", 2023)
        dfn = _compute_defense_tendencies(pbp_session, "KC", 2023)
        result = compute_strengths_weaknesses(
            off, dfn,
            compute_formation_breakdown(pbp_session, "KC", 2023),
            compute_run_scheme(pbp_session, "KC", 2023),
            compute_pass_detail(pbp_session, "KC", 2023),
            compute_defense_scheme(pbp_session, "KC", 2023),
        )
        assert isinstance(result["strengths"], list)
        assert isinstance(result["weaknesses"], list)
        assert isinstance(result["tendencies"], list)

    def test_tendencies_mention_shotgun(self, pbp_session):
        from api.coaching_pbp import (
            compute_formation_breakdown, compute_run_scheme, compute_pass_detail,
            compute_defense_scheme, compute_strengths_weaknesses,
        )
        from api.coaches import _compute_offense_tendencies, _compute_defense_tendencies
        off = _compute_offense_tendencies(pbp_session, "KC", 2023)
        dfn = _compute_defense_tendencies(pbp_session, "KC", 2023)
        result = compute_strengths_weaknesses(
            off, dfn,
            compute_formation_breakdown(pbp_session, "KC", 2023),
            compute_run_scheme(pbp_session, "KC", 2023),
            compute_pass_detail(pbp_session, "KC", 2023),
            compute_defense_scheme(pbp_session, "KC", 2023),
        )
        tendencies_text = " ".join(result["tendencies"]).lower()
        assert "shotgun" in tendencies_text


# ── Integration: /coaches/{name}/breakdown endpoint ──────────────────────────

class TestCoachBreakdownEndpoint:
    def test_returns_200_for_known_coach(self, breakdown_client):
        resp = breakdown_client.get("/coaches/Andy%20Reid/breakdown?season=2023")
        assert resp.status_code == 200

    def test_status_success(self, breakdown_client):
        resp = breakdown_client.get("/coaches/Andy%20Reid/breakdown?season=2023")
        assert resp.json()["status"] == "success"

    def test_coach_field_correct(self, breakdown_client):
        resp = breakdown_client.get("/coaches/Andy%20Reid/breakdown?season=2023")
        assert resp.json()["coach"] == "Andy Reid"

    def test_data_is_list(self, breakdown_client):
        resp = breakdown_client.get("/coaches/Andy%20Reid/breakdown?season=2023")
        assert isinstance(resp.json()["data"], list)

    def test_data_has_one_entry_for_season(self, breakdown_client):
        resp = breakdown_client.get("/coaches/Andy%20Reid/breakdown?season=2023")
        assert len(resp.json()["data"]) == 1

    def test_entry_has_offense_and_defense(self, breakdown_client):
        entry = breakdown_client.get("/coaches/Andy%20Reid/breakdown?season=2023").json()["data"][0]
        assert "offense" in entry
        assert "defense" in entry

    def test_offense_has_formation_block(self, breakdown_client):
        entry = breakdown_client.get("/coaches/Andy%20Reid/breakdown?season=2023").json()["data"][0]
        assert "formation" in entry["offense"]
        assert "shotgun_rate" in entry["offense"]["formation"]

    def test_offense_has_personnel_block(self, breakdown_client):
        entry = breakdown_client.get("/coaches/Andy%20Reid/breakdown?season=2023").json()["data"][0]
        assert "personnel" in entry["offense"]
        assert isinstance(entry["offense"]["personnel"], dict)

    def test_offense_has_run_scheme(self, breakdown_client):
        entry = breakdown_client.get("/coaches/Andy%20Reid/breakdown?season=2023").json()["data"][0]
        assert "run_scheme" in entry["offense"]
        assert "total_runs" in entry["offense"]["run_scheme"]

    def test_offense_has_passing_block(self, breakdown_client):
        entry = breakdown_client.get("/coaches/Andy%20Reid/breakdown?season=2023").json()["data"][0]
        assert "passing" in entry["offense"]
        assert "avg_air_yards" in entry["offense"]["passing"]

    def test_defense_has_scheme_block(self, breakdown_client):
        entry = breakdown_client.get("/coaches/Andy%20Reid/breakdown?season=2023").json()["data"][0]
        assert "scheme" in entry["defense"]
        assert "sack_rate" in entry["defense"]["scheme"]

    def test_strengths_weaknesses_tendencies_present(self, breakdown_client):
        entry = breakdown_client.get("/coaches/Andy%20Reid/breakdown?season=2023").json()["data"][0]
        assert "strengths" in entry
        assert "weaknesses" in entry
        assert "tendencies" in entry
        assert isinstance(entry["strengths"], list)
        assert isinstance(entry["weaknesses"], list)
        assert isinstance(entry["tendencies"], list)

    def test_unknown_coach_returns_404(self, breakdown_client):
        resp = breakdown_client.get("/coaches/Nobody%20Here/breakdown?season=2023")
        assert resp.status_code == 404

    def test_dan_campbell_away_coach(self, breakdown_client):
        """Away coach should also be found."""
        resp = breakdown_client.get("/coaches/Dan%20Campbell/breakdown?season=2023")
        assert resp.status_code == 200
        assert resp.json()["data"][0]["team"] == "DET"

    def test_no_season_filter_returns_all(self, breakdown_client):
        """Without season filter, returns all seasons the coach appears in."""
        resp = breakdown_client.get("/coaches/Andy%20Reid/breakdown")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) >= 1


# ── Graceful no_data when PBP table absent ────────────────────────────────────

class TestBreakdownNoPBPTable:
    def test_breakdown_with_empty_db_returns_404(self, client):
        """With no schedule data, coach lookup returns 404."""
        resp = client.get("/coaches/Andy%20Reid/breakdown?season=2023")
        assert resp.status_code == 404


# ── Personnel / formation string parsing ─────────────────────────────────────

class TestPersonnelParsing:
    def test_parse_11(self):
        from api.coaching_pbp import _parse_offense_personnel
        assert "11" in _parse_offense_personnel("1 RB, 1 TE, 3 WR")

    def test_parse_12(self):
        from api.coaching_pbp import _parse_offense_personnel
        assert "12" in _parse_offense_personnel("1 RB, 2 TE, 2 WR")

    def test_parse_21(self):
        from api.coaching_pbp import _parse_offense_personnel
        assert "21" in _parse_offense_personnel("2 RB, 1 TE, 2 WR")

    def test_parse_22(self):
        from api.coaching_pbp import _parse_offense_personnel
        assert "22" in _parse_offense_personnel("2 RB, 2 TE, 1 WR")

    def test_parse_10(self):
        from api.coaching_pbp import _parse_offense_personnel
        assert "10" in _parse_offense_personnel("1 RB, 0 TE, 4 WR")

    def test_empty_string(self):
        from api.coaching_pbp import _parse_offense_personnel
        assert _parse_offense_personnel("") == "unknown"

    def test_parse_defense_base_43(self):
        from api.coaching_pbp import _parse_defense_personnel
        assert "4-3" in _parse_defense_personnel("4 DL, 3 LB, 4 DB")

    def test_parse_defense_base_34(self):
        from api.coaching_pbp import _parse_defense_personnel
        assert "3-4" in _parse_defense_personnel("3 DL, 4 LB, 4 DB")

    def test_parse_defense_nickel(self):
        from api.coaching_pbp import _parse_defense_personnel
        result = _parse_defense_personnel("4 DL, 2 LB, 5 DB")
        assert "Nickel" in result

    def test_parse_defense_dime(self):
        from api.coaching_pbp import _parse_defense_personnel
        result = _parse_defense_personnel("4 DL, 1 LB, 6 DB")
        assert "Dime" in result

    def test_parse_defense_quarter(self):
        from api.coaching_pbp import _parse_defense_personnel
        result = _parse_defense_personnel("3 DL, 1 LB, 7 DB")
        assert "Quarter" in result
