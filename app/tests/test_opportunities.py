#!/usr/bin/env python3
"""
Tests for the /opportunities endpoints.

Uses a dedicated in-memory SQLite engine with:
  - ORM tables (player_rosters seeded for enrichment)
  - a play_by_play table (created via pandas to_sql)

Also unit-tests the pure compute helpers (no DB) so the ported Celery math is
pinned down independently of the HTTP layer.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.opportunities import (
    process_week_opportunities,
    compute_opportunities,
    add_roster_info,
    calculate_opportunity_trends,
)


# ── Sample PBP ────────────────────────────────────────────────────────────────

SAMPLE_PBP = [
    # Week 1 — KC
    {"season": 2024, "season_type": "REG", "week": 1, "posteam": "KC",
     "play_type": "pass", "down": 1, "yardline_100": 15, "air_yards": 25.0,
     "receiver_player_id": "REC_A", "rusher_player_id": None},
    {"season": 2024, "season_type": "REG", "week": 1, "posteam": "KC",
     "play_type": "pass", "down": 3, "yardline_100": 8, "air_yards": 5.0,
     "receiver_player_id": "REC_A", "rusher_player_id": None},
    {"season": 2024, "season_type": "REG", "week": 1, "posteam": "KC",
     "play_type": "run", "down": 1, "yardline_100": 3, "air_yards": None,
     "receiver_player_id": None, "rusher_player_id": "RUSH_B"},
    {"season": 2024, "season_type": "REG", "week": 1, "posteam": "KC",
     "play_type": "run", "down": 2, "yardline_100": 50, "air_yards": None,
     "receiver_player_id": None, "rusher_player_id": "RUSH_B"},
    # Week 2 — KC (gives REC_A a second week for trends)
    {"season": 2024, "season_type": "REG", "week": 2, "posteam": "KC",
     "play_type": "pass", "down": 2, "yardline_100": 40, "air_yards": 12.0,
     "receiver_player_id": "REC_A", "rusher_player_id": None},
    # Postseason play — must be excluded by the REG filter
    {"season": 2024, "season_type": "POST", "week": 20, "posteam": "KC",
     "play_type": "pass", "down": 1, "yardline_100": 10, "air_yards": 30.0,
     "receiver_player_id": "REC_A", "rusher_player_id": None},
]


# ── Unit tests: process_week_opportunities ────────────────────────────────────

class TestProcessWeek:
    def test_receiver_situational_counts(self):
        week1 = pd.DataFrame([p for p in SAMPLE_PBP if p["week"] == 1])
        recs = {r["player_id"]: r for r in process_week_opportunities(week1, 1, 2024)}
        a = recs["REC_A"]
        assert a["targets"] == 2
        assert a["touches"] == 2
        assert a["air_yards"] == 30.0
        assert a["red_zone_targets"] == 2      # yl 15 and 8 both <= 20
        assert a["end_zone_targets"] == 1      # yl 8 <= 10
        assert a["goal_line_touches"] == 1     # from the end-zone target
        assert a["third_down_targets"] == 1    # the down==3 pass
        assert a["deep_targets"] == 1          # air 25 >= 20
        assert a["short_targets"] == 1         # air 5 < 10
        assert a["target_share"] == 100.0      # 2 of 2 team targets

    def test_rusher_situational_counts(self):
        week1 = pd.DataFrame([p for p in SAMPLE_PBP if p["week"] == 1])
        recs = {r["player_id"]: r for r in process_week_opportunities(week1, 1, 2024)}
        b = recs["RUSH_B"]
        assert b["carries"] == 2
        assert b["touches"] == 2
        assert b["red_zone_carries"] == 1      # yl 3 <= 20 (yl 50 no)
        assert b["goal_line_carries"] == 1     # yl 3 <= 5
        assert b["goal_line_touches"] == 1
        assert b["carry_share"] == 100.0


# ── Unit tests: trends ────────────────────────────────────────────────────────

class TestTrends:
    def _enriched(self):
        pbp = pd.DataFrame(SAMPLE_PBP)
        pbp = pbp[pbp["season_type"] == "REG"]
        opp = compute_opportunities(pbp, 2024)
        roster = pd.DataFrame([
            {"player_id": "REC_A", "player_name": "Receiver A", "position": "WR", "team": "KC"},
            {"player_id": "RUSH_B", "player_name": "Rusher B", "position": "RB", "team": "KC"},
        ])
        return add_roster_info(opp, roster)

    def test_min_weeks_filter(self):
        trends = calculate_opportunity_trends(self._enriched())
        # RUSH_B only has 1 week -> excluded; REC_A has 2 weeks
        assert set(trends["player_id"]) == {"REC_A"}

    def test_trend_and_consistency_values(self):
        trends = calculate_opportunity_trends(self._enriched())
        row = trends[trends["player_id"] == "REC_A"].iloc[0]
        assert row["targets_avg"] == pytest.approx(1.5)
        assert row["targets_latest"] == 1
        assert row["targets_max"] == 2
        # 2 weeks: recent=last(1), early=first(2) -> (1-2)/2*100 = -50
        assert row["targets_trend"] == pytest.approx(-50.0)
        # cv = std([2,1]) / mean([2,1]) * 100 = 0.5/1.5*100
        assert row["targets_consistency"] == pytest.approx(0.5 / 1.5 * 100)

    def test_roster_enrichment_position(self):
        enriched = self._enriched()
        rec = enriched[enriched["player_id"] == "REC_A"].iloc[0]
        assert rec["position"] == "WR"
        assert rec["team"] == "KC"
        assert rec["player_display_name"] == "Receiver A"


# ── HTTP fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def _opp_engine():
    from database.models import Base
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    df = pd.DataFrame(SAMPLE_PBP)
    with engine.connect() as conn:
        df.to_sql("play_by_play", con=conn, if_exists="replace", index=False)
        conn.commit()
    return engine


@pytest.fixture(scope="module")
def opp_session(_opp_engine):
    from database.models import PlayerRoster
    Session = sessionmaker(bind=_opp_engine)
    session = Session()
    for pid, name, pos in [("REC_A", "Receiver A", "WR"), ("RUSH_B", "Rusher B", "RB")]:
        session.merge(PlayerRoster(
            player_id=pid, season=2024, week=1,
            player_name=name, player_display_name=name, position=pos, team="KC",
        ))
    session.commit()
    yield session
    session.close()


@pytest.fixture(scope="module")
def opp_client(_opp_engine, opp_session):
    from main import app
    from database.session import get_db

    def _override():
        yield opp_session

    app.dependency_overrides[get_db] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── HTTP tests ────────────────────────────────────────────────────────────────

class TestOpportunitiesEndpoint:
    def test_all_records(self, opp_client):
        body = opp_client.get("/opportunities/?season=2024").json()
        assert body["status"] == "success"
        # REC_A (2 weeks) + RUSH_B (1 week) = 3 player-week rows
        assert body["total_records"] == 3

    def test_week_filter(self, opp_client):
        body = opp_client.get("/opportunities/?season=2024&week=1").json()
        assert body["total_records"] == 2  # REC_A + RUSH_B in week 1

    def test_team_filter(self, opp_client):
        body = opp_client.get("/opportunities/?season=2024&team=KC").json()
        assert body["total_records"] == 3
        empty = opp_client.get("/opportunities/?season=2024&team=SF").json()
        assert empty["status"] == "no_data"

    def test_row_shape(self, opp_client):
        body = opp_client.get("/opportunities/?season=2024&week=1").json()
        rec = next(r for r in body["data"] if r["player_id"] == "REC_A")
        for key in ("player_name", "position", "team", "season", "week",
                    "targets", "target_share", "deep_targets", "goal_line_touches"):
            assert key in rec
        assert rec["position"] == "WR"


class TestTrendsEndpoint:
    def test_trends(self, opp_client):
        body = opp_client.get("/opportunities/trends/?season=2024").json()
        assert body["status"] == "success"
        assert body["total_players"] == 1  # only REC_A has >= 2 weeks
        assert body["data"][0]["player_id"] == "REC_A"
        assert body["data"][0]["targets_trend"] == pytest.approx(-50.0)
