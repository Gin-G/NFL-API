#!/usr/bin/env python3
"""
Shared test fixtures and configuration.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# Ensure app directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_teams_df():
    return pd.DataFrame([
        {
            "team_abbr": "KC", "team_name": "Kansas City Chiefs",
            "team_conf": "AFC", "team_division": "AFC West",
            "team_color": "#E31837", "team_color2": "#FFB81C",
        },
        {
            "team_abbr": "SF", "team_name": "San Francisco 49ers",
            "team_conf": "NFC", "team_division": "NFC West",
            "team_color": "#AA0000", "team_color2": "#B3995D",
        },
    ])


@pytest.fixture
def sample_schedules_df():
    return pd.DataFrame([
        {
            "season": 2023, "week": 1,
            "home_team": "KC", "away_team": "DET",
            "home_score": 21, "away_score": 20,
        },
        {
            "season": 2023, "week": 1,
            "home_team": "SF", "away_team": "PIT",
            "home_score": 30, "away_score": 7,
        },
        {
            "season": 2023, "week": 2,
            "home_team": "KC", "away_team": "JAX",
            "home_score": 17, "away_score": 9,
        },
    ])


@pytest.fixture
def sample_rosters_df():
    return pd.DataFrame([
        {
            "player_id": "00-0033873", "player_name": "Patrick Mahomes",
            "position": "QB", "team": "KC", "week": 1, "season": 2023,
            "height": "6-3", "weight": 230, "college": "Texas Tech",
            "rookie_year": 2017,
        },
        {
            "player_id": "00-0031280", "player_name": "Travis Kelce",
            "position": "TE", "team": "KC", "week": 1, "season": 2023,
            "height": "6-5", "weight": 256, "college": "Cincinnati",
            "rookie_year": 2013,
        },
    ])


@pytest.fixture
def sample_weekly_data_df():
    return pd.DataFrame([
        {
            "player_id": "00-0033873", "player_display_name": "Patrick Mahomes",
            "position": "QB", "recent_team": "KC", "week": 1, "season": 2023,
            "passing_yards": 315, "passing_tds": 3, "interceptions": 0,
            "rushing_yards": 10, "fantasy_points": 35.5,
        },
        {
            "player_id": "00-0031280", "player_display_name": "Travis Kelce",
            "position": "TE", "recent_team": "KC", "week": 1, "season": 2023,
            "receiving_yards": 124, "receiving_tds": 1, "receptions": 9,
            "fantasy_points": 21.4,
        },
    ])


@pytest.fixture
def sample_grades_df():
    return pd.DataFrame([
        {
            "player_name": "Patrick Mahomes", "position": "QB",
            "numeric_grade": 96.5, "letter_grade": "A+",
            "games_played": 17, "grade_category": "QB",
        },
        {
            "player_name": "Travis Kelce", "position": "TE",
            "numeric_grade": 91.2, "letter_grade": "A",
            "games_played": 15, "grade_category": "TE",
        },
    ])


# ---------------------------------------------------------------------------
# Mock coaching analytics
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_coaching_analytics():
    analytics = MagicMock()
    analytics.get_available_coaches.return_value = ["Andy Reid", "Kyle Shanahan"]
    analytics.coaching_data = {
        ("Andy Reid", 2023): {
            "teams": {"KC"},
            "games": [
                {"result": "W"}, {"result": "W"}, {"result": "L"},
                {"result": "W"}, {"result": None},
            ],
        },
        ("Kyle Shanahan", 2023): {
            "teams": {"SF"},
            "games": [
                {"result": "W"}, {"result": "W"}, {"result": "W"},
            ],
        },
    }
    analytics.analyze_roster_quality.return_value = {
        "overall_avg_grade": 75.0,
        "roster_tier": "Good",
        "qb_avg_grade": 85.0,
        "rb_avg_grade": 72.0,
        "wr_te_avg_grade": 78.0,
        "defense_avg_grade": 70.0,
    }
    analytics.get_letter_grade.side_effect = lambda score: (
        "A+" if score >= 95 else "A" if score >= 90 else "B+" if score >= 85
        else "B" if score >= 75 else "C"
    )
    return analytics


# ---------------------------------------------------------------------------
# Test client
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from main import app
    with TestClient(app) as c:
        yield c
