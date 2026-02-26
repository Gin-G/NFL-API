#!/usr/bin/env python3
"""
Tests for /coaches endpoints.
"""

from unittest.mock import patch, MagicMock


COACHING_UNAVAILABLE = {"player_grading": False, "coaching_analytics": False}
COACHING_AVAILABLE = {"player_grading": False, "coaching_analytics": True}


class TestCoachTendencies:
    def test_unknown_coach_returns_404(self, client):
        """Coach not in schedule DB → 404."""
        response = client.get("/coaches/Unknown%20Coach/tendencies")
        assert response.status_code == 404

    def test_unknown_coach_with_season_returns_404(self, client):
        """Coach not found even with season filter → 404."""
        response = client.get("/coaches/Nobody%20Here/tendencies?season=2024")
        assert response.status_code == 404

    def test_tendencies_url_does_not_crash(self, client):
        """Endpoint is registered and reachable (even if coach not found)."""
        response = client.get("/coaches/Andy%20Reid/tendencies?season=2024")
        # Either 404 (coach not in empty test DB) or 200 with data
        assert response.status_code in (200, 404)


class TestGetCoaches:
    def test_returns_503_when_unavailable(self, client):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_UNAVAILABLE):
            response = client.get("/coaches/")
        assert response.status_code == 503

    def test_returns_200_when_available(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            response = client.get("/coaches/")
        assert response.status_code == 200

    def test_status_success(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            body = client.get("/coaches/").json()
        assert body["status"] == "success"

    def test_total_coaches(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            body = client.get("/coaches/").json()
        assert body["total_coaches"] == 2

    def test_data_is_list(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            body = client.get("/coaches/").json()
        assert isinstance(body["data"], list)

    def test_coach_has_name_and_seasons(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            body = client.get("/coaches/").json()
        coach = body["data"][0]
        assert "name" in coach
        assert "seasons" in coach

    def test_filter_by_season(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            response = client.get("/coaches/?season=2023")
        assert response.status_code == 200

    def test_returns_500_on_error(self, client):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", side_effect=Exception("error")):
            response = client.get("/coaches/")
        assert response.status_code == 500


class TestGetCoachAnalysis:
    def test_returns_503_when_unavailable(self, client):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_UNAVAILABLE):
            response = client.get("/coaches/Andy Reid/analysis")
        assert response.status_code == 503

    def test_returns_200_for_existing_coach(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            response = client.get("/coaches/Andy Reid/analysis")
        assert response.status_code == 200

    def test_status_success(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            body = client.get("/coaches/Andy Reid/analysis").json()
        assert body["status"] == "success"

    def test_coach_name_in_response(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            body = client.get("/coaches/Andy Reid/analysis").json()
        assert body["coach"] == "Andy Reid"

    def test_has_offensive_and_defensive_analysis(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            body = client.get("/coaches/Andy Reid/analysis").json()
        assert "offensive_analysis" in body
        assert "defensive_analysis" in body

    def test_returns_404_for_missing_coach(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            response = client.get("/coaches/Unknown Coach/analysis")
        assert response.status_code == 404

    def test_returns_500_on_error(self, client):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", side_effect=Exception("error")):
            response = client.get("/coaches/Andy Reid/analysis")
        assert response.status_code == 500


class TestGetCoachGrades:
    def test_returns_503_when_unavailable(self, client):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_UNAVAILABLE):
            response = client.get("/coaches/Andy Reid/grades")
        assert response.status_code == 503

    def test_returns_200_for_existing_coach(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            response = client.get("/coaches/Andy Reid/grades")
        assert response.status_code == 200

    def test_status_success(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            body = client.get("/coaches/Andy Reid/grades").json()
        assert body["status"] == "success"

    def test_has_grades_key(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            body = client.get("/coaches/Andy Reid/grades").json()
        assert "grades" in body

    def test_returns_404_for_missing_coach(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            response = client.get("/coaches/Unknown Coach/grades")
        assert response.status_code == 404

    def test_empty_grades_when_no_season_data(self, client):
        """Grades is None when a coach has no entries in coaching_data."""
        analytics = MagicMock()
        analytics.get_available_coaches.return_value = ["Andy Reid"]
        analytics.coaching_data = {}  # coach present in list but no data
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=analytics):
            body = client.get("/coaches/Andy Reid/grades").json()
        assert body["status"] == "success"
        assert body["grades"] is None

    def test_returns_500_on_error(self, client):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", side_effect=Exception("error")):
            response = client.get("/coaches/Andy Reid/grades")
        assert response.status_code == 500


class TestCompareCoaches:
    def test_returns_503_when_unavailable(self, client):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_UNAVAILABLE):
            response = client.post("/coaches/compare", json=["Andy Reid", "Kyle Shanahan"])
        assert response.status_code == 503

    def test_returns_200_for_valid_coaches(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            response = client.post(
                "/coaches/compare",
                json=["Andy Reid", "Kyle Shanahan"],
            )
        assert response.status_code == 200

    def test_status_success(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            body = client.post(
                "/coaches/compare",
                json=["Andy Reid", "Kyle Shanahan"],
            ).json()
        assert body["status"] == "success"

    def test_has_comparison_matrix(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            body = client.post(
                "/coaches/compare",
                json=["Andy Reid", "Kyle Shanahan"],
            ).json()
        assert "comparison_matrix" in body

    def test_coaches_list_in_response(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            body = client.post(
                "/coaches/compare",
                json=["Andy Reid", "Kyle Shanahan"],
            ).json()
        assert body["coaches"] == ["Andy Reid", "Kyle Shanahan"]

    def test_returns_404_for_invalid_coach(self, client, mock_coaching_analytics):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            response = client.post(
                "/coaches/compare",
                json=["Andy Reid", "Unknown Coach"],
            )
        assert response.status_code == 404

    def test_returns_500_on_error(self, client):
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", side_effect=Exception("error")):
            response = client.post("/coaches/compare", json=["Andy Reid"])
        assert response.status_code == 500


class TestMultiYearCoaches:
    """Full career history: coaches with seasons across multiple years."""

    def _make_multi_year_analytics(self):
        from unittest.mock import MagicMock
        analytics = MagicMock()
        # Coach with 4-year career history
        analytics.get_available_coaches.return_value = ["Andy Reid"]
        analytics.coaching_data = {
            ("Andy Reid", 2020): {
                "teams": {"KC"},
                "games": [{"result": "W"}] * 14 + [{"result": "L"}] * 2,
            },
            ("Andy Reid", 2021): {
                "teams": {"KC"},
                "games": [{"result": "W"}] * 12 + [{"result": "L"}] * 5,
            },
            ("Andy Reid", 2022): {
                "teams": {"KC"},
                "games": [{"result": "W"}] * 14 + [{"result": "L"}] * 3,
            },
            ("Andy Reid", 2023): {
                "teams": {"KC"},
                "games": [{"result": "W"}] * 11 + [{"result": "L"}] * 6,
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
        analytics.get_letter_grade.return_value = "B"
        return analytics

    def test_coach_has_all_seasons(self, client):
        analytics = self._make_multi_year_analytics()
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=analytics):
            body = client.get("/coaches/").json()
        coach = body["data"][0]
        assert coach["name"] == "Andy Reid"
        assert len(coach["seasons"]) == 4

    def test_seasons_are_sorted_ascending(self, client):
        analytics = self._make_multi_year_analytics()
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=analytics):
            body = client.get("/coaches/").json()
        seasons = [s["season"] for s in body["data"][0]["seasons"]]
        assert seasons == sorted(seasons)

    def test_win_loss_record_per_season(self, client):
        analytics = self._make_multi_year_analytics()
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=analytics):
            body = client.get("/coaches/").json()
        season_2020 = next(s for s in body["data"][0]["seasons"] if s["season"] == 2020)
        assert season_2020["wins"] == 14
        assert season_2020["losses"] == 2
        assert season_2020["record"] == "14-2"

    def test_analysis_returns_all_seasons(self, client):
        analytics = self._make_multi_year_analytics()
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=analytics):
            body = client.get("/coaches/Andy Reid/analysis").json()
        assert len(body["seasons"]) == 4

    def test_grades_returns_all_seasons(self, client):
        analytics = self._make_multi_year_analytics()
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=analytics):
            body = client.get("/coaches/Andy Reid/grades").json()
        assert len(body["grades"]) == 4


class TestCoachesIsActive:
    """DB-path: coaches list includes is_active field, active/legacy split is correct."""

    def _populate(self, db_session):
        from database.models import Schedule
        # 2024 coach — will be active (max season in DB)
        db_session.add(Schedule(
            game_id="2024_01_KC_BAL_ia", season=2024, week=1,
            home_team="KC", away_team="BAL",
            home_score=27, away_score=20,
            home_coach="Andy Reid", away_coach="John Harbaugh",
        ))
        # 2020-only coach — will be legacy
        db_session.add(Schedule(
            game_id="2020_01_HOU_KC_ia", season=2020, week=1,
            home_team="HOU", away_team="KC",
            home_score=20, away_score=34,
            home_coach="Bill O'Brien", away_coach=None,
        ))
        db_session.flush()

    def test_coaches_list_has_is_active_field(self, client, db_session):
        self._populate(db_session)
        body = client.get("/coaches/").json()
        assert body["status"] == "success"
        for coach in body["data"]:
            assert "is_active" in coach

    def test_active_coach_has_is_active_true(self, client, db_session):
        self._populate(db_session)
        body = client.get("/coaches/").json()
        andy = next(c for c in body["data"] if c["name"] == "Andy Reid")
        assert andy["is_active"] is True

    def test_legacy_coach_has_is_active_false(self, client, db_session):
        self._populate(db_session)
        body = client.get("/coaches/").json()
        obrien = next(c for c in body["data"] if c["name"] == "Bill O'Brien")
        assert obrien["is_active"] is False


class TestBreakdownCache:
    """Breakdown endpoint returns from DB cache when CoachSeasonAnalytics is populated."""

    def _setup(self, db_session):
        import json
        from database.models import CoachSeasonAnalytics, Schedule
        db_session.add(Schedule(
            game_id="2024_01_KC_BAL_bc", season=2024, week=1,
            home_team="KC", away_team="BAL",
            home_score=27, away_score=20,
            home_coach="Andy Reid", away_coach="John Harbaugh",
        ))
        blob = {
            "season": 2024, "team": "KC",
            "offense": {
                "total_plays": 60, "pass_rate": 0.6,
                "pass_rate_by_down": {}, "avg_epa_per_play": 0.12,
                "red_zone_pass_rate": None, "fourth_down_attempts": 3,
                "fourth_down_conversion_rate": None, "two_point_attempts": 0,
                "two_point_success_rate": None, "third_down_conversion_rate": 0.45,
                "formation": {"shotgun_rate": 0.6, "no_huddle_rate": 0.05,
                              "play_action_rate": 0.2, "formation_breakdown": {}},
                "personnel": {},
                "run_scheme": {"total_runs": 24, "inside_rate": 0.5,
                               "outside_rate": 0.5, "by_location": {}},
                "passing": {"avg_air_yards": 8.0, "deep_pass_rate": 0.15,
                            "screen_rate": 0.1, "intermediate_rate": 0.4,
                            "scramble_rate": 0.03, "dropback_epa": 0.2,
                            "avg_yac": 5.0, "by_direction": {}},
                "fourth_down_sample": [],
                "third_down_sample": [],
            },
            "defense": {
                "total_plays": 58, "avg_epa_allowed_per_play": -0.05,
                "third_down_stop_rate": 0.6, "red_zone_td_rate_allowed": 0.3,
                "sack_rate": 0.08,
                "scheme": {"blitz_rate": 0.25, "avg_defenders_in_box": 6.5,
                           "blitz_epa_allowed": 0.1, "non_blitz_epa_allowed": -0.1,
                           "qb_hit_rate": 0.12, "sack_rate": 0.08,
                           "personnel_breakdown": {}},
            },
            "strengths": ["Pass-heavy offense"],
            "weaknesses": [],
            "tendencies": ["Shotgun-heavy"],
        }
        db_session.add(CoachSeasonAnalytics(
            coach_name="Andy Reid", season=2024, team="KC",
            breakdown_json=json.dumps(blob),
        ))
        db_session.flush()

    def test_breakdown_returns_200_from_cache(self, client, db_session):
        self._setup(db_session)
        response = client.get("/coaches/Andy%20Reid/breakdown?season=2024")
        assert response.status_code == 200

    def test_breakdown_status_success_from_cache(self, client, db_session):
        self._setup(db_session)
        body = client.get("/coaches/Andy%20Reid/breakdown?season=2024").json()
        assert body["status"] == "success"

    def test_breakdown_data_has_correct_team_and_season(self, client, db_session):
        self._setup(db_session)
        body = client.get("/coaches/Andy%20Reid/breakdown?season=2024").json()
        assert len(body["data"]) == 1
        assert body["data"][0]["team"] == "KC"
        assert body["data"][0]["season"] == 2024

    def test_breakdown_cached_strengths_returned(self, client, db_session):
        self._setup(db_session)
        body = client.get("/coaches/Andy%20Reid/breakdown?season=2024").json()
        assert body["data"][0]["strengths"] == ["Pass-heavy offense"]
