#!/usr/bin/env python3
"""
Tests for /coaches endpoints.
"""

from unittest.mock import patch


COACHING_UNAVAILABLE = {"player_grading": False, "coaching_analytics": False}
COACHING_AVAILABLE = {"player_grading": False, "coaching_analytics": True}


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

    def test_empty_grades_response(self, client, mock_coaching_analytics):
        mock_coaching_analytics.grade_coach_performance.return_value = {}
        with patch("api.coaches.check_grading_systems", return_value=COACHING_AVAILABLE), \
             patch("api.coaches.get_coaching_analytics", return_value=mock_coaching_analytics):
            body = client.get("/coaches/Andy Reid/grades").json()
        assert body["status"] == "success"
        assert body["data"] is None

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
