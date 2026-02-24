#!/usr/bin/env python3
"""
Tests for /schedules endpoints.
"""

from unittest.mock import patch


class TestGetSchedules:
    def test_returns_200(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            response = client.get("/schedules/")
        assert response.status_code == 200

    def test_status_success(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            body = client.get("/schedules/").json()
        assert body["status"] == "success"

    def test_returns_all_games_by_default(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            body = client.get("/schedules/").json()
        assert body["total_games"] == 3

    def test_data_is_list(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            body = client.get("/schedules/").json()
        assert isinstance(body["data"], list)

    def test_season_in_response(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            body = client.get("/schedules/?season=2023").json()
        assert body["season"] == 2023

    def test_filter_by_week(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            body = client.get("/schedules/?week=1").json()
        assert body["total_games"] == 2

    def test_filter_by_team(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            body = client.get("/schedules/?team=KC").json()
        assert body["total_games"] == 2

    def test_filter_by_team_case_insensitive(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            body = client.get("/schedules/?team=kc").json()
        assert body["total_games"] == 2

    def test_filter_week_and_team(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            body = client.get("/schedules/?week=1&team=SF").json()
        assert body["total_games"] == 1

    def test_returns_500_on_error(self, client):
        with patch("api.schedules.nfl.load_schedules", side_effect=Exception("error")):
            response = client.get("/schedules/")
        assert response.status_code == 500


class TestGetWeeklySchedule:
    def test_returns_200(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            response = client.get("/schedules/2023/week/1")
        assert response.status_code == 200

    def test_status_success(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            body = client.get("/schedules/2023/week/1").json()
        assert body["status"] == "success"

    def test_correct_season_and_week(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            body = client.get("/schedules/2023/week/1").json()
        assert body["season"] == 2023
        assert body["week"] == 1

    def test_games_filtered_to_week(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            body = client.get("/schedules/2023/week/1").json()
        assert len(body["games"]) == 2

    def test_empty_week(self, client, sample_schedules_df):
        with patch("api.schedules.nfl.load_schedules", return_value=sample_schedules_df):
            body = client.get("/schedules/2023/week/99").json()
        assert body["games"] == []

    def test_returns_500_on_error(self, client):
        with patch("api.schedules.nfl.load_schedules", side_effect=Exception("error")):
            response = client.get("/schedules/2023/week/1")
        assert response.status_code == 500
