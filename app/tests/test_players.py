#!/usr/bin/env python3
"""
Tests for /players endpoints.
"""

import pandas as pd
from unittest.mock import patch, MagicMock


GRADING_UNAVAILABLE = {"player_grading": False, "coaching_analytics": False}
GRADING_AVAILABLE = {"player_grading": True, "coaching_analytics": False}


class TestGetRosters:
    def test_returns_200(self, client, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            response = client.get("/players/rosters")
        assert response.status_code == 200

    def test_status_success(self, client, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            body = client.get("/players/rosters").json()
        assert body["status"] == "success"

    def test_total_players_count(self, client, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            body = client.get("/players/rosters").json()
        assert body["total_players"] == 2

    def test_data_is_list(self, client, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            body = client.get("/players/rosters").json()
        assert isinstance(body["data"], list)

    def test_filter_by_team(self, client, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            body = client.get("/players/rosters?team=KC").json()
        assert body["total_players"] == 2

    def test_filter_by_position(self, client, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            body = client.get("/players/rosters?position=QB").json()
        assert body["total_players"] == 1

    def test_filter_by_week(self, client, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            body = client.get("/players/rosters?week=1").json()
        assert body["total_players"] == 2

    def test_filter_no_match(self, client, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            body = client.get("/players/rosters?position=K").json()
        assert body["total_players"] == 0

    def test_returns_500_on_error(self, client):
        with patch("api.players.nfl.import_weekly_rosters", side_effect=Exception("error")):
            response = client.get("/players/rosters")
        assert response.status_code == 500


class TestGetPlayerStats:
    def test_returns_200(self, client, sample_weekly_data_df):
        with patch("api.players.nfl.import_weekly_data", return_value=sample_weekly_data_df):
            response = client.get("/players/stats")
        assert response.status_code == 200

    def test_status_success(self, client, sample_weekly_data_df):
        with patch("api.players.nfl.import_weekly_data", return_value=sample_weekly_data_df):
            body = client.get("/players/stats").json()
        assert body["status"] == "success"

    def test_total_records_count(self, client, sample_weekly_data_df):
        with patch("api.players.nfl.import_weekly_data", return_value=sample_weekly_data_df):
            body = client.get("/players/stats").json()
        assert body["total_records"] == 2

    def test_filter_by_position(self, client, sample_weekly_data_df):
        with patch("api.players.nfl.import_weekly_data", return_value=sample_weekly_data_df):
            body = client.get("/players/stats?position=QB").json()
        assert body["total_records"] == 1

    def test_filter_by_team(self, client, sample_weekly_data_df):
        with patch("api.players.nfl.import_weekly_data", return_value=sample_weekly_data_df):
            body = client.get("/players/stats?team=KC").json()
        assert body["total_records"] == 2

    def test_filter_by_player_id(self, client, sample_weekly_data_df):
        with patch("api.players.nfl.import_weekly_data", return_value=sample_weekly_data_df):
            body = client.get("/players/stats?player_id=00-0033873").json()
        assert body["total_records"] == 1

    def test_filter_by_week(self, client, sample_weekly_data_df):
        with patch("api.players.nfl.import_weekly_data", return_value=sample_weekly_data_df):
            body = client.get("/players/stats?week=1").json()
        assert body["total_records"] == 2

    def test_returns_500_on_error(self, client):
        with patch("api.players.nfl.import_weekly_data", side_effect=Exception("error")):
            response = client.get("/players/stats")
        assert response.status_code == 500


class TestGetPlayerDetails:
    def test_returns_200_when_found(self, client, sample_weekly_data_df, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_data", return_value=sample_weekly_data_df), \
             patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            response = client.get("/players/00-0033873")
        assert response.status_code == 200

    def test_status_success(self, client, sample_weekly_data_df, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_data", return_value=sample_weekly_data_df), \
             patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            body = client.get("/players/00-0033873").json()
        assert body["status"] == "success"

    def test_player_info_present(self, client, sample_weekly_data_df, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_data", return_value=sample_weekly_data_df), \
             patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            body = client.get("/players/00-0033873").json()
        assert "player_info" in body
        assert "season_stats" in body
        assert "games_played" in body

    def test_games_played_count(self, client, sample_weekly_data_df, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_data", return_value=sample_weekly_data_df), \
             patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            body = client.get("/players/00-0033873").json()
        assert body["games_played"] == 1

    def test_missing_player_returns_404(self, client, sample_weekly_data_df, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_data", return_value=sample_weekly_data_df), \
             patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            response = client.get("/players/NONEXISTENT")
        assert response.status_code == 404

    def test_returns_500_on_error(self, client):
        with patch("api.players.nfl.import_weekly_data", side_effect=Exception("error")):
            response = client.get("/players/00-0033873")
        assert response.status_code == 500


class TestGetPlayerGrades:
    def test_returns_503_when_grading_unavailable(self, client):
        with patch("api.players.check_grading_systems", return_value=GRADING_UNAVAILABLE):
            response = client.get("/players/grades")
        assert response.status_code == 503

    def test_returns_200_when_grading_available(self, client, sample_grades_df):
        mock_grader = MagicMock()
        mock_grader.calculate_all_grades.return_value = sample_grades_df
        mock_grader.identify_performance_outliers.return_value = sample_grades_df
        mock_grader.get_top_performers.return_value = sample_grades_df

        with patch("api.players.check_grading_systems", return_value=GRADING_AVAILABLE), \
             patch("api.players.get_player_grader", return_value=mock_grader):
            response = client.get("/players/grades")
        assert response.status_code == 200

    def test_status_success(self, client, sample_grades_df):
        mock_grader = MagicMock()
        mock_grader.calculate_all_grades.return_value = sample_grades_df
        mock_grader.identify_performance_outliers.return_value = sample_grades_df
        mock_grader.get_top_performers.return_value = sample_grades_df

        with patch("api.players.check_grading_systems", return_value=GRADING_AVAILABLE), \
             patch("api.players.get_player_grader", return_value=mock_grader):
            body = client.get("/players/grades").json()
        assert body["status"] == "success"

    def test_data_is_list(self, client, sample_grades_df):
        mock_grader = MagicMock()
        mock_grader.calculate_all_grades.return_value = sample_grades_df
        mock_grader.identify_performance_outliers.return_value = sample_grades_df
        mock_grader.get_top_performers.return_value = sample_grades_df

        with patch("api.players.check_grading_systems", return_value=GRADING_AVAILABLE), \
             patch("api.players.get_player_grader", return_value=mock_grader):
            body = client.get("/players/grades").json()
        assert isinstance(body["data"], list)

    def test_dict_return_from_grader(self, client, sample_grades_df):
        """Grader may return a dict of DataFrames; endpoint must handle it."""
        mock_grader = MagicMock()
        mock_grader.calculate_all_grades.return_value = {"QB": sample_grades_df}

        with patch("api.players.check_grading_systems", return_value=GRADING_AVAILABLE), \
             patch("api.players.get_player_grader", return_value=mock_grader):
            response = client.get("/players/grades")
        assert response.status_code == 200

    def test_empty_dict_from_grader(self, client):
        mock_grader = MagicMock()
        mock_grader.calculate_all_grades.return_value = {}

        with patch("api.players.check_grading_systems", return_value=GRADING_AVAILABLE), \
             patch("api.players.get_player_grader", return_value=mock_grader):
            body = client.get("/players/grades").json()
        assert body["status"] == "success"
        assert body["data"] == []

    def test_returns_500_on_grader_error(self, client):
        with patch("api.players.check_grading_systems", return_value=GRADING_AVAILABLE), \
             patch("api.players.get_player_grader", side_effect=Exception("grader error")):
            response = client.get("/players/grades")
        assert response.status_code == 500


class TestGetPlayerGradeDetails:
    def test_returns_503_when_grading_unavailable(self, client):
        with patch("api.players.check_grading_systems", return_value=GRADING_UNAVAILABLE):
            response = client.get("/players/grades/Patrick Mahomes")
        assert response.status_code == 503

    def test_returns_200_when_player_found(self, client, sample_grades_df):
        mock_grader = MagicMock()
        mock_grader.calculate_all_grades.return_value = sample_grades_df
        mock_grader.identify_performance_outliers.return_value = sample_grades_df

        with patch("api.players.check_grading_systems", return_value=GRADING_AVAILABLE), \
             patch("api.players.get_player_grader", return_value=mock_grader):
            response = client.get("/players/grades/Mahomes")
        assert response.status_code == 200

    def test_returns_404_when_player_not_found(self, client, sample_grades_df):
        mock_grader = MagicMock()
        mock_grader.calculate_all_grades.return_value = sample_grades_df
        mock_grader.identify_performance_outliers.return_value = sample_grades_df

        with patch("api.players.check_grading_systems", return_value=GRADING_AVAILABLE), \
             patch("api.players.get_player_grader", return_value=mock_grader):
            response = client.get("/players/grades/UnknownPlayerXYZ")
        assert response.status_code == 404

    def test_dict_grades_player_found(self, client, sample_grades_df):
        mock_grader = MagicMock()
        mock_grader.calculate_all_grades.return_value = {"QB": sample_grades_df}

        with patch("api.players.check_grading_systems", return_value=GRADING_AVAILABLE), \
             patch("api.players.get_player_grader", return_value=mock_grader):
            response = client.get("/players/grades/Mahomes")
        assert response.status_code == 200

    def test_dict_grades_empty_returns_404(self, client):
        mock_grader = MagicMock()
        mock_grader.calculate_all_grades.return_value = {"QB": pd.DataFrame()}

        with patch("api.players.check_grading_systems", return_value=GRADING_AVAILABLE), \
             patch("api.players.get_player_grader", return_value=mock_grader):
            response = client.get("/players/grades/Mahomes")
        assert response.status_code == 404
