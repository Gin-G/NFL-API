#!/usr/bin/env python3
"""
Tests for /teams endpoints.
"""

from unittest.mock import patch


class TestGetTeams:
    def test_returns_200(self, client, sample_teams_df):
        with patch("api.teams.nfl.import_team_desc", return_value=sample_teams_df):
            response = client.get("/teams/")
        assert response.status_code == 200

    def test_status_success(self, client, sample_teams_df):
        with patch("api.teams.nfl.import_team_desc", return_value=sample_teams_df):
            body = client.get("/teams/").json()
        assert body["status"] == "success"

    def test_total_teams_count(self, client, sample_teams_df):
        with patch("api.teams.nfl.import_team_desc", return_value=sample_teams_df):
            body = client.get("/teams/").json()
        assert body["total_teams"] == 2

    def test_data_is_list(self, client, sample_teams_df):
        with patch("api.teams.nfl.import_team_desc", return_value=sample_teams_df):
            body = client.get("/teams/").json()
        assert isinstance(body["data"], list)

    def test_data_contains_team_fields(self, client, sample_teams_df):
        with patch("api.teams.nfl.import_team_desc", return_value=sample_teams_df):
            body = client.get("/teams/").json()
        team = body["data"][0]
        assert "team_abbr" in team
        assert "team_name" in team

    def test_returns_500_on_error(self, client):
        with patch("api.teams.nfl.import_team_desc", side_effect=Exception("db error")):
            response = client.get("/teams/")
        assert response.status_code == 500


class TestGetTeamDetails:
    def test_existing_team_returns_200(self, client, sample_teams_df):
        with patch("api.teams.nfl.import_team_desc", return_value=sample_teams_df):
            response = client.get("/teams/KC")
        assert response.status_code == 200

    def test_existing_team_data(self, client, sample_teams_df):
        with patch("api.teams.nfl.import_team_desc", return_value=sample_teams_df):
            body = client.get("/teams/KC").json()
        assert body["status"] == "success"
        assert body["data"]["team_abbr"] == "KC"

    def test_team_abbr_case_insensitive(self, client, sample_teams_df):
        with patch("api.teams.nfl.import_team_desc", return_value=sample_teams_df):
            body = client.get("/teams/kc").json()
        assert body["data"]["team_abbr"] == "KC"

    def test_missing_team_returns_404(self, client, sample_teams_df):
        with patch("api.teams.nfl.import_team_desc", return_value=sample_teams_df):
            response = client.get("/teams/XXX")
        assert response.status_code == 404

    def test_returns_500_on_error(self, client):
        with patch("api.teams.nfl.import_team_desc", side_effect=Exception("error")):
            response = client.get("/teams/KC")
        assert response.status_code == 500
