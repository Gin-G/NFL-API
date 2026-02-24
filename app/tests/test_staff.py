#!/usr/bin/env python3
"""
Tests for GET /coaches/staff (SportRadar coaching staff endpoint).
"""

from unittest.mock import MagicMock, patch

SAMPLE_STAFF = [
    {
        "team_abbr": "KC",
        "team_name": "Kansas City Chiefs",
        "head_coach": "Andy Reid",
        "offensive_coordinator": "Matt Nagy",
        "defensive_coordinator": "Steve Spagnuolo",
        "special_teams_coordinator": "Dave Toub",
    },
    {
        "team_abbr": "SF",
        "team_name": "San Francisco 49ers",
        "head_coach": "Kyle Shanahan",
        "offensive_coordinator": None,
        "defensive_coordinator": "Nick Sorensen",
        "special_teams_coordinator": None,
    },
]


class TestGetCoachingStaff:
    # ── no API key configured ───────────────────────────────────────────
    def test_returns_200_when_not_configured(self, client):
        with patch("api.coaches.get_sportradar_coaches_client", return_value=None):
            response = client.get("/coaches/staff")
        assert response.status_code == 200

    def test_configured_false_when_no_key(self, client):
        with patch("api.coaches.get_sportradar_coaches_client", return_value=None):
            body = client.get("/coaches/staff").json()
        assert body["configured"] is False
        assert body["data"] == []

    def test_message_present_when_not_configured(self, client):
        with patch("api.coaches.get_sportradar_coaches_client", return_value=None):
            body = client.get("/coaches/staff").json()
        assert "message" in body

    # ── API key configured ──────────────────────────────────────────────
    def _mock_client(self):
        c = MagicMock()
        c.get_all_team_staff.return_value = SAMPLE_STAFF
        return c

    def test_returns_200_when_configured(self, client):
        with patch("api.coaches.get_sportradar_coaches_client", return_value=self._mock_client()):
            response = client.get("/coaches/staff")
        assert response.status_code == 200

    def test_configured_true_when_key_set(self, client):
        with patch("api.coaches.get_sportradar_coaches_client", return_value=self._mock_client()):
            body = client.get("/coaches/staff").json()
        assert body["configured"] is True

    def test_data_is_list(self, client):
        with patch("api.coaches.get_sportradar_coaches_client", return_value=self._mock_client()):
            body = client.get("/coaches/staff").json()
        assert isinstance(body["data"], list)

    def test_total_teams_count(self, client):
        with patch("api.coaches.get_sportradar_coaches_client", return_value=self._mock_client()):
            body = client.get("/coaches/staff").json()
        assert body["total_teams"] == 2

    def test_staff_entry_has_required_fields(self, client):
        with patch("api.coaches.get_sportradar_coaches_client", return_value=self._mock_client()):
            body = client.get("/coaches/staff").json()
        entry = body["data"][0]
        assert entry["team_abbr"] == "KC"
        assert entry["head_coach"] == "Andy Reid"
        assert entry["offensive_coordinator"] == "Matt Nagy"
        assert entry["defensive_coordinator"] == "Steve Spagnuolo"
        assert entry["special_teams_coordinator"] == "Dave Toub"

    def test_null_coordinator_allowed(self, client):
        with patch("api.coaches.get_sportradar_coaches_client", return_value=self._mock_client()):
            body = client.get("/coaches/staff").json()
        sf = next(e for e in body["data"] if e["team_abbr"] == "SF")
        assert sf["offensive_coordinator"] is None

    # ── team filter ─────────────────────────────────────────────────────
    def test_filter_by_team_returns_one(self, client):
        with patch("api.coaches.get_sportradar_coaches_client", return_value=self._mock_client()):
            body = client.get("/coaches/staff?team=KC").json()
        assert body["total_teams"] == 1
        assert body["data"][0]["team_abbr"] == "KC"

    def test_filter_by_team_case_insensitive(self, client):
        with patch("api.coaches.get_sportradar_coaches_client", return_value=self._mock_client()):
            body = client.get("/coaches/staff?team=kc").json()
        assert body["total_teams"] == 1

    def test_filter_unknown_team_returns_empty(self, client):
        with patch("api.coaches.get_sportradar_coaches_client", return_value=self._mock_client()):
            body = client.get("/coaches/staff?team=XXX").json()
        assert body["data"] == []
        assert body["total_teams"] == 0

    # ── error handling ──────────────────────────────────────────────────
    def test_returns_500_on_client_error(self, client):
        bad_client = MagicMock()
        bad_client.get_all_team_staff.side_effect = Exception("SportRadar down")
        with patch("api.coaches.get_sportradar_coaches_client", return_value=bad_client):
            response = client.get("/coaches/staff")
        assert response.status_code == 500
