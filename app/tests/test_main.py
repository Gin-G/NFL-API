#!/usr/bin/env python3
"""
Tests for the root, health, and debug endpoints defined in main.py.
"""

from unittest.mock import patch


class TestRootEndpoint:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_has_message(self, client):
        body = client.get("/").json()
        assert body["message"] == "NFL Analytics API"

    def test_has_version(self, client):
        body = client.get("/").json()
        assert body["version"] == "1.0.0"

    def test_has_status(self, client):
        body = client.get("/").json()
        assert body["status"] == "running"

    def test_endpoints_keys_present(self, client):
        body = client.get("/").json()
        endpoints = body["endpoints"]
        for key in ("teams", "schedules", "players", "coaches", "docs", "health"):
            assert key in endpoints


class TestHealthEndpoint:
    def test_returns_200(self, client):
        with patch("api.utils.check_grading_systems", return_value={
            "player_grading": False, "coaching_analytics": False
        }):
            response = client.get("/health")
        assert response.status_code == 200

    def test_status_healthy(self, client):
        with patch("api.utils.check_grading_systems", return_value={
            "player_grading": False, "coaching_analytics": False
        }):
            body = client.get("/health").json()
        assert body["status"] == "healthy"

    def test_has_timestamp(self, client):
        with patch("api.utils.check_grading_systems", return_value={
            "player_grading": False, "coaching_analytics": False
        }):
            body = client.get("/health").json()
        assert "timestamp" in body

    def test_has_systems(self, client):
        with patch("api.utils.check_grading_systems", return_value={
            "player_grading": True, "coaching_analytics": False
        }):
            body = client.get("/health").json()
        assert "systems" in body


class TestDebugEndpoint:
    def test_returns_200(self, client):
        with patch("api.utils.check_grading_systems", return_value={
            "player_grading": False, "coaching_analytics": False
        }):
            response = client.get("/debug/info")
        assert response.status_code == 200

    def test_has_expected_keys(self, client):
        with patch("api.utils.check_grading_systems", return_value={
            "player_grading": False, "coaching_analytics": False
        }):
            body = client.get("/debug/info").json()
        assert "python_path" in body
        assert "working_directory" in body
        assert "available_systems" in body
        assert "files" in body
