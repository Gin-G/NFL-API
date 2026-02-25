#!/usr/bin/env python3
"""
Tests for the /pbp router.

The play_by_play table is not created by Base.metadata.create_all (no ORM model),
so in the test DB it never exists. These tests verify the router degrades gracefully
to a no_data response instead of returning a 500.
"""

import pytest


# ---------------------------------------------------------------------------
# PBP endpoint — table-absent fallback
# ---------------------------------------------------------------------------

def test_pbp_no_filters_returns_no_data(client):
    """GET /pbp/ with no filters returns no_data when table doesn't exist."""
    resp = client.get("/pbp/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "no_data"
    assert data["total_plays"] == 0
    assert data["data"] == []


def test_pbp_with_season_filter_returns_no_data(client):
    """GET /pbp/?season=2024 returns no_data when table doesn't exist."""
    resp = client.get("/pbp/?season=2024")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "no_data"
    assert data["total_plays"] == 0


def test_pbp_with_game_id_returns_no_data(client):
    """GET /pbp/?game_id=... returns no_data when table doesn't exist."""
    resp = client.get("/pbp/?game_id=2024_01_KC_DET")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "no_data"


def test_pbp_with_multiple_filters_returns_no_data(client):
    """Combination of filters still returns no_data gracefully."""
    resp = client.get("/pbp/?season=2024&play_type=pass&down=4&team=KC")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "no_data"


def test_pbp_response_shape(client):
    """Response always has status, total_plays, and data keys."""
    resp = client.get("/pbp/?season=2023&week=1")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "total_plays" in body
    assert "data" in body


# ---------------------------------------------------------------------------
# PBP endpoint — limit param capped at 500
# ---------------------------------------------------------------------------

def test_pbp_limit_param_accepted(client):
    """A large limit param doesn't cause a server error."""
    resp = client.get("/pbp/?season=2024&limit=1000")
    assert resp.status_code == 200
    # Still no_data since table doesn't exist; we just confirm no crash
    assert resp.json()["status"] == "no_data"
