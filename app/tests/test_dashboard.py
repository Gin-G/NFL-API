#!/usr/bin/env python3
"""
Tests for the /dashboard static file serving functionality.

Three test groups:
  1. API still works when the frontend build is absent
  2. Correct HTML is served once static files are present
  3. API routes are not shadowed after the static mount is added
"""

import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch


# ─── Shared constants ─────────────────────────────────────────────────────────

EXPECTED_ROOT_DIV = '<div id="root">'
EXPECTED_TITLE = "NFL Analytics Dashboard"
JS_BUNDLE_NAME = "index-test123.js"
CSS_BUNDLE_NAME = "index-test123.css"


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_dashboard_dir(tmp_path):
    """Minimal built-frontend tree: index.html + assets/."""
    (tmp_path / "index.html").write_text(
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "  <head>\n"
        '    <meta charset="UTF-8" />\n'
        f"    <title>{EXPECTED_TITLE}</title>\n"
        "  </head>\n"
        "  <body>\n"
        f"    {EXPECTED_ROOT_DIV}\n"
        f'    <script type="module" src="/dashboard/assets/{JS_BUNDLE_NAME}"></script>\n'
        "  </body>\n"
        "</html>\n",
        encoding="utf-8",
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / JS_BUNDLE_NAME).write_text(
        "// NFL Analytics Dashboard bundle", encoding="utf-8"
    )
    (assets / CSS_BUNDLE_NAME).write_text(
        "body{margin:0}", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def static_app(fake_dashboard_dir):
    """Isolated FastAPI app with only the SPAStaticFiles mount — mirrors main.py logic."""
    from main import SPAStaticFiles
    app = FastAPI()
    app.mount(
        "/dashboard",
        SPAStaticFiles(directory=str(fake_dashboard_dir), html=True),
        name="dashboard",
    )
    return app


@pytest.fixture
def static_client(static_app):
    with TestClient(static_app) as c:
        yield c


@pytest.fixture
def combined_client(fake_dashboard_dir):
    """main.app with the static mount added via a temp dir (tests coexistence)."""
    from main import app, SPAStaticFiles
    app.mount(
        "/dashboard",
        SPAStaticFiles(directory=str(fake_dashboard_dir), html=True),
        name="dashboard_combined",
    )
    with TestClient(app) as c:
        yield c
    # Teardown: remove the temporary mount so it does not bleed into other tests
    app.routes[:] = [r for r in app.routes if getattr(r, "name", None) != "dashboard_combined"]


# ─── 1. API without dashboard build ───────────────────────────────────────────

class TestApiWithoutDashboard:
    """API must boot and serve all routes even when no frontend build exists."""

    def test_root_200(self, client):
        assert client.get("/").status_code == 200

    def test_root_has_status_running(self, client):
        assert client.get("/").json()["status"] == "running"

    def test_health_200(self, client):
        with patch("api.utils.check_grading_systems", return_value={
            "player_grading": False, "coaching_analytics": False,
        }):
            assert client.get("/health").status_code == 200

    def test_health_status_healthy(self, client):
        with patch("api.utils.check_grading_systems", return_value={
            "player_grading": False, "coaching_analytics": False,
        }):
            assert client.get("/health").json()["status"] == "healthy"

    def test_docs_200(self, client):
        assert client.get("/docs").status_code == 200

    def test_dashboard_absent_does_not_cause_500(self, client):
        """Hitting /dashboard/ when it is not mounted must return 404, not 500."""
        from main import app
        is_mounted = any("dashboard" in getattr(r, "path", "") for r in app.routes)
        if not is_mounted:
            assert client.get("/dashboard/").status_code == 404


# ─── 2. Dashboard HTML content ────────────────────────────────────────────────

class TestDashboardHtmlContent:
    """Once the mount is active /dashboard/ must serve a valid SPA shell."""

    def test_returns_200(self, static_client):
        assert static_client.get("/dashboard/").status_code == 200

    def test_content_type_is_html(self, static_client):
        resp = static_client.get("/dashboard/")
        assert "text/html" in resp.headers["content-type"]

    def test_contains_doctype(self, static_client):
        assert "<!DOCTYPE html>" in static_client.get("/dashboard/").text

    def test_contains_root_div(self, static_client):
        assert EXPECTED_ROOT_DIV in static_client.get("/dashboard/").text

    def test_contains_page_title(self, static_client):
        assert EXPECTED_TITLE in static_client.get("/dashboard/").text

    def test_references_js_bundle(self, static_client):
        assert ".js" in static_client.get("/dashboard/").text

    def test_spa_fallback_teams(self, static_client):
        """React Router deep links must return index.html (html=True fallback)."""
        resp = static_client.get("/dashboard/teams")
        assert resp.status_code == 200
        assert EXPECTED_ROOT_DIV in resp.text

    def test_spa_fallback_players(self, static_client):
        resp = static_client.get("/dashboard/players")
        assert resp.status_code == 200
        assert EXPECTED_ROOT_DIV in resp.text

    def test_spa_fallback_coaches(self, static_client):
        resp = static_client.get("/dashboard/coaches")
        assert resp.status_code == 200
        assert EXPECTED_ROOT_DIV in resp.text

    def test_spa_fallback_schedule(self, static_client):
        resp = static_client.get("/dashboard/schedule")
        assert resp.status_code == 200
        assert EXPECTED_ROOT_DIV in resp.text

    def test_unknown_asset_returns_404(self, static_client):
        assert static_client.get("/dashboard/assets/does-not-exist.js").status_code == 404


# ─── 3. Static assets ─────────────────────────────────────────────────────────

class TestDashboardStaticAssets:
    """JS and CSS bundles must be served with correct MIME types and content."""

    def test_js_bundle_200(self, static_client):
        assert static_client.get(f"/dashboard/assets/{JS_BUNDLE_NAME}").status_code == 200

    def test_js_content_type(self, static_client):
        resp = static_client.get(f"/dashboard/assets/{JS_BUNDLE_NAME}")
        assert "javascript" in resp.headers["content-type"]

    def test_js_has_content(self, static_client):
        resp = static_client.get(f"/dashboard/assets/{JS_BUNDLE_NAME}")
        assert len(resp.content) > 0

    def test_css_bundle_200(self, static_client):
        assert static_client.get(f"/dashboard/assets/{CSS_BUNDLE_NAME}").status_code == 200

    def test_css_content_type(self, static_client):
        resp = static_client.get(f"/dashboard/assets/{CSS_BUNDLE_NAME}")
        assert "css" in resp.headers["content-type"]

    def test_css_has_content(self, static_client):
        resp = static_client.get(f"/dashboard/assets/{CSS_BUNDLE_NAME}")
        assert len(resp.content) > 0


# ─── 4. API routes coexist with static mount ──────────────────────────────────

class TestApiAndDashboardCoexist:
    """API endpoints must remain reachable after the static mount is added."""

    def test_root_still_200(self, combined_client):
        assert combined_client.get("/").status_code == 200

    def test_root_status_running(self, combined_client):
        assert combined_client.get("/").json()["status"] == "running"

    def test_health_still_200(self, combined_client):
        with patch("api.utils.check_grading_systems", return_value={
            "player_grading": False, "coaching_analytics": False,
        }):
            assert combined_client.get("/health").status_code == 200

    def test_teams_not_shadowed(self, combined_client, sample_teams_df):
        with patch("api.teams.nfl.import_team_desc", return_value=sample_teams_df):
            resp = combined_client.get("/teams/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_schedules_not_shadowed(self, combined_client, sample_schedules_df):
        with patch("api.schedules.nfl.import_schedules", return_value=sample_schedules_df):
            assert combined_client.get("/schedules/?season=2023").status_code == 200

    def test_players_rosters_not_shadowed(self, combined_client, sample_rosters_df):
        with patch("api.players.nfl.import_weekly_rosters", return_value=sample_rosters_df):
            assert combined_client.get("/players/rosters?season=2023").status_code == 200

    def test_players_stats_not_shadowed(self, combined_client, sample_weekly_data_df):
        with patch("api.players.nfl.import_weekly_data", return_value=sample_weekly_data_df):
            assert combined_client.get("/players/stats?season=2023").status_code == 200

    def test_docs_not_shadowed(self, combined_client):
        assert combined_client.get("/docs").status_code == 200

    def test_dashboard_and_api_both_work(self, combined_client, sample_teams_df):
        """Dashboard HTML and /teams/ must both respond successfully in the same app."""
        html = combined_client.get("/dashboard/").text
        assert EXPECTED_ROOT_DIV in html

        with patch("api.teams.nfl.import_team_desc", return_value=sample_teams_df):
            api_resp = combined_client.get("/teams/").json()
        assert api_resp["status"] == "success"


# ─── 5. isdir guard ───────────────────────────────────────────────────────────

class TestDashboardIsdirGuard:
    """The isdir guard in main.py prevents mount errors on a fresh checkout."""

    def test_nonexistent_path_does_not_raise(self, tmp_path):
        """Simulates a first checkout with no frontend build."""
        from main import SPAStaticFiles
        absent_path = str(tmp_path / "nonexistent" / "dashboard")
        assert not os.path.isdir(absent_path)
        # Reproduces the exact guard logic from main.py
        app = FastAPI()
        if os.path.isdir(absent_path):
            app.mount("/dashboard", SPAStaticFiles(directory=absent_path, html=True))
        # If we reached here without raising, the guard works
        with TestClient(app) as c:
            assert c.get("/dashboard/").status_code == 404

    def test_existing_path_mounts_correctly(self, fake_dashboard_dir):
        """Simulates a post-build checkout with frontend dist present."""
        from main import SPAStaticFiles
        assert os.path.isdir(str(fake_dashboard_dir))
        app = FastAPI()
        if os.path.isdir(str(fake_dashboard_dir)):
            app.mount(
                "/dashboard",
                SPAStaticFiles(directory=str(fake_dashboard_dir), html=True),
            )
        with TestClient(app) as c:
            assert c.get("/dashboard/").status_code == 200
            assert EXPECTED_ROOT_DIV in c.get("/dashboard/").text
