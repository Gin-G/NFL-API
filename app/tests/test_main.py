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


class TestLightMigrations:
    """Columns added to a model after its table exists.

    `create_all` creates missing TABLES and never a missing COLUMN, so a model
    that grows a field deploys clean and then 500s on the first query that
    selects it. That is not hypothetical: `lineup_builds.max_per_team` shipped
    without its column and took every read of the table down with it, because
    `apply_light_migrations` was only being run by the cron scripts.
    """

    @staticmethod
    def _stale_db(tmp_path, table, drop):
        """A database whose table predates some of its model's columns."""
        from sqlalchemy import create_engine, inspect, text

        from database.models import Base, apply_light_migrations

        engine = create_engine(f"sqlite:///{tmp_path}/stale.db")
        Base.metadata.create_all(engine)
        with engine.begin() as conn:
            for column in drop:
                conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))
        missing = set(drop) - {c["name"] for c in inspect(engine).get_columns(table)}
        assert missing == set(drop), "the fixture did not actually drop them"
        apply_light_migrations(engine)
        return {c["name"] for c in inspect(engine).get_columns(table)}

    def test_the_build_columns_come_back(self, tmp_path):
        wanted = ["max_per_team", "stack_qb", "rb_with_qb", "ownership_weight",
                  "projection_computed_at"]
        assert set(wanted) <= self._stale_db(tmp_path, "lineup_builds", wanted)

    def test_the_fanduel_id_comes_back(self, tmp_path):
        assert "fanduel_id" in self._stale_db(tmp_path, "lineup_players",
                                              ["fanduel_id"])

    def test_running_it_twice_is_a_no_op(self, tmp_path):
        from sqlalchemy import create_engine, inspect

        from database.models import Base, apply_light_migrations

        engine = create_engine(f"sqlite:///{tmp_path}/twice.db")
        Base.metadata.create_all(engine)
        apply_light_migrations(engine)
        before = {c["name"] for c in inspect(engine).get_columns("lineup_builds")}
        apply_light_migrations(engine)          # must not raise on existing columns
        after = {c["name"] for c in inspect(engine).get_columns("lineup_builds")}
        assert before == after

    def test_startup_applies_them(self):
        # The web app is the only process most deploys run, so it is the one
        # that has to do this.
        import inspect as _inspect

        import main

        source = _inspect.getsource(main)
        assert "apply_light_migrations(engine)" in source


class TestDashboardCaching:
    """A deploy nobody can see is a deploy that did not happen.

    Vite fingerprints every asset, so the bundle name changes when its content
    does — but index.html is the file that NAMES the bundles, so it cannot be
    fingerprinted itself. Served with no Cache-Control at all, a browser
    caches it heuristically from Last-Modified and keeps asking for the
    previous deploy's bundle. That is exactly how the ownership controls and
    the FanDuel download shipped and stayed invisible.
    """

    @staticmethod
    def _headers(path):
        from starlette.responses import Response

        from main import SPAStaticFiles

        return SPAStaticFiles._cache_headers(path, Response("x")).headers

    def test_the_directory_path_is_never_cached(self):
        # THE regression. StaticFiles normalises the mount-relative path, so a
        # request for /dashboard/ — how anyone actually opens the page —
        # arrives as ".", not "" and not "index.html". The first version of
        # this enumerated the files that must not cache, "." matched none of
        # them, and the page went out with no Cache-Control at all.
        assert "no-cache" in self._headers(".")["cache-control"]

    def test_index_is_never_cached(self):
        assert "no-cache" in self._headers("index.html")["cache-control"]

    def test_the_spa_fallback_route_is_never_cached(self):
        # /dashboard/optimizer is served index.html's bytes under its own path.
        assert "no-cache" in self._headers("optimizer")["cache-control"]

    def test_the_mount_root_is_never_cached(self):
        assert "no-cache" in self._headers("")["cache-control"]

    def test_fingerprinted_assets_cache_forever(self):
        cache = self._headers("assets/index-CFHA0Osj.js")["cache-control"]
        assert "immutable" in cache and "max-age=31536000" in cache

    def test_a_stray_file_is_not_made_immutable(self):
        # Only /assets/ is fingerprinted; favicon.ico keeps its name forever,
        # so it has to revalidate like the rest.
        assert "immutable" not in self._headers("favicon.ico")["cache-control"]

    def test_every_path_gets_some_rule(self):
        # The point of inverting the rule: a path nobody anticipated should
        # revalidate rather than fall through with no header.
        for path in (".", "", "/", "index.html", "optimizer", "teams/DET",
                     "favicon.ico", "robots.txt", "assets/x.css", "./assets/x.js"):
            assert "cache-control" in self._headers(path), path

    def test_a_real_request_for_the_page_carries_the_header(self, client):
        # The unit tests above all passed while the live site still served
        # index.html with no Cache-Control, because they called the helper
        # directly and never asked what path a real request produces. This one
        # goes through the mount.
        import os

        import main

        if not os.path.isdir(os.path.join(os.path.dirname(main.__file__),
                                          "static", "dashboard")):
            import pytest as _pytest
            _pytest.skip("frontend not built in this checkout")
        response = client.get("/dashboard/")
        assert response.status_code == 200
        assert "no-cache" in response.headers["cache-control"]
