#!/usr/bin/env python3
"""The slate is uploaded once, and the lineups built from it are scored later.

Two things are pinned here. One: the FanDuel CSV — the only manual step in the
whole chain, because FanDuel publishes pricing nowhere fetchable — is stored on
upload and reused, so changing the objective or the cap does not mean
downloading it again. Two: every build is recorded and scored against what
actually happened, with the baseline that makes the number mean something (what
FanDuel's own FPPG would have built from the same pool).
"""

import io
from datetime import datetime

import pytest

from database.models import (FanduelSlate, LineupBuild, LineupEntry, LineupPlayer,
                             PlayerProjection, PlayerStat, Schedule)

pytest.importorskip("nfl_projections.optimizer",
                    reason="optimizer comes from nfl_projections (installed --no-deps in the image)")


@pytest.fixture(autouse=True)
def _no_opportunity_fetch(monkeypatch):
    from api import optimizer as api_opt

    monkeypatch.setattr(api_opt, "_opportunity", lambda season, week: None)


@pytest.fixture(autouse=True)
def _clean(db_session):
    """The engine is session-scoped and these tests commit."""
    for model in (LineupPlayer, LineupEntry, LineupBuild, FanduelSlate,
                  PlayerProjection, PlayerStat, Schedule):
        db_session.query(model).delete()
    db_session.commit()
    yield


# A slate deep enough to fill QB / 2 RB / 3 WR / TE / FLEX / DEF with
# alternatives, priced well under the cap so these tests are about tracking
# rather than about the cap.
SLATE_ROWS = [
    # name, position, roster position, salary, FPPG, team
    ("Josh Allen", "QB", "QB", 4500, 22.0, "BUF"),
    ("Jared Goff", "QB", "QB", 3750, 18.0, "DET"),
    ("Bijan Robinson", "RB", "RB", 4250, 18.5, "ATL"),
    ("Jahmyr Gibbs", "RB", "RB", 4100, 18.0, "DET"),
    ("Derrick Henry", "RB", "RB", 3900, 17.0, "BAL"),
    ("Kyren Williams", "RB", "RB", 3250, 13.0, "LA"),
    ("Puka Nacua", "WR", "WR", 4000, 17.0, "LA"),
    ("Amon-Ra St. Brown", "WR", "WR", 3950, 16.5, "DET"),
    ("Ja'Marr Chase", "WR", "WR", 3850, 16.0, "CIN"),
    ("Chris Olave", "WR", "WR", 3400, 14.0, "NO"),
    ("Trey McBride", "TE", "TE", 3000, 13.0, "ARI"),
    ("Dallas Goedert", "TE", "TE", 2600, 11.0, "PHI"),
    ("Ravens", "D", "D", 2100, 8.5, "BAL"),
    ("Broncos", "D", "D", 1950, 8.0, "DEN"),
]

GAMES = [("BUF", "DET"), ("ATL", "BAL"), ("LA", "CIN"), ("NO", "ARI"),
         ("PHI", "NYJ"), ("DEN", "KC")]


def _csv(rows=None) -> bytes:
    rows = rows if rows is not None else SLATE_ROWS
    lines = ["Id,Nickname,Position,Roster Position,Salary,FPPG,Team,Injury Indicator"]
    for i, (name, pos, slot, salary, fppg, team) in enumerate(rows):
        lines.append(f"{i},{name},{pos},{slot},{salary},{fppg},{team},")
    return "\n".join(lines).encode()


def _project(db, season=2026, week=6):
    """The published board: every non-defense in the slate, two points above
    FanDuel's FPPG so the model and the baseline disagree about the order."""
    for i, (name, pos, _, _, fppg, team) in enumerate(SLATE_ROWS):
        if pos == "D":
            continue
        points = fppg + 2
        db.add(PlayerProjection(
            season=season, week=week, player_id=f"00-{i:04d}", player_name=name,
            position=pos, team=team, projected_points=points, floor=points * 0.4,
            median=points * 0.95, ceiling=points * 1.8, prediction_type="veteran_ml",
            model_version="1.5.0", computed_at=datetime.utcnow()))
    db.commit()


def _schedule(db, season=2026, week=6, final=True):
    for i, (away, home) in enumerate(GAMES):
        db.add(Schedule(
            game_id=f"{season}_{week:02d}_{away}_{home}", season=season, week=week,
            game_type="REG", gameday=f"{season}-10-18", gametime="13:00",
            away_team=away, home_team=home,
            away_score=20 if final else None, home_score=17 if final else None))
    db.commit()


def _post(client, csv=_csv, **form):
    """Build lineups. `csv=None` sends no file at all — the saved-slate path."""
    data = {"season": 2026, "week": 6, "num_lineups": 1, **form}
    files = None
    if csv is not None:
        raw = csv() if callable(csv) else csv
        files = {"file": ("slate.csv", io.BytesIO(raw), "text/csv")}
    return client.post("/optimizer/lineups", data=data, files=files)


class TestTheSlateIsUploadedOnce:
    def test_an_upload_is_saved(self, client, db_session):
        _project(db_session)
        body = _post(client).json()

        assert body["slate_source"] == "upload"
        assert body["slate_saved"] is True
        assert body["saved_slate"]["players"] == len(SLATE_ROWS)
        assert body["saved_slate"]["contest"] == "classic"
        assert set(body["saved_slate"]["teams"]) == {t for row in SLATE_ROWS
                                                    for t in [row[5]]}
        (stored,) = db_session.query(FanduelSlate).all()
        assert stored.slate_key == "classic"
        assert "Josh Allen" in stored.content

    def test_a_later_build_needs_no_file(self, client, db_session):
        _project(db_session)
        first = _post(client).json()
        again = _post(client, csv=None, objective="ceiling").json()

        assert again["slate_source"] == "stored"
        assert again["objective"] == "ceiling"
        assert again["slate_players"] == first["slate_players"]
        assert len(again["data"][0]["players"]) == 9

    def test_a_re_upload_replaces_rather_than_piles_up(self, client, db_session):
        """Prices move during the week, so the newest download is the slate."""
        _project(db_session)
        _post(client)
        before = db_session.query(FanduelSlate).one().content_hash

        cheaper = [(n, p, s, salary - 500, f, t)
                   for n, p, s, salary, f, t in SLATE_ROWS]
        _post(client, csv=_csv(cheaper))

        rows = db_session.query(FanduelSlate).all()
        assert len(rows) == 1
        assert rows[0].content_hash != before
        assert "4000," in rows[0].content          # Josh Allen, re-priced

    def test_building_from_a_saved_slate_marks_it_used(self, client, db_session):
        _project(db_session)
        _post(client)
        db_session.query(FanduelSlate).one().used_at = None
        db_session.commit()

        _post(client, csv=None)
        assert db_session.query(FanduelSlate).one().used_at is not None

    def test_nothing_saved_says_so(self, client, db_session):
        _project(db_session)
        resp = _post(client, csv=None)
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "week 6" in detail and "once" in detail

    def test_an_untouched_file_input_is_not_an_upload(self, client, db_session):
        """What a browser posts when the file field was never touched: a file
        part with an empty name. Treated as an upload it fails as "that file is
        empty" on every build after the first."""
        _project(db_session)
        _post(client)
        resp = client.post("/optimizer/lineups",
                           data={"season": 2026, "week": 6, "num_lineups": 1},
                           files={"file": ("", b"", "application/octet-stream")})
        assert resp.status_code == 200
        assert resp.json()["slate_source"] == "stored"

    def test_save_slate_false_builds_without_keeping_it(self, client, db_session):
        _project(db_session)
        body = _post(client, save_slate="false").json()
        assert body["slate_saved"] is False
        assert db_session.query(FanduelSlate).count() == 0

    def test_a_showdown_file_is_its_own_slate(self, client, db_session):
        """One classic list per week, but a Showdown file per game."""
        _project(db_session)
        _post(client)
        showdown = [("Josh Allen", "QB", "MVP", 15000, 22.0, "BUF"),
                    ("Jared Goff", "QB", "AnyFLEX", 11000, 18.0, "DET"),
                    ("Jahmyr Gibbs", "RB", "AnyFLEX", 10000, 18.0, "DET"),
                    ("Amon-Ra St. Brown", "WR", "AnyFLEX", 9500, 16.5, "DET"),
                    ("Dallas Goedert", "TE", "AnyFLEX", 6000, 11.0, "DET"),
                    ("Lions", "D", "AnyFLEX", 4000, 8.0, "DET")]
        _post(client, csv=_csv(showdown), num_lineups=1)

        keys = {r.slate_key for r in db_session.query(FanduelSlate).all()}
        assert keys == {"classic", "showdown:BUF@DET"}

    def test_slate_id_picks_one(self, client, db_session):
        _project(db_session)
        _post(client)
        slate_id = db_session.query(FanduelSlate).one().id
        body = _post(client, csv=None, slate_id=slate_id).json()
        assert body["saved_slate"]["slate_id"] == slate_id

    def test_a_slate_id_from_another_week_is_refused(self, client, db_session):
        _project(db_session, week=6)
        _post(client)
        slate_id = db_session.query(FanduelSlate).one().id
        _project(db_session, week=7)

        resp = _post(client, csv=None, week=7, slate_id=slate_id)
        assert resp.status_code == 400
        assert "week 6" in resp.json()["detail"]

    def test_listing_and_forgetting(self, client, db_session):
        _project(db_session)
        _post(client)
        listed = client.get("/optimizer/slates", params={"season": 2026}).json()
        assert listed["count"] == 1
        slate_id = listed["data"][0]["slate_id"]

        assert client.delete(f"/optimizer/slates/{slate_id}").status_code == 200
        assert db_session.query(FanduelSlate).count() == 0
        assert client.delete(f"/optimizer/slates/{slate_id}").status_code == 404

    def test_a_saved_slate_outlives_the_week_it_was_for(self, client, db_session):
        """Keyed by week, so last week's slate is not silently reused."""
        _project(db_session, week=6)
        _post(client)
        _project(db_session, week=7)
        resp = _post(client, csv=None, week=7)
        assert resp.status_code == 400


class TestTheBuildIsRecorded:
    def test_a_build_writes_its_lineups(self, client, db_session):
        _project(db_session)
        body = _post(client, num_lineups=2, max_usage_percentage=100).json()

        build = db_session.query(LineupBuild).one()
        assert body["build_id"] == build.id
        assert body["tracked"] is True
        assert build.num_requested == 2
        assert build.num_built == len(body["data"])
        assert build.objective == "mean" and build.salary_cap == 60000
        assert build.slate_id == db_session.query(FanduelSlate).one().id

        entries = db_session.query(LineupEntry).filter(
            LineupEntry.build_id == build.id).all()
        assert len(entries) == len(body["data"])
        assert {e.lineup_index for e in entries} == set(range(1, len(entries) + 1))
        for entry, lineup in zip(sorted(entries, key=lambda e: e.lineup_index),
                                 body["data"]):
            assert entry.salary == lineup["salary"]
            assert entry.projected_points == pytest.approx(lineup["projected"], abs=0.01)
            assert entry.actual_points is None        # the games have not happened

    def test_the_players_are_recorded_with_what_we_said(self, client, db_session):
        _project(db_session)
        _post(client)
        entry = db_session.query(LineupEntry).one()
        players = db_session.query(LineupPlayer).filter(
            LineupPlayer.lineup_id == entry.id).all()

        assert len(players) == 9
        defense = next(p for p in players if p.position == "D")
        assert defense.player_id is None and defense.from_model == 0
        skill = [p for p in players if p.position != "D"]
        assert all(p.player_id and p.from_model == 1 for p in skill)
        assert all(p.projected_points and p.salary for p in players)

    def test_pressing_build_twice_is_not_two_samples(self, client, db_session):
        _project(db_session)
        first = _post(client).json()["build_id"]
        second = _post(client).json()["build_id"]

        assert second != first
        (build,) = db_session.query(LineupBuild).all()
        assert build.id == second
        # ... and the superseded build's lineups went with it.
        assert db_session.query(LineupEntry).count() == 1
        assert db_session.query(LineupPlayer).count() == 9

    def test_different_settings_are_different_builds(self, client, db_session):
        _project(db_session)
        _post(client, objective="mean")
        _post(client, objective="ceiling")
        _post(client, objective="ceiling", salary_cap=55000)

        objectives = [(b.objective, b.salary_cap)
                      for b in db_session.query(LineupBuild).all()]
        assert sorted(objectives) == [("ceiling", 55000), ("ceiling", 60000),
                                      ("mean", 60000)]

    def test_a_scored_build_is_never_superseded(self, client, db_session):
        """Once it is history it stays history, however much is rebuilt after."""
        _project(db_session)
        _post(client)
        scored = db_session.query(LineupBuild).one()
        scored.scored_at = datetime.utcnow()
        scored.fppg_points = 101.5
        db_session.commit()

        _post(client)
        builds = db_session.query(LineupBuild).all()
        assert len(builds) == 2
        assert any(b.id == scored.id and b.fppg_points == 101.5 for b in builds)

    def test_track_false_records_nothing(self, client, db_session):
        _project(db_session)
        body = _post(client, track="false").json()
        assert body["build_id"] is None and body["tracked"] is False
        assert db_session.query(LineupBuild).count() == 0

    def test_a_failed_build_is_not_recorded(self, client, db_session):
        _project(db_session)
        assert _post(client, salary_cap=1000).status_code == 422
        assert db_session.query(LineupBuild).count() == 0
        # ... but the slate it was built from is kept, which is the point.
        assert db_session.query(FanduelSlate).count() == 1


class TestPerformanceBeforeTheGames:
    def test_nothing_built_says_so(self, client, db_session):
        body = client.get("/optimizer/performance", params={"season": 2026}).json()
        assert body["status"] == "no_data"

    def test_a_built_week_is_listed_unscored(self, client, db_session):
        _project(db_session)
        _post(client, num_lineups=2, max_usage_percentage=100)
        body = client.get("/optimizer/performance", params={"season": 2026}).json()

        assert body["status"] == "success"
        (week,) = body["weeks"]
        assert week["week"] == 6 and week["builds"] == 1
        assert week["lineups"] >= 1
        assert week["scored_lineups"] == 0
        assert week["actual"] is None and week["fppg_baseline"] is None
        assert body["totals"]["beat_fppg_rate"] is None
