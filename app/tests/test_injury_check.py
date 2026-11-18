"""The injury check: what ESPN says this week that the board does not.

2026 week 10: Kenneth Walker was Questionable on Friday's nflverse report and
ruled out Sunday; nflverse never changed, ESPN had him Out.
"""
from datetime import datetime

import pandas as pd
import pytest

from database.models import FanduelSlate, PlayerProjection, Schedule, SlatePlayerRule

pytest.importorskip("nfl_projections.injuries")

import api.optimizer as optmod                                     # noqa: E402

SEASON, WEEK = 2026, 10
SLATE = [("Kenneth Walker III", "RB", 9800, "KC", "w"), ("Emmett Johnson", "RB", 5100, "KC", "j"),
         ("Jaydn Ott", "RB", 4000, "KC", "o"), ("Rashee Rice", "WR", 6900, "KC", "r"),
         ("Hurt Guy", "WR", 5000, "LA", "h"), ("Back Guy", "WR", 4500, "LA", "b")]
POINTS = {"w": 20.4, "j": 6.0, "o": 1.3, "r": 10.5, "h": 0.0, "b": 8.0}


def _espn(*rows):
    return pd.DataFrame([{"player_name": n, "team": t, "position": p, "status": s,
                          "injury_type": "Groin", "espn_id": "", "player_id": pid,
                          "updated": pd.Timestamp(d), "comment": "news"}
                         for n, t, p, s, pid, d in rows])


@pytest.fixture
def week(db_session, monkeypatch):
    for model in (SlatePlayerRule, FanduelSlate, PlayerProjection, Schedule):
        db_session.query(model).delete()
    for (name, pos, _, team, pid) in SLATE:
        pts = POINTS[pid]
        db_session.add(PlayerProjection(
            season=SEASON, week=WEEK, player_id=pid, player_name=name, position=pos,
            team=team, projected_points=pts, floor=pts * .3, median=pts, ceiling=pts * 2,
            prediction_type="injured_out" if pts == 0 else "veteran_ml",
            model_version="t", computed_at=datetime.utcnow()))
    lines = ["Id,Nickname,Position,Roster Position,Salary,FPPG,Team,Injury Indicator"]
    lines += [f"{i},{n},{p},{p},{s},10,{t}," for i, (n, p, s, t, _) in enumerate(SLATE)]
    db_session.add(FanduelSlate(season=SEASON, week=WEEK, contest="classic",
                                content="\n".join(lines), uploaded_at=datetime.utcnow()))
    for gid, wk, day, home, away in (("g9", 9, "2026-11-09", "KC", "DEN"),
                                     ("g10", 10, "2099-11-15", "KC", "LV"),
                                     ("g10b", 10, "2026-11-12", "LA", "SF")):
        db_session.add(Schedule(game_id=gid, season=SEASON, week=wk, game_type="REG",
                                gameday=day, gametime="13:00", home_team=home,
                                away_team=away))
    db_session.commit()
    feed = _espn(
        ("Kenneth Walker III", "KC", "RB", "OUT", "w", "2026-11-15T16:30Z"),
        ("Rashee Rice", "KC", "WR", "QUESTIONABLE", "", "2026-11-14T16:30Z"),
        ("Hurt Guy", "LA", "WR", "ACTIVE", "h", "2026-11-12T12:00Z"),
        ("Jaydn Ott", "KC", "RB", "OUT", "o", "2026-11-09T12:00Z"),       # last week
    )
    monkeypatch.setattr(optmod, "_espn_frame", lambda refresh=False: feed)
    yield


def _check(client):
    r = client.get("/optimizer/injury-check", params={"season": SEASON, "week": WEEK})
    assert r.status_code == 200, r.text
    return r.json()


def test_the_sunday_scratch_is_listed_with_his_backup(client, week):
    from nfl_projections import roles
    body = _check(client)
    walker = next(c for c in body["changes"] if c["nickname"] == "Kenneth Walker III")
    assert walker["kind"] == "out" and walker["board_points"] == 20.4
    assert not walker["applied"] and not walker["started"]
    johnson = walker["promotes"][0]
    assert johnson["nickname"] == "Emmett Johnson" and johnson["from"] == 6.0
    assert johnson["to"] == pytest.approx(6.0 * roles.promotion_multiplier("RB", 2, (1,)),
                                          abs=0.01)
    assert body["pending"] == 1


def test_last_weeks_news_and_unchanged_players_are_not_listed(client, week):
    names = {c["nickname"] for c in _check(client)["changes"]}
    assert "Jaydn Ott" not in names and "Emmett Johnson" not in names


def test_a_cleared_player_and_a_questionable_one(client, week):
    by = {c["nickname"]: c for c in _check(client)["changes"]}
    assert by["Hurt Guy"]["kind"] == "cleared"
    assert by["Hurt Guy"]["started"]                 # LA kicked off Thursday
    assert by["Rashee Rice"]["kind"] == "questionable"


def test_once_ruled_out_it_shows_as_applied(client, week):
    r = client.post("/optimizer/pool", params={"season": SEASON, "week": WEEK},
                    json=[{"nickname": "Kenneth Walker III", "rule": "out"}])
    assert r.status_code == 200, r.text
    body = _check(client)
    walker = next(c for c in body["changes"] if c["nickname"] == "Kenneth Walker III")
    assert walker["applied"] and body["pending"] == 0
    assert walker["promotes"][0]["nickname"] == "Emmett Johnson"   # still explained


def test_no_feed_is_a_502(client, week, monkeypatch):
    def boom(refresh=False):
        raise ConnectionError("down")
    monkeypatch.setattr(optmod, "_espn_frame", boom)
    r = client.get("/optimizer/injury-check", params={"season": SEASON, "week": WEEK})
    assert r.status_code == 502 and "down" in r.json()["detail"]
