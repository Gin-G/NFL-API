#!/usr/bin/env python3
"""The price-tier targets endpoint."""

import hashlib
from datetime import datetime

import pytest

from database.models import FanduelSlate, PlayerProjection, ProjectionRun

SEASON, WEEK = 2026, 9

SLATE_CSV = (
    "Id,Position,First Name,Nickname,Last Name,FPPG,Played,Salary,Game,Team,"
    "Opponent,Injury Indicator,Injury Details,Tier,,,Roster Position\n"
    "1,RB,Alpha,Alpha Back,Back,27.3,7,10200,DET@MIA,DET,MIA,,,,,,RB/FLEX\n"
    "2,RB,Cheap,Cheap Back,Back,9.6,7,5800,ARI@SEA,ARI,SEA,,,,,,RB/FLEX\n"
    "3,RB,Hurt,Hurt Back,Back,18.0,7,8700,LV@SF,SF,LV,O,Hamstring,,,,RB/FLEX\n"
    "4,QB,Alpha,Alpha Arm,Arm,21.7,7,7700,DET@MIA,DET,MIA,,,,,,QB\n"
    "5,WR,Quest,Quest Catch,Catch,11.0,7,6200,NYG@PHI,NYG,PHI,Q,Knee,,,,WR/FLEX\n"
    "6,D,A,A Defense,Defense,7.0,8,4800,NYG@PHI,NYG,PHI,,,,,,DEF\n"
)

BOARD = [
    ("p1", "Alpha Back", "RB", "DET", 24.3, 8.2, 22.0, 44.3),
    ("p2", "Cheap Back", "RB", "ARI", 14.5, 2.7, 13.0, 27.0),
    ("p3", "Hurt Back", "RB", "SF", 16.9, 4.0, 15.0, 30.0),
    ("p4", "Alpha Arm", "QB", "DET", 18.5, 6.1, 17.0, 33.6),
    ("p5", "Quest Catch", "WR", "NYG", 10.2, 2.0, 9.0, 20.0),
    ("p6", "A Defense", "D", "NYG", 7.5, 1.0, 7.0, 16.0),
]


@pytest.fixture(autouse=True)
def _seed(db_session):
    for model in (FanduelSlate, PlayerProjection, ProjectionRun):
        db_session.query(model).delete()
    db_session.add(FanduelSlate(
        season=SEASON, week=WEEK, contest="classic", slate_key="classic",
        filename="slate.csv", content=SLATE_CSV,
        content_hash=hashlib.sha256(SLATE_CSV.encode()).hexdigest(),
        row_count=6, teams="DET,MIA,ARI,SEA,SF,LV,NYG,PHI",
        uploaded_at=datetime(2026, 11, 5, 12, 0)))
    for pid, name, pos, team, pts, lo, md, hi in BOARD:
        db_session.add(PlayerProjection(
            season=SEASON, week=WEEK, player_id=pid, player_name=name,
            position=pos, team=team, projected_points=pts, floor=lo,
            median=md, ceiling=hi))
    db_session.commit()
    yield


def _get(client, **params):
    q = "&".join(f"{k}={v}" for k, v in
                 {"season": SEASON, "week": WEEK, **params}.items())
    return client.get(f"/dfs/targets?{q}")


class TestTargets:
    def test_it_groups_by_position_and_tier(self, client):
        body = _get(client).json()
        assert body["status"] == "success", body
        seen = {(g["position"], g["tier"]) for g in body["groups"]}
        assert ("RB", "premium") in seen
        assert ("RB", "value") in seen
        assert ("QB", "premium") in seen

    def test_the_quantiles_actually_arrive(self, client):
        """Reading them under the wrong attribute returns None silently."""
        body = _get(client).json()
        rb = _player(body, "Alpha Back")
        assert rb["floor"] == 8.2
        assert rb["ceiling"] == 44.3
        assert "Floor 8.2 to ceiling 44.3" in rb["summary"]

    def test_value_orders_within_a_tier(self, client):
        body = _get(client).json()
        assert _player(body, "Cheap Back")["value"] > _player(body, "Alpha Back")["value"]

    def test_an_unrosterable_player_is_absent(self, client):
        body = _get(client).json()
        with pytest.raises(KeyError):
            _player(body, "Hurt Back")
        assert "O" in body["excluded_injury_flags"]

    def test_a_questionable_player_is_present_and_flagged(self, client):
        body = _get(client).json()
        p = _player(body, "Quest Catch")
        assert p["injury_flag"] == "Q"
        assert "questionable" in p["summary"]

    def test_a_defense_comes_back_for_DEF(self, client):
        body = _get(client, positions="DEF").json()
        names = [p["player"] for g in body["groups"] for p in g["players"]]
        assert names == ["A Defense"]

    def test_the_bands_are_reported_in_dollars(self, client):
        body = _get(client, premium_min=8000, value_max=6000).json()
        bands = {(g["position"], g["tier"]): g["band"] for g in body["groups"]}
        assert bands[("RB", "premium")] == "$8,000 and up"
        assert bands[("RB", "value")] == "under $6,000"


class TestItSaysWhatItDidNotUse:
    def test_the_response_states_that_context_moved_nobody(self, client):
        body = _get(client).json()
        assert "did not affect any ordering" in body["context_only"]
        assert "never an input" in body["no_vegas"]
        assert body["ranked_by"] == "our projected points per $1,000 of salary"


class TestTheSlateTrap:
    def test_it_says_how_many_games_the_file_holds(self, client):
        body = _get(client).json()
        assert body["slate"]["games_in_file"] == 4
        assert "not the main slate" in body["slate_note"]

    def test_restricting_to_games_drops_the_rest(self, client):
        body = _get(client, games="DET@MIA").json()
        names = {p["player"] for g in body["groups"] for p in g["players"]}
        assert names == {"Alpha Back", "Alpha Arm"}
        assert "Restricted to 1 game" in body["slate_note"]

    def test_a_game_not_on_the_slate_is_refused(self, client):
        r = _get(client, games="KC@BUF")
        assert r.status_code == 400
        assert "no game(s) KC@BUF" in r.json()["detail"]


class TestProvenanceIsReportedFirst:
    def test_a_board_that_benched_nobody_from_the_report_says_so(
            self, client, db_session):
        db_session.add(ProjectionRun(
            season=SEASON, week=WEEK, as_of_week=8,
            computed_at=datetime(2026, 11, 4, 21, 32),
            injury_report_available=True, injury_report_rows=28,
            injury_report_players=18, ir_players=106, players_zeroed=106,
            questionable=18))
        db_session.commit()
        body = _get(client).json()
        prov = body["provenance"]
        assert prov["players_zeroed"] == 106
        assert any("came off roster status" in w for w in prov["warnings"])

    def test_no_run_recorded_is_null_not_a_claim(self, client):
        assert _get(client).json()["provenance"] is None


class TestRefusals:
    def test_no_slate(self, client, db_session):
        db_session.query(FanduelSlate).delete()
        db_session.commit()
        r = _get(client)
        assert r.status_code == 404
        assert "Upload the contest player list" in r.json()["detail"]

    def test_no_board(self, client, db_session):
        db_session.query(PlayerProjection).delete()
        db_session.commit()
        assert _get(client).status_code == 404


def _player(body, name):
    for g in body["groups"]:
        for p in g["players"]:
            if p["player"] == name:
                return p
    raise KeyError(name)
