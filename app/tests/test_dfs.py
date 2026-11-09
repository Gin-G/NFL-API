#!/usr/bin/env python3
"""The price-tier targets endpoint."""

import hashlib
from datetime import datetime

import pytest

from database.models import (FanduelSlate, PlayerProjection, PlayerStat,
                             ProjectionRun)

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
    for model in (FanduelSlate, PlayerProjection, PlayerStat,
                  ProjectionRun):
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


def _played(db_session, name, weeks_points, player_id="p1"):
    """Give a player a weekly record, in the shape nflverse stores it."""
    for week, points in weeks_points:
        # Built from receptions (0.5 each) and the remainder in yards, because
        # receptions carry no bonus threshold: 312 rushing yards would be 31.2
        # points PLUS the 100-yard bonus, and the fixture would not say what it
        # meant to say.
        catches = int(round(points / 0.5)) - 1
        yards = round((points - 0.5 * catches) * 10)
        db_session.add(PlayerStat(
            season=SEASON, week=week, season_type="REG",
            player_id=player_id, player_name=name,
            player_display_name=name, position="RB", recent_team="DET",
            receptions=catches, receiving_yards=yards))
    db_session.commit()


class TestWhatHeHasActuallyScored:
    """The range is description, not a band: measured on 617 player-weeks it
    covers less of the next result than the model's floor/ceiling does."""

    def test_the_range_is_on_the_row_and_in_the_summary(self, client, db_session):
        _played(db_session, "Alpha Back",
                [(1, 31.2), (2, 8.4), (3, 19.0), (4, 22.6), (5, 11.1),
                 (6, 27.3), (7, 4.2), (8, 14.0)])
        player = _player(_get(client).json(), "Alpha Back")
        assert player["history"] == {"games": 8, "low": 4.2, "high": 31.2,
                                     "median": 16.5}
        assert "Has scored 4.2 to 31.2 in 8 games" in player["summary"]

    def test_it_only_counts_weeks_before_the_slate(self, client, db_session):
        """A range holding the result being projected is hindsight."""
        _played(db_session, "Alpha Back", [(1, 10.0), (2, 12.0), (3, 14.0),
                                           (WEEK, 99.0)])
        player = _player(_get(client).json(), "Alpha Back")
        assert player["history"]["games"] == 3
        assert player["history"]["high"] == 14.0

    def test_a_player_with_no_record_has_no_range(self, client, db_session):
        _played(db_session, "Alpha Back", [(1, 10.0), (2, 12.0), (3, 14.0)])
        assert _player(_get(client).json(), "Cheap Back")["history"] is None

    def test_it_can_be_turned_off(self, client, db_session):
        _played(db_session, "Alpha Back", [(1, 10.0), (2, 12.0), (3, 14.0)])
        body = _get(client, history="false").json()
        assert _player(body, "Alpha Back")["history"] is None
        assert "not attached" in body["history_note"]

    def test_the_note_says_it_does_not_rank(self, client, db_session):
        body = _get(client).json()
        assert "never enters a projection, a value or a rank" in body["history_note"]

    def test_the_range_does_not_reorder_anything(self, client, db_session):
        plain = _get(client).json()
        _played(db_session, "Cheap Back", [(1, 70.0), (2, 72.0), (3, 75.0)],
                player_id="p2")
        _played(db_session, "Alpha Back", [(1, 0.1), (2, 0.2), (3, 0.3)])
        loud = _get(client).json()

        def order(body):
            return [(g["position"], g["tier"], [p["player"] for p in g["players"]])
                    for g in body["groups"]]

        assert order(plain) == order(loud)

    def test_projecting_above_his_best_is_said_outright(self, client, db_session):
        _played(db_session, "Alpha Back", [(1, 2.0), (2, 3.0), (3, 4.0)])
        player = _player(_get(client).json(), "Alpha Back")
        assert "We project above anything done this season." in player["summary"]


class TestAPositionWithNoGroup:
    """DEF vanished from the response without a word, because the board holds
    no defenses — ten groups where eleven were asked for reads as a bad week."""

    def test_a_board_with_defenses_has_a_def_group(self, client):
        body = _get(client).json()
        assert any(g["position"] == "D" for g in body["groups"])
        # TE really is absent from this slate, which is the other half of the
        # point: the list names what is missing rather than claiming nothing is.
        assert "D" not in body["positions_absent"]
        assert body["positions_absent"] == ["TE"]

    def test_without_them_it_says_so_and_says_why(self, client, db_session):
        db_session.query(PlayerProjection).filter(
            PlayerProjection.position == "D").delete()
        db_session.commit()
        body = _get(client).json()
        assert not any(g["position"] == "D" for g in body["groups"])
        assert "D" in body["positions_absent"]
        assert "not projected by the model" in body["positions_absent_note"]
        assert "FanDuel's own FPPG" in body["positions_absent_note"]

    def test_a_position_not_asked_for_is_not_reported_absent(self, client):
        body = _get(client, positions="QB,RB").json()
        assert body["positions_requested"] == ["QB", "RB"]
        assert body["positions_absent"] == []


def _season_weeks(db_session, rows):
    """Weekly stat lines for the consistency table."""
    for name, pid, position, team, weeks in rows:
        for week, (carries, targets, rec_yards) in enumerate(weeks, 1):
            db_session.add(PlayerStat(
                season=SEASON, week=week, season_type="REG",
                player_id=pid, player_name=name,
                player_display_name=name, position=position, recent_team=team,
                carries=carries, rushing_yards=carries * 4.0,
                targets=targets, receptions=targets,
                receiving_yards=rec_yards))
    db_session.commit()


STEADY_WEEKS = [(18, 3, 20), (19, 2, 18), (17, 3, 22),
                (18, 3, 19), (19, 2, 21), (18, 3, 20)]
SPIKY_WEEKS = [(4, 1, 5), (28, 6, 60), (3, 0, 0),
               (26, 5, 55), (2, 1, 8), (30, 4, 50)]


class TestConsistency:
    """Ranked on volume, because scoring spread does not persist."""

    def _two_backs(self, db_session):
        _season_weeks(db_session, [
            ("Steady Back", "s1", "RB", "DET", STEADY_WEEKS),
            ("Spiky Back", "s2", "RB", "CAR", SPIKY_WEEKS)])

    def test_the_steady_back_comes_first(self, client, db_session):
        self._two_backs(db_session)
        body = client.get("/dfs/consistency",
                          params={"season": SEASON, "min_games": 4}).json()
        assert body["status"] == "success"
        assert [r["player"] for r in body["data"]] == ["Steady Back", "Spiky Back"]
        assert body["data"][0]["volume_steadiness"] > \
            body["data"][1]["volume_steadiness"]

    def test_a_row_carries_volume_trend_and_scoring(self, client, db_session):
        self._two_backs(db_session)
        row = client.get("/dfs/consistency",
                         params={"season": SEASON}).json()["data"][0]
        for key in ("opportunities_per_game", "opportunity_low", "opportunity_high",
                    "opportunity_cv", "volume_steadiness", "opportunity_trend",
                    "points_per_game", "points_low", "points_high", "points_cv",
                    "hit_rate_15", "summary"):
            assert key in row, key
        assert row["games"] == 6
        assert "opportunities a game" in row["summary"]

    def test_it_says_why_it_does_not_rank_on_scoring(self, client, db_session):
        self._two_backs(db_session)
        body = client.get("/dfs/consistency", params={"season": SEASON}).json()
        assert "volume steadiness" in body["ranked_by"]
        assert "not a property of a player" in body["why_not_scoring"]
        assert "+0.412" in body["what_does_measure"]
        assert "read `games` before the rank" in body["small_sample_warning"]

    def test_a_bit_part_is_not_the_steadiest_player_alive(self, client, db_session):
        """Two targets every week is a tiny spread and a useless player."""
        self._two_backs(db_session)
        _season_weeks(db_session, [
            ("Tiny Role", "s3", "WR", "ARI", [(0, 2, 10)] * 6)])
        body = client.get("/dfs/consistency",
                          params={"season": SEASON}).json()
        assert "Tiny Role" not in [r["player"] for r in body["data"]]

    def test_the_volume_floor_can_be_moved(self, client, db_session):
        self._two_backs(db_session)
        _season_weeks(db_session, [
            ("Tiny Role", "s3", "WR", "ARI", [(0, 2, 10)] * 6)])
        body = client.get("/dfs/consistency",
                          params={"season": SEASON, "min_opportunities": 0}).json()
        assert "Tiny Role" in [r["player"] for r in body["data"]]

    def test_only_weeks_before_the_slate_are_counted(self, client, db_session):
        self._two_backs(db_session)
        body = client.get("/dfs/consistency",
                          params={"season": SEASON, "week": 5,
                                  "min_games": 4}).json()
        assert all(r["games"] == 4 for r in body["data"])

    def test_positions_can_be_narrowed(self, client, db_session):
        self._two_backs(db_session)
        _season_weeks(db_session, [
            ("A Receiver", "s4", "WR", "LAR", STEADY_WEEKS)])
        body = client.get("/dfs/consistency",
                          params={"season": SEASON, "positions": "WR"}).json()
        assert {r["position"] for r in body["data"]} == {"WR"}

    def test_too_few_games_is_refused_with_the_reason(self, client, db_session):
        _season_weeks(db_session, [
            ("Short Back", "s5", "RB", "LV", STEADY_WEEKS[:2])])
        response = client.get("/dfs/consistency",
                              params={"season": SEASON, "min_games": 6})
        assert response.status_code == 404
        assert "games" in response.json()["detail"]

    def test_a_season_with_no_stats_says_so(self, client, db_session):
        response = client.get("/dfs/consistency", params={"season": 1999})
        assert response.status_code == 404
        assert "No player stats stored" in response.json()["detail"]

    def test_scoring_noise_cannot_reorder_the_table(self, client, db_session):
        """The whole design, end to end: change the scoring, keep the usage,
        and the ranking must not move."""
        self._two_backs(db_session)
        before = [r["player"] for r in client.get(
            "/dfs/consistency", params={"season": SEASON}).json()["data"]]
        # Same carries and targets, wildly different yardage.
        for row in db_session.query(PlayerStat).filter(
                PlayerStat.player_name == "Steady Back").all():
            row.receiving_yards = 0.0 if row.week % 2 else 220.0
        db_session.commit()
        after_body = client.get("/dfs/consistency",
                                params={"season": SEASON}).json()
        assert [r["player"] for r in after_body["data"]] == before
        # ... and the scoring columns really did move.
        assert after_body["data"][0]["points_sd"] > 0
