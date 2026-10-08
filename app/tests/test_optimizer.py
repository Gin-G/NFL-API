#!/usr/bin/env python3
"""Tests for the FanDuel lineup optimizer endpoint.

The slate CSV is the input because FanDuel's pricing is not fetchable anywhere
free and live — every alternative was dead, historical, licensed or
session-gated. So the endpoint has to be forgiving about what gets uploaded and
honest about what it could not match, which is most of what these pin.
"""

import io
from datetime import datetime

import pytest

from database.models import PlayerProjection

pytest.importorskip("nfl_projections.optimizer",
                    reason="optimizer comes from nfl_projections (installed --no-deps in the image)")


@pytest.fixture(autouse=True)
def _clean(db_session):
    db_session.query(PlayerProjection).delete()
    db_session.commit()
    yield


# A FanDuel slate export, trimmed to the columns the optimizer reads. Enough
# bodies to fill QB/2RB/3WR/TE/FLEX/DEF with alternatives at each spot.
#
# "Roster Position" carries FanDuel's own flex-eligibility strings — RB/FLEX,
# not RB — which is what nfl_projections.optimizer matches its slots against,
# and what a real export contains.
#
# Salaries are deliberately well under the $60k cap in aggregate. Priced near
# the cap, only one legal lineup fits and any test about lineup DIVERSITY is
# really testing the cap.
SLATE_ROWS = [
    ("Josh Allen", "QB", "QB", 4500, 22.0, "BUF", ""),
    ("Jared Goff", "QB", "QB", 3750, 18.0, "DET", ""),
    ("Bijan Robinson", "RB", "RB/FLEX", 4250, 18.5, "ATL", ""),
    ("Jahmyr Gibbs", "RB", "RB/FLEX", 4100, 18.0, "DET", ""),
    ("Derrick Henry", "RB", "RB/FLEX", 3900, 17.0, "BAL", ""),
    ("Kyren Williams", "RB", "RB/FLEX", 3250, 13.0, "LA", ""),
    ("Puka Nacua", "WR", "WR/FLEX", 4000, 17.0, "LA", ""),
    ("Amon-Ra St. Brown", "WR", "WR/FLEX", 3950, 16.5, "DET", ""),
    ("Ja'Marr Chase", "WR", "WR/FLEX", 3850, 16.0, "CIN", ""),
    ("Chris Olave", "WR", "WR/FLEX", 3400, 14.0, "NO", ""),
    ("Davante Adams", "WR", "WR/FLEX", 3300, 13.5, "LA", ""),
    ("Trey McBride", "TE", "TE/FLEX", 3000, 13.0, "ARI", ""),
    ("Dallas Goedert", "TE", "TE/FLEX", 2600, 11.0, "PHI", ""),
    ("Hurt Guy", "WR", "WR/FLEX", 2000, 9.0, "NYJ", "O"),
    ("Ravens", "D", "DEF", 2100, 8.5, "BAL", ""),
    ("Broncos", "D", "DEF", 1950, 8.0, "DEN", ""),
]


def _slate_csv(rows=None):
    rows = rows if rows is not None else SLATE_ROWS
    lines = ["Id,Nickname,Position,Roster Position,Salary,FPPG,Team,Injury Indicator"]
    for i, (name, pos, roster_pos, salary, fppg, team, inj) in enumerate(rows):
        lines.append(f"{i},{name},{pos},{roster_pos},{salary},{fppg},{team},{inj}")
    return io.BytesIO("\n".join(lines).encode())


def _project(db, names_points, season=2026, week=5):
    """Store a published board. Salaries come from the CSV, points from here."""
    for i, (name, points) in enumerate(names_points):
        db.add(PlayerProjection(
            season=season, week=week, player_id=f"00-{i:04d}", player_name=name,
            position="RB", team="DAL", projected_points=points,
            floor=points * 0.4, median=points * 0.95, ceiling=points * 1.8,
            prediction_type="veteran_ml", model_version="1.5.0",
            computed_at=datetime.utcnow()))
    db.commit()


def _board():
    return [(name, fppg + 2) for name, _, roster_pos, _, fppg, _, _ in SLATE_ROWS
            if roster_pos != "DEF"]


def _post(client, csv=None, **form):
    data = {"season": 2026, "week": 5, **form}
    return client.post("/optimizer/lineups",
                       files={"file": ("slate.csv", csv or _slate_csv(), "text/csv")},
                       data=data)


class TestLineups:
    def test_builds_a_legal_lineup(self, client, db_session):
        _project(db_session, _board())
        body = _post(client, num_lineups=1).json()

        assert body["status"] == "success"
        (lineup,) = body["data"]
        assert len(lineup["players"]) == 9        # QB, 2RB, 3WR, TE, FLEX, DEF
        assert lineup["salary"] <= 60000
        slots = [p["roster_position"] for p in lineup["players"]]
        assert slots.count("QB") == 1 and slots.count("DEF") == 1

    def test_reports_how_much_of_the_slate_it_matched(self, client, db_session):
        _project(db_session, _board())
        body = _post(client, num_lineups=1).json()
        # Everyone but the two defenses should join to the board.
        assert body["matched_to_projections"] == len(SLATE_ROWS) - 2
        assert body["slate_players"] == len(SLATE_ROWS)

    def test_defenses_come_from_fanduels_own_fppg(self, client, db_session):
        """The model projects no defenses, so the DEF slot has to fall back."""
        _project(db_session, _board())
        body = _post(client, num_lineups=1).json()
        defense = next(p for p in body["data"][0]["players"]
                       if p["roster_position"] == "DEF")
        assert defense["from_model"] is False
        assert defense["projected"] in (8.5, 8.0)

    def test_several_lineups_are_not_one_lineup_repeated(self, client, db_session):
        # 100% exposure: this fixture slate is 16 players, and the default 50%
        # cap means a player may appear in at most half the lineups, which a
        # slate this thin cannot satisfy. A real slate is hundreds deep.
        _project(db_session, _board())
        body = _post(client, num_lineups=3, max_usage_percentage=100).json()
        signatures = {tuple(sorted(p["player_name"] for p in l["players"]))
                      for l in body["data"]}
        assert len(signatures) == len(body["data"]) > 1

    def test_says_so_when_the_exposure_cap_limits_the_count(self, client, db_session):
        _project(db_session, _board())
        body = _post(client, num_lineups=3, max_usage_percentage=50).json()
        assert body["requested"] == 3
        assert body["count"] < 3
        assert "exposure cap" in body["note"]

    def test_no_note_when_it_delivered_what_was_asked(self, client, db_session):
        _project(db_session, _board())
        assert _post(client, num_lineups=1).json()["note"] is None

    def test_injured_players_are_dropped(self, client, db_session):
        _project(db_session, _board())
        body = _post(client, num_lineups=3).json()
        names = {p["player_name"] for l in body["data"] for p in l["players"]}
        assert "Hurt Guy" not in names        # Injury Indicator "O"

    def test_exclusions_are_honoured(self, client, db_session):
        _project(db_session, _board())
        body = _post(client, num_lineups=2, exclude="Josh Allen").json()
        names = {p["player_name"] for l in body["data"] for p in l["players"]}
        assert "Josh Allen" not in names
        assert "Jared Goff" in names

    def test_ceiling_objective_is_accepted(self, client, db_session):
        _project(db_session, _board())
        assert _post(client, num_lineups=1, objective="ceiling").json()["objective"] == (
            "ceiling")


class TestBadInput:
    def test_a_csv_that_is_not_a_slate_export(self, client, db_session):
        _project(db_session, _board())
        resp = _post(client, csv=io.BytesIO(b"name,points\nSomebody,12\n"))
        assert resp.status_code == 400
        assert "Nickname" in resp.json()["detail"]

    def test_an_empty_file(self, client, db_session):
        _project(db_session, _board())
        assert _post(client, csv=io.BytesIO(b"")).status_code == 400

    def test_a_week_with_no_projections(self, client, db_session):
        _project(db_session, _board(), week=5)
        resp = _post(client, week=9)
        assert resp.status_code == 404
        assert "week 9" in resp.json()["detail"]

    def test_constraints_nothing_can_satisfy(self, client, db_session):
        _project(db_session, _board())
        resp = _post(client, num_lineups=1, salary_cap=1000)
        assert resp.status_code == 422
        assert "salary cap" in resp.json()["detail"]

    def test_absurd_lineup_counts_are_refused(self, client, db_session):
        _project(db_session, _board())
        assert _post(client, num_lineups=500).status_code == 400


class TestSlateCoverage:
    def test_describes_the_board_before_any_upload(self, client, db_session):
        _project(db_session, [("A", 10.0), ("B", 12.0)])
        body = client.get("/optimizer/slate-coverage?season=2026&week=5").json()
        assert body["projected_players"] == 2
        assert body["by_position"] == {"RB": 2}
        assert "DEF is not projected" in body["note"]

    def test_no_projections_says_so(self, client):
        assert client.get("/optimizer/slate-coverage?season=2026&week=5").json()["status"] == (
            "no_data")
