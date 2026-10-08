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

from database.models import PlayerProjection, Schedule

pytest.importorskip("nfl_projections.optimizer",
                    reason="optimizer comes from nfl_projections (installed --no-deps in the image)")


@pytest.fixture(autouse=True)
def _clean(db_session):
    """The engine is session-scoped and these tests commit, so both tables have
    to be cleared or the second schedule insert collides on its primary key."""
    db_session.query(PlayerProjection).delete()
    db_session.query(Schedule).delete()
    db_session.commit()
    yield


# A FanDuel slate export, trimmed to the columns the optimizer reads. Enough
# bodies to fill QB/2RB/3WR/TE/FLEX/DEF with alternatives at each spot.
#
# "Roster Position" here carries BARE positions — RB, not RB/FLEX — because
# that is what the first real slate upload contained, and matching the
# flex-qualified spelling exactly produced zero lineups with nothing to say why.
# TestBothRosterFormats covers the other spelling.
#
# Salaries are deliberately well under the $60k cap in aggregate. Priced near
# the cap, only one legal lineup fits and any test about lineup DIVERSITY is
# really testing the cap.
SLATE_ROWS = [
    ("Josh Allen", "QB", "QB", 4500, 22.0, "BUF", ""),
    ("Jared Goff", "QB", "QB", 3750, 18.0, "DET", ""),
    ("Bijan Robinson", "RB", "RB", 4250, 18.5, "ATL", ""),
    ("Jahmyr Gibbs", "RB", "RB", 4100, 18.0, "DET", ""),
    ("Derrick Henry", "RB", "RB", 3900, 17.0, "BAL", ""),
    ("Kyren Williams", "RB", "RB", 3250, 13.0, "LA", ""),
    ("Puka Nacua", "WR", "WR", 4000, 17.0, "LA", ""),
    ("Amon-Ra St. Brown", "WR", "WR", 3950, 16.5, "DET", ""),
    ("Ja'Marr Chase", "WR", "WR", 3850, 16.0, "CIN", ""),
    ("Chris Olave", "WR", "WR", 3400, 14.0, "NO", ""),
    ("Davante Adams", "WR", "WR", 3300, 13.5, "LA", ""),
    ("Trey McBride", "TE", "TE", 3000, 13.0, "ARI", ""),
    ("Dallas Goedert", "TE", "TE", 2600, 11.0, "PHI", ""),
    ("Hurt Guy", "WR", "WR", 2000, 9.0, "NYJ", "O"),
    ("Ravens", "D", "D", 2100, 8.5, "BAL", ""),
    ("Broncos", "D", "D", 1950, 8.0, "DEN", ""),
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
    return [(name, fppg + 2) for name, pos, _, _, fppg, _, _ in SLATE_ROWS
            if pos != "D"]


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
        assert slots.count("QB") == 1 and slots.count("D") == 1

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
                       if p["roster_position"] == "D")
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


class TestBothRosterFormats:
    """Either spelling of Roster Position has to work — the export has used both."""

    def test_flex_qualified_spelling_also_builds(self, client, db_session):
        rows = [(name, pos,
                 {"RB": "RB/FLEX", "WR": "WR/FLEX", "TE": "TE/FLEX", "D": "DEF"}.get(pos, pos),
                 salary, fppg, team, inj)
                for name, pos, _, salary, fppg, team, inj in SLATE_ROWS]
        _project(db_session, _board())
        body = _post(client, csv=_slate_csv(rows), num_lineups=1).json()
        assert len(body["data"][0]["players"]) == 9

    def test_an_unrecognised_format_says_which_slots_are_empty(self, client, db_session):
        # Garbage in the roster column: every slot is 0 eligible, and the error
        # should say so rather than blame the salary cap.
        rows = [(name, "ZZ", "ZZ", salary, fppg, team, inj)
                for name, _, _, salary, fppg, team, inj in SLATE_ROWS]
        _project(db_session, _board())
        resp = _post(client, csv=_slate_csv(rows), num_lineups=1)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "QB (0 eligible)" in detail
        assert "format we do not recognise" in detail


class TestSlates:
    """The export is always the whole Thursday-Monday list; the contest may not be."""

    @staticmethod
    def _schedule(db):
        from database.models import Schedule

        games = [
            ("2026_05_SEA_DEN", "2026-10-15", "20:15", "SEA", "DEN"),    # Thursday
            ("2026_05_HOU_JAX", "2026-10-18", "09:30", "HOU", "JAX"),    # London
            ("2026_05_CHI_ATL", "2026-10-18", "13:00", "CHI", "ATL"),    # main
            ("2026_05_BUF_LV", "2026-10-18", "16:25", "BUF", "LV"),      # main
            ("2026_05_DAL_GB", "2026-10-18", "20:20", "DAL", "GB"),      # Sunday night
            ("2026_05_WAS_SF", "2026-10-19", "20:15", "WAS", "SF"),      # Monday
        ]
        for gid, day, time, away, home in games:
            db.add(Schedule(game_id=gid, season=2026, week=5, game_type="REG",
                            gameday=day, gametime=time, away_team=away, home_team=home))
        db.commit()

    def _teams(self, db, slate):
        from api.optimizer import _slate_teams

        self._schedule(db)
        teams, games = _slate_teams(db, 2026, 5, slate)
        return teams, games

    def test_main_is_the_one_and_four_oclock_windows(self, db_session):
        teams, games = self._teams(db_session, "main")
        assert teams == {"CHI", "ATL", "BUF", "LV"}
        assert len(games) == 2        # no London, no Sunday night, no Thu/Mon

    def test_sunday_includes_london_and_sunday_night(self, db_session):
        teams, _ = self._teams(db_session, "sunday")
        assert teams == {"HOU", "JAX", "CHI", "ATL", "BUF", "LV", "DAL", "GB"}

    def test_primetime_is_thursday_sunday_night_and_monday(self, db_session):
        teams, _ = self._teams(db_session, "primetime")
        assert teams == {"SEA", "DEN", "DAL", "GB", "WAS", "SF"}

    def test_all_means_no_filtering(self, db_session):
        teams, games = self._teams(db_session, "all")
        assert teams is None and games == []

    def test_filtering_drops_the_other_games_players(self, client, db_session):
        self._schedule(db_session)
        _project(db_session, _board())
        # Fixture teams are BUF/DET/ATL/BAL/LA/CIN/NO/ARI/PHI/NYJ/DEN — of those,
        # only ATL and BUF are in the main slate, so a lineup cannot be filled.
        resp = _post(client, num_lineups=1, slate="main")
        assert resp.status_code == 422

    def test_the_response_names_the_games(self, client, db_session):
        self._schedule(db_session)
        _project(db_session, _board())
        body = _post(client, num_lineups=1, slate="all").json()
        assert body["slate"] == "all"
        assert body["players_in_file"] == len(SLATE_ROWS)


class TestShowdownRouting:
    """A Showdown export is detected and routed, not forced through classic rules."""

    @staticmethod
    def _showdown_csv():
        rows = [
            ("CeeDee Lamb", "WR", 12600, 18900, 25.7),
            ("Dak Prescott", "QB", 11800, 17700, 21.3),
            ("Bucky Irving", "RB", 11200, 16800, 10.2),
            ("Mike Evans", "WR", 9800, 14700, 14.1),
            ("Jake Ferguson", "TE", 7600, 11400, 9.4),
            ("Cheap Flyer", "WR", 4500, 6750, 5.0),
            ("Deep Guy", "TE", 3500, 5250, 3.0),
            ("Dallas Cowboys", "D", 4200, 6300, 7.5),
        ]
        lines = ["Id,Position,Nickname,FPPG,Salary,MVP 1.5x Salary,Game,Team,"
                 "Opponent,Injury Indicator,Roster Position"]
        for i, (name, pos, salary, mvp_salary, fppg) in enumerate(rows):
            lines.append(f"{i},{pos},{name},{fppg},{salary},{mvp_salary},TB@DAL,DAL,TB,,"
                         f"MVP - 1.5X Points/AnyFLEX")
        return io.BytesIO("\n".join(lines).encode())

    def _board(self, db):
        _project(db, [("CeeDee Lamb", 18.0), ("Dak Prescott", 19.0),
                      ("Bucky Irving", 12.0), ("Mike Evans", 13.0),
                      ("Jake Ferguson", 9.0), ("Cheap Flyer", 6.0), ("Deep Guy", 4.0)])

    def test_it_builds_a_five_player_lineup(self, client, db_session):
        self._board(db_session)
        body = _post(client, csv=self._showdown_csv(), num_lineups=1).json()
        assert body["contest"] == "showdown"
        assert body["slate"] == "single_game"
        (lineup,) = body["data"]
        assert len(lineup["players"]) == 5
        assert lineup["players"][0]["roster_position"] == "MVP"
        assert lineup["salary"] <= 60000

    def test_the_slate_filter_is_ignored_for_one_game(self, client, db_session):
        # "main" would drop every team if applied; a Showdown file has one game.
        self._board(db_session)
        body = _post(client, csv=self._showdown_csv(), num_lineups=1, slate="main").json()
        assert body["contest"] == "showdown"
        assert body["games"] == []

    def test_an_impossible_cap_explains_mvp_pricing(self, client, db_session):
        self._board(db_session)
        resp = _post(client, csv=self._showdown_csv(), num_lineups=1, salary_cap=5000)
        assert resp.status_code == 422
        assert "MVP pricing" in resp.json()["detail"]

    def test_a_classic_file_still_goes_through_classic(self, client, db_session):
        _project(db_session, _board())
        assert _post(client, num_lineups=1).json()["contest"] == "classic"

    def test_one_kicker_by_default(self, client, db_session):
        self._board(db_session)
        body = _post(client, csv=self._showdown_csv(), num_lineups=1).json()
        assert body["position_limits"] == {"K": 1, "D": 1}

    def test_the_limit_can_be_turned_off(self, client, db_session):
        self._board(db_session)
        body = _post(client, csv=self._showdown_csv(), num_lineups=1,
                     one_kicker=False).json()
        assert body["position_limits"] is None

    def test_classic_is_unaffected_by_it(self, client, db_session):
        _project(db_session, _board())
        assert _post(client, num_lineups=1).json()["position_limits"] is None
