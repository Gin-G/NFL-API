#!/usr/bin/env python3
"""Hand-uploaded third-party matchup tables: upload, storage, comparison, gate.

Fixtures are the real shape of a FantasyGuru week 9 2026 export, including the
two columns both headed `Team`.
"""

import io
from datetime import datetime

import pytest

from database.models import (ExternalSnapshot, ExternalTeamLine,
                             ExternalWrMatchup, PlayerProjection, PlayerStat)
from scripts import grade_external

SEASON = 2026
WEEK = 9

WR_CSV = (
    "Rank,Defensive Back,Team,Coverage Type,Advantage,Primary Receiver,Team\n"
    "1,Jason Marshall,MIA,45% Cover-3 Zone,65.38,Jameson Williams,DET\n"
    "2,Paulson Adebo,NYG,45% Quarters Zone,62.86,Darius Cooper,PHI\n"
    "3,Travis Hunter,JAC,100% Shadow Coverage,51.20,Zay Flowers,BAL\n"
)
TEAM_CSV = ("Team,Offensive Line,Defensive Front\n"
            "Baltimore,35.97,38.69\n"
            "Jacksonville,49.54,50.45\n"
            "LA Chargers,40.10,31.00\n")
LINE_CSV = ("AWAY,O-LINE (AWAY),D-LINE (AWAY),O-LINE ADV (AWAY),"
            "O-LINE ADV (HOME),O-LINE (HOME),D-LINE (HOME),HOME\n"
            "JAC,49.54,50.45,10.848,-14.486,35.97,38.69,BAL\n")


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in (ExternalWrMatchup, ExternalTeamLine, ExternalSnapshot,
                  PlayerProjection, PlayerStat):
        db_session.query(model).delete()
    db_session.commit()
    yield


@pytest.fixture(autouse=True)
def _offline_player_ids(monkeypatch):
    """Keep the id mapping off the network.

    The router maps names onto gsis ids from an nflverse roster download. It
    already soft-fails without one, but a suite that passes differently with
    and without network is not a suite.
    """
    from nfl_projections import external as ext

    ROSTER = {"zay flowers": "00-0039337",
              "jameson williams": "00-0037240",
              "darius cooper": "00-0040024"}

    def fake(frame, columns, roster=None, season=None):
        pairs = (columns.items() if hasattr(columns, "items")
                 else [(c, None) for c in columns])
        out = frame.copy()
        for name_col, _team in pairs:
            if name_col not in out.columns:
                continue
            out[f"{name_col}_id"] = [ROSTER.get(str(v).lower()) for v in out[name_col]]
            out[f"{name_col}_match"] = ["team" if ROSTER.get(str(v).lower()) else None
                                        for v in out[name_col]]
        return out

    monkeypatch.setattr(ext, "attach_player_ids", fake)


def _upload(client, text, *, week=WEEK, kind=None, filename="table.csv",
            season=SEASON, source="fantasyguru"):
    data = {"week": str(week), "season": str(season), "source": source}
    if kind:
        data["kind"] = kind
    return client.post("/external/upload", data=data,
                       files={"file": (filename, io.BytesIO(text.encode()),
                                       "text/csv")})


class TestUpload:
    def test_team_lines_are_shredded_onto_our_abbreviations(self, client, db_session):
        body = _upload(client, TEAM_CSV).json()
        assert body["status"] == "success"
        assert body["kind"] == "team_lines"
        assert body["teams"] == 3
        teams = {t.team: t for t in db_session.query(ExternalTeamLine).all()}
        assert set(teams) == {"BAL", "JAX", "LAC"}          # JAC -> JAX
        assert teams["LAC"].team_raw == "LA Chargers"       # raw spelling kept
        assert teams["BAL"].offensive_line == 35.97

    def test_wr_coverage_keeps_both_teams_and_parses_the_shell(self, client,
                                                               db_session):
        body = _upload(client, WR_CSV).json()
        assert body["kind"] == "wr_coverage"
        assert body["pairings"] == 3
        rows = {r.fg_rank: r for r in db_session.query(ExternalWrMatchup).all()}
        # The duplicate-header trap: both teams have to survive.
        assert (rows[1].defender_team, rows[1].receiver_team) == ("MIA", "DET")
        assert rows[2].shell == "COVER_4"          # Quarters
        assert rows[1].man_zone == "ZONE_COVERAGE"
        assert rows[3].shadow is True and rows[3].shell is None
        assert rows[3].defender_team == "JAX"      # JAC normalised

    def test_player_ids_are_attached_and_how_is_recorded(self, client, db_session):
        _upload(client, WR_CSV)
        row = db_session.query(ExternalWrMatchup).filter_by(fg_rank=3).one()
        assert row.receiver_id == "00-0039337"
        assert row.receiver_match == "team"

    def test_unmapped_names_are_reported_not_dropped(self, client, db_session):
        body = _upload(client, WR_CSV).json()
        # The defenders are not in the stub roster, so they come back unmapped.
        assert body["unmapped"]
        assert body["pairings"] == 3            # every row still stored
        snap = db_session.query(ExternalSnapshot).one()
        assert snap.unmapped and "Jason Marshall" in snap.unmapped

    def test_the_file_is_kept_verbatim(self, client, db_session):
        _upload(client, TEAM_CSV)
        snap = db_session.query(ExternalSnapshot).one()
        assert snap.content == TEAM_CSV
        assert snap.row_count == 3
        assert len(snap.content_hash) == 64

    def test_a_bom_does_not_survive_into_the_stored_text(self, client, db_session):
        _upload(client, "﻿" + TEAM_CSV)
        snap = db_session.query(ExternalSnapshot).one()
        assert not snap.content.startswith("﻿")
        assert db_session.query(ExternalTeamLine).count() == 3

    def test_a_reupload_replaces_rather_than_accumulating(self, client, db_session):
        _upload(client, TEAM_CSV)
        changed = TEAM_CSV.replace("35.97", "40.00")
        body = _upload(client, changed).json()
        assert body["replaced"] == 3
        assert db_session.query(ExternalSnapshot).count() == 1
        assert db_session.query(ExternalTeamLine).count() == 3
        bal = db_session.query(ExternalTeamLine).filter_by(team="BAL").one()
        assert bal.offensive_line == 40.00

    def test_the_kind_is_detected_from_the_header(self, client):
        assert _upload(client, WR_CSV).json()["kind"] == "wr_coverage"
        assert _upload(client, TEAM_CSV).json()["kind"] == "team_lines"

    def test_a_kind_that_disagrees_with_the_file_is_refused(self, client, db_session):
        r = _upload(client, TEAM_CSV, kind="wr_coverage")
        assert r.status_code == 400
        assert "header says team_lines" in r.json()["detail"]
        assert db_session.query(ExternalSnapshot).count() == 0   # nothing stored

    def test_an_unrecognised_file(self, client):
        r = _upload(client, "a,b,c\n1,2,3\n")
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "heads a known export" in detail
        # The refusal names every kind it tried, so the next step is obvious.
        for kind in ("wr_coverage", "team_lines", "line_matchups"):
            assert kind in detail

    def test_an_empty_file(self, client):
        r = _upload(client, "")
        assert r.status_code == 400

    def test_an_absurdly_large_file(self, client):
        r = _upload(client, "Team,Offensive Line,Defensive Front\n"
                    + "Baltimore,1,2\n" * 80000)
        assert r.status_code == 413

    def test_an_unknown_source(self, client):
        r = _upload(client, TEAM_CSV, source="somebody_else")
        assert r.status_code == 400


class TestTheDerivedExportIsCheckedNotStored:
    """Their per-game table is team_lines plus the schedule. Storing it as its
    own fact would let the two drift apart."""

    def test_it_is_not_stored(self, client, db_session):
        body = _upload(client, LINE_CSV).json()
        assert body["status"] == "checked_not_stored"
        assert db_session.query(ExternalSnapshot).count() == 0
        assert db_session.query(ExternalTeamLine).count() == 0

    def test_their_adv_is_checked_against_its_own_arithmetic(self, client):
        body = _upload(client, LINE_CSV).json()
        assert body["adv_self_consistent"] is True
        assert body["sides"] == 2

    def test_a_tampered_adv_is_reported(self, client):
        body = _upload(client, LINE_CSV.replace("10.848", "99.000")).json()
        assert body["adv_self_consistent"] is False
        assert body["adv_disagreements"][0]["team"] == "JAX"

    def test_it_is_checked_against_a_stored_team_lines_week(self, client):
        _upload(client, TEAM_CSV)
        body = _upload(client, LINE_CSV).json()
        assert body["matched_against_team_lines"] is True
        assert body["team_line_drift"] == []

    def test_drift_from_the_stored_week_is_named(self, client):
        _upload(client, TEAM_CSV.replace("35.97", "11.11"))
        body = _upload(client, LINE_CSV).json()
        assert body["team_line_drift"][0]["team"] == "BAL"


class TestReadBack:
    def test_snapshots_list_what_the_harness_is_waiting_for(self, client):
        _upload(client, TEAM_CSV, week=9)
        _upload(client, WR_CSV, week=9)
        _upload(client, WR_CSV, week=10)
        body = client.get(f"/external/snapshots?season={SEASON}").json()
        assert body["count"] == 3
        assert body["weeks_held"] == [9, 10]
        assert body["weeks_with_wr_coverage"] == [9, 10]

    def test_team_lines_read_back_best_line_first(self, client):
        _upload(client, TEAM_CSV)
        body = client.get(f"/external/team-lines?season={SEASON}&week={WEEK}").json()
        assert [r["team"] for r in body["data"]] == ["JAX", "LAC", "BAL"]

    def test_wr_matchups_say_they_are_gated(self, client):
        _upload(client, WR_CSV)
        body = client.get(f"/external/wr-matchups?season={SEASON}&week={WEEK}").json()
        assert body["gated"] is True
        assert body["shadow_assignments"] == 1
        assert "feeds /projections" in body["note"]

    def test_shadow_only_filter(self, client):
        _upload(client, WR_CSV)
        body = client.get(f"/external/wr-matchups?season={SEASON}&week={WEEK}"
                          "&shadow_only=true").json()
        assert body["count"] == 1
        assert body["data"][0]["receiver"] == "Zay Flowers"

    def test_nothing_uploaded(self, client):
        assert client.get(f"/external/team-lines?season={SEASON}"
                          ).json()["status"] == "no_data"


class TestCompareWithRatings:
    """The path that does the work.

    The no-upload guard above returns before `/trenches` is ever consulted, so
    on its own it certified nothing about the comparison. A wrong call into
    `_grades` was reported as `no_ratings` and read exactly like a quiet week.
    """

    @staticmethod
    def _grades_frame():
        import pandas as pd
        return pd.DataFrame([
            {"team": "BAL", "run_block": 55.0, "pass_block": 48.0,
             "run_stop": 61.0, "pass_rush": 44.0},
            {"team": "JAX", "run_block": 62.0, "pass_block": 70.0,
             "run_stop": 40.0, "pass_rush": 52.0},
            {"team": "LAC", "run_block": 30.0, "pass_block": 35.0,
             "run_stop": 50.0, "pass_rush": 66.0},
        ])

    def test_grades_really_are_one_frame_not_a_tuple(self, db_session,
                                                     monkeypatch):
        """Pin the contract the endpoint depends on.

        `_grades` returns a DataFrame, and unpacking one iterates its columns
        rather than failing in any way that points at the cause -- which is
        how the mistake survived a green suite. Called for real, with the
        nflverse read stubbed out so this stays an offline test.
        """
        import pandas as pd
        from api import trenches as trenches_api

        monkeypatch.setattr(trenches_api, "load_trench_plays",
                            lambda db, seasons: (pd.DataFrame(), pd.DataFrame()))
        trenches_api._cache.clear()
        got = trenches_api._grades(db_session, SEASON, WEEK)
        assert isinstance(got, pd.DataFrame)
        assert not isinstance(got, tuple)

    def test_correlations_come_out_against_real_shaped_grades(
            self, client, monkeypatch):
        from api import trenches as trenches_api
        monkeypatch.setattr(trenches_api, "_grades",
                            lambda db, season, week: self._grades_frame())
        _upload(client, TEAM_CSV)
        body = client.get(
            f"/external/compare/trenches?season={SEASON}&week={WEEK}").json()
        assert body["status"] == "success", body
        assert body["teams"] == 3
        corr = body["correlations"]
        assert corr["their_oline_vs_our_run_block"] is not None
        assert corr["their_dfront_vs_our_pass_rush"] is not None
        assert body["self_correlation"] is not None
        assert all(r["our_run_block"] is not None for r in body["data"])

    def test_a_broken_lookup_says_so_instead_of_reading_as_a_quiet_week(
            self, client, monkeypatch):
        from api import trenches as trenches_api

        def boom(db, season, week):
            raise RuntimeError("pbp table went away")

        monkeypatch.setattr(trenches_api, "_grades", boom)
        _upload(client, TEAM_CSV)
        body = client.get(
            f"/external/compare/trenches?season={SEASON}&week={WEEK}").json()
        assert body["status"] == "no_ratings"
        assert "RuntimeError" in body["reason"]
        assert "pbp table went away" in body["reason"]

    def test_an_unrated_week_is_distinguishable_from_a_broken_one(
            self, client, monkeypatch):
        import pandas as pd
        from api import trenches as trenches_api
        monkeypatch.setattr(trenches_api, "_grades",
                            lambda db, season, week: pd.DataFrame())
        _upload(client, TEAM_CSV)
        body = client.get(
            f"/external/compare/trenches?season={SEASON}&week={WEEK}").json()
        assert body["status"] == "no_ratings"
        assert body["reason"] == "no rated weeks"

    def test_coverage_grades_really_are_a_pair(self, monkeypatch):
        """The other module's `_grades` does return two frames.

        Pinned because `/external` calls both, and the one difference between
        them is what the unpack bug turned on.
        """
        import pandas as pd
        from api import coverage as coverage_api

        monkeypatch.setattr(coverage_api, "_corner_weeks",
                            lambda season: pd.DataFrame())
        coverage_api._cache.clear()
        got = coverage_api._grades(SEASON, WEEK)
        assert isinstance(got, tuple) and len(got) == 2

    def test_a_broken_corner_lookup_also_says_why(self, client, monkeypatch):
        from api import coverage as coverage_api

        def boom(season, week):
            raise RuntimeError("charting went away")

        monkeypatch.setattr(coverage_api, "_grades", boom)
        _upload(client, WR_CSV)
        body = client.get(
            f"/external/compare/coverage?season={SEASON}&week={WEEK}").json()
        assert body["status"] == "no_grades"
        assert "charting went away" in body["reason"]


class TestCompareWithoutAnUpload:
    def test_trenches(self, client):
        body = client.get(f"/external/compare/trenches?season={SEASON}&week={WEEK}").json()
        assert body["status"] == "no_data"
        assert "/external/upload" in body["message"]

    def test_coverage(self, client):
        body = client.get(f"/external/compare/coverage?season={SEASON}&week={WEEK}").json()
        assert body["status"] == "no_data"


class TestSpearman:
    def test_perfect_agreement(self):
        from api.external import _spearman
        assert _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0

    def test_perfect_disagreement(self):
        from api.external import _spearman
        assert _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0

    def test_too_few_pairs_is_none_not_zero(self):
        from api.external import _spearman
        assert _spearman([1, 2], [1, 2]) is None

    def test_a_constant_column_is_none(self):
        from api.external import _spearman
        assert _spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None

    def test_nulls_are_dropped_pairwise(self):
        from api.external import _spearman
        assert _spearman([1, 2, 3, 4, None], [10, 20, 30, 40, 50]) == 1.0


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def _board(db, week, player_id, receptions, yards):
    db.add(PlayerProjection(
        season=SEASON, week=week, player_id=player_id, player_name=player_id,
        position="WR", team="BAL", projected_points=12.0,
        receptions=receptions, receiving_yards=yards,
        prediction_type="veteran_ml", model_version="1.6.0",
        computed_at=datetime.utcnow()))


def _actual(db, week, player_id, receptions, yards):
    db.add(PlayerStat(player_id=player_id, season=SEASON, week=week,
                      season_type="REG", receptions=receptions,
                      receiving_yards=yards))


def _pairing(db, week, rank, player_id, advantage, shadow=False):
    db.add(ExternalWrMatchup(
        source="fantasyguru", season=SEASON, week=week, fg_rank=rank,
        defender=f"CB{rank}", defender_team="JAX", receiver=player_id,
        receiver_id=player_id, receiver_team="BAL", advantage=advantage,
        shadow=shadow, uploaded_at=datetime.utcnow()))


def _slate(db, weeks, per_week=25, effect=0.0):
    """Synthetic weeks. `effect` adds a real advantage->residual slope.

    The effect is centred on the mean advantage so the board stays UNBIASED,
    which is the condition the harness is built for: it applies its correction
    with the intercept dropped, precisely so a fit cannot earn its keep by
    re-centring the board instead of ordering it. A fixture with a biased
    board would be testing something the method refuses to do.
    """
    centre = 30.0 + (per_week - 1) / 2.0
    for w in weeks:
        for i in range(per_week):
            pid = f"p{w}_{i}"
            adv = 30.0 + i
            _board(db, w, pid, receptions=4.0, yards=50.0)
            _actual(db, w, pid, receptions=4.0 + effect * (adv - centre),
                    yards=50.0 + 10 * effect * (adv - centre))
            _pairing(db, w, i + 1, pid, adv, shadow=(i == 0))
    db.commit()


class TestTheGateRefusesThinData:
    def test_one_week_fits_nothing(self, db_session):
        _slate(db_session, [9])
        report = grade_external.run(db_session, SEASON)
        assert report["status"] == "insufficient_weeks"
        assert report["gate"] == "closed"
        assert "needs 4" in report["message"]
        assert "targets" not in report

    def test_no_snapshots_at_all(self, db_session):
        report = grade_external.run(db_session, SEASON)
        assert report["status"] == "insufficient_weeks"
        assert report["weeks_held"] == []

    def test_a_pairing_with_no_actual_yet_is_dropped_with_a_reason(self, db_session):
        _board(db_session, 9, "p1", 4.0, 50.0)
        _pairing(db_session, 9, 1, "p1", 55.0)
        db_session.commit()
        report = grade_external.run(db_session, SEASON)
        assert report["rows_dropped"] == 1
        assert "no actual stat line" in report["drop_reasons"][0]["why"]

    def test_an_unmapped_receiver_is_dropped_with_a_reason(self, db_session):
        db_session.add(ExternalWrMatchup(
            source="fantasyguru", season=SEASON, week=9, fg_rank=1,
            defender="CB", receiver="Nobody", receiver_id=None, advantage=50.0))
        db_session.commit()
        report = grade_external.run(db_session, SEASON)
        assert "not mapped" in report["drop_reasons"][0]["why"]


class TestTheGateStaysClosedOnWeakEvidence:
    def test_four_weeks_of_noise_is_inconclusive_not_negative(self, db_session):
        _slate(db_session, [6, 7, 8, 9], effect=0.0)
        report = grade_external.run(db_session, SEASON)
        assert report["status"] == "measured"
        assert report["gate"] == "closed"
        assert "INCONCLUSIVE" in report["verdict"]

    def test_every_target_carries_its_own_power_arithmetic(self, db_session):
        _slate(db_session, [6, 7, 8, 9])
        report = grade_external.run(db_session, SEASON)
        for target in report["targets"]:
            power = target["power"]
            assert power["pairings"] == 100
            # n=100 against the n=16,380 test the method comes from.
            assert power["ci_halfwidth_vs_reference"] == 12.8
            assert "absence of evidence" in power["reading"]

    def test_a_real_effect_at_low_power_still_does_not_open_the_gate(self,
                                                                     db_session):
        """A signal that pays only where the interval is 12x too wide is the
        shape of a signal that does not pay."""
        _slate(db_session, [6, 7, 8, 9], effect=0.05)
        report = grade_external.run(db_session, SEASON)
        assert any(t["paid"] for t in report["targets"])
        assert report["gate"] == "closed"
        assert "too wide to act on" in report["verdict"]

    def test_the_gate_can_open_once_the_sample_is_there(self, db_session):
        """20 seasons' worth of weeks, so the power test is satisfied too.
        Proves the gate is a threshold and not a permanent refusal."""
        _slate(db_session, list(range(1, 19)), per_week=120, effect=0.05)
        report = grade_external.run(db_session, SEASON)
        assert report["gate"] == "open"
        assert "Beat the board out of sample" in report["verdict"]


class TestShadowIsReportedSeparately:
    def test_the_assignment_column_gets_its_own_verdict(self, db_session):
        _slate(db_session, [6, 7, 8, 9])
        shadow = grade_external.run(db_session, SEASON)["shadow"]
        assert shadow["status"] == "insufficient"
        assert shadow["shadow_rows"] == 4          # ~1 a week
        assert shadow["accrual_per_week"] == 1.0
        assert shadow["weeks_still_needed"] == 96
        assert "seasons, not weeks" in shadow["reading"]

    def test_it_is_never_folded_into_the_aggregate(self, db_session):
        """The whole point: 100 pairings and 4 shadow rows must not be
        reported as one adequately-powered number."""
        _slate(db_session, [6, 7, 8, 9])
        report = grade_external.run(db_session, SEASON)
        assert report["shadow"]["shadow_rows"] == 4
        assert report["targets"][0]["n"] == 100


class TestLineRatings:
    def test_a_rating_that_never_moves_is_flagged_as_a_season_descriptor(
            self, client, db_session):
        _upload(client, TEAM_CSV, week=8)
        _upload(client, TEAM_CSV, week=9)
        lines = grade_external._grade_lines(db_session, SEASON, "fantasyguru")
        assert lines["status"] == "measured"
        assert lines["weeks"] == [8, 9]
        assert lines["teams_whose_rating_moved"] == 0
        assert "not a weekly matchup signal" in lines["note"]

    def test_a_moving_rating_is_counted(self, client, db_session):
        _upload(client, TEAM_CSV, week=8)
        _upload(client, TEAM_CSV.replace("35.97", "44.00"), week=9)
        lines = grade_external._grade_lines(db_session, SEASON, "fantasyguru")
        assert lines["teams_whose_rating_moved"] == 1

    def test_nothing_uploaded(self, db_session):
        assert grade_external._grade_lines(
            db_session, SEASON, "fantasyguru")["status"] == "no_data"


class TestTheFilenameDoesNotMatter:
    """Uploads are identified by content, so any name must behave the same.

    The source renames its exports freely and the browser decorates
    duplicates, so a pipeline keyed on the filename would break on a week
    where nothing was actually wrong.
    """

    @pytest.mark.parametrize("name", [
        "table.csv", "table (1).csv", "table (17).csv",
        "FG Week 9 - WR Coverage FINAL(2).csv", "download", "",
        "wr_coverage.csv", "team_lines.csv",
    ])
    def test_any_name_loads_the_same_rows(self, client, name):
        body = _upload(client, WR_CSV, filename=name).json()
        assert body["kind"] == "wr_coverage", body
        assert body["stored"] == 3

    def test_a_name_that_contradicts_the_contents_is_ignored(self, client):
        """Named for the wrong export; the header is what counts."""
        body = _upload(client, TEAM_CSV, filename="wr_coverage.csv").json()
        assert body["kind"] == "team_lines"
        assert body["teams"] == 3

    def test_a_title_row_above_the_header_still_uploads(self, client):
        body = _upload(client,
                       "FantasyGuru Weekly Matchups - Week 9\n\n" + WR_CSV).json()
        assert body["kind"] == "wr_coverage"
        assert body["stored"] == 3

    def test_a_reordered_export_still_uploads(self, client):
        text = ("Primary Receiver,Team,Advantage,Coverage Type,"
                "Defensive Back,Team,Rank\n"
                "Zay Flowers,BAL,51.20,100% Shadow Coverage,"
                "Travis Hunter,JAC,3\n")
        body = _upload(client, text).json()
        assert body["kind"] == "wr_coverage"
        assert body["stored"] == 1
        rows = client.get(f"/external/wr-matchups?season={SEASON}&week={WEEK}"
                          ).json()["data"]
        assert rows[0]["defender"] == "Travis Hunter"
        assert rows[0]["defender_team"] == "JAX"
        assert rows[0]["receiver"] == "Zay Flowers"
        assert rows[0]["receiver_team"] == "BAL"

    def test_an_unrecognised_file_says_the_name_is_irrelevant(self, client):
        resp = _upload(client, "a,b,c\n1,2,3\n", filename="team_lines.csv")
        assert resp.status_code == 400
        assert "filename is not used" in resp.json()["detail"]
