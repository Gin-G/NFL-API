#!/usr/bin/env python3
"""Scoring the lineups the optimizer built, against what they actually scored.

The projections already have a prospective record. This is the one for the
thing the board is for: the optimizer reads only the ORDER of the projections,
so it can be right about a lineup while being wrong about every point in it.

The rule that differs from projection accuracy, and the reason this file
exists: a player with no stat line whose team has played scores ZERO here, not
"skip". A projection for an inactive player is not a projection error, but a
lineup with an inactive player really did score nothing from that slot.
"""

import io
from datetime import datetime

import pytest

from database.models import (FanduelSlate, LineupBuild, LineupEntry, LineupPlayer,
                             PlayerProjection, PlayerStat, Schedule)

pytest.importorskip("nfl_projections.optimizer",
                    reason="optimizer comes from nfl_projections (installed --no-deps in the image)")

from scripts import score_lineups                                    # noqa: E402

from .test_lineup_tracking import GAMES, SLATE_ROWS, _csv, _post, _schedule  # noqa: E402

SEASON, WEEK = 2026, 6
DST = {"BAL": 12.0, "DEN": 9.0}


@pytest.fixture(autouse=True)
def _no_opportunity_fetch(monkeypatch):
    from api import optimizer as api_opt

    monkeypatch.setattr(api_opt, "_opportunity", lambda season, week: None)


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in (LineupPlayer, LineupEntry, LineupBuild, FanduelSlate,
                  PlayerProjection, PlayerStat, Schedule):
        db_session.query(model).delete()
    db_session.commit()
    yield


@pytest.fixture(autouse=True)
def _defenses(monkeypatch):
    """Team stats are a network read, so the job soft-fails on them; here they
    are injected. TestWhatCannotBeScored covers them being unavailable."""
    monkeypatch.setattr(score_lineups, "_dst_actuals",
                        lambda db, season, week: dict(DST))


def _board(db, season=SEASON, week=WEEK):
    """A board that disagrees with FanDuel's FPPG about the ORDER.

    Projecting FPPG plus a constant would make the model's lineup and the FPPG
    baseline the same nine players, and the comparison the whole record rests
    on would be vacuous.
    """
    for i, (name, pos, _, salary, fppg, team) in enumerate(SLATE_ROWS):
        if pos == "D":
            continue
        # Cheap players projected high, expensive ones low: a different order.
        points = 30.0 - fppg
        db.add(PlayerProjection(
            season=season, week=week, player_id=f"00-{i:04d}", player_name=name,
            position=pos, team=team, projected_points=points, floor=points * 0.4,
            median=points * 0.95, ceiling=points * 1.8, prediction_type="veteran_ml",
            model_version="1.5.0", computed_at=datetime.utcnow()))
    db.commit()


def _stats(db, season=SEASON, week=WEEK, skip=()):
    """A stat line for every non-defense on the slate, except `skip`."""
    for i, (name, pos, _, _, fppg, team) in enumerate(SLATE_ROWS):
        if pos == "D" or name in skip:
            continue
        db.add(PlayerStat(
            player_id=f"00-{i:04d}", season=season, week=week, player_name=name,
            player_display_name=name, position=pos, recent_team=team,
            season_type="REG", receptions=float(i % 5), receiving_yards=float(10 + i * 3),
            receiving_tds=float(i % 2), rushing_yards=float(i), carries=float(i)))
    db.commit()


def _points(name) -> float:
    """What that player's stat line is worth, on the scoring the record uses."""
    from api.utils import fanduel_points

    i = next(i for i, row in enumerate(SLATE_ROWS) if row[0] == name)
    return fanduel_points({"receptions": float(i % 5), "receiving_yards": float(10 + i * 3),
                           "receiving_tds": float(i % 2), "rushing_yards": float(i),
                           "carries": float(i)})


def _build(client, db, **form):
    _board(db)
    _schedule(db)
    body = _post(client, num_lineups=1, **form).json()
    assert body["status"] == "success", body
    return db.query(LineupBuild).one()


def _entry(db, build):
    return db.query(LineupEntry).filter(LineupEntry.build_id == build.id).one()


def _players(db, entry):
    return db.query(LineupPlayer).filter(
        LineupPlayer.lineup_id == entry.id).order_by(LineupPlayer.id).all()


class TestScoringALineup:
    def test_the_lineup_is_worth_what_its_players_scored(self, client, db_session):
        build = _build(client, db_session)
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)

        entry = _entry(db_session, build)
        players = _players(db_session, entry)
        expected = sum(DST[p.team] if p.position == "D" else _points(p.player_name)
                       for p in players)
        assert entry.actual_points == pytest.approx(expected, abs=0.02)
        assert entry.scored_players == 9 and entry.unscored_players == 0
        assert all(p.actual_points is not None for p in players)

    def test_the_defense_is_scored_from_team_stats(self, client, db_session):
        build = _build(client, db_session)
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)

        defense = next(p for p in _players(db_session, _entry(db_session, build))
                       if p.position == "D")
        assert defense.actual_points == DST[defense.team]

    def test_a_player_who_did_not_play_scored_nothing(self, client, db_session):
        """Zero, not skipped. The slot really did produce nothing."""
        build = _build(client, db_session)
        entry_players = _players(db_session, _entry(db_session, build))
        absent = next(p for p in entry_players if p.position != "D").player_name
        _stats(db_session, skip=(absent,))
        score_lineups.run(db_session, SEASON, week=WEEK)

        entry = _entry(db_session, build)
        scored = next(p for p in _players(db_session, entry)
                      if p.player_name == absent)
        assert scored.actual_points == 0.0
        assert entry.unscored_players == 0        # it is a score, not a gap
        assert entry.actual_points < sum(
            DST[p.team] if p.position == "D" else _points(p.player_name)
            for p in entry_players)

    def test_a_scored_build_is_frozen(self, client, db_session):
        build = _build(client, db_session)
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)
        first = build.scored_at
        assert first is not None

        assert score_lineups.run(db_session, SEASON, week=WEEK) == []
        assert db_session.query(LineupBuild).one().scored_at == first

    def test_rescore_recomputes_deliberately(self, client, db_session):
        build = _build(client, db_session)
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)
        before = _entry(db_session, build).actual_points

        db_session.query(PlayerStat).delete()
        db_session.commit()
        _stats(db_session)
        summaries = score_lineups.run(db_session, SEASON, week=WEEK, rescore=True)
        assert len(summaries) == 1
        assert _entry(db_session, build).actual_points == pytest.approx(before, abs=0.02)

    def test_a_week_with_no_stats_is_left_alone(self, client, db_session):
        build = _build(client, db_session)
        assert score_lineups.run(db_session, SEASON, week=WEEK) == []
        assert build.scored_at is None


class TestWhatCannotBeScored:
    def test_a_player_whose_game_has_not_happened_is_not_a_zero(self, client, db_session):
        """The distinction the whole scoring rule turns on.

        No stat line and his team has played: he did not play, and that is a
        real zero. No stat line and his game is not final: nothing is known
        yet, and calling it zero would invent a bad week.
        """
        _board(db_session)
        _schedule(db_session, final=False)
        _post(client, num_lineups=1)
        build = db_session.query(LineupBuild).one()
        absent = next(p for p in _players(db_session, _entry(db_session, build))
                      if p.position != "D").player_name
        _stats(db_session, skip=(absent,))
        score_lineups.run(db_session, SEASON, week=WEEK)

        entry = _entry(db_session, build)
        waiting = next(p for p in _players(db_session, entry)
                       if p.player_name == absent)
        assert waiting.actual_points is None
        assert entry.unscored_players == 1
        assert build.scored_at is None

    def test_defenses_without_team_stats_leave_it_partial(self, client, db_session,
                                                          monkeypatch):
        monkeypatch.setattr(score_lineups, "_dst_actuals", lambda db, s, w: {})
        build = _build(client, db_session)
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)

        entry = _entry(db_session, build)
        assert entry.unscored_players == 1
        assert entry.scored_players == 8
        assert build.scored_at is None            # retried, not written off
        assert build.fppg_points is None          # and no flattering baseline

    def test_the_outage_clearing_finishes_the_job(self, client, db_session, monkeypatch):
        monkeypatch.setattr(score_lineups, "_dst_actuals", lambda db, s, w: {})
        build = _build(client, db_session)
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)
        assert build.scored_at is None

        monkeypatch.setattr(score_lineups, "_dst_actuals", lambda db, s, w: dict(DST))
        score_lineups.run(db_session, SEASON, week=WEEK)
        entry = _entry(db_session, build)
        assert entry.unscored_players == 0
        assert db_session.query(LineupBuild).one().scored_at is not None


class TestTheBaselines:
    def test_fanduels_own_fppg_is_the_denominator(self, client, db_session):
        build = _build(client, db_session)
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)

        assert build.fppg_points is not None
        assert build.fppg_projected is not None
        # The board was deliberately given a different order, so the two
        # lineups are different nine players and can actually be compared.
        assert build.fppg_points != _entry(db_session, build).actual_points

    def test_the_best_lineup_the_slate_held_beats_everything(self, client, db_session):
        build = _build(client, db_session)
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)

        assert build.best_points >= _entry(db_session, build).actual_points
        assert build.best_points >= build.fppg_points

    def test_the_baseline_chooses_from_the_same_pool(self, client, db_session):
        """Otherwise it is a baseline for a slate nobody was choosing from —
        and both filters that trip people up are the builder's own: the
        exclusions, and FanDuel's injury flag."""
        _board(db_session)
        _schedule(db_session)
        injured = SLATE_ROWS + [("Hurt Guy", "WR", "WR", 2000, 9.0, "NYJ")]
        csv = _csv(injured).replace(b"Hurt Guy,WR,WR,2000,9.0,NYJ,",
                                   b"Hurt Guy,WR,WR,2000,9.0,NYJ,O")
        _post(client, csv=csv, num_lineups=1, exclude="Kyren Williams")
        build = db_session.query(LineupBuild).one()
        slate = db_session.query(FanduelSlate).one()

        pool = score_lineups._pool(db_session, build, slate)
        names = set(pool["Nickname"])
        assert "Kyren Williams" not in names      # excluded by the request
        assert "Hurt Guy" not in names            # out, per FanDuel's own flag
        assert "Bijan Robinson" in names

        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)
        assert build.fppg_points is not None

    def test_an_older_package_costs_the_baselines_and_nothing_else(self, client,
                                                                   db_session,
                                                                   monkeypatch):
        """The image installs nfl_projections from a branch tarball at build
        time, so the exact solver the baselines need can simply not be there
        yet. The lineups still get scored, and the build stays open so the next
        run fills the baselines in."""
        from nfl_projections import optimizer as opt

        build = _build(client, db_session)
        _stats(db_session)
        monkeypatch.delattr(opt, "best_lineup")
        score_lineups.run(db_session, SEASON, week=WEEK)

        assert _entry(db_session, build).actual_points is not None
        assert build.fppg_points is None
        assert build.scored_at is None

    def test_a_deleted_slate_costs_the_baseline_not_the_record(self, client, db_session):
        build = _build(client, db_session)
        db_session.query(FanduelSlate).delete()
        db_session.commit()
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)

        assert _entry(db_session, build).actual_points is not None
        assert build.fppg_points is None


class TestShowdown:
    """One game, five players, and an MVP who scores 1.5x. Scoring him flat
    would understate every Showdown lineup ever built."""

    SLATE = [("Josh Allen", "QB", "MVP", 15000, 22.0, "BUF"),
             ("Jared Goff", "QB", "AnyFLEX", 11000, 18.0, "DET"),
             ("Jahmyr Gibbs", "RB", "AnyFLEX", 10000, 18.0, "DET"),
             ("Amon-Ra St. Brown", "WR", "AnyFLEX", 9500, 16.5, "DET"),
             ("Dallas Goedert", "TE", "AnyFLEX", 6000, 11.0, "DET"),
             ("Bijan Robinson", "RB", "AnyFLEX", 8000, 18.5, "ATL")]

    def _build(self, client, db):
        _board(db)
        _schedule(db)
        body = _post(client, csv=_csv(self.SLATE), num_lineups=1).json()
        assert body["contest"] == "showdown", body
        return db.query(LineupBuild).one()

    def test_the_mvp_is_paid_at_one_and_a_half(self, client, db_session):
        build = self._build(client, db_session)
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)

        players = _players(db_session, _entry(db_session, build))
        mvp = next(p for p in players if (p.slot or "").upper() == "MVP")
        assert mvp.actual_points == pytest.approx(_points(mvp.player_name) * 1.5, abs=0.02)
        flex = [p for p in players if (p.slot or "").upper() != "MVP"]
        assert all(p.actual_points == pytest.approx(_points(p.player_name), abs=0.02)
                   for p in flex)

    def test_it_is_scored_without_a_classic_baseline(self, client, db_session):
        """A classic solve on a Showdown pool would be a baseline for a contest
        nobody entered."""
        build = self._build(client, db_session)
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)

        assert _entry(db_session, build).actual_points is not None
        assert build.fppg_points is None and build.best_points is None
        assert build.scored_at is not None      # and it still gets frozen


class TestThePerformanceRecord:
    def test_it_reports_what_happened(self, client, db_session):
        build = _build(client, db_session)
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)

        body = client.get("/optimizer/performance", params={"season": SEASON}).json()
        (week,) = body["weeks"]
        entry = _entry(db_session, build)
        assert week["scored_lineups"] == 1
        assert week["actual"] == pytest.approx(entry.actual_points, abs=0.02)
        assert week["projected"] == pytest.approx(entry.projected_points, abs=0.02)
        assert week["fppg_baseline"] == build.fppg_points
        assert week["best_possible"] == build.best_points
        assert week["compared_to_fppg"] == 1
        assert week["beat_fppg"] == int(entry.actual_points > build.fppg_points)
        assert body["totals"]["beat_fppg_rate"] in (0.0, 1.0)

    def test_one_week_comes_with_the_lineups_themselves(self, client, db_session):
        _build(client, db_session)
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)

        body = client.get("/optimizer/performance",
                          params={"season": SEASON, "week": WEEK}).json()
        (detail,) = body["builds"]
        (lineup,) = detail["lineups"]
        assert len(lineup["players"]) == 9
        assert all(p["actual"] is not None for p in lineup["players"])
        assert detail["fppg_baseline"] is not None

    def test_a_partial_lineup_stays_out_of_the_averages(self, client, db_session,
                                                        monkeypatch):
        monkeypatch.setattr(score_lineups, "_dst_actuals", lambda db, s, w: {})
        _build(client, db_session)
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)

        body = client.get("/optimizer/performance", params={"season": SEASON}).json()
        (week,) = body["weeks"]
        assert week["lineups"] == 1
        assert week["scored_lineups"] == 0
        assert week["actual"] is None


def _divided_board(db, season=SEASON, week=WEEK):
    """A board whose MEAN and CEILING rank the slate differently.

    The default board scales floor/median/ceiling from the mean, so every
    objective picks the same nine men and comparing them is vacuous. Here the
    mean prefers the cheap players and the ceiling prefers the expensive ones,
    which is the disagreement the comparison is supposed to measure.
    """
    for i, (name, pos, _, _, fppg, team) in enumerate(SLATE_ROWS):
        if pos == "D":
            continue
        mean = 30.0 - fppg
        db.add(PlayerProjection(
            season=season, week=week, player_id=f"00-{i:04d}", player_name=name,
            position=pos, team=team, projected_points=mean, floor=mean * 0.4,
            median=mean * 0.95, ceiling=fppg * 2.0,     # the other way round
            prediction_type="veteran_ml", model_version="1.5.0",
            computed_at=datetime.utcnow()))
    db.commit()


class TestWhichObjectiveWon:
    """The record is only useful if it can say which setting to use next week."""

    def _two_objectives(self, client, db):
        _divided_board(db)
        _schedule(db)
        assert _post(client, num_lineups=1, objective="mean").json()["status"] == "success"
        assert _post(client, csv=None, num_lineups=1,
                     objective="ceiling").json()["status"] == "success"
        _stats(db)
        score_lineups.run(db, SEASON, week=WEEK)

    def test_the_objectives_are_ranked_by_what_they_scored(self, client, db_session):
        self._two_objectives(client, db_session)

        body = client.get("/optimizer/performance", params={"season": SEASON}).json()
        objectives = {o["objective"]: o for o in body["by_objective"]}
        assert set(objectives) == {"mean", "ceiling"}
        assert all(o["scored_lineups"] == 1 for o in objectives.values())

        # Ranked best-first by what the lineups actually scored.
        actuals = [o["actual"] for o in body["by_objective"]]
        assert actuals == sorted(actuals, reverse=True)

        verdict = body["objective_verdict"]
        assert verdict["comparable"] is True
        assert verdict["objectives_scored"] == 2
        assert verdict["best"] == body["by_objective"][0]["objective"]
        assert verdict["worst"] == body["by_objective"][-1]["objective"]
        assert verdict["spread"] == pytest.approx(
            body["by_objective"][0]["actual"] - body["by_objective"][-1]["actual"], abs=0.02)

    def test_the_two_objectives_really_built_different_lineups(self, client, db_session):
        """Otherwise the ranking above would be comparing a lineup with itself."""
        self._two_objectives(client, db_session)

        body = client.get("/optimizer/performance",
                          params={"season": SEASON, "week": WEEK}).json()
        rosters = [{p["player_name"] for p in lineup["players"]}
                   for build in body["builds"] for lineup in build["lineups"]]
        assert len(rosters) == 2
        assert rosters[0] != rosters[1]

    def test_one_objective_is_not_a_comparison(self, client, db_session):
        _build(client, db_session, objective="mean")
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)

        verdict = client.get("/optimizer/performance",
                             params={"season": SEASON}).json()["objective_verdict"]
        assert verdict["comparable"] is False
        assert verdict["objectives_scored"] == 1
        assert verdict["worst"] is None and verdict["spread"] is None

    def test_a_week_that_has_not_been_scored_cannot_rank_anything(self, client, db_session):
        _divided_board(db_session)
        _schedule(db_session)
        _post(client, num_lineups=1, objective="mean")
        _post(client, csv=None, num_lineups=1, objective="ceiling")

        body = client.get("/optimizer/performance", params={"season": SEASON}).json()
        assert [o["actual"] for o in body["by_objective"]] == [None, None]
        verdict = body["objective_verdict"]
        assert verdict["comparable"] is False
        assert verdict["best"] is None

    def test_an_unscored_objective_sorts_last_rather_than_losing(self, client,
                                                                 db_session):
        """A missing number is not a bad number."""
        self._two_objectives(client, db_session)
        # A third build made after the scoring run, so it has no actuals yet --
        # which is what happens whenever you build again mid-week.
        assert _post(client, csv=None, num_lineups=1,
                     objective="floor").json()["status"] == "success"

        body = client.get("/optimizer/performance", params={"season": SEASON}).json()
        assert body["by_objective"][-1]["objective"] == "floor"
        assert body["by_objective"][-1]["actual"] is None
        assert body["objective_verdict"]["best"] != "floor"
        assert body["objective_verdict"]["objectives_scored"] == 2

    def test_a_single_week_says_so(self, client, db_session):
        """One slate cannot separate objectives; the flag is what stops the UI
        claiming it did."""
        self._two_objectives(client, db_session)

        verdict = client.get("/optimizer/performance",
                             params={"season": SEASON}).json()["objective_verdict"]
        assert verdict["one_week_only"] is True
        assert verdict["weeks_scored"] == 1

    def test_a_showdown_is_not_ranked_against_a_classic(self, client, db_session):
        """Five players against nine: the contest would decide the comparison.

        The real record hit this -- a scored Showdown pooled with an unscored
        classic week put "ceiling" top on 25 five-man lineups.
        """
        _divided_board(db_session)
        _schedule(db_session)
        # A classic build and a Showdown build, both on the ceiling objective.
        assert _post(client, num_lineups=1, objective="ceiling"
                     ).json()["contest"] == "classic"
        assert _post(client, csv=_csv(TestShowdown.SLATE), num_lineups=1,
                     objective="mean").json()["contest"] == "showdown"
        _stats(db_session)
        score_lineups.run(db_session, SEASON, week=WEEK)

        body = client.get("/optimizer/performance", params={"season": SEASON}).json()
        rows = {(o["objective"], o["contest"]): o for o in body["by_objective"]}
        assert ("ceiling", "classic") in rows
        assert ("mean", "showdown") in rows

        # One contest only, so the two cannot be declared a winner and a loser.
        verdict = body["objective_verdict"]
        assert verdict["comparable"] is False
        assert verdict["objectives_scored"] == 1
        assert verdict["contest"] in ("classic", "showdown")
