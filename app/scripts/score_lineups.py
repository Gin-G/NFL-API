#!/usr/bin/env python3
"""Score the lineups the optimizer built against what they actually scored.

The projections have had a prospective record since scripts.score_projections:
what the board said before kickoff, scored afterwards, never rewritten. This
does the same for the thing the board is actually *for*. A projection's MAE and
a lineup's score are not the same claim — the optimizer reads only the ORDER of
the projections, so it can be right about a lineup while being wrong about
every point in it, and vice versa.

Two baselines come with each build, because a lineup's raw score means nothing
on its own:

  * `fppg_points` — the lineup FanDuel's own season-average FPPG would have
    built from the same pool under the same cap, scored on what happened. This
    is the honest denominator, and it is the one number that says whether the
    model earned its keep: FPPG is free and on the slate already.
  * `best_points` — the best lineup the slate held, in hindsight. Nobody hits
    it. It says how much was on the table.

Both are only reproducible because the slate CSV is stored (`fanduel_slates`),
which is also why the upload only happens once.

Scoring rules, and why:
  * A player with no stat line whose team has played scores 0, not "skip".
    Unlike projection accuracy — which excludes players who didn't play,
    because a projection for a player who was inactive is not a projection
    error — a LINEUP with an inactive player really did score nothing from
    that slot. That is the cost of the pick.
  * A lineup is only averaged when every slot could be scored. A slot that
    cannot be scored (a defense on a week nflverse team stats are unavailable,
    a game not yet played) leaves it partial and out of the averages.
  * A build is frozen once it is fully scored. Until then the job retries it,
    so a late stat load or an nflverse outage self-heals on the next run.

Usage (from app/):
    python -m scripts.score_lineups                     # every unscored build
    python -m scripts.score_lineups --season 2026 --week 6
    python -m scripts.score_lineups --rescore           # recompute scored builds
"""
import argparse
import io
import logging
import os
import re
import sys
from datetime import datetime

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("score_lineups")

DEF_POSITIONS = {"D", "DEF", "DST", "D/ST"}

# Showdown's MVP scores 1.5x, which is the whole shape of that contest. Named
# here rather than imported so scoring does not depend on the package version
# the image happens to carry.
MVP_MULTIPLIER = 1.5

# FanDuel's team codes against nflverse's. Left unmapped, a defense silently
# scores nothing at all, which looks like missing data rather than a code.
TEAM_ALIASES = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS", "WFT": "WAS", "ARZ": "ARI",
                "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA", "OAK": "LV",
                "SD": "LAC", "STL": "LA"}

# FanDuel's points-allowed tiers for a defense.
PA_TIERS = ((0, 0, 10), (1, 6, 7), (7, 13, 4), (14, 20, 1),
            (21, 27, 0), (28, 34, -1), (35, 999, -4))


def _fp(stat) -> float:
    """FanDuel points for a stat row, on the same scale as the projections."""
    from api.utils import fanduel_points

    return fanduel_points({c.name: getattr(stat, c.name)
                           for c in stat.__table__.columns})


def _norm(name) -> str:
    """Letters only, lower case — for the players whose name never joined to the
    board, who therefore have no player_id to look up."""
    name = re.sub(r"\s+(Jr\.?|Sr\.?|I{2,}|IV|V)$", "", str(name or "").strip())
    return re.sub(r"[^a-z]", "", name.lower())


def _team(code) -> str:
    code = str(code or "").strip().upper()
    return TEAM_ALIASES.get(code, code)


def _played_teams(db, season: int, week: int) -> set:
    """Teams whose game that week is final — which is what separates "he scored
    nothing" from "his game has not happened yet"."""
    from database.models import Schedule

    played = set()
    for game in db.query(Schedule).filter(
            Schedule.season == season, Schedule.week == week).all():
        if game.home_score is not None and game.away_score is not None:
            played |= {_team(game.home_team), _team(game.away_team)}
    return played


def _player_actuals(db, season: int, week: int):
    """``(by player_id, by normalized name)`` of FanDuel points for the week."""
    from sqlalchemy import or_

    from database.models import PlayerStat

    rows = db.query(PlayerStat).filter(
        PlayerStat.season == season, PlayerStat.week == week
    ).filter(or_(PlayerStat.season_type == "REG",
                 PlayerStat.season_type.is_(None))).all()
    by_id, by_name = {}, {}
    for row in rows:
        points = _fp(row)
        by_id[row.player_id] = points
        for name in (row.player_name, row.player_display_name):
            key = _norm(name)
            if key:
                by_name.setdefault(key, points)
    return by_id, by_name


def _dst_actuals(db, season: int, week: int) -> dict:
    """``{team: FanDuel points}`` for defenses, or ``{}`` if it cannot be had.

    Points allowed comes from our own schedule (what the opponent scored);
    everything else from nflverse team stats, which is a network read — so this
    soft-fails, the lineups holding a defense stay partial, and the next run
    picks them up.
    """
    try:
        import nflreadpy as nfl

        frame = nfl.load_team_stats(seasons=[season], summary_level="week")
        frame = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    except Exception as exc:
        logger.warning("Team stats unavailable (%s); defenses stay unscored", exc)
        return {}

    import pandas as pd

    from database.models import Schedule

    try:
        frame = frame[(frame["season"] == season) & (frame["week"] == week)]
        if "season_type" in frame.columns:
            frame = frame[frame["season_type"] == "REG"]
        if frame.empty:
            logger.warning("No team stats for %s week %s yet; defenses stay unscored",
                           season, week)
            return {}

        allowed = {}
        for game in db.query(Schedule).filter(
                Schedule.season == season, Schedule.week == week).all():
            if game.home_score is None or game.away_score is None:
                continue
            allowed[_team(game.home_team)] = float(game.away_score)
            allowed[_team(game.away_team)] = float(game.home_score)

        def num(row, *names):
            for name in names:
                if name in frame.columns:
                    value = pd.to_numeric(row.get(name), errors="coerce")
                    if value == value:          # not NaN
                        return float(value)
            return 0.0

        def tier(points):
            for low, high, value in PA_TIERS:
                if low <= points <= high:
                    return value
            return -4

        out = {}
        for _, row in frame.iterrows():
            team = _team(row.get("team"))
            if team not in allowed:
                continue        # game not final, so neither is the defense's score
            sacks = num(row, "def_sacks")
            picks = num(row, "def_interceptions")
            fumbles = num(row, "fumble_recovery_opp")
            # def_tds covers interception and fumble returns; special_teams_tds is
            # the team's own return TDs -- it reconciles exactly with the sum of
            # its players' special_teams_tds, where pt_return_tds does not (it
            # disagrees on 39 of 544 team-weeks), so that one is left out.
            tds = num(row, "def_tds") + num(row, "special_teams_tds")
            safeties = num(row, "def_safeties")
            blocks = (num(row, "def_punt_blocks") + num(row, "def_pat_blocks")
                      + num(row, "def_fg_blocks"))
            out[team] = round(sacks + picks * 2 + fumbles * 2 + tds * 6
                              + safeties * 2 + blocks * 2 + tier(allowed[team]), 2)
        return out
    except Exception as exc:
        logger.warning("Could not score defenses (%s); they stay unscored", exc)
        return {}


class Actuals:
    """Everything one week's scoring needs, loaded once for all its builds."""

    def __init__(self, db, season: int, week: int):
        self.by_id, self.by_name = _player_actuals(db, season, week)
        self.dst = _dst_actuals(db, season, week) if self.by_id else {}
        self.played = _played_teams(db, season, week)

    @property
    def usable(self) -> bool:
        return bool(self.by_id)

    def points(self, *, position, player_id, player_name, team):
        """``(points, scorable)`` for one slot."""
        team = _team(team)
        if str(position or "").upper() in DEF_POSITIONS:
            points = self.dst.get(team)
            return (points, True) if points is not None else (None, False)
        if player_id and player_id in self.by_id:
            return self.by_id[player_id], True
        key = _norm(player_name)
        if key and key in self.by_name:
            return self.by_name[key], True
        if team in self.played:
            # On the slate, his team has played, and he has no stat line: he did
            # not play. In a lineup that is a real zero, not a missing value.
            return 0.0, True
        return None, False


# ---------------------------------------------------------------------------
# The baselines, rebuilt from the stored slate
# ---------------------------------------------------------------------------

def _pool(db, build, slate):
    """The pool the build was choosing from, or None if it cannot be rebuilt."""
    import pandas as pd

    if slate is None or not slate.content:
        return None
    try:
        frame = pd.read_csv(io.StringIO(slate.content))
    except Exception as exc:
        logger.warning("build %s: saved slate unreadable (%s)", build.id, exc)
        return None

    if build.slate and build.slate not in ("all", "single_game"):
        from api.optimizer import _slate_teams

        teams, _ = _slate_teams(db, build.season, build.week, build.slate)
        if teams and "Team" in frame.columns:
            frame = frame[frame["Team"].astype(str).str.upper().isin(teams)]

    # The same filters the builder applies before it solves, so the baseline is
    # choosing from the same players and not a kinder pool. The injury statuses
    # come from the package rather than a copy of the list, so the two cannot
    # drift apart silently.
    try:
        from nfl_projections.optimizer import EXCLUDED_INJURY_STATUSES as excluded_status
    except Exception:
        excluded_status = ["IR", "O", "D"]
    if "Injury Indicator" in frame.columns:
        flags = frame["Injury Indicator"].fillna("").astype(str).str.upper()
        frame = frame[~flags.isin(list(excluded_status))]
    if build.exclude_players:
        excluded = {n.strip() for n in build.exclude_players.split(",") if n.strip()}
        frame = frame[~frame["Nickname"].isin(excluded)]
    return frame if len(frame) else None


def _baselines(db, build, slate, actuals) -> dict:
    """``{fppg_points, fppg_projected, best_points}``, as far as they can be had.

    Showdown is left alone: its roster rules and 1.5x MVP are a different
    optimisation, and a classic solve on that pool would be a baseline for a
    contest nobody entered.
    """
    out = {"fppg_points": None, "fppg_projected": None, "best_points": None}
    if build.contest == "showdown":
        return out
    frame = _pool(db, build, slate)
    if frame is None:
        return out
    try:
        import pandas as pd

        from nfl_projections import optimizer as opt
    except Exception as exc:
        logger.warning("build %s: no optimizer available for baselines (%s)", build.id, exc)
        return out

    if "FPPG" not in frame.columns:
        return out
    # FPPG among them: the builder drops those rows too, so leaving them in
    # would let the hindsight-best lineup roster someone it could never have
    # picked.
    frame = frame.dropna(subset=["Salary", "FPPG", "Roster Position", "Nickname"]).copy()
    if frame.empty:
        return out

    scored, scorable = [], True
    for _, row in frame.iterrows():
        points, ok = actuals.points(position=row.get("Position"), player_id=None,
                                    player_name=row.get("Nickname"), team=row.get("Team"))
        scored.append(points if ok else 0.0)
        scorable = scorable and ok
    frame["actual_points"] = scored
    if not scorable:
        # Half the slate scored and half of it zero would make both baselines
        # look beatable for no reason. Better to have no baseline than a flattering one.
        logger.info("build %s: part of the slate cannot be scored; baselines skipped",
                    build.id)
        return out

    cap = build.salary_cap or 60000

    def solve(points_column):
        pool = frame.copy()
        pool["lineup_points"] = pd.to_numeric(pool[points_column], errors="coerce").fillna(0.0)
        found = opt.best_lineup(pool, cap)
        if found is None:
            return None
        picks, _, _ = found
        return pool.loc[picks]

    try:
        fppg = solve("FPPG")
        if fppg is not None:
            out["fppg_points"] = round(float(fppg["actual_points"].sum()), 2)
            out["fppg_projected"] = round(float(
                pd.to_numeric(fppg["FPPG"], errors="coerce").fillna(0).sum()), 2)
        best = solve("actual_points")
        if best is not None:
            out["best_points"] = round(float(best["actual_points"].sum()), 2)
    except Exception as exc:
        # `best_lineup` is new (the exact solver), and the image installs
        # nfl_projections from a branch tarball at build time -- so an older
        # image gets AttributeError here. The lineups are still scored; only
        # the baselines go missing, and the next run fills them in.
        logger.warning("build %s: baselines not computed (%s)", build.id, exc)
        return {"fppg_points": None, "fppg_projected": None, "best_points": None}
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_build(db, build, actuals) -> dict:
    """Score one build's lineups. Returns a summary; commits what it writes."""
    from database.models import FanduelSlate, LineupEntry, LineupPlayer

    entries = db.query(LineupEntry).filter(
        LineupEntry.build_id == build.id).order_by(LineupEntry.lineup_index).all()
    if not entries:
        return {"build_id": build.id, "status": "no_lineups", "lineups": 0}

    showdown = build.contest == "showdown"
    complete, scored_slots, missing_slots = 0, 0, 0
    for entry in entries:
        players = db.query(LineupPlayer).filter(
            LineupPlayer.lineup_id == entry.id).order_by(LineupPlayer.id).all()
        total, scored, unscored = 0.0, 0, 0
        for player in players:
            points, ok = actuals.points(
                position=player.position, player_id=player.player_id,
                player_name=player.player_name, team=player.team)
            if ok and showdown and str(player.slot or "").upper() == "MVP":
                points *= MVP_MULTIPLIER      # as the contest pays him
            player.actual_points = round(points, 2) if ok else None
            if ok:
                total += points
                scored += 1
            else:
                unscored += 1
        entry.actual_points = round(total, 2) if scored else None
        entry.scored_players = scored
        entry.unscored_players = unscored
        entry.scored_at = datetime.utcnow()
        scored_slots += scored
        missing_slots += unscored
        complete += int(unscored == 0)

    slate = None
    if build.slate_id is not None:
        slate = db.query(FanduelSlate).filter(
            FanduelSlate.id == build.slate_id).one_or_none()
    baselines = _baselines(db, build, slate, actuals)
    build.fppg_points = baselines["fppg_points"]
    build.fppg_projected = baselines["fppg_projected"]
    build.best_points = baselines["best_points"]

    # Frozen only when there is nothing left to learn about it. A partial score
    # is written (it is correct as far as it goes) but not frozen, so the next
    # run picks up a late stat load or an nflverse outage that has cleared.
    #
    # A missing baseline counts as unfinished too, but only when it was
    # obtainable: a slate that has been deleted is never coming back, and
    # Showdown has no classic baseline by design, so neither should hold a
    # build open forever.
    baseline_due = (slate is not None and build.contest != "showdown"
                    and baselines["fppg_points"] is None)
    done = complete == len(entries) and not baseline_due
    build.scored_at = datetime.utcnow() if done else None
    db.commit()

    return {
        "build_id": build.id, "week": build.week,
        "status": "scored" if done else "partial",
        "lineups": len(entries), "complete": complete,
        "slots_scored": scored_slots, "slots_unscored": missing_slots,
        "actual": _mean([e.actual_points for e in entries if not (e.unscored_players or 0)]),
        "projected": _mean([e.projected_points for e in entries
                            if not (e.unscored_players or 0)]),
        "fppg_baseline": baselines["fppg_points"],
        "best_possible": baselines["best_points"],
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def pending_builds(db, season: int, week: int = None, rescore: bool = False):
    """Builds worth scoring: unfrozen, and for a week that has been played."""
    from database.models import LineupBuild, PlayerStat

    q = db.query(LineupBuild).filter(LineupBuild.season == season)
    if week is not None:
        q = q.filter(LineupBuild.week == week)
    if not rescore:
        q = q.filter(LineupBuild.scored_at.is_(None))
    builds = q.order_by(LineupBuild.week, LineupBuild.id).all()

    played = {w for (w,) in db.query(PlayerStat.week).filter(
        PlayerStat.season == season).distinct().all()}
    return [b for b in builds if b.week in played]


def run(db, season: int, week: int = None, rescore: bool = False) -> list:
    from database.models import AnalyticsJobStatus

    builds = pending_builds(db, season, week=week, rescore=rescore)
    if not builds:
        logger.info("Nothing to score for %s%s", season,
                    f" week {week}" if week else "")
        return []

    job = AnalyticsJobStatus(
        job_type="lineup_accuracy", status="running",
        started_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        total_entries=len(builds), processed_entries=0, skipped_entries=0,
        failed_entries=0, current_season=season)
    db.add(job)
    db.commit()

    summaries = []
    try:
        for build_week in sorted({b.week for b in builds}):
            actuals = Actuals(db, season, build_week)
            if not actuals.usable:
                logger.info("week %s has no stats loaded yet; skipping", build_week)
                continue
            for build in [b for b in builds if b.week == build_week]:
                summary = score_build(db, build, actuals)
                summaries.append(summary)
                logger.info("build %s: %s", build.id, summary)
                job.processed_entries = (job.processed_entries or 0) + 1
                if summary["status"] == "partial":
                    job.skipped_entries = (job.skipped_entries or 0) + 1
                job.updated_at = datetime.utcnow()
                db.commit()
        job.status = "completed"
    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)[:500]
        raise
    finally:
        job.updated_at = datetime.utcnow()
        db.commit()

    logger.info("Scored %d build(s) of %s", len(summaries), season)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score built FanDuel lineups against actual results")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--rescore", action="store_true",
                        help="Recompute builds already scored (off by default: a "
                             "scored build is frozen)")
    args = parser.parse_args()

    from api.utils import get_current_nfl_season
    from database.models import Base
    from database.session import SessionLocal, engine

    season = args.season or get_current_nfl_season()
    Base.metadata.create_all(engine)    # creates the lineup tables on first run

    db = SessionLocal()
    try:
        run(db, season, week=args.week, rescore=args.rescore)
    except Exception as exc:
        logger.error("Lineup scoring failed: %s", exc, exc_info=True)
        raise SystemExit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
