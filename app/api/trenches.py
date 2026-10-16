"""Trench report: line-of-scrimmage ratings and weekly line matchups.

Four 0-100 grades per team (50 = league average, 15 points to a standard
deviation, the same scale as /ratings): run blocking and pass protection for
the offence, run stopping and pass rush for the defence. A matchup rating puts
one unit against the one it faces, so a strong rushing front against a soft one
reads in the eighties and the reverse reads in the twenties.

The rating math lives in `nfl_projections.trenches` so the research harness and
this endpoint cannot drift apart; this module is the DB adapter and the cache.
Grades for a given week use only plays from earlier weeks, so a matchup shown
before kickoff is the same number the backtest graded afterwards.

**Read the number with its backtest.** /trenches/backtest reports what each band
actually delivered. Summary of the research (nfl-data-py EXPERIMENTS.md, "The
trench report"): the bands separate rushing outcomes cleanly, but most of that
separation is which offence is running, not which matchup it drew — the
opponent half is worth about a fifth of a yard per carry. The endpoint returns
both so a dashboard cannot show the flattering half alone.
"""
import logging
import time
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.session import get_db
from database.models import Schedule

logger = logging.getLogger(__name__)
router = APIRouter()

_PBP_COLUMNS = [
    "season", "week", "posteam", "defteam", "yards_gained",
    "rush_attempt", "qb_dropback", "qb_scramble", "qb_kneel",
    "two_point_attempt", "sack", "qb_hit",
]
# Ratings move once a week, and a season of play-by-play is a few hundred
# thousand rows, so the work is cached for an hour rather than per request.
_TTL = 3600
_cache: dict = {}


def _trenches():
    """The research module, or a 503 explaining what is missing.

    The API image installs nfl-projections without its heavy dependencies, so
    an import error here means a packaging problem, not a data problem, and it
    should say so rather than returning an empty table.
    """
    try:
        from nfl_projections import trenches
        return trenches
    except ImportError as exc:        # pragma: no cover - packaging failure
        logger.error("nfl_projections.trenches unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Line ratings need the nfl-projections package in this image.",
        )


def _cached(key, build):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    value = build()
    _cache[key] = (time.time(), value)
    return value


def load_trench_plays(db: Session, seasons: list[int]):
    """Rushing and dropback frames for whole seasons, DB first.

    Whole seasons rather than a week at a time: every rating is built from the
    weeks before the one being shown, so a week-scoped query would have to be
    re-issued for each week of a backtest.

    A season the loader has not ingested comes back empty and the caller
    answers 404. Downloading it here instead would put a few hundred megabytes
    of nflverse traffic inside a web request — the fallback exists only for the
    case where the table is absent altogether, which is a dev database.
    """
    trenches = _trenches()
    placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    params = {f"s{i}": int(s) for i, s in enumerate(seasons)}
    cols = ", ".join(_PBP_COLUMNS)
    try:
        sql = (f"SELECT {cols} FROM play_by_play "
               f"WHERE season IN ({placeholders}) AND season_type = 'REG' "
               f"AND posteam IS NOT NULL AND defteam IS NOT NULL")
        rows = db.execute(text(sql), params).mappings().all()
    except Exception as exc:
        logger.warning("No play_by_play table (%s); reading nflverse directly", exc)
        try:
            return trenches.plays(pbp=trenches.load_pbp(seasons))
        except Exception as load_exc:
            logger.warning("nflverse play-by-play unavailable: %s", load_exc)
            return _empty_plays()
    if not rows:
        return _empty_plays()
    frame = pd.DataFrame([dict(r) for r in rows])
    frame["season_type"] = "REG"
    return trenches.plays(pbp=frame)


def _empty_plays():
    rush = pd.DataFrame(columns=["season", "week", "posteam", "defteam",
                                 "aly", "stuffed", "yards_gained"])
    drop = pd.DataFrame(columns=["season", "week", "posteam", "defteam", "pressured"])
    return rush, drop


def _grades(db: Session, season: int, through_week: Optional[int]):
    """Unit grades for a season, optionally as they stood entering a week."""
    trenches = _trenches()

    def build():
        rush, drop = load_trench_plays(db, [season - 1, season])
        if rush.empty:
            return pd.DataFrame(columns=list(trenches.UNITS))
        prior_rush = rush[rush.season == season - 1]
        prior_drop = drop[drop.season == season - 1]
        prior = None
        if len(prior_rush) > 2000 and len(prior_drop) > 2000:
            prior = trenches.unit_ratings(prior_rush, prior_drop)
        return trenches.ratings_through(rush, drop, season, through_week, prior=prior)

    return _cached(("grades", season, through_week), build)


def _rows(frame: pd.DataFrame):
    """Rows as JSON-safe dicts, for a frame whose columns are already right.

    `.where(notna, None)` alone is not enough: on a float64 column pandas puts
    the NaN straight back, and json refuses it with "Out of range float values
    are not JSON compliant". Going through object dtype is what makes the null
    stick. `trenches.matchup` returns nan for a team missing from the grades
    frame, so this path does reach nulls.
    """
    if frame is None or frame.empty:
        return []
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def _records(grades: pd.DataFrame):
    """The same, for a frame that carries the team in its index."""
    return _rows(grades.reset_index().rename(columns={"index": "team"}))


@router.get("/ratings")
def unit_ratings(season: int = Query(..., ge=1999),
                 through_week: Optional[int] = Query(None, ge=1, le=22),
                 db: Session = Depends(get_db)):
    """0-100 grades for all four line units, 50 = league average.

    `through_week` gives the grades as they stood entering that week, which is
    what a matchup shown before kickoff was built from.
    """
    grades = _grades(db, season, through_week)
    if grades.empty:
        raise HTTPException(status_code=404,
                            detail=f"No play-by-play for {season} yet.")
    return {"season": season, "through_week": through_week,
            "scale": "0-100, 50 = league average, 15 points per standard deviation",
            "units": _records(grades)}


@router.get("/matchups")
def weekly_matchups(season: int = Query(..., ge=1999),
                    week: int = Query(..., ge=1, le=22),
                    db: Session = Depends(get_db)):
    """Both sides of every game in a week, run and pass.

    Ratings are built from weeks before `week`, so this is usable before the
    games are played and is the same number the backtest grades afterwards.
    """
    trenches = _trenches()
    grades = _grades(db, season, week)
    if grades.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough {season} football yet to rate lines.")
    games = db.query(Schedule).filter(Schedule.season == season,
                                      Schedule.week == week).all()
    if not games:
        raise HTTPException(status_code=404,
                            detail=f"No schedule for {season} week {week}.")
    schedule = pd.DataFrame([{"season": g.season, "week": g.week,
                              "home_team": g.home_team, "away_team": g.away_team}
                             for g in games])
    rows = trenches.matchups_for_week(grades, schedule, season, week)
    rows = rows.sort_values("run_matchup", ascending=False)
    return {
        "season": season, "week": week,
        "basis": str(grades.get("basis", pd.Series(["in-season"])).iloc[0]),
        "bands": [{"from": lo, "to": hi, "label": label}
                  for lo, hi, label in trenches.BANDS],
        "matchups": _rows(rows.reset_index(drop=True)),
    }


@router.get("/backtest")
def backtest(seasons: int = Query(3, ge=1, le=10),
             through_season: Optional[int] = Query(None, ge=1999),
             first_week: int = Query(6, ge=2, le=18),
             db: Session = Depends(get_db)):
    """What each rating band actually delivered, on games already played.

    Every rating is rebuilt from the weeks before the game it is graded against,
    so this is a prospective test and not a fit. Two ladders come back:
    `matchup_bands` for the combined number the dashboard shows, and
    `opponent_bands` for the opponent's front on its own.

    The second is the one to read. A team's own line is already inside its
    players' recent form, so the combined rating partly re-states what the
    projection already knows; the opponent's front is the new information.
    `ypc_vs_own` is each team's yards per carry in the game minus its own
    average for the season, which is what strips team quality out.
    """
    trenches = _trenches()

    def build():
        rows = db.execute(text("SELECT DISTINCT season FROM play_by_play "
                               "WHERE season_type = 'REG' ORDER BY season")).all()
        available = [int(r[0]) for r in rows]
        if through_season is not None:
            available = [s for s in available if s <= through_season]
        if not available:
            raise HTTPException(status_code=404, detail="No play-by-play loaded.")
        chosen = available[-seasons:]
        rush, drop = load_trench_plays(db, chosen)
        per_game, matchup_bands = trenches.backtest(rush, drop, seasons=chosen,
                                                    first_week=first_week)
        if per_game.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Not enough football in {chosen} to backtest from week {first_week}.")
        return {
            "seasons": chosen,
            "first_week": first_week,
            "team_games": int(len(per_game)),
            "matchup_bands": _records(matchup_bands.reset_index(drop=True)),
            "opponent_bands": _records(
                trenches.bands(per_game, opponent_only=True).reset_index(drop=True)),
            "correlations": trenches.correlations(per_game),
        }

    return _cached(("backtest", seasons, through_season, first_week), build)
