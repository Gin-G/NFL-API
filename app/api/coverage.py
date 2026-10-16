"""Coverage report: which corners get beaten, and which receivers face them.

A 0-100 grade per cornerback, 50 = league average and 15 points to a standard
deviation — the same scale as /ratings and /trenches. High is GOOD coverage, so
the corners to attack are the low ones.

The grade math lives in `nfl_projections.coverage` so the research harness and
this endpoint cannot drift apart; this module is the adapter, the receiver join
and the cache. Grades for a given week use only charting from earlier weeks, so
a matchup shown before kickoff is the number the week was actually graded on.

**Read this page as context, not as a projection input.** That is measured, not
cautious. nfl-data-py EXPERIMENTS.md, "Per-corner coverage grades": every grade
here was fitted against the production board's own residual leave-one-season-out
and scored on MAE and on top-N ordering, and nothing paid. Wide receiver was
actively harmful (-0.0032 MAE, interval clear of zero on the wrong side), tight
end and running back came out at exactly zero, and quarterback was the only
positive and straddled zero. So /projections is not adjusted by anything on this
page, and `coverage_lean` below is a label for a human reading a slate.

**What a corner ranking can and cannot know.** Correlating a corner against
himself (odd weeks against even, pooled over his career) splits the metrics
sharply: how often he is thrown at (0.728) and how deep (0.733) are strong
traits, completion % allowed (0.519) is a real one, and **yards allowed per
target is not a trait at all (0.188, and negative on tighter samples)**. Yards
is the number a corner ranking most wants and the one that does not persist,
because a corner sees a median of 26 targets a season and yards per completion
rides on a handful of long plays. So the grade is built from completion rate
alone, heavily shrunk, with volume and depth shown beside it. A corner who is
avoided and a corner who is good look identical in a rate; only the volume
column separates them.

**The assignment is the part nobody has.** nflverse publishes no shadow-coverage
field, so this endpoint can say a secondary has one weak corner and cannot say
which receiver draws him. `cover_spread` is the honest form of a "pick on him"
read — how unequal a secondary is, which is the half of that read that does not
need the assignment.
"""
import logging
import time
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database.session import get_db
from database.models import PlayerProjection, Schedule

logger = logging.getLogger(__name__)
router = APIRouter()

# PFR charting is about a megabyte a season, three orders of magnitude smaller
# than play-by-play, so unlike /trenches this can be fetched inside a request.
# The cache is about not doing it per page load, not about request size.
_TTL = 3600
_cache: dict = {}
# Positions whose matchup the page is about.
_RECEIVING = ("WR", "TE", "RB")


def _coverage():
    """The research module, or a 503 explaining what is missing."""
    try:
        from nfl_projections import coverage
        return coverage
    except ImportError as exc:        # pragma: no cover - packaging failure
        logger.error("nfl_projections.coverage unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Coverage ratings need the nfl-projections package in this image.",
        )


def _cached(key, build):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    value = build()
    _cache[key] = (time.time(), value)
    return value


def _corner_weeks(season: int):
    """Per-corner charting for this season and last, cached."""
    coverage = _coverage()

    def build():
        charting = coverage.load_charting([season - 1, season])
        if charting.empty:
            return pd.DataFrame()
        try:
            return coverage.corner_weeks(charting)
        except Exception as exc:
            logger.warning("coverage position map unavailable: %s", exc)
            return pd.DataFrame()

    return _cached(("weeks", season), build)


def _grades(season: int, through_week: Optional[int]):
    coverage = _coverage()

    def build():
        cw = _corner_weeks(season)
        if cw.empty:
            return pd.DataFrame(), pd.DataFrame()
        cg = coverage.corner_grades(cw, season, through_week)
        return cg, coverage.team_coverage(cg)

    return _cached(("grades", season, through_week), build)


def _records(frame: pd.DataFrame):
    if frame is None or frame.empty:
        return []
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _bands(coverage):
    return [{"from": lo, "to": hi, "label": label}
            for lo, hi, label in coverage.BANDS]


@router.get("/corners")
def corner_grades(season: int = Query(..., ge=2018),
                  through_week: Optional[int] = Query(None, ge=1, le=22),
                  min_targets: int = Query(10, ge=0, le=200),
                  limit: int = Query(200, ge=1, le=500)):
    """Every graded cornerback, worst coverage first.

    `through_week` gives the grades as they stood entering that week.
    `min_targets` filters on THIS season's charted targets — a corner carried
    by last season's grade alone has a number but has not been tested yet, and
    a ranking that puts him on top is naming a player nobody is throwing at.
    """
    coverage = _coverage()
    cg, _ = _grades(season, through_week)
    if cg.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No PFR coverage charting for {season} yet.")
    rows = cg[cg.targets >= min_targets].head(limit)
    return {
        "season": season, "through_week": through_week,
        "scale": "0-100, 50 = league average, 15 points per standard deviation;"
                 " high = good coverage",
        "shrinkage_targets": coverage.SHRINK_TARGETS,
        "not_a_trait": "yards_per_target is shown for reference only — it does"
                       " not persist (career odd/even r = 0.188)",
        "bands": _bands(coverage),
        "corners": _records(rows),
    }


@router.get("/secondaries")
def team_grades(season: int = Query(..., ge=2018),
                through_week: Optional[int] = Query(None, ge=1, le=22)):
    """Every secondary, softest first, with its weakest regular corner.

    `cover_spread` is the gap in completion rate between a secondary's best and
    worst regular corner. A unit can average out fine and still have one side
    worth attacking, and the average cannot see that — so read the two columns
    together rather than ranking on the grade alone.
    """
    coverage = _coverage()
    _, tc = _grades(season, through_week)
    if tc.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough {season} coverage charting to rate secondaries.")
    return {
        "season": season, "through_week": through_week,
        "scale": "0-100, 50 = league average; high = good coverage",
        "bands": _bands(coverage),
        "secondaries": _records(tc),
    }


@router.get("/matchups")
def weekly_matchups(season: int = Query(..., ge=2018),
                    week: int = Query(..., ge=1, le=22),
                    top_receivers: int = Query(4, ge=1, le=10),
                    db: Session = Depends(get_db)):
    """Every game from the receiving side: who faces which secondary.

    Grades are built from weeks before `week`, so this is readable before
    kickoff. Receivers come from /projections for the same week, ordered by
    projected points, and each carries a `coverage_lean` label.

    **`coverage_lean` is a note, not a multiplier.** `projected_points` is the
    board's own number and is not adjusted here. The lean exists because the
    page would otherwise invite the reader to do the adjustment themselves on
    the position where it was measured harmful: for WR it is deliberately
    `context only`, since the WR coverage matchup has now failed three
    independent tests (team-level yards allowed, per-corner grades, and the
    board's own residual).
    """
    coverage = _coverage()
    cg, tc = _grades(season, week)
    if tc.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough {season} coverage charting to rate secondaries.")
    games = db.query(Schedule).filter(Schedule.season == season,
                                      Schedule.week == week).all()
    if not games:
        raise HTTPException(status_code=404,
                            detail=f"No schedule for {season} week {week}.")
    schedule = pd.DataFrame([{"season": g.season, "week": g.week,
                              "home_team": g.home_team, "away_team": g.away_team}
                             for g in games])
    rows = coverage.matchups_for_week(tc, schedule, season, week)
    if rows.empty:
        raise HTTPException(status_code=404,
                            detail=f"No rateable games in {season} week {week}.")

    projections = db.query(PlayerProjection).filter(
        PlayerProjection.season == season,
        PlayerProjection.week == week,
        PlayerProjection.position.in_(_RECEIVING)).all()
    by_team: dict = {}
    for p in projections:
        by_team.setdefault(p.team, []).append(p)

    corners = cg.set_index("team") if "team" in cg.columns else None
    out = []
    for row in rows.to_dict(orient="records"):
        team = row["team"]
        receivers = sorted(by_team.get(team, []),
                           key=lambda p: p.projected_points or 0.0, reverse=True)
        row["receivers"] = [{
            "player_id": p.player_id, "player_name": p.player_name,
            "position": p.position, "projected_points": p.projected_points,
            "ceiling": p.ceiling,
            "coverage_lean": _lean(p.position, row["opp_cover_grade"],
                                   row.get("opp_cover_spread")),
        } for p in receivers[:top_receivers]]
        if corners is not None and row["opponent"] in corners.index:
            opp = corners.loc[[row["opponent"]]].nsmallest(3, "cover_grade")
            row["opp_corners"] = _records(
                opp[[c for c in ("player_name", "targets", "targets_per_game",
                                 "adot", "cmp_pct_allowed", "cover_grade", "band")
                     if c in opp.columns]])
        out.append(row)
    return {
        "season": season, "week": week,
        "bands": _bands(coverage),
        "lean_is_advisory": True,
        "matchups": out,
    }


def _lean(position: str, grade: float, spread: Optional[float]) -> str:
    """A label for a human, chosen by what was measured for that position.

    WR is `context only` on purpose. The other two are the positions where a
    coverage signal at least showed up in the board's residual (TE on corner
    spread, RB on completion rate allowed), though neither paid when it was
    scored — so even these say `slight`, and nothing says a number.
    """
    if position == "WR":
        return "context only"
    if grade is None or pd.isna(grade):
        return "unknown"
    if position == "TE":
        if spread is not None and not pd.isna(spread) and spread >= 0.08:
            return "slight up (uneven secondary)"
        return "neutral"
    if position == "RB":
        if grade <= 40:
            return "slight up (soft underneath)"
        if grade >= 60:
            return "slight down"
        return "neutral"
    return "neutral"
