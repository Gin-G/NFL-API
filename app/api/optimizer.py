"""FanDuel lineup optimizer, driven by the published projections.

The projections this builds on are already in the database; what FanDuel never
publishes anywhere fetchable is its PRICING. Every free source turned out to be
dead (RotoGuru), historical-only (nfldfs), someone else's product to scrape
(RotoWire, FantasyPros' salary page) or behind a session (FanDuel's own DFS
API), so the slate CSV you download from the contest page is the input: it
carries salary, roster position, injury flag and FanDuel's own FPPG.

That FPPG matters for more than completeness — it fills the DEF slot, which the
model does not project at all.

The optimizer itself lives in nfl_projections.optimizer, imported here without
TensorFlow (the package defers its model imports), so the web image stays small.
"""
import io
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from database.models import PlayerProjection, Schedule
from database.session import get_db

from .utils import get_current_nfl_season

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_CSV_BYTES = 5 * 1024 * 1024        # a full slate export is ~100 KB
REQUIRED_COLUMNS = ("Nickname", "Salary", "Position", "Roster Position")

# Slates, as FanDuel sells them. The export is the whole Thursday-Monday player
# list whichever contest you are entering, and it carries no kickoff times — so
# the windows come from our own schedule, matched on team.
#
# "main" is the 1pm and 4pm Sunday windows: no Thursday or Monday game, no
# Sunday night game, and not the 9:30am London game either.
SLATES = {
    "all": "Every game in the file (Thursday through Monday)",
    "main": "Sunday main — the 1pm and 4pm windows only",
    "sunday": "Every Sunday game, including the early London game and Sunday night",
    "primetime": "Thursday, Sunday night and Monday night",
}


def _slate_teams(db, season: int, week: int, slate: str):
    """``(teams, games)`` for a slate, or ``(None, [])`` to mean no filtering."""
    if slate == "all":
        return None, []
    games = db.query(Schedule).filter(
        Schedule.season == season, Schedule.week == week,
        Schedule.game_type == "REG").all()
    if not games:
        return None, []

    def minutes(game):
        try:
            hh, mm = str(game.gametime).split(":")[:2]
            return int(hh) * 60 + int(mm)
        except (ValueError, AttributeError):
            return None

    def weekday(game):
        try:
            return datetime.strptime(str(game.gameday)[:10], "%Y-%m-%d").strftime("%A")
        except ValueError:
            return ""

    keep, listed = set(), []
    for game in games:
        day, kick = weekday(game), minutes(game)
        if kick is None:
            continue
        if slate == "main":
            wanted = day == "Sunday" and 13 * 60 <= kick < 18 * 60
        elif slate == "sunday":
            wanted = day == "Sunday"
        elif slate == "primetime":
            wanted = (day in ("Thursday", "Monday")) or (day == "Sunday" and kick >= 18 * 60)
        else:
            wanted = True
        if wanted:
            keep.update({game.home_team, game.away_team})
            listed.append({"game": f"{game.away_team}@{game.home_team}",
                           "day": day, "kickoff_et": game.gametime})
    return keep, sorted(listed, key=lambda g: (g["day"], g["kickoff_et"]))


# Our stored column names -> what nfl_projections.optimizer expects to find.
PROJECTION_COLUMNS = {
    "projected_points": "fanduel_fantasy_points",
    "median": "projection_median",
    "floor": "floor",
    "ceiling": "ceiling",
}


def _projection_frame(db, season: int, week: int) -> pd.DataFrame:
    """The published board for one week, shaped the way the optimizer reads it."""
    rows = db.query(PlayerProjection).filter(
        PlayerProjection.season == season, PlayerProjection.week == week).all()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([{
        "player_id": r.player_id,
        "player_name": r.player_name,
        "position": r.position,
        "team": r.team,
        **{dest: getattr(r, src) for src, dest in PROJECTION_COLUMNS.items()},
    } for r in rows])
    return frame


async def _read_csv(upload: UploadFile) -> pd.DataFrame:
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="That file is empty.")
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"That file is {len(raw) // 1024} KB; a FanDuel slate export is ~100 KB.")
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read that as CSV: {exc}")
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(f"That does not look like a FanDuel slate export — no {', '.join(missing)} "
                    f"column. Download the player list from the contest page."))
    return frame


@router.post("/lineups")
async def build_lineups(
    file: UploadFile = File(..., description="FanDuel slate CSV (the contest player list)"),
    season: Optional[int] = Form(None),
    week: Optional[int] = Form(None),
    num_lineups: int = Form(5),
    objective: str = Form("mean"),
    salary_cap: int = Form(60000),
    max_usage_percentage: int = Form(50),
    exclude: str = Form(""),
    slate: str = Form("all"),
    db: Session = Depends(get_db),
):
    """Build lineups from a FanDuel slate CSV and the week's published projections.

    `objective` is mean / median / floor / ceiling. Mean is the default because
    it is the one that was measured: EXPERIMENTS #13 built mean- and
    ceiling-optimised lineups across 2025 weeks 4-18 and mean won on every
    metric, including the tournament-style best-of-five — the ceiling objective
    scored lower at the same variance.

    `max_usage_percentage` caps how many of the returned lineups one player can
    appear in, which is what keeps five lineups from being one lineup five times.
    """
    from nfl_projections import optimizer as opt

    season = season or get_current_nfl_season()
    if week is None:
        weeks = [w for (w,) in db.query(PlayerProjection.week).filter(
            PlayerProjection.season == season).distinct().all()]
        if not weeks:
            raise HTTPException(status_code=404,
                                detail=f"No projections stored for {season}.")
        week = min(weeks)        # the upcoming week is the lowest unplayed one
    if num_lineups < 1 or num_lineups > 50:
        raise HTTPException(status_code=400, detail="num_lineups must be 1-50.")

    projections = _projection_frame(db, season, week)
    if projections.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No projections stored for {season} week {week}; run the projections job.")

    if slate not in SLATES:
        raise HTTPException(status_code=400,
                            detail=f"slate must be one of {', '.join(SLATES)}.")

    salaries = await _read_csv(file)
    slate_size = len(salaries)
    # The export is always the full Thursday-Monday list, so a narrower contest
    # means dropping the games it does not include.
    teams, games = _slate_teams(db, season, week, slate)
    if teams is not None and "Team" in salaries.columns:
        salaries = salaries[salaries["Team"].astype(str).str.upper().isin(teams)]
        if salaries.empty:
            raise HTTPException(
                status_code=422,
                detail=(f"No players from the {slate} slate are in that file. Either the CSV "
                        f"is for a different week, or its team codes do not match ours."))

    merged = opt.merge_fanduel_salaries(salaries, projections)
    matched = int(merged["fanduel_fantasy_points"].notna().sum())

    excludes = [n.strip() for n in exclude.split(",") if n.strip()]
    try:
        lineups = opt.optimize_lineups(
            merged, num_lineups=num_lineups, salary_cap=salary_cap,
            exclude_players=excludes or None,
            max_usage_percentage=max_usage_percentage, objective=objective)
    except Exception as exc:
        logger.exception("Lineup optimisation failed")
        raise HTTPException(status_code=422, detail=f"Could not build lineups: {exc}")
    if not lineups:
        # Say WHICH slot could not be filled. The first real slate upload failed
        # with a generic version of this message and the cause — bare "RB" roster
        # positions where flex-qualified ones were expected — was invisible.
        counts = opt.slot_counts(merged)
        short = {slot: counts.get(slot, 0) for slot, need in opt.ROSTER_SLOTS.items()
                 if counts.get(slot, 0) < need}
        if short:
            detail = ("No lineup fits: the slate is short at "
                      + ", ".join(f"{slot} ({have} eligible)" for slot, have in short.items())
                      + ". If every slot reads 0, the CSV's Roster Position column is in a "
                      "format we do not recognise — send it over.")
        else:
            detail = (f"Every slot has players, so the salary cap (${salary_cap:,}) or the "
                      f"exclusions are what nothing fits under.")
        raise HTTPException(status_code=422, detail=detail)

    out = []
    for i, lineup in enumerate(lineups, start=1):
        players = [{
            "player_name": r.get("Nickname"),
            "roster_position": r.get("Roster Position"),
            "team": r.get("Team") if "Team" in r else r.get("team"),
            "salary": int(r["Salary"]) if pd.notna(r.get("Salary")) else None,
            "projected": round(float(r["lineup_points"]), 2) if pd.notna(
                r.get("lineup_points")) else None,
            "from_model": bool(pd.notna(r.get("fanduel_fantasy_points"))),
        } for _, r in lineup.iterrows()]
        out.append({
            "lineup": i,
            "salary": int(sum(p["salary"] or 0 for p in players)),
            "projected": round(sum(p["projected"] or 0 for p in players), 2),
            "players": players,
        })

    return {
        "status": "success",
        "season": season,
        "week": week,
        "objective": objective,
        "salary_cap": salary_cap,
        "requested": num_lineups,
        "slate": slate,
        "slate_description": SLATES[slate],
        "games": games,
        # Fewer than requested is normal rather than an error: the exposure cap
        # limits how often a player may repeat, so a thin slate (or a short
        # one) runs out of distinct lineups. Saying so beats silently returning
        # one lineup when five were asked for.
        "note": (None if len(out) >= num_lineups else
                 f"Only {len(out)} of {num_lineups} lineups fit a "
                 f"{max_usage_percentage}% exposure cap on this slate; raise the cap "
                 f"or ask for fewer."),
        "slate_players": len(salaries),
        "players_in_file": slate_size,
        # How much of the slate our board actually covers. A low number means the
        # name join missed, or the CSV is for a week we have not projected.
        "matched_to_projections": matched,
        "count": len(out),
        "data": out,
    }


@router.get("/slate-coverage")
def slate_coverage(
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """What the optimizer has to work with before a CSV is uploaded: how many
    players are projected for the week, by position."""
    season = season or get_current_nfl_season()
    q = db.query(PlayerProjection).filter(PlayerProjection.season == season)
    if week is not None:
        q = q.filter(PlayerProjection.week == week)
    rows = q.all()
    if not rows:
        return {"status": "no_data", "season": season, "week": week,
                "message": "No projections stored for that week."}
    weeks = sorted({r.week for r in rows})
    week = week if week is not None else min(weeks)
    rows = [r for r in rows if r.week == week]
    by_position: dict = {}
    for r in rows:
        by_position[r.position] = by_position.get(r.position, 0) + 1
    return {
        "status": "success", "season": season, "week": week,
        "projected_players": len(rows), "by_position": by_position,
        "computed_at": max((r.computed_at for r in rows if r.computed_at), default=None),
        "note": ("DEF is not projected by the model; those slots use FanDuel's own "
                 "FPPG from the uploaded CSV."),
    }
