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


# Distinguishes "the slate cannot field a lineup" (None) from "this package
# version cannot tell us" when explaining why nothing fit.
_UNKNOWN = object()


# Named here rather than imported from the package, so a version skew cannot
# turn a missing constant into a failed lineup build.
_OPPORTUNITY_COLUMNS = ("opp_per_game", "opp_share", "xfp_per_game",
                        "volume_td_gap", "share_jump", "prior_games")


def _attach_opportunity(pool: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Join the volume context on here rather than inside the package call.

    Deliberately NOT routed through `merge_fanduel_salaries`, which grew an
    optional third argument for this: the image installs nfl_projections from
    the main branch at build time, so an image built while the package was
    older would raise `TypeError: takes 2 positional arguments but 3 were
    given` at request time and every lineup build would 422. A plain merge
    here needs nothing from the package but the context frame itself, which
    `_opportunity` already soft-fails.
    """
    try:
        context = _opportunity(season, week)
        if context is None or getattr(context, "empty", True):
            return pool
        cols = [c for c in _OPPORTUNITY_COLUMNS if c in context.columns]
        if "player_id" not in pool.columns or not cols:
            return pool
        return pool.merge(
            context[["player_id"] + cols].drop_duplicates("player_id"),
            on="player_id", how="left")
    except Exception as exc:
        # The whole path, not just the fetch inside `_opportunity`: the join can
        # fail too (a duplicated column, a dtype that will not merge), and none
        # of it is worth somebody's lineups. Optional context that can break the
        # build is not optional.
        logger.warning("opportunity context not attached (%s); omitting", exc)
        return pool


def _opportunity(season: int, week: int):
    """Volume context for the pool, or None if it cannot be had.

    Context only, and the boundary is measured rather than careful: the board
    already carries targets, carries, target_share, air_yards_share, wopr and
    an opportunity_score as input features, so against its own error on five
    seasons of backtests both opportunity share and volume-implied points come
    in null. The one residual -- volume-implied points minus points actually
    scored -- is real but buys 0.2 points of top-24 overlap in 4 weeks out of
    83, and the optimizer reads nothing but the order. So these columns are
    shown beside the lineup and never enter the objective.

    Soft-failing on purpose: a lineup build must not die because nflverse is
    slow. The columns simply come back empty.
    """
    try:
        from nfl_projections import opportunity

        return opportunity.opportunity_context(season, week)
    except Exception as exc:
        logger.warning("opportunity context unavailable (%s); omitting", exc)
        return None


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
    one_kicker: bool = Form(True),
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

    # Showdown is its own contest, not a classic slate with fewer games: five
    # players from one game, one MVP at 1.5x points and 1.5x salary. Run through
    # the classic optimizer it yields nothing, since no row names a classic slot
    # — which is exactly what the first real upload here did.
    showdown = opt.is_showdown(salaries)
    teams, games = (None, [])
    if not showdown:
        # The export is always the full Thursday-Monday list, so a narrower
        # contest means dropping the games it does not include.
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
    merged = _attach_opportunity(merged, season, week)

    excludes = [n.strip() for n in exclude.split(",") if n.strip()]
    extra = {}
    if showdown and one_kicker:
        # Showdown lets a lineup take every kicker and defense in the game, and
        # FanDuel's season-average FPPG — all these positions have, since the
        # model does not project them — makes them look cheap and dependable.
        # The first real slate duly produced a lineup with both kickers in it.
        extra["position_limits"] = {"K": 1, "D": 1}
    build = opt.optimize_showdown if showdown else opt.optimize_lineups
    try:
        lineups = build(
            merged, num_lineups=num_lineups, salary_cap=salary_cap,
            exclude_players=excludes or None,
            max_usage_percentage=max_usage_percentage, objective=objective, **extra)
    except Exception as exc:
        logger.exception("Lineup optimisation failed")
        raise HTTPException(status_code=422, detail=f"Could not build lineups: {exc}")
    if not lineups and showdown:
        raise HTTPException(
            status_code=422,
            detail=(f"No Showdown lineup fits. Five players at MVP pricing have to come in "
                    f"under ${salary_cap:,}, and {len(merged)} players are in this pool "
                    f"after exclusions."))
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
            # Say by HOW MUCH. The old wording named the cap and the exclusions
            # and left the user to guess which, on a slate where a lineup
            # plainly fitted -- the builder was greedy and simply could not
            # find it. Now that the solve is exact this message only fires when
            # nothing fits, so it can afford to be specific.
            # Three outcomes, not two: `None` means the slate genuinely cannot
            # field a lineup at any price, while "could not work it out" means
            # an older nfl_projections in the image has no such function.
            # Collapsing them blamed overlapping candidates for what was
            # actually the cap.
            cheapest = _UNKNOWN
            try:
                cheapest = opt.cheapest_lineup_cost(merged)
            except AttributeError:
                pass        # older nfl_projections in the image
            if cheapest is _UNKNOWN:
                detail = (f"No lineup fits under ${salary_cap:,}: the cap, the "
                          f"exclusions, or the exposure cap "
                          f"({max_usage_percentage}%) is what nothing fits under.")
            elif cheapest is None:
                detail = ("Every slot has players, but no legal lineup can be assembled "
                          "from them — some slot's only candidates are already used by "
                          "another slot.")
            elif cheapest > salary_cap:
                detail = (f"The cheapest legal lineup on this slate costs ${cheapest:,}, "
                          f"which is over the ${salary_cap:,} cap. Raise the cap or drop "
                          f"an exclusion.")
            else:
                detail = (f"No lineup fits under ${salary_cap:,}. The cheapest legal "
                          f"lineup costs ${cheapest:,}, so the exclusions or the "
                          f"exposure cap ({max_usage_percentage}%) are what nothing "
                          f"fits under.")
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
            # Volume context. `opportunities` is targets + carries per game
            # (pass attempts for a quarterback), `opportunity_share` his cut of
            # the team's, and `volume_td_gap` what his volume says he should
            # have scored minus what he did -- positive is the receiver who gets
            # the ball and has not found the end zone.
            **{key: (round(float(r[col]), 3) if pd.notna(r.get(col)) else None)
               for key, col in (("opportunities", "opp_per_game"),
                                ("opportunity_share", "opp_share"),
                                ("volume_points", "xfp_per_game"),
                                ("volume_td_gap", "volume_td_gap"),
                                ("share_jump", "share_jump"))
               if col in lineup.columns},
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
        "contest": "showdown" if showdown else "classic",
        "position_limits": extra.get("position_limits"),
        "slate": "single_game" if showdown else slate,
        "slate_description": ("Showdown — one game, five players, one MVP at 1.5x"
                              if showdown else SLATES[slate]),
        "opportunity_is_context_only": True,
        "opportunity_note": (
            "Volume is shown, not scored. Opportunity per game correlates 0.974 "
            "with itself across odd and even weeks; touchdowns per opportunity "
            "manage 0.178. But the board already carries volume as input "
            "features, so against its own error opportunity share and "
            "volume-implied points are both null, and the one residual "
            "(volume_td_gap) is worth 0.2 points of top-24 overlap in 4 weeks "
            "of 83. A positive share_jump is a FADE, not a buy: on the clearest "
            "cases receivers scored 3.688 against a 4.470 projection."),
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
