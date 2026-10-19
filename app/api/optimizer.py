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
import hashlib
import io
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from database.models import (FanduelSlate, LineupBuild, LineupEntry, LineupPlayer,
                             PlayerProjection, Schedule)
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


def _csv_frame(text: str, source: str = "That file") -> pd.DataFrame:
    """Parse and validate a slate CSV. Shared by the upload and the saved copy,
    so a slate that was readable on upload cannot become unreadable later
    without saying so."""
    try:
        frame = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"{source} could not be read as CSV: {exc}")
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(f"{source} does not look like a FanDuel slate export — no "
                    f"{', '.join(missing)} column. Download the player list from the "
                    f"contest page."))
    return frame


async def _read_csv(upload: UploadFile):
    """``(frame, text)`` for an upload. The text goes to the database verbatim."""
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="That file is empty.")
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"That file is {len(raw) // 1024} KB; a FanDuel slate export is ~100 KB.")
    # utf-8-sig, not utf-8: FanDuel's export has carried a byte-order mark, and
    # left in the stored text it would reappear in the first column's NAME on
    # every later read.
    text = raw.decode("utf-8-sig", "replace")
    return _csv_frame(text), text


# ---------------------------------------------------------------------------
# The saved slate
#
# Uploading the contest CSV is the one thing in this chain a person has to do by
# hand, so it is done once per slate and reused: every build stores what it was
# given, and a build with no file reaches for the stored copy.
# ---------------------------------------------------------------------------

def _slate_teams_listed(frame: pd.DataFrame):
    if "Team" not in frame.columns:
        return []
    return sorted({str(t).upper() for t in frame["Team"].dropna().unique() if str(t).strip()})


def _slate_key(frame: pd.DataFrame, contest: str) -> str:
    """What makes two uploads the same slate.

    A week has exactly one classic player list, but a Showdown file per game —
    so the game's teams are part of the key, and re-uploading Chiefs-Bills
    replaces Chiefs-Bills rather than last week's other Showdown.
    """
    if contest != "showdown":
        return "classic"
    teams = _slate_teams_listed(frame)
    return "showdown:" + "@".join(teams) if teams else "showdown"


def _save_slate(db, season: int, week: int, contest: str, filename: Optional[str],
                text: str, frame: pd.DataFrame) -> Optional[FanduelSlate]:
    """Store (or replace) the slate for this week. Never raises: a failed save
    costs the convenience, not the lineups the caller asked for."""
    try:
        key = _slate_key(frame, contest)
        row = db.query(FanduelSlate).filter(
            FanduelSlate.season == season, FanduelSlate.week == week,
            FanduelSlate.slate_key == key).one_or_none()
        if row is None:
            row = FanduelSlate(season=season, week=week, slate_key=key)
            db.add(row)
        row.contest = contest
        row.filename = (filename or "slate.csv")[:255]
        row.content = text
        row.content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row.row_count = int(len(frame))
        row.teams = ",".join(_slate_teams_listed(frame))
        row.uploaded_at = datetime.utcnow()
        db.commit()
        return row
    except Exception as exc:
        logger.warning("Could not save the slate CSV (%s); this build still runs", exc)
        db.rollback()
        return None


def _stored_slate(db, season: int, week: int,
                  slate_id: Optional[int] = None) -> Optional[FanduelSlate]:
    """The slate to use when no file was uploaded.

    Classic first when a week holds both: it is the contest the optimizer is
    normally asked about, and a Showdown file is one game. `slate_id` picks a
    specific one, which is how a Showdown slate gets chosen at all.
    """
    if slate_id is not None:
        return db.query(FanduelSlate).filter(FanduelSlate.id == slate_id).one_or_none()
    rows = db.query(FanduelSlate).filter(
        FanduelSlate.season == season, FanduelSlate.week == week).all()
    if not rows:
        return None
    classic = [r for r in rows if r.contest == "classic"]
    return max(classic or rows, key=lambda r: r.uploaded_at or datetime.min)


def _slate_summary(row: FanduelSlate) -> dict:
    return {
        "slate_id": row.id,
        "season": row.season,
        "week": row.week,
        "contest": row.contest,
        "slate_key": row.slate_key,
        "filename": row.filename,
        "players": row.row_count,
        "teams": [t for t in (row.teams or "").split(",") if t],
        "uploaded_at": row.uploaded_at,
        "used_at": row.used_at,
        "content_hash": (row.content_hash or "")[:12],
    }


# ---------------------------------------------------------------------------
# The record of what was built
#
# Stored so the optimizer can be scored the way the projections are: what it
# said before kickoff, against what happened. scripts.score_lineups fills in
# the actuals, and GET /optimizer/performance reads them back.
# ---------------------------------------------------------------------------

def _text_or_none(value):
    if value is None or value != value:        # None or NaN
        return None
    text = str(value).strip()
    return text or None


def _float_or_none(value):
    try:
        if value is None or value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _drop_superseded(db, settings: dict) -> None:
    """Forget an earlier UNSCORED build with exactly these settings.

    Pressing Build twice is not two samples. The same slate, cap, objective,
    exposure and exclusions produce the same lineups — or, if the board has
    been recomputed in between, the newer ones are the ones that matter — so
    the record keeps the latest rather than averaging a click count. A build
    that has already been scored is history and is never touched.
    """
    stale = db.query(LineupBuild).filter_by(scored_at=None, **settings).all()
    for build in stale:
        entries = db.query(LineupEntry).filter(LineupEntry.build_id == build.id).all()
        for entry in entries:
            db.query(LineupPlayer).filter(LineupPlayer.lineup_id == entry.id).delete()
            db.delete(entry)
        db.delete(build)


def _record_build(db, lineups, *, season: int, week: int, slate_id: Optional[int],
                  contest: str, slate: str, objective: str, salary_cap: int,
                  max_usage_percentage: int, excludes, num_requested: int):
    """Write the build and its lineups; return the build id, or None.

    Wrapped whole in try/except on purpose, the same way the volume context is:
    bookkeeping that can fail somebody's lineups is worse than no bookkeeping.
    """
    try:
        settings = dict(
            season=season, week=week, slate_id=slate_id, contest=contest, slate=slate,
            objective=objective, salary_cap=int(salary_cap),
            max_usage_percentage=int(max_usage_percentage),
            exclude_players=",".join(excludes) if excludes else None,
            num_requested=int(num_requested))
        _drop_superseded(db, settings)
        build = LineupBuild(**settings, num_built=len(lineups),
                            built_at=datetime.utcnow())
        db.add(build)
        db.flush()
        for index, lineup in enumerate(lineups, start=1):
            def column(name):
                # `lineup.get(missing)` is None, and pd.to_numeric(None) is a
                # scalar NaN rather than a Series -- so .fillna on it raises.
                if name not in lineup.columns:
                    return pd.Series(0.0, index=lineup.index)
                return pd.to_numeric(lineup[name], errors="coerce")

            salaries, points = column("Salary"), column("lineup_points")
            entry = LineupEntry(
                build_id=build.id, lineup_index=index,
                salary=int(salaries.fillna(0).sum()),
                projected_points=round(float(points.fillna(0).sum()), 2))
            db.add(entry)
            db.flush()
            for _, row in lineup.iterrows():
                db.add(LineupPlayer(
                    lineup_id=entry.id,
                    slot=_text_or_none(row.get("Roster Position")),
                    position=_text_or_none(row.get("Position")),
                    player_id=_text_or_none(row.get("player_id")),
                    player_name=_text_or_none(row.get("Nickname")),
                    team=_text_or_none(row.get("Team")),
                    salary=int(row["Salary"]) if pd.notna(row.get("Salary")) else None,
                    projected_points=_float_or_none(row.get("lineup_points")),
                    from_model=int(pd.notna(row.get("fanduel_fantasy_points"))),
                    actual_points=None))
        if slate_id is not None:
            row = db.query(FanduelSlate).filter(FanduelSlate.id == slate_id).one_or_none()
            if row is not None:
                row.used_at = datetime.utcnow()
        db.commit()
        return build.id
    except Exception as exc:
        logger.warning("Could not record this build for scoring (%s)", exc)
        db.rollback()
        return None


@router.post("/lineups")
async def build_lineups(
    file: Optional[UploadFile] = File(
        None, description="FanDuel slate CSV. Optional: the week's saved slate is used "
                          "when no file is sent."),
    season: Optional[int] = Form(None),
    week: Optional[int] = Form(None),
    num_lineups: int = Form(5),
    objective: str = Form("mean"),
    salary_cap: int = Form(60000),
    max_usage_percentage: int = Form(50),
    exclude: str = Form(""),
    slate: str = Form("all"),
    one_kicker: bool = Form(True),
    slate_id: Optional[int] = Form(None),
    save_slate: bool = Form(True),
    track: bool = Form(True),
    db: Session = Depends(get_db),
):
    """Build lineups from a FanDuel slate CSV and the week's published projections.

    The CSV is uploaded ONCE. Every upload is stored against (season, week,
    contest) and a request with no file reuses it, so changing the objective or
    the cap does not mean going back to FanDuel for the same download. A later
    upload for the same week replaces it, which is what you want when prices
    move midweek. `slate_id` picks a specific saved slate (how a Showdown file
    gets chosen when a classic list is also stored), `save_slate=false` builds
    from a file without keeping it, and `track=false` builds without recording
    it for scoring.

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

    uploaded = file is not None and bool((getattr(file, "filename", "") or "").strip())
    slate_row = None
    if uploaded:
        salaries, csv_text = await _read_csv(file)
    else:
        slate_row = _stored_slate(db, season, week, slate_id)
        if slate_row is None:
            raise HTTPException(
                status_code=400,
                detail=(f"No slate CSV saved for {season} week {week}. Upload the "
                        f"contest player list once — every build after that reuses it."))
        if slate_row.season != season or slate_row.week != week:
            # Only reachable via an explicit slate_id, and worth saying rather
            # than quietly pricing one week's players off another week's slate.
            raise HTTPException(
                status_code=400,
                detail=(f"Saved slate {slate_row.id} is for {slate_row.season} week "
                        f"{slate_row.week}, not {season} week {week}."))
        csv_text = slate_row.content or ""
        salaries = _csv_frame(csv_text, "The saved slate")
    slate_size = len(salaries)

    # Showdown is its own contest, not a classic slate with fewer games: five
    # players from one game, one MVP at 1.5x points and 1.5x salary. Run through
    # the classic optimizer it yields nothing, since no row names a classic slot
    # — which is exactly what the first real upload here did.
    showdown = opt.is_showdown(salaries)

    # Save before building, not after: an upload that turns out to fit no lineup
    # is still the file for this week, and re-downloading it to try a different
    # cap is the work this is here to remove.
    if uploaded and save_slate:
        slate_row = _save_slate(db, season, week, "showdown" if showdown else "classic",
                                getattr(file, "filename", None), csv_text, salaries)
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

    build_id = None
    if track:
        build_id = _record_build(
            db, lineups, season=season, week=week,
            slate_id=slate_row.id if slate_row is not None else None,
            contest="showdown" if showdown else "classic",
            slate="single_game" if showdown else slate, objective=objective,
            salary_cap=salary_cap, max_usage_percentage=max_usage_percentage,
            excludes=excludes, num_requested=num_lineups)

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
        # Where the salaries came from, and the row they are kept in. "stored"
        # means no file was sent and the saved slate was used.
        "slate_source": "upload" if uploaded else "stored",
        "slate_saved": slate_row is not None,
        **({"saved_slate": _slate_summary(slate_row)} if slate_row is not None else {}),
        # The row in `lineup_builds` these lineups were written to, which is
        # what GET /optimizer/performance scores once the week is played.
        "build_id": build_id,
        "tracked": build_id is not None,
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
        # Who these lineups actually lean on. The cap bounds this within the
        # build; GET /optimizer/exposure is the same table across every build
        # of the week, where nothing bounds it.
        "exposure": _exposure_rows([{
            "player_name": p["player_name"],
            "team": p["team"],
            "position": p["roster_position"],
            "slot": p["roster_position"],
            "salary": p["salary"],
            "projected": p["projected"],
            "from_model": p["from_model"],
        } for lineup in out for p in lineup["players"]], len(out)),
        "exposure_cap": max_usage_percentage,
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


@router.get("/slates")
def list_slates(
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """The slate CSVs on file. A build with no upload uses the newest classic one
    for its week."""
    season = season or get_current_nfl_season()
    q = db.query(FanduelSlate).filter(FanduelSlate.season == season)
    if week is not None:
        q = q.filter(FanduelSlate.week == week)
    rows = sorted(q.all(), key=lambda r: (r.week or 0, r.slate_key or ""), reverse=True)
    return {
        "status": "success",
        "season": season,
        "week": week,
        "count": len(rows),
        "data": [_slate_summary(r) for r in rows],
        "note": ("A build with no file uploaded uses the newest classic slate for the "
                 "week; pass slate_id to use a specific one (a Showdown file, say)."),
    }


@router.delete("/slates/{slate_id}")
def delete_slate(slate_id: int, db: Session = Depends(get_db)):
    """Forget a saved slate. The builds made from it keep their record."""
    row = db.query(FanduelSlate).filter(FanduelSlate.id == slate_id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No saved slate {slate_id}.")
    summary = _slate_summary(row)
    db.delete(row)
    db.commit()
    return {"status": "deleted", "slate": summary}


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def _perf_bucket(**fields) -> dict:
    return {**fields, "builds": 0, "lineups": 0, "complete": 0,
            "projected": [], "actual": [], "best_actual": None, "worst_actual": None,
            "fppg_points": [], "best_points": [], "beat_fppg": 0, "compared": 0}


def _perf_add(bucket: dict, build, rows) -> None:
    """Fold one build's lineups into a bucket.

    Only lineups whose every slot could be scored are averaged: a partial
    lineup is not a low score, it is an unknown one.
    """
    complete = [e for e in rows
                if e.actual_points is not None and not (e.unscored_players or 0)]
    bucket["builds"] += 1
    bucket["lineups"] += len(rows)
    bucket["complete"] += len(complete)
    bucket["projected"].extend(e.projected_points for e in complete)
    bucket["actual"].extend(e.actual_points for e in complete)
    for entry in complete:
        if bucket["best_actual"] is None or entry.actual_points > bucket["best_actual"]:
            bucket["best_actual"] = entry.actual_points
        if bucket["worst_actual"] is None or entry.actual_points < bucket["worst_actual"]:
            bucket["worst_actual"] = entry.actual_points
    if build.fppg_points is not None:
        bucket["fppg_points"].append(build.fppg_points)
        for entry in complete:
            bucket["compared"] += 1
            bucket["beat_fppg"] += int(entry.actual_points > build.fppg_points)
    if build.best_points is not None:
        bucket["best_points"].append(build.best_points)


def _perf_summary(bucket: dict) -> dict:
    projected, actual = _mean(bucket["projected"]), _mean(bucket["actual"])
    return {
        "builds": bucket["builds"],
        "lineups": bucket["lineups"],
        "scored_lineups": bucket["complete"],
        "projected": projected,
        "actual": actual,
        "error": (round(projected - actual, 2)
                  if projected is not None and actual is not None else None),
        "best_lineup_actual": bucket["best_actual"],
        "worst_lineup_actual": bucket["worst_actual"],
        "fppg_baseline": _mean(bucket["fppg_points"]),
        "best_possible": _mean(bucket["best_points"]),
        "beat_fppg": bucket["beat_fppg"],
        "compared_to_fppg": bucket["compared"],
        "beat_fppg_rate": (round(bucket["beat_fppg"] / bucket["compared"], 3)
                           if bucket["compared"] else None),
    }


def _exposure_rows(records, lineups_counted: int):
    """Each player's share of the lineups counted, most-used first.

    Exposure is the readback of the only knob the builder has for diversity:
    `max_usage_percentage` caps how much of ONE build a player may occupy. It
    says nothing across builds, so the same chalk can sit in every lineup of
    every build of the week and no constraint ever notices. That is what this
    table is for -- the cap is a promise about one build, exposure is what
    actually happened across all of them.

    `records` is one dict per filled slot: player_name, team, position, slot,
    salary, projected, actual, from_model, build_id. A player is keyed by name
    and team, which is also what makes a defense a single row.
    """
    bucket: dict = {}
    for rec in records:
        name = (rec.get("player_name") or "").strip()
        if not name:
            continue
        team = (rec.get("team") or "").strip().upper() or None
        row = bucket.setdefault((name.lower(), team or ""), {
            "player_name": name,
            "team": team,
            "position": rec.get("position"),
            "salary": rec.get("salary"),
            "lineups": 0,
            "slots": {},
            "projected": [],
            "actual": [],
            "builds": set(),
            "from_model": False,
        })
        row["lineups"] += 1
        slot = (rec.get("slot") or "").strip()
        if slot:
            row["slots"][slot] = row["slots"].get(slot, 0) + 1
        for key in ("projected", "actual"):
            if rec.get(key) is not None:
                row[key].append(float(rec[key]))
        if rec.get("build_id") is not None:
            row["builds"].add(rec["build_id"])
        # Salary and position are properties of the slate, identical on every
        # repeat; the first non-null wins so a sparse record cannot blank them.
        for key in ("salary", "position"):
            if row[key] is None and rec.get(key) is not None:
                row[key] = rec[key]
        row["from_model"] = row["from_model"] or bool(rec.get("from_model"))

    out = []
    for row in bucket.values():
        out.append({
            "player_name": row["player_name"],
            "position": row["position"],
            "team": row["team"],
            "salary": row["salary"],
            "lineups": row["lineups"],
            "exposure": (round(100.0 * row["lineups"] / lineups_counted, 1)
                         if lineups_counted else None),
            "builds": len(row["builds"]) or None,
            # Showdown pays the MVP 1.5x, so which slot he filled is part of
            # the exposure, not a detail: 4 lineups as MVP is a different bet
            # from 4 as a flex.
            "slots": row["slots"] or None,
            "projected": _mean(row["projected"]),
            "actual": _mean(row["actual"]),
            "from_model": row["from_model"],
        })
    # Most-used first, and a tie goes to the more expensive player: that is the
    # one whose exposure costs the most to be wrong about.
    out.sort(key=lambda r: (-r["lineups"], -(r["salary"] or 0), r["player_name"]))
    return out


@router.get("/exposure")
def lineup_exposure(
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(None, description="One week; omit for the whole season"),
    build_id: Optional[int] = Query(None, description="One build instead of all of them"),
    db: Session = Depends(get_db),
):
    """Which players the lineups are built on, and at what percentage.

    Every recorded build is pooled, so this answers the question the per-build
    exposure cap cannot: across everything entered this week, how much of the
    slate is really one player? A 30% cap on five separate builds still permits
    a man in all of them.

    Exposure is a share of LINEUPS, not of builds -- 6 of 20 lineups is 30%
    whether those 20 came from one build or four.
    """
    season = season or get_current_nfl_season()
    builds = db.query(LineupBuild).filter(LineupBuild.season == season)
    if week is not None:
        builds = builds.filter(LineupBuild.week == week)
    if build_id is not None:
        builds = builds.filter(LineupBuild.id == build_id)
    builds = sorted(builds.all(), key=lambda b: (b.week or 0, b.id))
    if not builds:
        return {"status": "no_data", "season": season, "week": week,
                "build_id": build_id,
                "message": ("No lineups have been built for that season yet. Exposure is "
                            "recorded with every build.")}

    by_id = {b.id: b for b in builds}
    entries = db.query(LineupEntry).filter(
        LineupEntry.build_id.in_(list(by_id))).all()
    records = []
    if entries:
        build_of = {e.id: e.build_id for e in entries}
        for player in db.query(LineupPlayer).filter(
                LineupPlayer.lineup_id.in_([e.id for e in entries])).all():
            records.append({
                "player_name": player.player_name,
                "team": player.team,
                "position": player.position,
                "slot": player.slot,
                "salary": player.salary,
                "projected": player.projected_points,
                "actual": player.actual_points,
                "from_model": bool(player.from_model),
                "build_id": build_of.get(player.lineup_id),
            })

    rows = _exposure_rows(records, len(entries))
    caps = sorted({b.max_usage_percentage for b in builds
                   if b.max_usage_percentage is not None})
    # A player over every cap in force is the headline: no single build broke
    # its promise, and he is still in more of the week than any of them allowed.
    tightest = caps[0] if caps else None
    over_cap = [r for r in rows if tightest is not None
                and (r["exposure"] or 0) > tightest] if len(builds) > 1 else []
    return {
        "status": "success",
        "season": season,
        "week": week,
        "build_id": build_id,
        "builds": len(builds),
        "lineups": len(entries),
        "players_used": len(rows),
        "exposure_caps": caps,
        "weeks": sorted({b.week for b in builds if b.week is not None}),
        "over_cap": [r["player_name"] for r in over_cap],
        "data": rows,
        "note": ("exposure is the share of all %d lineups, pooled across %d build(s). "
                 "The per-build cap (%s) binds inside a build only, so a name in "
                 "over_cap broke no constraint -- it is the concentration the caps "
                 "cannot see. `actual` fills in once the week is scored."
                 % (len(entries), len(builds),
                    ", ".join(f"{c}%" for c in caps) or "none recorded")),
    }


@router.get("/performance")
def lineup_performance(
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(None, description="Include the lineups themselves for one week"),
    db: Session = Depends(get_db),
):
    """How the lineups this optimizer built actually scored.

    The same discipline as the projection accuracy record: the lineups were
    built before kickoff and are scored afterwards, never rewritten. Two
    baselines come with them, because a lineup's raw score says nothing on its
    own — a 110-point week is good or bad depending on the slate.

      * `fppg_points` — what FanDuel's own season-average FPPG would have built
        from the same pool under the same cap, actually scored. This is the
        honest denominator: beating it is the whole claim the board makes.
      * `best_points` — the best lineup the slate held, in hindsight. Nobody
        hits it; it says how much was on the table.

    `complete` counts only lineups where every slot could be scored. A defense
    on a week team stats were unavailable, or a player who did not play, leaves
    a lineup partial and out of the averages.
    """
    season = season or get_current_nfl_season()
    builds = db.query(LineupBuild).filter(LineupBuild.season == season)
    if week is not None:
        builds = builds.filter(LineupBuild.week == week)
    builds = sorted(builds.all(), key=lambda b: (b.week or 0, b.id))
    if not builds:
        return {"status": "no_data", "season": season, "week": week,
                "message": ("No lineups have been built for that season yet. Builds are "
                            "recorded automatically; scoring runs once the week's "
                            "actuals land.")}

    entries = {}
    if builds:
        ids = [b.id for b in builds]
        for entry in db.query(LineupEntry).filter(LineupEntry.build_id.in_(ids)).all():
            entries.setdefault(entry.build_id, []).append(entry)
    for rows in entries.values():
        rows.sort(key=lambda e: e.lineup_index or 0)

    by_week: dict = {}
    by_objective: dict = {}
    for build in builds:
        rows = entries.get(build.id, [])
        _perf_add(by_week.setdefault(build.week, _perf_bucket(week=build.week)),
                  build, rows)
        # The same lineups cut the other way: which objective actually won.
        # Keyed by contest as well, because a Showdown lineup is five players
        # and a classic one is nine -- their raw points are not the same scale,
        # and pooling them would have the contest deciding the comparison.
        key = (build.objective or "unknown", build.contest or "unknown")
        _perf_add(by_objective.setdefault(key, _perf_bucket(
            objective=key[0], contest=key[1])), build, rows)

    weeks = [{"week": b["week"], **_perf_summary(b)}
             for b in sorted(by_week.values(), key=lambda b: b["week"] or 0)]

    scored = [w for w in weeks if w["scored_lineups"]]
    compared = sum(w["compared_to_fppg"] for w in weeks)
    beat = sum(w["beat_fppg"] for w in weeks)
    totals = {
        "builds": sum(w["builds"] for w in weeks),
        "lineups": sum(w["lineups"] for w in weeks),
        "scored_lineups": sum(w["scored_lineups"] for w in weeks),
        "weeks_scored": len(scored),
        "projected": _mean([w["projected"] for w in scored]),
        "actual": _mean([w["actual"] for w in scored]),
        "fppg_baseline": _mean([w["fppg_baseline"] for w in scored]),
        "best_possible": _mean([w["best_possible"] for w in scored]),
        "beat_fppg": beat,
        "compared_to_fppg": compared,
        "beat_fppg_rate": round(beat / compared, 3) if compared else None,
    }

    # Which objective actually won, ranked by what the lineups scored. Ordered
    # best-first among those with scored lineups; an unscored objective sorts
    # last rather than appearing to have lost.
    objectives = [{"objective": b["objective"], "contest": b["contest"],
                   **_perf_summary(b)} for b in by_objective.values()]
    objectives.sort(key=lambda o: (o["actual"] is None, -(o["actual"] or 0)))
    # A verdict is only drawn inside ONE contest, whichever has the most scored
    # lineups: ranking a Showdown against a classic would measure the contest.
    ranked = [o for o in objectives if o["scored_lineups"]]
    contest = None
    if ranked:
        totals_by_contest: dict = {}
        for row in ranked:
            totals_by_contest[row["contest"]] = (
                totals_by_contest.get(row["contest"], 0) + row["scored_lineups"])
        contest = max(totals_by_contest, key=lambda c: totals_by_contest[c])
    judged = [o for o in ranked if o["contest"] == contest]
    # Two objectives have to have scored lineups before one can be said to have
    # beaten the other, and a handful of lineups from one slate is a sample of
    # one week, not evidence about an objective. The flags say which it is; the
    # caller decides whether to phrase a verdict.
    verdict = {
        "comparable": len(judged) >= 2,
        "contest": contest,
        "objectives_scored": len(judged),
        "scored_lineups": sum(o["scored_lineups"] for o in judged),
        "weeks_scored": len(scored),
        "best": judged[0]["objective"] if judged else None,
        "worst": judged[-1]["objective"] if len(judged) >= 2 else None,
        "spread": (round(judged[0]["actual"] - judged[-1]["actual"], 2)
                   if len(judged) >= 2 else None),
        # One week cannot separate objectives that differ by a point or two:
        # the same slate, the same pool, largely the same players.
        "one_week_only": len(scored) <= 1,
    }

    detail = []
    if week is not None:
        lineup_ids = [e.id for rows in entries.values() for e in rows]
        players: dict = {}
        if lineup_ids:
            for player in db.query(LineupPlayer).filter(
                    LineupPlayer.lineup_id.in_(lineup_ids)).all():
                players.setdefault(player.lineup_id, []).append(player)
        for build in builds:
            detail.append({
                "build_id": build.id,
                "built_at": build.built_at,
                "objective": build.objective,
                "contest": build.contest,
                "slate": build.slate,
                "salary_cap": build.salary_cap,
                "exclude_players": build.exclude_players,
                "fppg_baseline": build.fppg_points,
                "fppg_projected": build.fppg_projected,
                "best_possible": build.best_points,
                "scored_at": build.scored_at,
                "lineups": [{
                    "lineup": entry.lineup_index,
                    "salary": entry.salary,
                    "projected": entry.projected_points,
                    "actual": entry.actual_points,
                    "unscored_players": entry.unscored_players,
                    "players": [{
                        "slot": p.slot,
                        "player_name": p.player_name,
                        "team": p.team,
                        "salary": p.salary,
                        "projected": p.projected_points,
                        "actual": p.actual_points,
                        "from_model": bool(p.from_model),
                    } for p in sorted(players.get(entry.id, []), key=lambda p: p.id)],
                } for entry in entries.get(build.id, [])],
            })

    return {
        "status": "success",
        "season": season,
        "week": week,
        "weeks": weeks,
        "totals": totals,
        "by_objective": objectives,
        "objective_verdict": verdict,
        **({"builds": detail} if week is not None else {}),
        "note": ("fppg_baseline is the lineup FanDuel's own FPPG would have built from "
                 "the same pool under the same cap, scored on what happened — the "
                 "honest denominator. best_possible is the slate's best lineup in "
                 "hindsight. Only lineups whose every slot could be scored are averaged."),
    }
