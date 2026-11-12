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
import json
import logging
from collections import Counter
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from database.models import (FanduelSlate, LineupBuild, LineupEntry, LineupPlayer,
                             PlayerProjection, PlayerStat, Schedule, SlatePlayerRule)
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
    # Stamped on the frame, not a column: the build records WHEN the board it
    # read was computed, because a stale one is otherwise invisible.
    stamps = [r.computed_at for r in rows if r.computed_at is not None]
    frame.attrs["computed_at"] = max(stamps) if stamps else None
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


def _attach_defense(pool: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Give the DEF rows a real projection, in place of FanDuel's FPPG.

    FPPG is what the slot has always used, for the honest reason that the
    board projects no defenses — and it is worse than nothing. Over 2019-2026,
    ranking defenses by their season average picks a weekly top three worth
    7.2 actual points where `nfl_projections.defense` picks one worth 8.5, and
    FPPG's error is worse than a flat league average's. The slot is one ninth
    of a lineup and this is about a point and a quarter of it.

    Soft-fails the same way the volume context does: a defense projection that
    can break somebody's build is not worth having. The optimizer falls back
    to FPPG per row wherever `defense_points` is missing.
    """
    try:
        from nfl_projections import defense as dst

        if "Team" not in pool.columns:
            return pool
        projected = dst.project(season, week)
        if projected is None or projected.empty:
            return pool
        # FanDuel writes JAC, LAR and WSH where nflverse writes JAX, LA and
        # WAS. Unnormalised, those three defenses quietly keep their FPPG.
        normalize = getattr(dst, "normalize_team",
                            lambda code: str(code or "").strip().upper())
        teams = projected.assign(team=projected["team"].map(normalize)) \
                         .set_index("team")["projected_points"]
        slots = pool.get("Roster Position", pd.Series("", index=pool.index)).astype(str)
        is_def = slots.str.upper().str.replace("/", "", regex=False).isin({"DEF", "D", "DST"})
        pool = pool.copy()
        pool["defense_points"] = np.where(
            is_def, pool["Team"].map(normalize).map(teams).astype(float), np.nan)
        logger.info("defense projections attached for %d of %d DEF rows",
                    int(pool.loc[is_def, "defense_points"].notna().sum()), int(is_def.sum()))
        return pool
    except Exception as exc:
        logger.warning("defense projections not attached (%s); the slot keeps FPPG", exc)
        return pool


def _attach_ownership(pool: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Add projected roster percentages — how much of the field owns each player.

    Footballguys publishes these per site and week, free, and they are the only
    read we have on what everyone else is doing. Two things make them worth
    handling carefully.

    They cover ONE slate: the Sunday 13:00-16:25 windows. Attach them to a pool
    that also holds Thursday, the London game, Sunday night and Monday and eight
    teams come back with no ownership at all, every one of their players looking
    beautifully contrarian. `ownership.coverage` is checked and logged for
    exactly that reason.

    And inside the slate, an omission is information rather than a zero. On 2026
    week 6 every one of the five main-slate players our board projected above 10
    points and the page did not list was inactive, while our own `exp_games` had
    all five at 1.00. So unlisted stays NaN, is never charged for, and is
    reported separately as something to check.

    Soft-fails like the rest of the context: no ownership is a build without a
    tilt, not a failed build.
    """
    try:
        from nfl_projections import ownership as own_mod

        published = own_mod.load(week, warn=False)
        if published is None or published.empty:
            logger.info("no roster percentages published yet for week %s", week)
            return pool
        pool = own_mod.attach(pool, published, warn=False)
        report = own_mod.coverage(pool)
        logger.info("ownership attached: %s of %s rated players listed (%.0f%%)",
                    report["listed"], report["players"],
                    100 * (report["listed_share"] or 0))
        if report["players"] and report["listed_share"] < 0.8:
            logger.warning(
                "only %.0f%% of rated players have ownership — is this pool a "
                "wider slate than the Sunday 13:00-16:25 games?",
                100 * report["listed_share"])
        return pool
    except Exception as exc:
        logger.warning("ownership not attached (%s); the build ignores the field", exc)
        return pool


def _unlisted_note(pool: pd.DataFrame, threshold: float = 10.0) -> list:
    """Players the board rates that the field has not rostered — check the news."""
    try:
        from nfl_projections import ownership as own_mod

        if "ownership_listed" not in pool.columns:
            return []
        flagged = own_mod.flag_unlisted(pool, "fanduel_fantasy_points",
                                        threshold=threshold)
        return [{"player_name": r.Nickname, "team": getattr(r, "Team", None),
                 "position": getattr(r, "Position", None),
                 "salary": int(r.Salary) if pd.notna(getattr(r, "Salary", None)) else None,
                 "projected": round(float(r.fanduel_fantasy_points), 2)}
                for r in flagged.itertuples()]
    except Exception as exc:
        logger.warning("could not flag unlisted players (%s)", exc)
        return []


def _ownership_summary(pool: pd.DataFrame) -> dict:
    """What the field read covered, so a tilt can be judged on its coverage."""
    if "ownership_listed" not in pool.columns:
        return {"available": False, "reason": "not published for this week yet"}
    try:
        from nfl_projections import ownership as own_mod

        report = own_mod.coverage(pool)
        return {"available": True, "rated_players": report["players"],
                "with_ownership": report["listed"],
                "covered_share": (None if report["players"] == 0
                                  else round(report["listed_share"], 3)),
                "source": "footballguys.com projected roster percentages",
                "slate_note": ("The source projects the Sunday 13:00-16:25 "
                               "games. A build on a wider slate will show a low "
                               "covered_share, and the teams it skips will look "
                               "contrarian when they are only unmeasured.")}
    except Exception:
        return {"available": False, "reason": "could not be summarised"}


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


# --- the player pool: lock, exclude, exposure -----------------------------
#
# Three things the global exposure cap cannot do: require a player, name a
# player, or survive the request that asked for it. They are one dial -- 0% is
# an exclusion, 100% is a lock -- so they share a table and a parser.

def _injury_flag(value) -> Optional[str]:
    """The slate's injury cell as a flag, or None.

    `str(value or "")` is the trap: an empty cell comes back from read_csv as a
    float NaN, and NaN is TRUTHY -- so that expression yields the string "NAN"
    and every healthy player on the slate was reported as carrying a flag. The
    fixtures used empty strings, where real exports leave the cell blank.
    """
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip().upper()
    return text or None


def _rule_kind(low, high) -> str:
    """The word for a rule, for a table a person reads."""
    if high is not None and high <= 0:
        return "exclude"
    if low is not None and low >= 100:
        return "lock"
    if low is not None and high is not None and low == high:
        return "target"
    if low is not None and high is not None:
        return "band"
    return "floor" if low is not None else "cap"


def _rule_payload(row: SlatePlayerRule) -> dict:
    return {"min_exposure": row.min_exposure, "max_exposure": row.max_exposure,
            "kind": _rule_kind(row.min_exposure, row.max_exposure),
            "note": row.note,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None}


def _saved_rules(db, slate_id: Optional[int]) -> dict:
    """``{nickname: {"min": x, "max": y}}`` stored against one slate."""
    if slate_id is None:
        return {}
    out = {}
    for row in db.query(SlatePlayerRule).filter(
            SlatePlayerRule.slate_id == slate_id).all():
        if row.min_exposure is None and row.max_exposure is None:
            continue
        out[row.nickname] = {"min": row.min_exposure, "max": row.max_exposure}
    return out


def _parse_pool_spec(spec: str) -> dict:
    """``"Walker:lock, Allgeier:exclude, Hubbard:40, Gibbs:20-60"`` as rules.

    A compact form because the whole request is multipart form fields and a
    nested JSON body cannot ride alongside an uploaded file. Names hold no
    colons and no commas, so splitting on them is safe; the value is whatever
    follows the LAST colon.
    """
    rules = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise HTTPException(
                status_code=400,
                detail=(f"{part!r} is not a pool rule. Write them as "
                        "'Name:lock', 'Name:exclude', 'Name:40' or "
                        "'Name:20-60', separated by commas."))
        name, _, value = part.rpartition(":")
        name, value = name.strip(), value.strip().lower()
        if not name:
            raise HTTPException(status_code=400,
                                detail=f"{part!r} names no player.")
        if value in ("lock", "exclude"):
            rules[name] = value
            continue
        if "-" in value:
            low, _, high = value.partition("-")
            try:
                rules[name] = {"min": float(low), "max": float(high)}
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"{value!r} for {name} is not a percentage range.")
            continue
        try:
            rules[name] = float(value)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(f"{value!r} for {name} is not a percentage, 'lock' or "
                        "'exclude'."))
    return rules


def _entered_lineups(db, season: int, week: int, *, slate_id=None):
    """Every lineup already recorded for the week, as sets of nicknames.

    This is what stops six builds being one build six times. 2026 week 9
    entered 45 lineups and held 31 distinct, because the builder is exact and
    deterministic: re-run with the same settings it returns the same nine men.
    """
    builds = db.query(LineupBuild).filter(LineupBuild.season == season,
                                          LineupBuild.week == week)
    if slate_id is not None:
        builds = builds.filter(LineupBuild.slate_id == slate_id)
    build_ids = [b.id for b in builds.all()]
    if not build_ids:
        return []
    entries = db.query(LineupEntry).filter(
        LineupEntry.build_id.in_(build_ids)).all()
    if not entries:
        return []
    by_lineup: dict = {}
    for player in db.query(LineupPlayer).filter(
            LineupPlayer.lineup_id.in_([e.id for e in entries])).all():
        by_lineup.setdefault(player.lineup_id, set()).add(player.player_name)
    return [names for names in by_lineup.values() if names]


def _player_history(db, season: int, week: int) -> pd.DataFrame:
    """Each player's own weekly FanDuel points before `week`, for reading.

    Description, not prediction: measured on 2026 weeks 5-9, a band of a
    player's lowest and highest covers 71.0% of his next result against the
    model band's 81.7%. It is here because it says in one line what the player
    has actually done, which neither a projection nor a band shows.
    """
    from .utils import fanduel_points

    rows = db.query(PlayerStat).filter(
        PlayerStat.season == season, PlayerStat.week < week).all()
    if not rows:
        return pd.DataFrame()
    records = []
    for row in rows:
        if (row.season_type or "REG") != "REG":
            continue
        name = row.player_display_name or row.player_name
        if not name:
            continue
        records.append({
            "player_name": name, "week": row.week,
            "fantasy_points": fanduel_points(
                {c.name: getattr(row, c.name) for c in row.__table__.columns}),
        })
    return pd.DataFrame(records)


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
                  max_usage_percentage: int, excludes, num_requested: int,
                  max_per_team=None, stack_qb=None, rb_with_qb=None,
                  ownership_weight=None, projection_computed_at=None,
                  player_rules=None, pool_exposure=False, deduped_against=0,
                  duplicates_skipped=0):
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
            num_requested=int(num_requested),
            max_per_team=max_per_team, stack_qb=stack_qb, rb_with_qb=rb_with_qb,
            ownership_weight=ownership_weight,
            player_rules=json.dumps(player_rules, sort_keys=True) if player_rules
            else None,
            pool_exposure=int(bool(pool_exposure)))
        # The vintage is recorded but is NOT part of what makes two builds the
        # same request: rebuilding after the board moves supersedes the older
        # lineups, which is the whole point of rebuilding.
        #
        # A build that AVOIDED lineups already entered is a different matter. It
        # is additive by construction -- it returned the next-best set precisely
        # because the first set exists -- so superseding would delete the better
        # lineups it had just stepped around, leaving only the ones it settled
        # for. Two clicks of Build then make the week worse than one.
        if not deduped_against:
            _drop_superseded(db, settings)
        build = LineupBuild(**settings, num_built=len(lineups),
                            built_at=datetime.utcnow(),
                            deduped_against=int(deduped_against or 0),
                            duplicates_skipped=int(duplicates_skipped or 0),
                            projection_computed_at=projection_computed_at)
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
                    fanduel_id=_text_or_none(row.get("Id")),
                    player_name=_text_or_none(row.get("Nickname")),
                    team=_text_or_none(row.get("Team")),
                    salary=int(row["Salary"]) if pd.notna(row.get("Salary")) else None,
                    projected_points=_float_or_none(row.get("lineup_points")),
                    from_model=int(pd.notna(row.get("fanduel_fantasy_points"))
                                   or pd.notna(row.get("defense_points"))),
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
    # Optional, not 5: with a contest preset the lineup count is part of the
    # preset, and a Form default would silently win over it every time.
    num_lineups: Optional[int] = Form(None),
    objective: str = Form("mean"),
    salary_cap: int = Form(60000),
    max_usage_percentage: int = Form(50),
    exclude: str = Form(""),
    lock: str = Form("", description="Nicknames in EVERY lineup, comma-separated."),
    pool: str = Form(
        "", description="Per-player rules: 'Walker:lock, Allgeier:exclude, "
                        "Hubbard:40, Gibbs:20-60'. A bare number is a target "
                        "share of the lineups, not a cap."),
    use_saved_rules: bool = Form(
        True, description="Apply the pool saved against this slate "
                          "(GET/POST /optimizer/pool). Request rules win."),
    dedupe: bool = Form(
        True, description="Never return a lineup already entered this week."),
    pool_exposure: bool = Form(
        False, description="Count lineups already entered toward the exposure "
                           "percentages, making them a share of the week."),
    slate: str = Form("all"),
    one_kicker: bool = Form(True),
    max_per_team: int = Form(4),
    stack_qb: int = Form(0),
    rb_with_qb: bool = Form(True),
    ownership_weight: float = Form(0.0),
    contest_type: Optional[str] = Form(None),
    aggression: int = Form(50),
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

    It cannot, however, require a player, name a player, or outlive the request
    — so `pool`, `lock` and the saved pool do that. They are one dial: 0% is an
    exclusion, 100% is a lock, and `Hubbard:40` means he appears in about four
    lineups in ten, enforced as a floor as well as a ceiling. A rule saved
    against the slate (POST /optimizer/pool) applies to every later build
    without being resent, and anything in this request wins over it.

    `dedupe` is on because the builder is exact and deterministic: re-run with
    the same settings it returns the same nine men. 2026 week 9 entered 45
    lineups across six builds and held 31 distinct ones. With it on, the lineups
    already recorded for the week are never returned again — the build steps
    down to the next-best set instead — and such a build never supersedes the
    earlier one, since it only exists because that one does.

    `pool_exposure` is off by default because it changes what a percentage
    means. On, the percentages count the whole week rather than this build,
    which is the only thing that can hold a player down across builds: in week
    9 six builds each honoured a 46-49% cap and pooled to 55.6% on seven
    players, who then returned 66.54 points across seven of nine slots.

    The remaining three say how the nine picks may relate to each other, which
    is the one thing a board of per-player projections cannot express — it is
    made one player at a time and has no idea that nine of them can be one bet
    on one offence. In 2026 week 7 the board put Carolina in 16% of every slot
    it filled; Carolina lost 43-13, and lineups holding two Panthers averaged
    75.1 against 101.6 for lineups holding none.

      `max_per_team` — at most this many of the nine from one team, the defense
        counted as one of its own. Defaults to 4, which is FanDuel's own limit:
        before this the builder could return a lineup the site would refuse.
      `stack_qb` — require this many WR/TE from the quarterback's team.
      `rb_with_qb` — false forbids a running back from the quarterback's team.
      `contest_type` — "cash" or "gpp". Given one, the build settings come
      from `nfl_projections.optimizer.contest_settings` and `aggression`
      (0-100) moves them together; anything sent explicitly still wins. Cash
      aggression is "how much do I trust the board": at 0 it hedges our own
      error with more lineups and a looser exposure cap, at 100 it plays the
      board's single best lineup. Measured both ways -- 2026 week 6 rewarded
      committing (105.6 to 123.5 across the dial) and week 8 punished it
      (106.9 to 88.9) -- so the dial spans the question rather than answering
      it. GPP aggression buys a stack first, then exposure spread, then a
      charge against the field, in order of how well each is evidenced.

      `ownership_weight` — points charged per percent of projected ownership,
      which tilts the build toward players the field has not found. 0 ignores
      the field. It costs expected points to buy that tilt: on the 2026 week 6
      main slate a charge of 0.10 took the best five from 125.1 actual to
      109.3, so it is a tournament trade, not an improvement. Players with no
      published ownership are charged nothing rather than treated as free
      leverage — see `_attach_ownership`.
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
    if num_lineups is not None and (num_lineups < 1 or num_lineups > 50):
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
    merged = _attach_defense(merged, season, week)
    merged = _attach_ownership(merged, season, week)

    excludes = [n.strip() for n in exclude.split(",") if n.strip()]
    extra = {}
    if showdown and one_kicker:
        # Showdown lets a lineup take every kicker and defense in the game, and
        # FanDuel's season-average FPPG — all these positions have, since the
        # model does not project them — makes them look cheap and dependable.
        # The first real slate duly produced a lineup with both kickers in it.
        extra["position_limits"] = {"K": 1, "D": 1}
    if not showdown:
        extra.update(max_per_team=max_per_team or None, stack_qb=stack_qb,
                     rb_with_qb=rb_with_qb)
        if contest_type:
            # The preset supplies the whole shape; whatever the caller sent
            # explicitly still wins, which is why it is applied underneath.
            try:
                preset = opt.contest_settings(contest_type, aggression,
                                              num_lineups=num_lineups)
            except (ValueError, AttributeError) as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            objective = preset.pop("objective", objective)
            num_lineups = preset.pop("num_lineups", num_lineups)
            max_usage_percentage = preset.pop("max_usage_percentage",
                                              max_usage_percentage)
            extra.update({k: v for k, v in preset.items()
                          if k not in ("price_step",)})
            # The preset asks for an ownership charge at high aggression, and
            # the page is only published from the Friday before a slate. A
            # Tuesday build must come back without the tilt, not fail.
            if extra.get("ownership_weight") and "ownership" not in merged.columns:
                logger.info("no ownership for week %s yet; %s build runs without "
                            "the fade", week, contest_type)
                extra.pop("ownership_weight", None)
                extra.pop("ownership_exempt_rank", None)
        if ownership_weight and "ownership" in merged.columns:
            extra["ownership_weight"] = float(ownership_weight)
        elif ownership_weight:
            logger.warning("ownership_weight=%s ignored: no ownership for this "
                           "week yet", ownership_weight)
    if num_lineups is None:
        num_lineups = 5

    # The pool: what is saved against this slate, then whatever this request
    # said, which wins. `lock` and `exclude` are the same dial at its ends.
    rules = {}
    if use_saved_rules and slate_row is not None:
        rules.update(_saved_rules(db, slate_row.id))
    rules.update(_parse_pool_spec(pool))
    for nickname in [n.strip() for n in lock.split(",") if n.strip()]:
        rules[nickname] = "lock"
    for nickname in excludes:
        rules.setdefault(nickname, "exclude")

    seen = _entered_lineups(db, season, week) if dedupe else []
    report: dict = {}
    pool_kwargs = {}
    if not showdown:
        # optimize_showdown takes neither, and sending them would be a
        # TypeError rather than a quiet no-op on an older package in the image.
        pool_kwargs = {"player_rules": rules or None, "seen_lineups": seen or None,
                       "pool_exposure": bool(pool_exposure), "report": report}
    elif rules or seen:
        logger.info("pool rules and dedupe are classic-only; ignored for a "
                    "showdown build")

    build = opt.optimize_showdown if showdown else opt.optimize_lineups
    try:
        lineups = build(
            merged, num_lineups=num_lineups, salary_cap=salary_cap,
            exclude_players=excludes or None,
            max_usage_percentage=max_usage_percentage, objective=objective,
            **extra, **pool_kwargs)
    except TypeError as exc:
        # A pool rule the package in this image cannot take is worth saying
        # plainly; silently building without it would hand back lineups that
        # ignore a lock.
        if pool_kwargs and ("player_rules" in str(exc) or "seen_lineups" in str(exc)
                            or "pool_exposure" in str(exc) or "report" in str(exc)):
            logger.exception("nfl_projections in this image has no player pool")
            raise HTTPException(
                status_code=503,
                detail=("This build of the service cannot apply a player pool "
                        "or dedupe lineups; its nfl_projections predates them. "
                        "Rebuild the image."))
        logger.exception("Lineup optimisation failed")
        raise HTTPException(status_code=422, detail=f"Could not build lineups: {exc}")
    except ValueError as exc:
        # The pool's own refusals -- a misspelled lock, a locked player the
        # slate says is out, locks over the cap -- are the user's to fix and
        # say exactly what is wrong.
        logger.info("player pool refused the build: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Lineup optimisation failed")
        raise HTTPException(status_code=422, detail=f"Could not build lineups: {exc}")
    if not lineups and showdown:
        raise HTTPException(
            status_code=422,
            detail=(f"No Showdown lineup fits. Five players at MVP pricing have to come in "
                    f"under ${salary_cap:,}, and {len(merged)} players are in this pool "
                    f"after exclusions."))
    if not lineups and report.get("duplicates_skipped"):
        # Dedupe emptied it, which is a different thing from nothing fitting:
        # every lineup this slate and these settings can produce is already
        # entered. "No lineup fits under the cap" would send the reader to the
        # cap, which is innocent.
        raise HTTPException(
            status_code=422,
            detail=(f"Every lineup this build found is already entered for "
                    f"{season} week {week} — {report['duplicates_skipped']} of "
                    f"them, against the {len(seen)} lineups already recorded. "
                    "This slate has no further distinct lineup at these "
                    "settings: change the objective, the cap or the pool to "
                    "open different ones up, or send `dedupe=false` to allow a "
                    "repeat."))
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
            excludes=excludes, num_requested=num_lineups,
            max_per_team=None if showdown else (int(max_per_team) or None),
            stack_qb=None if showdown else int(stack_qb),
            rb_with_qb=None if showdown else int(bool(rb_with_qb)),
            ownership_weight=None if showdown else float(ownership_weight or 0.0),
            projection_computed_at=projections.attrs.get("computed_at"),
            player_rules=report.get("rules_applied") or None,
            pool_exposure=bool(pool_exposure) and not showdown,
            deduped_against=len(seen),
            duplicates_skipped=report.get("duplicates_skipped", 0))

    out = []
    for i, lineup in enumerate(lineups, start=1):
        players = [{
            "player_name": r.get("Nickname"),
            "roster_position": r.get("Roster Position"),
            "team": r.get("Team") if "Team" in r else r.get("team"),
            "salary": int(r["Salary"]) if pd.notna(r.get("Salary")) else None,
            "projected": round(float(r["lineup_points"]), 2) if pd.notna(
                r.get("lineup_points")) else None,
            # A projected defense is from a model too, even though it comes
            # from a different one than the players do.
            "from_model": bool(pd.notna(r.get("fanduel_fantasy_points"))
                               or pd.notna(r.get("defense_points"))),
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
        "ownership_weight": float(ownership_weight or 0.0),
        "ownership": _ownership_summary(merged),
        # Players the board rates that the field has not rostered at all. Not
        # excluded — checked. Almost always the field knows he is inactive (on
        # 2026 week 6, all five such players were), but in week 7 Kirk Cousins
        # was unlisted at a projected 18.4 because the field expected Penix to
        # start, and he went for 20.4.
        "unlisted_but_projected": _unlisted_note(merged),
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
        # What the pool asked for and what it got. A lock that silently did
        # nothing is the failure mode worth designing against, so the rules in
        # force are echoed rather than assumed.
        "player_rules": report.get("rules_applied") or {},
        "pool_exposure": bool(pool_exposure) and not showdown,
        "deduped_against": len(seen),
        "duplicates_skipped": report.get("duplicates_skipped", 0),
        "pool_warnings": report.get("warnings") or [],
        "dedupe_note": (
            None if not dedupe else
            (f"{len(seen)} lineup(s) already recorded for this week were held "
             "out, so these are additional bets rather than the same roster "
             "re-solved."
             if seen else
             "Nothing was entered for this week yet, so there was nothing to "
             "dedupe against.")),
        "data": out,
    }


@router.get("/builds/{build_id}/upload.csv")
def build_upload_csv(
    build_id: int,
    contest_id: Optional[str] = Query(
        None, description="Rewrite the contest half of every FanDuel id. A "
                          "FanDuel id is '<contest>-<player>', so lineups "
                          "priced from one contest's player list are rejected "
                          "by another. The contest id is the number in the "
                          "file name of the list or template you downloaded."),
    include_names: bool = Query(
        True, description="Write 'id:Name' rather than the bare id. FanDuel "
                          "accepts either; names make the file readable."),
    db: Session = Depends(get_db),
):
    """A build as a CSV FanDuel's bulk entry will accept.

    Nine columns in FanDuel's own order, one row per lineup, up to their limit
    of 250. The solver does not record which flexible player it treated as the
    FLEX -- FanDuel scores it the same either way -- so the quotas are filled
    first and the surplus back, receiver or tight end takes the FLEX.
    """
    from nfl_projections import optimizer as opt

    build = db.query(LineupBuild).filter(LineupBuild.id == build_id).one_or_none()
    if build is None:
        raise HTTPException(status_code=404, detail=f"No build {build_id}.")
    if build.contest == "showdown":
        raise HTTPException(
            status_code=400,
            detail="Showdown lineups are five players with an MVP, not the nine "
                   "classic columns; FanDuel takes a different upload for them.")
    entries = sorted(
        db.query(LineupEntry).filter(LineupEntry.build_id == build_id).all(),
        key=lambda e: e.lineup_index or 0)
    if not entries:
        raise HTTPException(status_code=404,
                            detail=f"Build {build_id} has no lineups stored.")

    players = db.query(LineupPlayer).filter(
        LineupPlayer.lineup_id.in_([e.id for e in entries])).all()
    by_lineup: dict = {}
    for player in players:
        by_lineup.setdefault(player.lineup_id, []).append(player)

    lineups, missing = [], set()
    for entry in entries:
        rows = by_lineup.get(entry.id, [])
        for player in rows:
            if not player.fanduel_id:
                missing.add(player.player_name)
        lineups.append(pd.DataFrame([{
            "Id": player.fanduel_id,
            "Nickname": player.player_name,
            "Position": player.position,
            "Roster Position": player.slot,
            "Salary": player.salary,
            "lineup_points": player.projected_points,
        } for player in rows]))
    if missing:
        # Builds made before the id was recorded cannot be exported, and saying
        # so beats handing back a file FanDuel rejects without explanation.
        raise HTTPException(
            status_code=409,
            detail=(f"Build {build_id} predates FanDuel ids being stored "
                    f"({len(missing)} players have none). Rebuild it and the "
                    "download will work."))

    try:
        text = opt.write_fanduel_upload(lineups, contest_id=contest_id,
                                        include_names=include_names)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    name = f"fanduel-week{build.week}-build{build_id}-{len(lineups)}-lineups.csv"
    return Response(content=text, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/pool")
def player_pool(
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(None),
    slate_id: Optional[int] = Query(None, description="A specific saved slate."),
    position: Optional[str] = Query(
        None, description="Comma-separated, e.g. 'RB,WR'. Omit for all."),
    search: Optional[str] = Query(None, description="Substring of the name."),
    only_rules: bool = Query(False, description="Only players with a rule set."),
    history: bool = Query(True, description="Attach each player's own weekly range."),
    db: Session = Depends(get_db),
):
    """The uploaded slate as a table: every player, what we think, what you said.

    This is the thing the optimizer's settings could not express. The global
    exposure cap applies to everyone equally and cannot require anybody, so
    knowing you want a particular back left no way to say it. Here each row
    carries its own rule — locked in, faded out, or held to a share of the
    lineups — and the rule is stored against the slate, so every later build
    obeys it without being told again.

    `entered` is how much of the week that player already has, pooled across
    every recorded build, because the decision being made here is usually a
    reaction to it. `history` is what he has actually scored, which is
    description and not prediction: as a band, a player's own low and high cover
    71.0% of his next result against the model band's 81.7% (2026 weeks 5-9,
    n=617), and both ends drift outward all season as the sample grows rather
    than as the player changes. It is here because it reads well, not because it
    ranks.
    """
    from nfl_projections import optimizer as opt

    season = season or get_current_nfl_season()
    if week is None:
        weeks = [w for (w,) in db.query(FanduelSlate.week).filter(
            FanduelSlate.season == season).distinct().all() if w is not None]
        if not weeks:
            raise HTTPException(
                status_code=404,
                detail=f"No slate CSV saved for {season}. Upload a contest "
                       "player list to /optimizer/lineups first.")
        week = max(weeks)

    slate_row = _stored_slate(db, season, week, slate_id)
    if slate_row is None:
        raise HTTPException(
            status_code=404,
            detail=(f"No slate CSV saved for {season} week {week}. Salaries are "
                    "the one thing here that cannot be derived, so the pool "
                    "needs the uploaded contest player list."))
    salaries = _csv_frame(slate_row.content or "", "The saved slate")
    projections = _projection_frame(db, season, week)
    merged = opt.merge_fanduel_salaries(salaries, projections) \
        if not projections.empty else salaries.assign(fanduel_fantasy_points=np.nan)
    merged = _attach_defense(merged, season, week)
    merged = _attach_ownership(merged, season, week)

    ranges = {}
    if history:
        frame = _player_history(db, season, week)
        if not frame.empty:
            points = pd.to_numeric(frame["fantasy_points"], errors="coerce")
            keyed = pd.DataFrame({
                "_key": frame["player_name"].map(opt.normalize_name),
                "_p": points}).dropna()
            stats = keyed.groupby("_key")["_p"].agg(["size", "min", "max", "median"])
            ranges = {k: row for k, row in stats.iterrows()}

    rules = {r.nickname: r for r in db.query(SlatePlayerRule).filter(
        SlatePlayerRule.slate_id == slate_row.id).all()}
    entered = Counter(n for names in _entered_lineups(db, season, week)
                      for n in names)
    lineups_entered = len(_entered_lineups(db, season, week))

    wanted = None
    if position:
        wanted = {p.strip().upper() for p in position.split(",") if p.strip()}
    needle = (search or "").strip().lower()

    rows = []
    for _, r in merged.iterrows():
        nickname = str(r.get("Nickname") or "")
        if not nickname:
            continue
        pos = str(r.get("Position") or "").upper()
        if wanted and pos not in wanted:
            continue
        if needle and needle not in nickname.lower():
            continue
        rule = rules.get(nickname)
        if only_rules and rule is None:
            continue
        salary = pd.to_numeric(r.get("Salary"), errors="coerce")
        proj = pd.to_numeric(r.get("fanduel_fantasy_points"), errors="coerce")
        fppg = pd.to_numeric(r.get("FPPG"), errors="coerce")
        used = entered.get(nickname, 0)
        row = {
            "nickname": nickname,
            "position": pos,
            "roster_position": r.get("Roster Position"),
            "team": r.get("Team"),
            "opponent": r.get("Opponent"),
            "game": r.get("Game"),
            "salary": int(salary) if pd.notna(salary) else None,
            "fppg": round(float(fppg), 2) if pd.notna(fppg) else None,
            "injury_flag": _injury_flag(r.get("Injury Indicator")),
            "projected_points": round(float(proj), 2) if pd.notna(proj) else None,
            # DEF is not projected by the model at all; its slot runs on
            # FanDuel's own FPPG, which is why `from_model` is on the row.
            "from_model": bool(pd.notna(proj)),
            "rosterable": (_injury_flag(r.get("Injury Indicator"))
                           not in opt.EXCLUDED_INJURY_STATUSES),
            "entered_lineups": used,
            "entered_exposure": (round(100.0 * used / lineups_entered, 1)
                                 if lineups_entered else None),
            "rule": _rule_payload(rule) if rule is not None else None,
        }
        for key, column in (("floor", "floor"), ("ceiling", "ceiling"),
                            ("median", "projection_median"),
                            ("ownership", "ownership")):
            value = pd.to_numeric(r.get(column), errors="coerce")
            row[key] = round(float(value), 2) if pd.notna(value) else None
        if pd.notna(proj) and pd.notna(salary) and salary:
            row["value"] = round(float(proj) / (float(salary) / 1000.0), 3)
        else:
            row["value"] = None
        stats = ranges.get(opt.normalize_name(nickname))
        row["history"] = (None if stats is None else {
            "games": int(stats["size"]), "low": round(float(stats["min"]), 1),
            "high": round(float(stats["max"]), 1),
            "median": round(float(stats["median"]), 1)})
        rows.append(row)

    rows.sort(key=lambda r: (r["position"], -(r["value"] or -1)))
    return {
        "status": "success",
        "season": season,
        "week": week,
        "slate": _slate_summary(slate_row),
        "count": len(rows),
        "players_in_file": len(salaries),
        "lineups_entered": lineups_entered,
        "rules_set": len(rules),
        "positions": sorted({r["position"] for r in rows}),
        "note": ("A rule is a share of the lineups: 0 is an exclusion, 100 is a "
                 "lock, 40 is about four in ten and binds as a floor as well as "
                 "a ceiling. `entered_exposure` is what the week already holds, "
                 "pooled across every build. `history` is what he has scored, "
                 "for reading only -- it never enters a projection or a rank."),
        "data": rows,
    }


@router.post("/pool")
def set_player_rules(
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(None),
    slate_id: Optional[int] = Query(None),
    rules: List[dict] = Body(
        ..., description="One object per player: {nickname, min_exposure, "
                         "max_exposure, note}, or {nickname, rule: 'lock'}. "
                         "Nulls for both clear the rule."),
    db: Session = Depends(get_db),
):
    """Set, change or clear the pool rules for one slate.

    Stored against the slate rather than the week, because a re-upload replaces
    the slate: prices move midweek and a rule set against Wednesday's file
    should not silently attach itself to Saturday's different contest. Every
    nickname is checked against the stored CSV first — FanDuel's spelling is the
    one that counts, and a rule naming nobody would be a lock that quietly does
    nothing.
    """
    from nfl_projections import optimizer as opt

    season = season or get_current_nfl_season()
    if week is None:
        weeks = [w for (w,) in db.query(FanduelSlate.week).filter(
            FanduelSlate.season == season).distinct().all() if w is not None]
        if not weeks:
            raise HTTPException(status_code=404,
                                detail=f"No slate CSV saved for {season}.")
        week = max(weeks)
    slate_row = _stored_slate(db, season, week, slate_id)
    if slate_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No slate CSV saved for {season} week {week}.")

    salaries = _csv_frame(slate_row.content or "", "The saved slate")
    on_slate = {str(n): _injury_flag(f) for n, f in zip(
        salaries["Nickname"],
        salaries.get("Injury Indicator", pd.Series("", index=salaries.index)))}

    parsed, cleared = {}, []
    for item in rules:
        nickname = str(item.get("nickname") or "").strip()
        if not nickname:
            raise HTTPException(status_code=400,
                                detail="Every rule needs a `nickname`.")
        if nickname not in on_slate:
            raise HTTPException(
                status_code=422,
                detail=(f"{nickname!r} is not on slate {slate_row.id}. FanDuel's "
                        "spelling is the one that counts -- check it against "
                        "GET /optimizer/pool."))
        word = str(item.get("rule") or "").strip().lower()
        if word in ("clear", "none", "free", ""):
            low, high = item.get("min_exposure"), item.get("max_exposure")
        elif word == "lock":
            low, high = 100.0, 100.0
        elif word == "exclude":
            low, high = 0.0, 0.0
        else:
            try:
                low = high = float(word)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=(f"rule {word!r} for {nickname} is not a percentage, "
                            "'lock', 'exclude' or 'clear'."))
        for value, label in ((low, "min_exposure"), (high, "max_exposure")):
            if value is not None and not 0 <= float(value) <= 100:
                raise HTTPException(
                    status_code=400,
                    detail=f"{label} for {nickname} is {value}; it is a "
                           "percentage, 0 to 100.")
        if low is not None and high is not None and float(low) > float(high):
            raise HTTPException(
                status_code=400,
                detail=(f"{nickname} would have a {low}% floor above its "
                        f"{high}% ceiling."))
        if (low is not None and float(low) > 0
                and on_slate[nickname] in opt.EXCLUDED_INJURY_STATUSES):
            raise HTTPException(
                status_code=422,
                detail=(f"{nickname} is marked {on_slate[nickname]} on this "
                        "slate, so he cannot be required. FanDuel will not pay "
                        "him."))
        if low is None and high is None:
            cleared.append(nickname)
        parsed[nickname] = (low, high, item.get("note"))

    for nickname, (low, high, note) in parsed.items():
        row = db.query(SlatePlayerRule).filter(
            SlatePlayerRule.slate_id == slate_row.id,
            SlatePlayerRule.nickname == nickname).first()
        if low is None and high is None:
            if row is not None:
                db.delete(row)
            continue
        if row is None:
            row = SlatePlayerRule(slate_id=slate_row.id, season=season,
                                  week=week, nickname=nickname)
            db.add(row)
        row.min_exposure = None if low is None else float(low)
        row.max_exposure = None if high is None else float(high)
        row.note = note
        row.updated_at = datetime.utcnow()
    db.commit()

    kept = db.query(SlatePlayerRule).filter(
        SlatePlayerRule.slate_id == slate_row.id).all()
    return {
        "status": "success",
        "season": season,
        "week": week,
        "slate_id": slate_row.id,
        "set": len([n for n in parsed if n not in cleared]),
        "cleared": cleared,
        "rules": {r.nickname: _rule_payload(r) for r in kept},
        "note": ("Every build for this slate now applies these unless it sends "
                 "`use_saved_rules=false`, and a rule in the request wins."),
    }


@router.delete("/pool")
def clear_player_rules(
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(None),
    slate_id: Optional[int] = Query(None),
    nickname: Optional[str] = Query(
        None, description="One player. Omit to clear the whole pool."),
    db: Session = Depends(get_db),
):
    """Clear one player's rule, or the whole pool for a slate."""
    season = season or get_current_nfl_season()
    if week is None:
        weeks = [w for (w,) in db.query(FanduelSlate.week).filter(
            FanduelSlate.season == season).distinct().all() if w is not None]
        week = max(weeks) if weeks else None
    slate_row = _stored_slate(db, season, week, slate_id) if week else None
    if slate_row is None:
        raise HTTPException(status_code=404,
                            detail=f"No slate CSV saved for {season} week {week}.")
    query = db.query(SlatePlayerRule).filter(
        SlatePlayerRule.slate_id == slate_row.id)
    if nickname:
        query = query.filter(SlatePlayerRule.nickname == nickname)
    rows = query.all()
    for row in rows:
        db.delete(row)
    db.commit()
    return {"status": "success", "season": season, "week": week,
            "slate_id": slate_row.id, "cleared": len(rows),
            "nickname": nickname}


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


def _round(value, places: int = 2):
    """Points as something to read. The columns are stored unrounded, and a
    lineup slot rendering as 24.299999999999997 is the result."""
    return None if value is None else round(float(value), places)


def _perf_bucket(**fields) -> dict:
    return {**fields, "builds": 0, "lineups": 0, "complete": 0,
            "projected": [], "actual": [], "best_actual": None, "worst_actual": None,
            "fppg_points": [], "best_points": [], "beat_fppg": 0, "compared": 0,
            "weeks": set(), "objectives": set()}


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
    if complete:
        # Week coverage per bucket, which is what says whether an objective's
        # average is a season's evidence or one slate's. A build with nothing
        # scored yet does not buy its objective a week.
        bucket["weeks"].add(build.week)
        bucket["objectives"].add(build.objective or "unknown")
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
    # A lineup's stored total is the sum of whatever column its objective
    # ranked on, so only a mean build's total is a forecast of its score. A
    # ceiling build's is a sum of nine 90th percentiles -- week 9 of 2026 read
    # "projected 217, actual 87, error 130", which was the word "ceiling" and
    # not a miss. `projected` is still reported, labelled with its scale; the
    # error is only drawn where subtracting the two means something.
    objectives = bucket["objectives"]
    scale = (next(iter(objectives)) if len(objectives) == 1
             else "mixed" if objectives else None)
    return {
        "builds": bucket["builds"],
        "lineups": bucket["lineups"],
        "scored_lineups": bucket["complete"],
        "weeks_scored": len(bucket["weeks"]),
        "projected": projected,
        "projected_scale": scale,
        "actual": actual,
        "error": (round(projected - actual, 2)
                  if scale == "mean" and projected is not None and actual is not None
                  else None),
        "best_lineup_actual": bucket["best_actual"],
        "worst_lineup_actual": bucket["worst_actual"],
        "fppg_baseline": _mean(bucket["fppg_points"]),
        "best_possible": _mean(bucket["best_points"]),
        "beat_fppg": bucket["beat_fppg"],
        "compared_to_fppg": bucket["compared"],
        "beat_fppg_rate": (round(bucket["beat_fppg"] / bucket["compared"], 3)
                           if bucket["compared"] else None),
    }


def _complete_ranked(rows):
    """A build's fully scored lineups, best-projected first.

    Inside one build every total is on the same scale, so the stored total is
    a fair ranking there whatever the objective was.
    """
    complete = [e for e in rows
                if e.actual_points is not None and not (e.unscored_players or 0)]
    return sorted(complete, key=lambda e: (-(e.projected_points or 0.0),
                                           e.lineup_index or 0))


def _matched_depth(builds, entries) -> dict:
    """Objectives compared only where they were built to the same depth.

    A 25-lineup build reaches further down the pool than a 5-lineup one --
    lineup 25 is worse than lineup 1 by construction -- so the pooled
    `by_objective` average measures how many lineups each objective was asked
    for as much as which objective asked. Here a cell is one (week, contest,
    lineups requested); inside it every build is cut to its top k, k being the
    fewest scored lineups any build in the cell has, and an objective built
    twice counts once. Only cells holding two or more objectives appear.
    """
    cells: dict = {}
    for build in builds:
        ranked = _complete_ranked(entries.get(build.id, []))
        if not ranked:
            continue
        key = (build.week, build.contest or "unknown",
               build.num_requested or build.num_built or len(ranked))
        cells.setdefault(key, []).append((build.objective or "unknown", ranked))

    out_cells, pairs = [], {}
    for (week, contest, depth), members in sorted(
            cells.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1], kv[0][2])):
        if len({objective for objective, _ in members}) < 2:
            continue
        k = min(len(ranked) for _, ranked in members)
        per_objective: dict = {}
        for objective, ranked in members:
            per_objective.setdefault(objective, []).append(
                sum(e.actual_points for e in ranked[:k]) / k)
        scores = {o: round(sum(v) / len(v), 2) for o, v in per_objective.items()}
        out_cells.append({"week": week, "contest": contest, "requested": depth,
                          "top_k": k, "actual": scores})
        names = sorted(scores)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                pairs.setdefault((contest, first, second), []).append(
                    round(scores[first] - scores[second], 2))

    return {
        "cells": out_cells,
        "pairs": [{
            "contest": contest, "first": first, "second": second,
            "cells": len(gaps), "gaps": gaps,
            "mean_gap": round(sum(gaps) / len(gaps), 2),
            "first_ahead": sum(1 for g in gaps if g > 0),
        } for (contest, first, second), gaps in sorted(pairs.items())],
        "note": ("Each cell is one week at one requested depth, every build cut to "
                 "its top_k lineups. A gap is first minus second. The spread between "
                 "two builds of the SAME objective has run larger than these gaps, "
                 "so a pair needs many cells with one sign before it means anything."),
    }


DEPTH_HEAD = 5       # the lineups a small entry would actually play
DEPTH_TAIL_FROM = 11  # where week 6 of 2026 crossed under the FPPG baseline


def _depth_record(builds, entries) -> dict:
    """What going further down a build has cost, week by week.

    The one lineup-level regularity the record has is that a build's own
    projected rank orders its lineups. This is the readback of that: the mean
    actual of each build's top five against its lineups ranked eleventh and
    below, classic builds deep enough to have both, averaged inside a week so
    a week is one observation -- its builds share a slate, a board and most of
    their players.
    """
    by_week: dict = {}
    for build in builds:
        if (build.contest or "classic") != "classic":
            continue
        ranked = _complete_ranked(entries.get(build.id, []))
        if len(ranked) < DEPTH_TAIL_FROM:
            continue
        head = ranked[:DEPTH_HEAD]
        tail = ranked[DEPTH_TAIL_FROM - 1:]
        by_week.setdefault(build.week, []).append((
            sum(e.actual_points for e in head) / len(head),
            sum(e.actual_points for e in tail) / len(tail)))

    weeks = []
    for week in sorted(by_week, key=lambda w: w or 0):
        rows = by_week[week]
        head = sum(h for h, _ in rows) / len(rows)
        tail = sum(t for _, t in rows) / len(rows)
        weeks.append({"week": week, "builds": len(rows), "top_5": round(head, 2),
                      "rank_11_plus": round(tail, 2), "gap": round(head - tail, 2)})
    gaps = [w["gap"] for w in weeks]
    return {
        "weeks": weeks,
        "weeks_counted": len(weeks),
        "mean_gap": round(sum(gaps) / len(gaps), 2) if gaps else None,
        "weeks_top_ahead": sum(1 for g in gaps if g > 0),
        "note": ("top_5 minus rank_11_plus, per build, averaged within the week. A "
                 "week is one observation however many lineups it holds; read the "
                 "count of weeks before the mean."),
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
    best_weeks = judged[0]["weeks_scored"] if judged else None
    worst_weeks = judged[-1]["weeks_scored"] if len(judged) >= 2 else None
    verdict = {
        "comparable": len(judged) >= 2,
        "contest": contest,
        "objectives_scored": len(judged),
        "scored_lineups": sum(o["scored_lineups"] for o in judged),
        "weeks_scored": len(scored),
        "best": judged[0]["objective"] if judged else None,
        "best_weeks": best_weeks,
        "worst": judged[-1]["objective"] if len(judged) >= 2 else None,
        "worst_weeks": worst_weeks,
        "spread": (round(judged[0]["actual"] - judged[-1]["actual"], 2)
                   if len(judged) >= 2 else None),
        # One week cannot separate objectives that differ by a point or two:
        # the same slate, the same pool, largely the same players. It is the
        # RANKED ENDS that have to clear a week each, not the season: four
        # scored weeks with the winner appearing in only one of them is still
        # a one-slate verdict, and the live record hit exactly that -- "floor"
        # took the title on week 7 alone, a week whose FPPG baseline was low
        # enough that every objective beat it.
        "one_week_only": (len(scored) <= 1
                          or (best_weeks is not None and best_weeks <= 1)
                          or (worst_weeks is not None and worst_weeks <= 1)),
        # This ranking pools every lineup an objective produced, so it also
        # measures how deep each objective was built. `matched_depth` is the
        # same question with that taken out.
        "basis": "pooled",
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
                "max_per_team": build.max_per_team,
                "stack_qb": build.stack_qb,
                "rb_with_qb": (None if build.rb_with_qb is None
                               else bool(build.rb_with_qb)),
                "ownership_weight": build.ownership_weight,
                "projection_computed_at": build.projection_computed_at,
                "fppg_baseline": _round(build.fppg_points),
                "fppg_projected": _round(build.fppg_projected),
                "best_possible": _round(build.best_points),
                "scored_at": build.scored_at,
                "lineups": [{
                    "lineup": entry.lineup_index,
                    "salary": entry.salary,
                    "projected": _round(entry.projected_points),
                    "actual": _round(entry.actual_points),
                    "unscored_players": entry.unscored_players,
                    "players": [{
                        "slot": p.slot,
                        "player_name": p.player_name,
                        "team": p.team,
                        "salary": p.salary,
                        "projected": _round(p.projected_points),
                        "actual": _round(p.actual_points),
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
        "matched_depth": _matched_depth(builds, entries),
        "depth": _depth_record(builds, entries),
        **({"builds": detail} if week is not None else {}),
        "note": ("fppg_baseline is the lineup FanDuel's own FPPG would have built from "
                 "the same pool under the same cap, scored on what happened — the "
                 "honest denominator. best_possible is the slate's best lineup in "
                 "hindsight. Only lineups whose every slot could be scored are averaged."),
    }
