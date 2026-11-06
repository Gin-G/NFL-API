"""Hand-uploaded third-party matchup tables, stored and compared.

The first source is FantasyGuru's weekly "smash" tables, exported by a
subscriber. **There is no fetcher here and there must never be one**: the
source's terms are personal use only, so the only way data arrives is a person
uploading the CSV they exported. `nfl_projections.external` has a test that
fails if a network library is ever imported into the parser.

**Nothing on these pages feeds /projections.** That is a gate with a condition
attached, not a general disclaimer: the house rule is that a signal earns its
way in by beating the production board's own residual out of sample, and this
one has not been tested, because testing it needs several weeks of snapshots
that do not exist yet. `scripts/grade_external.py` is the test and it refuses
to run until the weeks are there. Until it passes, these are comparison
numbers for a human reading a slate, in the same spirit as /coverage and
/trenches.

There is prior reason for scepticism about most of what is here, and it is
worth stating next to the data rather than in a footnote. Our own WR coverage
matchups failed three separate tests (/coverage's own page has the numbers:
wide receiver came out at -0.0032 MAE, actively harmful, with the interval
clear of zero), and the sharp-edge alt-line backtest found matchup added
nothing beyond role and cushion. The line ratings look substantially like a
run-game measure -- their O-line Spearman 0.66 against our run_block but 0.42
against pass_block, their D-front 0.73 against run_stop and 0.22 against
pass_rush -- and their own two sides correlate 0.6 with each other, which is
what team strength leaking into both would look like.

What is genuinely untested is the **assignment**: which corner a receiver
draws, and whether he is shadowed. nflverse publishes no such field, so that
column is the one thing here that is not a second opinion about something
already measured. It is the reason this is worth storing at all.
"""
import csv
import hashlib
import io
import logging
from datetime import datetime
from typing import Optional, Union

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Query,
                     UploadFile)
from sqlalchemy.orm import Session

from database.models import (ExternalSnapshot, ExternalTeamLine,
                             ExternalWrMatchup)
from database.session import get_db
from .utils import get_current_nfl_season

logger = logging.getLogger(__name__)
router = APIRouter()

# The exports are a few kilobytes. A megabyte means the wrong file.
MAX_CSV_BYTES = 1_000_000
SOURCES = ("fantasyguru",)
# line_matchups is accepted and CHECKED rather than stored: every value in it
# is derived from team_lines plus the schedule (its ADV columns are one side's
# O-line minus the other's D-front, verified to 0.02 across all 30 sides of
# week 9). Storing it as its own fact would let the two drift apart.
DERIVED_KINDS = ("line_matchups",)


def _external():
    """The parsing module, or a 503 saying what is missing."""
    try:
        from nfl_projections import external
        return external
    except ImportError as exc:        # pragma: no cover - packaging failure
        logger.error("nfl_projections.external unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Uploads need the nfl-projections package in this image.")


async def _read_upload(upload) -> str:
    """The CSV text of an upload, whether or not it arrived with a filename.

    A multipart part with no filename is, per the spec, an ordinary form
    field, and Starlette hands it over as a `str` rather than an `UploadFile`.
    Since nothing here cares what a file is called, a nameless one is a
    perfectly good upload and is read as the CSV it is -- which also means the
    text can simply be posted as a form field.
    """
    raw = upload if isinstance(upload, (str, bytes)) else await upload.read()
    if isinstance(raw, str):
        raw = raw.encode()
    if not raw:
        raise HTTPException(status_code=400, detail="That file is empty.")
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"That file is {len(raw) // 1024} KB; these exports are a "
                   "few KB. Check you uploaded the right file.")
    # utf-8-sig: a browser-exported CSV routinely carries a byte-order mark,
    # and left in the stored text it reappears in the first column's name.
    return raw.decode("utf-8-sig", "replace")


def _data_rows(text: str) -> int:
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    return max(len(rows) - 1, 0)


@router.post("/upload")
async def upload_snapshot(
    file: Union[UploadFile, str] = File(
        ..., description="A CSV exported by hand from the source. The filename "
                         "is ignored, and a part with no filename is fine."),
    season: Optional[int] = Form(None),
    week: int = Form(..., description="The week the table is about"),
    source: str = Form("fantasyguru"),
    kind: Optional[str] = Form(
        None, description="wr_coverage | team_lines | line_matchups; detected if omitted"),
    db: Session = Depends(get_db),
):
    """Store one weekly export and shred it onto our own ids.

    **The filename is not used for anything.** These files arrive as
    `table.csv`, `table (1).csv` and whatever the browser calls the next one,
    and the source is free to rename its exports whenever it likes, so the
    kind is worked out from the contents: the header is located by the columns
    it carries rather than assumed to be the first line, and columns are read
    by name, so a title row, a blank leading line, a reordered export or an
    added column all still load. The filename is kept on the snapshot as a
    label and never consulted again.

    So `kind` is optional and there is nothing to get wrong from a menu.
    Passing it turns detection into an assertion instead: a file that
    disagrees with what you said it was is refused rather than stored under
    the wrong name, because a WR table filed as team lines would read as an
    empty week rather than as a mistake.
    """
    external = _external()
    season = season or get_current_nfl_season()
    if source not in SOURCES:
        raise HTTPException(status_code=400,
                            detail=f"Unknown source {source!r}; expected one of "
                                   f"{', '.join(SOURCES)}.")
    text = await _read_upload(file)

    # Detected from the file's contents, never its name: these arrive called
    # table.csv, table (1).csv and whatever the browser names the next one.
    detected = external.detect_kind(text)
    if detected is None:
        head = [ln[:120] for ln in text.splitlines()[:3]] or ["(empty)"]
        raise HTTPException(
            status_code=400,
            detail=f"Nothing in the first {external.HEADER_SEARCH_ROWS} rows "
                   "of that file heads a known export "
                   f"({', '.join(external.KINDS)}). The filename is not used "
                   "and does not matter; the header is what identifies it. "
                   f"First rows seen: {head}")
    if kind and kind != detected:
        raise HTTPException(
            status_code=400,
            detail=f"You said {kind}, but the header says {detected}. Nothing "
                   "was stored.")
    kind = detected

    try:
        frame = external.parse(kind, text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # A derived export is checked and discarded: it is a consistency test on
    # team_lines, not a fact of its own.
    if kind in DERIVED_KINDS:
        return _check_derived(db, frame, source, season, week, kind, text)

    snapshot = _store_snapshot(db, source, season, week, kind,
                               getattr(file, "filename", None) or None,
                               text, _data_rows(text))
    if kind == "team_lines":
        result = _store_team_lines(db, frame, snapshot, source, season, week)
    else:
        result = _store_wr_matchups(db, frame, snapshot, source, season, week,
                                    external)
    snapshot.parsed_rows = result["stored"]
    snapshot.unmapped = ",".join(result["unmapped"]) or None
    db.commit()
    return {"status": "success", "source": source, "season": season,
            "week": week, "kind": kind, "snapshot_id": snapshot.id,
            "content_hash": snapshot.content_hash,
            "replaced": result.get("replaced", 0), **result}


def _store_snapshot(db, source, season, week, kind, filename, text, row_count):
    """Upsert the verbatim file. A re-upload replaces, keyed on the week."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    existing = db.query(ExternalSnapshot).filter_by(
        source=source, season=season, week=week, kind=kind).one_or_none()
    if existing is None:
        existing = ExternalSnapshot(source=source, season=season, week=week,
                                    kind=kind)
        db.add(existing)
    existing.filename = filename
    existing.content = text
    existing.content_hash = digest
    existing.row_count = row_count
    existing.uploaded_at = datetime.utcnow()
    db.flush()
    return existing


def _store_team_lines(db, frame, snapshot, source, season, week):
    replaced = db.query(ExternalTeamLine).filter_by(
        source=source, season=season, week=week).delete()
    unmapped, stored = [], 0
    for row in frame.to_dict("records"):
        if not row.get("team"):
            unmapped.append(str(row.get("team_raw") or "?"))
            continue
        db.add(ExternalTeamLine(
            source=source, season=season, week=week, team=row["team"],
            team_raw=row.get("team_raw"),
            offensive_line=row.get("offensive_line"),
            defensive_front=row.get("defensive_front"),
            snapshot_id=snapshot.id, uploaded_at=datetime.utcnow()))
        stored += 1
    db.flush()
    return {"stored": stored, "replaced": replaced, "unmapped": unmapped,
            "teams": stored}


def _store_wr_matchups(db, frame, snapshot, source, season, week, external):
    """Shred the corner/receiver table, mapping both names onto gsis ids."""
    try:
        frame = external.attach_player_ids(
            frame, {"defender": "defender_team", "receiver": "receiver_team"},
            season=season)
    except Exception as exc:
        # A roster download failing should not lose the upload: the names and
        # the pairing are the data, the ids are a convenience for joining.
        logger.warning("Could not map player ids for %s week %s: %s",
                       source, week, exc)
    replaced = db.query(ExternalWrMatchup).filter_by(
        source=source, season=season, week=week).delete()
    unmapped, stored, fuzzy = [], 0, 0
    for i, row in enumerate(frame.to_dict("records"), start=1):
        for side in ("defender", "receiver"):
            if row.get(f"{side}_id") is None:
                unmapped.append(f"{row.get(side)} ({row.get(side + '_team') or '?'})")
            elif row.get(f"{side}_match") == "team_fuzzy":
                fuzzy += 1
        db.add(ExternalWrMatchup(
            source=source, season=season, week=week,
            fg_rank=row.get("fg_rank") or i,
            defender=row.get("defender"), defender_id=row.get("defender_id"),
            defender_team=row.get("defender_team"),
            defender_match=row.get("defender_match"),
            receiver=row.get("receiver"), receiver_id=row.get("receiver_id"),
            receiver_team=row.get("receiver_team"),
            receiver_match=row.get("receiver_match"),
            coverage_type=row.get("coverage_type"),
            shell_pct=row.get("shell_pct"), shell=row.get("shell"),
            man_zone=row.get("man_zone"), shadow=bool(row.get("shadow")),
            advantage=row.get("advantage"),
            snapshot_id=snapshot.id, uploaded_at=datetime.utcnow()))
        stored += 1
    db.flush()
    return {"stored": stored, "replaced": replaced, "unmapped": unmapped,
            "pairings": stored, "fuzzy_name_matches": fuzzy,
            "shadow_assignments": int(frame.get("shadow", []).sum())
            if "shadow" in frame.columns else 0}


def _check_derived(db, frame, source, season, week, kind, text):
    """Report whether a derived export agrees with what we already store.

    Nothing is written. The answer that matters is whether the source's own
    numbers are self-consistent and whether they match the team_lines upload
    for the same week; if they do, this file holds no fact of its own.
    """
    rows = frame.to_dict("records")
    disagree = [r for r in rows if not r.get("adv_agrees")]
    stored = {t.team: t for t in db.query(ExternalTeamLine).filter_by(
        source=source, season=season, week=week).all()}
    drifted = []
    for r in rows:
        held = stored.get(r.get("team"))
        if held is None or r.get("offensive_line") is None:
            continue
        if abs((held.offensive_line or 0) - r["offensive_line"]) > 0.02:
            drifted.append({"team": r["team"], "in_team_lines": held.offensive_line,
                            "in_this_file": r["offensive_line"]})
    return {
        "status": "checked_not_stored",
        "source": source, "season": season, "week": week, "kind": kind,
        "sides": len(rows),
        # Their ADV is one side's O-line minus the other's D-front. If that
        # holds, the file is a view of team_lines and nothing here is new.
        "adv_self_consistent": not disagree,
        "adv_disagreements": [
            {"team": r.get("team"), "stated": r.get("adv_stated"),
             "derived": r.get("adv_derived")} for r in disagree],
        "matched_against_team_lines": len(stored) > 0,
        "team_line_drift": drifted,
        "note": ("Derived from team_lines plus the schedule, so it is checked "
                 "rather than stored. Upload the 32-row team table and the "
                 "25-row coverage table; this one is a view of the first."),
    }


def _row(obj, drop=("uploaded_at", "snapshot_id")):
    out = {c.name: getattr(obj, c.name) for c in obj.__table__.columns
           if c.name not in drop}
    return out


@router.get("/snapshots")
def list_snapshots(
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(None),
    source: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """What has been uploaded, newest week first, with what failed to map."""
    season = season or get_current_nfl_season()
    q = db.query(ExternalSnapshot).filter(ExternalSnapshot.season == season)
    if week is not None:
        q = q.filter(ExternalSnapshot.week == week)
    if source:
        q = q.filter(ExternalSnapshot.source == source)
    rows = q.order_by(ExternalSnapshot.week.desc(),
                      ExternalSnapshot.kind).all()
    weeks = sorted({r.week for r in rows})
    return {
        "status": "success" if rows else "no_data",
        "season": season,
        "count": len(rows),
        "weeks_held": weeks,
        # What the grading harness is waiting for.
        "weeks_with_wr_coverage": sorted(
            {r.week for r in rows if r.kind == "wr_coverage"}),
        "data": [{
            "snapshot_id": r.id, "source": r.source, "kind": r.kind,
            "season": r.season, "week": r.week, "filename": r.filename,
            "row_count": r.row_count, "parsed_rows": r.parsed_rows,
            "content_hash": r.content_hash,
            "unmapped": r.unmapped.split(",") if r.unmapped else [],
            "uploaded_at": r.uploaded_at.isoformat() if r.uploaded_at else None,
        } for r in rows],
    }


@router.get("/team-lines")
def get_team_lines(
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(None),
    source: str = Query("fantasyguru"),
    db: Session = Depends(get_db),
):
    """The source's O-line / D-front ratings for a week."""
    season = season or get_current_nfl_season()
    q = db.query(ExternalTeamLine).filter(ExternalTeamLine.season == season,
                                          ExternalTeamLine.source == source)
    if week is not None:
        q = q.filter(ExternalTeamLine.week == week)
    rows = q.order_by(ExternalTeamLine.offensive_line.desc()).all()
    return {"status": "success" if rows else "no_data", "season": season,
            "week": week, "source": source, "count": len(rows),
            "data": [_row(r) for r in rows]}


@router.get("/wr-matchups")
def get_wr_matchups(
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(None),
    source: str = Query("fantasyguru"),
    shadow_only: bool = Query(False, description="Only travelling assignments"),
    db: Session = Depends(get_db),
):
    """The corner-vs-receiver pairings for a week, best advantage first.

    `shadow` is the column worth reading: it is an assignment, which nflverse
    does not publish, so it is the only thing here that is not another
    estimate of something /coverage already measures.
    """
    season = season or get_current_nfl_season()
    q = db.query(ExternalWrMatchup).filter(ExternalWrMatchup.season == season,
                                           ExternalWrMatchup.source == source)
    if week is not None:
        q = q.filter(ExternalWrMatchup.week == week)
    if shadow_only:
        q = q.filter(ExternalWrMatchup.shadow.is_(True))
    rows = q.order_by(ExternalWrMatchup.fg_rank).all()
    return {
        "status": "success" if rows else "no_data",
        "season": season, "week": week, "source": source, "count": len(rows),
        "shadow_assignments": sum(1 for r in rows if r.shadow),
        "unresolved_names": [f"{r.defender}/{r.receiver}" for r in rows
                             if r.defender_id is None or r.receiver_id is None],
        "gated": True,
        "note": ("Comparison only. Nothing here feeds /projections until "
                 "scripts/grade_external.py has 4+ weeks and passes."),
        "data": [_row(r) for r in rows],
    }


# ---------------------------------------------------------------------------
# Comparisons: do they agree with us, and where do they not
# ---------------------------------------------------------------------------
#
# This is the job the user actually does with these tables — read two
# independent reports against each other, where agreement confirms a read and
# disagreement denies it. So the endpoints below are built to answer "where do
# we disagree", not to produce a blended number. There is no blended number
# anywhere in this module, by design: a blend would be a signal, and a signal
# has to pass scripts/grade_external.py first.


def _spearman(xs, ys) -> Optional[float]:
    """Rank correlation, or None with fewer than three usable pairs.

    Rank rather than Pearson because the two scales are unrelated — their
    ratings are not on our 0-100 grade scale and need not be linear in it. All
    that is being asked is whether the two orderings agree.
    """
    import pandas as pd

    frame = pd.DataFrame({"x": xs, "y": ys}).dropna()
    if len(frame) < 3 or frame.x.nunique() < 2 or frame.y.nunique() < 2:
        return None
    value = frame.x.corr(frame.y, method="spearman")
    return None if pd.isna(value) else round(float(value), 3)


@router.get("/compare/trenches")
def compare_trenches(
    season: Optional[int] = Query(None),
    week: int = Query(..., description="The week the upload is about"),
    source: str = Query("fantasyguru"),
    db: Session = Depends(get_db),
):
    """Their line ratings against /trenches, team by team.

    Our ratings split the line four ways (run block, pass block, run stop,
    pass rush) where theirs are two numbers, and the split is where the
    disagreement lives: on week 9 2026 their O-line ranked 0.66 with our run
    block and 0.42 with our pass block, their D-front 0.73 with run stop and
    0.22 with pass rush. Read as a run-game measure they largely agree with
    us; read as a pass-protection measure they do not.

    `self_correlation` is the other thing worth watching: their own O-line and
    D-front correlate about 0.6 across teams, which is what it looks like when
    overall team strength is baked into both sides rather than the two being
    measured independently.
    """
    import pandas as pd

    season = season or get_current_nfl_season()
    theirs = db.query(ExternalTeamLine).filter_by(
        source=source, season=season, week=week).all()
    if not theirs:
        return {"status": "no_data", "season": season, "week": week,
                "message": f"No {source} team-lines upload for {season} week "
                           f"{week}. POST it to /external/upload first."}

    from . import trenches as trenches_api

    # `_grades` hands back a plain DataFrame. Unpacking it would iterate its
    # columns instead of raising anything obvious, so it is bound whole.
    unavailable = None
    try:
        ours = trenches_api._grades(db, season, week)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("trench grades unavailable: %s", exc)
        unavailable = f"{type(exc).__name__}: {exc}"
        ours = pd.DataFrame()
    if ours is None or getattr(ours, "empty", True):
        # Say which it was. An empty frame means the weeks are not rated yet;
        # a raised exception means the lookup itself broke, and reporting the
        # two identically is how a broken call reads as a quiet "no data".
        return {"status": "no_ratings", "season": season, "week": week,
                "message": ("No /trenches ratings for this week to compare "
                            "against."),
                "reason": unavailable or "no rated weeks"}

    ours = ours.set_index("team") if "team" in ours.columns else ours
    pairs, rows = [], []
    for t in theirs:
        mine = ours.loc[t.team].to_dict() if t.team in ours.index else {}
        row = {
            "team": t.team,
            "their_oline": t.offensive_line,
            "their_dfront": t.defensive_front,
            "our_run_block": mine.get("run_block"),
            "our_pass_block": mine.get("pass_block"),
            "our_run_stop": mine.get("run_stop"),
            "our_pass_rush": mine.get("pass_rush"),
        }
        rows.append(row)
        pairs.append(row)
    frame = pd.DataFrame(pairs)

    correlations = {
        "their_oline_vs_our_run_block": _spearman(frame.their_oline, frame.our_run_block),
        "their_oline_vs_our_pass_block": _spearman(frame.their_oline, frame.our_pass_block),
        "their_dfront_vs_our_run_stop": _spearman(frame.their_dfront, frame.our_run_stop),
        "their_dfront_vs_our_pass_rush": _spearman(frame.their_dfront, frame.our_pass_rush),
    }
    self_corr = _spearman(frame.their_oline, frame.their_dfront)

    # Where the two orderings part company, which is the actionable half.
    disagreements = []
    if frame.their_oline.notna().any() and frame.our_run_block.notna().any():
        f = frame.dropna(subset=["their_oline", "our_run_block"]).copy()
        f["their_rank"] = f.their_oline.rank(ascending=False)
        f["our_rank"] = f.our_run_block.rank(ascending=False)
        f["rank_gap"] = (f.their_rank - f.our_rank).abs()
        for r in f.nlargest(5, "rank_gap").to_dict("records"):
            disagreements.append({
                "team": r["team"], "their_oline_rank": int(r["their_rank"]),
                "our_run_block_rank": int(r["our_rank"]),
                "rank_gap": int(r["rank_gap"]),
                "direction": ("they rate the line lower"
                              if r["their_rank"] > r["our_rank"]
                              else "they rate the line higher"),
            })

    return {
        "status": "success", "season": season, "week": week, "source": source,
        "teams": len(rows),
        "correlations": correlations,
        "self_correlation": self_corr,
        "self_correlation_note": (
            "Their O-line against their own D-front. A high value means team "
            "strength is in both sides rather than each being measured alone."),
        "biggest_disagreements": disagreements,
        "gated": True,
        "data": rows,
    }


@router.get("/compare/coverage")
def compare_coverage(
    season: Optional[int] = Query(None),
    week: int = Query(..., description="The week the upload is about"),
    source: str = Query("fantasyguru"),
    db: Session = Depends(get_db),
):
    """Their corner advantage against /coverage's own corner grades.

    The signs run opposite by construction: their Advantage is high when the
    matchup favours the RECEIVER, our cover_grade is high when the corner is
    good. So agreement shows up as a NEGATIVE correlation, and the figure to
    beat is the -0.33 measured on week 9 2026 — the right direction and loose.

    Our side of several of these pairings rests on very little: a corner needs
    15 charted targets to be listed at all, and some are listed on barely
    more. `our_targets` is on every row so a disagreement can be read as
    "they are wrong" or "we do not know yet", which are different findings.
    """
    import pandas as pd

    season = season or get_current_nfl_season()
    theirs = db.query(ExternalWrMatchup).filter_by(
        source=source, season=season, week=week).order_by(
        ExternalWrMatchup.fg_rank).all()
    if not theirs:
        return {"status": "no_data", "season": season, "week": week,
                "message": f"No {source} WR coverage upload for {season} week "
                           f"{week}. POST it to /external/upload first."}

    from nfl_projections.utils import normalize_player_name

    from . import coverage as coverage_api

    # Unlike `trenches._grades`, this one really is a pair: per-corner grades
    # and the team roll-up. The two modules differing is worth the comment.
    unavailable = None
    try:
        cg, _ = coverage_api._grades(season, week)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("corner grades unavailable: %s", exc)
        unavailable = f"{type(exc).__name__}: {exc}"
        cg = pd.DataFrame()
    if cg is None or getattr(cg, "empty", True):
        return {"status": "no_grades", "season": season, "week": week,
                "message": "No /coverage corner grades for this week.",
                "reason": unavailable or "no graded corners"}

    # Joined on the normalised name, and on (name, team) first: our grades are
    # keyed on PFR ids and theirs on gsis ids, with no published bridge
    # between them that does not cost another download.
    by_pair, by_name = {}, {}
    for rec in cg.to_dict("records"):
        key = normalize_player_name(rec.get("player_name"))
        if not key:
            continue
        by_name.setdefault(key, rec)
        by_pair[(key, rec.get("team"))] = rec

    rows, matched = [], 0
    for t in theirs:
        key = normalize_player_name(t.defender)
        mine = by_pair.get((key, t.defender_team)) or by_name.get(key)
        if mine:
            matched += 1
        rows.append({
            "fg_rank": t.fg_rank,
            "defender": t.defender, "defender_team": t.defender_team,
            "receiver": t.receiver, "receiver_team": t.receiver_team,
            "their_advantage": t.advantage,
            "coverage_type": t.coverage_type,
            "shell": t.shell, "man_zone": t.man_zone, "shadow": t.shadow,
            "our_cover_grade": (round(float(mine["cover_grade"]), 1)
                                if mine and mine.get("cover_grade") is not None
                                else None),
            "our_band": mine.get("band") if mine else None,
            # How much our own number rests on. 15 targets is the floor to be
            # listed at all, so a figure near it is not a disagreement.
            "our_targets": (int(mine["targets"]) if mine
                            and mine.get("targets") is not None else None),
        })

    frame = pd.DataFrame(rows)
    thin = [r["defender"] for r in rows
            if r["our_targets"] is not None and r["our_targets"] < 20]
    return {
        "status": "success", "season": season, "week": week, "source": source,
        "pairings": len(rows),
        "matched_to_our_grades": matched,
        "unmatched": [r["defender"] for r in rows
                      if r["our_cover_grade"] is None],
        "spearman_advantage_vs_cover_grade": _spearman(
            frame.their_advantage, frame.our_cover_grade),
        "expected_sign": "negative",
        "expected_sign_note": (
            "Their Advantage is high when the matchup favours the receiver; "
            "our cover_grade is high when the corner is good. Agreement is a "
            "negative correlation."),
        "our_grade_rests_on_under_20_targets": thin,
        "shadow_assignments": [
            {"defender": r["defender"], "receiver": r["receiver"]}
            for r in rows if r["shadow"]],
        "shadow_note": (
            "The one column here nflverse has no equivalent for. /coverage can "
            "say a secondary has a weak corner and cannot say who draws him."),
        "gated": True,
        "data": rows,
    }
