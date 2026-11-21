"""Players to Target: expert picks set beside our own board.

The user subscribes to weekly position articles (FantasyGuru's QB/RB/WR/TE
"Coach" pieces). He exports them, the picks are extracted into a CSV he checks,
and he uploads it here. **Personal use only, nothing fetches anything**, and
only the pick is stored -- never the article's text.

Each pick lands in one of three piles:

  agree      our board would play him too: inside our top N at his position by
             projected points, or by points per $1,000 of salary
  skeptical  our board would not: outside the top N both ways
  no board   we have no projection for him (not rostered, a name we could not
             match, or ruled out)

N is a main-slate starting depth: 8 QBs, 16 RBs, 24 WRs, 8 TEs. The numbers
behind every verdict are on the row, so the line can be read rather than
trusted.

**Display only.** None of it reaches /projections or the optimizer's
objective. A skeptical row is a disagreement, and after the games it is a
graded one: the page shows what each pile actually scored against our board,
which is the record that will say whether the articles know something the
board does not.
"""
import csv
import io
import json
import logging
from datetime import datetime
from typing import Optional, Union

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.models import ExpertPick, PlayerStat, TargetLineup, TargetNote
from database.session import get_db
from .utils import fanduel_points, get_current_nfl_season

logger = logging.getLogger(__name__)
router = APIRouter()

SOURCES = ("fantasyguru",)
POSITIONS = ("QB", "RB", "WR", "TE")
# Main-slate starting depth: how far down our board "we would play him" runs.
DEPTH = {"QB": 8, "RB": 16, "WR": 24, "TE": 8}
TAGS = ("cash", "gpp", "punt", "favorite_cash", "favorite_gpp", "mention")
MAX_CSV_BYTES = 200_000
MAX_NOTE_CHARS = 20_000


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------

def _tags(raw) -> str:
    """Normalise a tag cell: ``"Cash + GPP"`` -> ``"cash;gpp"``. Unknown words
    are refused rather than dropped, so a typo is seen at upload."""
    words = [w.strip().lower().replace(" ", "_")
             for w in str(raw or "").replace("+", ";").replace(",", ";").split(";")]
    words = [w for w in words if w]
    bad = [w for w in words if w not in TAGS]
    if bad:
        raise ValueError(f"unknown tag(s) {bad}; expected {', '.join(TAGS)}")
    return ";".join(dict.fromkeys(words))


def parse_picks(text: str) -> list[dict]:
    """Rows of a picks CSV. Needs `player` and `position`; `team`, `tags`,
    `article`, `author`, `published`, `salary_fd` are optional."""
    reader = csv.DictReader(io.StringIO(text))
    cols = {c.strip().lower() for c in (reader.fieldnames or [])}
    missing = {"player", "position"} - cols
    if missing:
        raise ValueError(f"the CSV needs columns {sorted(missing)}; it has {sorted(cols)}")
    rows = []
    for n, raw in enumerate(reader, start=2):
        r = {str(k).strip().lower(): (v or "").strip() for k, v in raw.items() if k}
        if not r.get("player"):
            continue
        pos = r.get("position", "").upper()
        if pos not in POSITIONS:
            raise ValueError(f"line {n}: position {pos!r} is not one of {POSITIONS}")
        try:
            tags = _tags(r.get("tags"))
        except ValueError as exc:
            raise ValueError(f"line {n}: {exc}")
        salary = r.get("salary_fd", "").replace("$", "").replace(",", "")
        rows.append({
            "player_name": r["player"], "position": pos,
            "team": r.get("team", "").upper() or None,
            "tags": tags or None,
            "article": r.get("article") or f"{pos} article",
            "author": r.get("author") or None,
            "published": r.get("published") or None,
            "salary_fd": int(float(salary)) if salary else None,
        })
    if not rows:
        raise ValueError("no picks in that file")
    return rows


@router.post("/picks")
async def upload_picks(
    file: Union[UploadFile, str] = File(..., description="The picks CSV."),
    season: Optional[int] = Form(None),
    week: int = Form(...),
    source: str = Form("fantasyguru"),
    db: Session = Depends(get_db),
):
    """Store a week's picks. Re-uploading an article REPLACES that article's
    picks for the week, so a corrected file overwrites rather than doubles."""
    from api.optimizer import _projection_frame
    from nfl_projections.names import Matcher

    season = season or get_current_nfl_season()
    if source not in SOURCES:
        raise HTTPException(400, f"Unknown source {source!r}; expected {SOURCES}.")
    raw = file if isinstance(file, (str, bytes)) else await file.read()
    if isinstance(raw, str):
        raw = raw.encode()
    if not raw:
        raise HTTPException(400, "That file is empty.")
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(413, "That file is far larger than a week of picks.")
    try:
        rows = parse_picks(raw.decode("utf-8-sig", "replace"))
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    board = _projection_frame(db, season, week)
    matcher = Matcher(board) if not board.empty else None
    for article in {r["article"] for r in rows}:
        db.query(ExpertPick).filter_by(season=season, week=week, source=source,
                                       article=article).delete()
    unmatched = []
    for r in rows:
        hit = matcher.match(r["player_name"], r["team"], r["position"]) if matcher else None
        pid = board.at[hit, "player_id"] if hit is not None else None
        if pid is None:
            unmatched.append(r["player_name"])
        db.add(ExpertPick(season=season, week=week, source=source, player_id=pid,
                          uploaded_at=datetime.utcnow(), **r))
    db.commit()
    return {"status": "success", "season": season, "week": week, "stored": len(rows),
            "articles": sorted({r["article"] for r in rows}),
            "unmatched": unmatched,
            "note": ("Unmatched players are kept and shown as 'no board'; they "
                     "usually have no projection this week." if unmatched else None)}


@router.delete("/picks")
def delete_picks(season: int = Query(...), week: int = Query(...),
                 article: Optional[str] = Query(None), db: Session = Depends(get_db)):
    q = db.query(ExpertPick).filter_by(season=season, week=week)
    if article:
        q = q.filter_by(article=article)
    n = q.delete()
    db.commit()
    return {"status": "success", "deleted": n}


# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------

class NoteBody(BaseModel):
    season: Optional[int] = None
    week: int
    body: str


def _note(db, season, week) -> Optional[dict]:
    n = db.query(TargetNote).filter_by(season=season, week=week).one_or_none()
    if n is None or not (n.body or "").strip():
        return None
    return {"body": n.body, "updated_at": n.updated_at}


@router.put("/notes")
def put_notes(note: NoteBody, db: Session = Depends(get_db)):
    """Set the week's notes (replaces them). An empty body clears them.

    A summary in our own words of what the week's articles agree and disagree
    on, never their text. Plain text: `## ` heads a section, `- ` a bullet."""
    season = note.season or get_current_nfl_season()
    body = note.body.strip()
    if len(body) > MAX_NOTE_CHARS:
        raise HTTPException(413, f"Notes are capped at {MAX_NOTE_CHARS:,} characters; "
                                 "this is a summary, not the articles.")
    row = db.query(TargetNote).filter_by(season=season, week=note.week).one_or_none()
    if not body:
        if row is not None:
            db.delete(row)
            db.commit()
        return {"status": "success", "season": season, "week": note.week, "notes": None}
    if row is None:
        row = TargetNote(season=season, week=note.week)
        db.add(row)
    row.body, row.updated_at = body, datetime.utcnow()
    db.commit()
    return {"status": "success", "season": season, "week": note.week,
            "notes": _note(db, season, note.week)}


# --------------------------------------------------------------------------
# Lineups
# --------------------------------------------------------------------------

LINEUP_KINDS = ("cash", "gpp")
MAX_RATIONALE_CHARS = 3_000


class LineupPlayerBody(BaseModel):
    slot: str
    player: str
    player_id: Optional[str] = None
    position: Optional[str] = None
    team: Optional[str] = None
    salary: Optional[int] = None
    projected: Optional[float] = None
    ceiling: Optional[float] = None


class LineupBody(BaseModel):
    season: Optional[int] = None
    week: int
    kind: str
    title: str = ""
    rationale: str = ""
    players: list[LineupPlayerBody] = []


def _lineups(db, season, week) -> list[dict]:
    """The week's stored lineups, each graded once its players have stats.

    A player with no id (a defense) has no stat row here, so a lineup's total
    is only reported once every player WITH an id has played; `unscored`
    says how many slots that total leaves out."""
    rows = db.query(TargetLineup).filter_by(season=season, week=week).all()
    out = []
    for r in sorted(rows, key=lambda r: LINEUP_KINDS.index(r.kind)
                    if r.kind in LINEUP_KINDS else 9):
        try:
            players = json.loads(r.players or "[]")
        except ValueError:
            players = []
        ids = [p["player_id"] for p in players if p.get("player_id")]
        actual = _actuals(db, season, week, ids) if ids else {}
        for p in players:
            p["actual"] = actual.get(p.get("player_id"))
        scored = [p for p in players if p.get("player_id")]
        done = bool(scored) and all(p["actual"] is not None for p in scored)
        out.append({
            "kind": r.kind, "title": r.title, "rationale": r.rationale,
            "players": players,
            "salary": sum(p.get("salary") or 0 for p in players),
            "projected": round(sum(p.get("projected") or 0 for p in players), 1),
            "ceiling": round(sum(p.get("ceiling") or 0 for p in players), 1),
            "actual": round(sum(p["actual"] or 0 for p in players), 1) if done else None,
            "unscored": sum(1 for p in players if p["actual"] is None) if done else None,
            "updated_at": r.updated_at,
        })
    return out


@router.put("/lineups")
def put_lineup(lineup: LineupBody, db: Session = Depends(get_db)):
    """Set the week's cash or tournament lineup (replaces it). No players clears it.

    Display only: built from the article pool with our board choosing, shown and
    graded on the page, never fed to the projections or the optimizer."""
    season = lineup.season or get_current_nfl_season()
    if lineup.kind not in LINEUP_KINDS:
        raise HTTPException(400, f"kind must be one of {', '.join(LINEUP_KINDS)}.")
    if len(lineup.players) > 9:
        raise HTTPException(400, "A FanDuel lineup is nine players.")
    if len(lineup.rationale) > MAX_RATIONALE_CHARS:
        raise HTTPException(413, f"The rationale is capped at {MAX_RATIONALE_CHARS:,} "
                                 "characters.")
    row = db.query(TargetLineup).filter_by(season=season, week=lineup.week,
                                           kind=lineup.kind).one_or_none()
    if not lineup.players:
        if row is not None:
            db.delete(row)
            db.commit()
        return {"status": "success", "season": season, "week": lineup.week,
                "lineups": _lineups(db, season, lineup.week)}
    if row is None:
        row = TargetLineup(season=season, week=lineup.week, kind=lineup.kind)
        db.add(row)
    row.title = lineup.title.strip()[:120]
    row.rationale = lineup.rationale.strip()
    row.players = json.dumps([p.model_dump() for p in lineup.players])
    row.updated_at = datetime.utcnow()
    db.commit()
    return {"status": "success", "season": season, "week": lineup.week,
            "lineups": _lineups(db, season, lineup.week)}


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

def _slate_frame(db, season, week, board):
    """The week's FanDuel slate joined to the board (salary per player id), or None."""
    from api.dfs import _slate_row
    from api.optimizer import _csv_frame
    from nfl_projections import optimizer as opt

    slate = _slate_row(db, season, week, None)
    if slate is None or not slate.content or board.empty:
        return None
    try:
        frame = opt.merge_fanduel_salaries(_csv_frame(slate.content, "The saved slate"),
                                           board)
    except Exception as exc:          # pragma: no cover - a bad saved slate
        logger.warning("slate unavailable for targets: %s", exc)
        return None
    frame = frame[frame["player_id"].notna()].copy()
    frame["Salary"] = pd.to_numeric(frame["Salary"], errors="coerce")
    frame = frame.sort_values("Salary").drop_duplicates("player_id", keep="first")
    return frame


def rank_board(board: pd.DataFrame, slate: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Per player id: projection, position rank by points, salary, value and
    position rank by value (value only for players on the slate)."""
    b = board[board["position"].isin(POSITIONS)].copy()
    b["fanduel_fantasy_points"] = pd.to_numeric(b["fanduel_fantasy_points"], errors="coerce")
    b = b[b["fanduel_fantasy_points"].notna()]
    b["proj_rank"] = (b.groupby("position")["fanduel_fantasy_points"]
                       .rank(ascending=False, method="min").astype(int))
    b["salary"] = None
    b["value"] = None
    b["value_rank"] = None
    if slate is not None and not slate.empty:
        sal = slate.set_index("player_id")["Salary"]
        b["salary"] = b["player_id"].map(sal)
        on = b["salary"].notna() & (b["salary"] > 0)
        b.loc[on, "value"] = b.loc[on, "fanduel_fantasy_points"] / (b.loc[on, "salary"] / 1000.0)
        v = pd.to_numeric(b["value"], errors="coerce")
        b["value_rank"] = v.groupby(b["position"]).rank(ascending=False, method="min")
    return b.set_index("player_id")


def verdict(position: str, proj_rank, value_rank) -> tuple[str, str]:
    """(pile, why) for one pick from where our board ranks him."""
    n = DEPTH[position]
    parts = [f"our #{int(proj_rank)} {position} by points"]
    if value_rank is not None and not pd.isna(value_rank):
        parts.append(f"#{int(value_rank)} by value")
    why = ", ".join(parts)
    in_points = proj_rank <= n
    in_value = value_rank is not None and not pd.isna(value_rank) and value_rank <= n
    if in_points or in_value:
        return "agree", f"{why} (top {n} counts)"
    return "skeptical", f"{why}: outside our top {n} both ways"


def _actuals(db, season, week, ids) -> dict:
    rows = db.query(PlayerStat).filter(PlayerStat.season == season,
                                       PlayerStat.week == week,
                                       PlayerStat.player_id.in_(list(ids))).all()
    return {r.player_id: round(fanduel_points(
        {c.name: getattr(r, c.name) for c in r.__table__.columns}), 2) for r in rows}


@router.get("/")
def players_to_target(season: Optional[int] = Query(None), week: int = Query(...),
                      db: Session = Depends(get_db)):
    """The week's picks in three piles, with our board's numbers on every row,
    and -- once the games are played -- what each pile actually scored."""
    from api.optimizer import _projection_frame

    season = season or get_current_nfl_season()
    picks = db.query(ExpertPick).filter_by(season=season, week=week).all()
    if not picks:
        return {"status": "success", "season": season, "week": week, "count": 0,
                "agree": [], "skeptical": [], "no_board": [], "scored": None,
                "depth": DEPTH, "articles": [], "notes": _note(db, season, week),
                "lineups": _lineups(db, season, week)}

    board = _projection_frame(db, season, week)
    ranked = rank_board(board, _slate_frame(db, season, week, board)) \
        if not board.empty else pd.DataFrame()

    # One row per player: the same man in two articles is one target with
    # both articles' tags.
    merged: dict = {}
    for p in picks:
        key = p.player_id or f"name:{p.player_name}|{p.team}"
        row = merged.setdefault(key, {
            "player_id": p.player_id, "player": p.player_name,
            "position": p.position, "team": p.team, "tags": [], "articles": [],
            "article_salary": p.salary_fd})
        row["tags"] += [t for t in (p.tags or "").split(";") if t and t not in row["tags"]]
        if p.article not in row["articles"]:
            row["articles"].append(p.article)

    ids = [r["player_id"] for r in merged.values() if r["player_id"]]
    actual = _actuals(db, season, week, ids) if ids else {}
    piles = {"agree": [], "skeptical": [], "no_board": []}
    for row in merged.values():
        pid = row["player_id"]
        if pid is None or ranked.empty or pid not in ranked.index:
            row.update(pile="no_board", why="no projection on our board this week",
                       projected=None, floor=None, ceiling=None, proj_rank=None,
                       salary=row["article_salary"], value=None, value_rank=None,
                       actual=actual.get(pid), vs_board=None)
            piles["no_board"].append(row)
            continue
        b = ranked.loc[pid]
        pile, why = verdict(row["position"], b["proj_rank"], b["value_rank"])
        proj = float(b["fanduel_fantasy_points"])
        salary = b["salary"] if pd.notna(b["salary"]) else row["article_salary"]
        got = actual.get(pid)
        row.update(
            pile=pile, why=why, team=row["team"] or b.get("team"),
            projected=round(proj, 1),
            floor=None if pd.isna(b.get("floor")) else round(float(b["floor"]), 1),
            ceiling=None if pd.isna(b.get("ceiling")) else round(float(b["ceiling"]), 1),
            proj_rank=int(b["proj_rank"]),
            salary=int(salary) if salary is not None and pd.notna(salary) else None,
            value=None if b["value"] is None or pd.isna(b["value"]) else round(float(b["value"]), 2),
            value_rank=None if b["value_rank"] is None or pd.isna(b["value_rank"]) else int(b["value_rank"]),
            actual=got, vs_board=None if got is None else round(got - proj, 1))
        piles[pile].append(row)

    for pile in piles.values():
        pile.sort(key=lambda r: (POSITIONS.index(r["position"]), -(r["projected"] or 0)))

    scored = {}
    for name in ("agree", "skeptical"):
        done = [r for r in piles[name] if r["actual"] is not None]
        if done:
            scored[name] = {
                "n": len(done),
                "mean_actual": round(sum(r["actual"] for r in done) / len(done), 2),
                "mean_projected": round(sum(r["projected"] for r in done) / len(done), 2),
                "mean_vs_board": round(sum(r["vs_board"] for r in done) / len(done), 2),
                "beat_board": sum(1 for r in done if r["vs_board"] > 0),
            }
    return {
        "status": "success", "season": season, "week": week,
        "count": sum(len(v) for v in piles.values()), **piles,
        "scored": scored or None, "depth": DEPTH,
        "articles": sorted({p.article for p in picks}),
        "notes": _note(db, season, week),
        "lineups": _lineups(db, season, week),
        "rule": ("Agree: inside our top N at the position by projected points or "
                 "by points per $1,000. Skeptical: outside both. N = "
                 + ", ".join(f"{k} {v}" for k, v in DEPTH.items())
                 + ". Display only; nothing here reaches the projections or "
                   "the optimizer's objective."),
    }
