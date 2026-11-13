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
import logging
from datetime import datetime
from typing import Optional, Union

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from database.models import ExpertPick, PlayerStat
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
                "depth": DEPTH, "articles": []}

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
        "rule": ("Agree: inside our top N at the position by projected points or "
                 "by points per $1,000. Skeptical: outside both. N = "
                 + ", ".join(f"{k} {v}" for k, v in DEPTH.items())
                 + ". Display only; nothing here reaches the projections or "
                   "the optimizer's objective."),
    }
