#!/usr/bin/env python3
"""A slate read down in price tiers, the way a human builds a lineup.

This serves `nfl_projections.dfs_targets`: the board joined onto the FanDuel
slate, split into price bands per position, ordered by points per dollar, with
a sentence per player. See that module for why the ordering uses nothing but
our own projection and the salary -- no Vegas, no matchup, no named overrides
-- and why opponent strength rides along in a separate `context` block that is
explicitly marked as not having moved anybody.

The provenance block is not decoration. A targets list is read at lineup time,
which is days after the board was computed, and the difference between those
two moments is where the injuries are. It is reported first for that reason.
"""

import io
import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database.models import (FanduelSlate, PlayerProjection, PlayerStat,
                             ProjectionRun)
from database.session import get_db
from .utils import get_current_nfl_season

logger = logging.getLogger(__name__)
router = APIRouter()

# Kept small and explicit: the response is read by a person deciding where to
# spend $60,000, so every number in it should be traceable to one of these.
# model attribute -> name used here. The quantile columns are stored as
# plain floor/median/ceiling on PlayerProjection; reading them under any other
# name yields None for every row without raising, which is why they are
# spelled out rather than assumed.
BOARD_COLUMNS = {
    "projected_points": "projected_points",
    "floor": "floor",
    "median": "median",
    "ceiling": "ceiling",
}


def _targets():
    """The research module, or a 503 explaining what is missing."""
    try:
        from nfl_projections import dfs_targets

        return dfs_targets
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"nfl_projections.dfs_targets unavailable: {exc}")


def _name_key():
    try:
        from nfl_projections.utils import normalize_player_name

        return normalize_player_name
    except ImportError as exc:
        raise HTTPException(status_code=503,
                            detail=f"nfl_projections.utils unavailable: {exc}")


def _board(db, season: int, week: int) -> pd.DataFrame:
    rows = db.query(PlayerProjection).filter(
        PlayerProjection.season == season,
        PlayerProjection.week == week).all()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([{
        "player_id": r.player_id,
        "player_name": r.player_name,
        "position": r.position,
        "team": r.team,
        **{dest: getattr(r, src, None) for src, dest in BOARD_COLUMNS.items()},
    } for r in rows])
    stamps = [r.computed_at for r in rows if r.computed_at is not None]
    frame.attrs["computed_at"] = max(stamps) if stamps else None
    return frame


def _history(db, season: int, week: int) -> pd.DataFrame:
    """Each player's weekly FanDuel points BEFORE `week`, one row per game.

    Weeks strictly before the slate's: a range that already contains the result
    being projected would be hindsight dressed as context.
    """
    from .utils import fanduel_points

    rows = db.query(PlayerStat).filter(PlayerStat.season == season,
                                       PlayerStat.week < week).all()
    records = []
    for row in rows:
        if (row.season_type or "REG") != "REG":
            continue
        name = row.player_display_name or row.player_name
        if not name:
            continue
        records.append({
            "player_name": name,
            "week": row.week,
            "fantasy_points": fanduel_points(
                {c.name: getattr(row, c.name) for c in row.__table__.columns}),
        })
    return pd.DataFrame(records)


def _slate_row(db, season: int, week: int, slate_id: Optional[int]):
    if slate_id is not None:
        return db.query(FanduelSlate).filter(
            FanduelSlate.id == slate_id).one_or_none()
    rows = db.query(FanduelSlate).filter(
        FanduelSlate.season == season, FanduelSlate.week == week).all()
    if not rows:
        return None
    classic = [r for r in rows if r.contest == "classic"]
    return max(classic or rows,
               key=lambda r: r.uploaded_at or pd.Timestamp.min.to_pydatetime())


def _fpa_context(db, season: int, week: int):
    """``{(team, position): {...}}`` of our schedule-adjusted opponent grades.

    Soft-fails to an empty map: a targets list must not die because the
    adjustment is unavailable, and every caller treats it as optional context.
    """
    try:
        from api.fpa import compute_fpa_detail

        detail = compute_fpa_detail(db, season, through_week=week)
    except Exception as exc:
        logger.warning("fpa context unavailable (%s); omitting", exc)
        return {}
    out = {}
    rows = detail.get("data", detail) if isinstance(detail, dict) else detail
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    for pos in ("qb", "rb", "wr", "te"):
        col = f"{pos}_grade"
        if col not in frame.columns:
            continue
        ranked = frame[["team", col]].dropna().copy()
        # Rank 1 = softest, which is the direction a person reads it in.
        ranked["rank"] = ranked[col].rank(ascending=False, method="min")
        for _, r in ranked.iterrows():
            band = frame.loc[frame.team == r["team"], f"{pos}_band"]
            out[(r["team"], pos.upper())] = {
                "grade": float(r[col]),
                "rank": int(r["rank"]),
                "band": (band.iloc[0] if len(band) else None),
            }
    return out


def _run_block(db, season: int, week: int):
    run = db.query(ProjectionRun).filter(
        ProjectionRun.season == season,
        ProjectionRun.week == week).order_by(
            ProjectionRun.computed_at.desc()).first()
    if run is None:
        return None
    stale = []
    if not run.injury_report_available:
        stale.append("No injury report had been filed when this board was "
                     "computed.")
    elif (run.players_zeroed or 0) <= (run.ir_players or 0):
        stale.append(
            f"Every one of the {run.players_zeroed or 0} players this board "
            f"zeroed came off roster status, not the injury report: the "
            f"{run.injury_report_players or 0} players the report named all "
            "parsed as questionable, which benches nobody.")
    return {
        "run_id": run.id,
        "computed_at": run.computed_at.isoformat() if run.computed_at else None,
        "as_of_week": run.as_of_week,
        "injury_report_rows": run.injury_report_rows,
        "players_zeroed": run.players_zeroed,
        "ir_players": run.ir_players,
        "questionable": run.questionable,
        "depth_chart_as_of": run.depth_chart_as_of.isoformat()
                             if run.depth_chart_as_of else None,
        "warnings": stale,
    }


@router.get("/targets")
def dfs_targets(
    season: Optional[int] = Query(None),
    week: int = Query(..., description="The slate's week"),
    slate_id: Optional[int] = Query(
        None, description="A specific saved slate; the newest classic by default"),
    premium_min: int = Query(7500, description="Floor of the premium band, in $"),
    value_max: int = Query(6000, description="Ceiling of the value band, in $"),
    limit_per_tier: int = Query(8, ge=1, le=50),
    positions: str = Query("QB,RB,WR,TE,DEF"),
    games: Optional[str] = Query(
        None, description="Restrict to these games, e.g. 'DET@MIA,LV@SF'. A "
                          "FanDuel week file holds every game, so a main-slate "
                          "read has to say which ones it means."),
    history: bool = Query(
        True, description="Attach the lowest, highest and median each player "
                          "has actually scored this season. Reading material: "
                          "it never enters a projection, a value or a rank."),
    db: Session = Depends(get_db),
):
    """The slate grouped into price tiers per position, best value first.

    Ordering is our projection divided by salary and nothing else. Opponent
    strength appears per player under `context`, and the response repeats at
    the top that it moved nobody — a matchup number printed beside a ranking
    is read as having produced it unless it says otherwise.
    """
    targets = _targets()
    name_key = _name_key()
    season = season or get_current_nfl_season()

    slate_row = _slate_row(db, season, week, slate_id)
    if slate_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No FanDuel slate saved for {season} week {week}. Upload "
                   "the contest player list to /optimizer/lineups first; "
                   "salaries are the one thing here that cannot be derived.")

    board = _board(db, season, week)
    if board.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No published projections for {season} week {week}.")

    try:
        slate = pd.read_csv(io.StringIO(slate_row.content))
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"Saved slate could not be read: {exc}")

    all_games = (sorted(slate["Game"].dropna().unique())
                 if "Game" in slate.columns else [])
    chosen = None
    if games:
        chosen = [g.strip().upper() for g in games.split(",") if g.strip()]
        unknown = [g for g in chosen if g not in {str(x).upper() for x in all_games}]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"This slate has no game(s) {', '.join(unknown)}. It "
                       f"holds: {', '.join(str(g) for g in all_games)}")
        slate = slate[slate["Game"].astype(str).str.upper().isin(chosen)]

    wanted = targets.canonical_positions(positions.split(","))
    frame = targets.build(board, slate, name_key=name_key,
                          premium_min=premium_min, value_max=value_max,
                          positions=wanted,
                          history=_history(db, season, week) if history else None)
    if frame.empty:
        return {"status": "no_overlap", "season": season, "week": week,
                "message": "The board and the slate have no players in common."}

    fpa = _fpa_context(db, season, week)
    unmatched = sorted(frame[frame.projected_points.isna()]["Nickname"].tolist())

    groups = []
    for pos in wanted:
        for tier in targets.TIER_ORDER:
            sel = frame[(frame.slate_position == pos) & (frame.tier == tier)]
            if sel.empty:
                continue
            players = []
            for _, r in sel.head(limit_per_tier).iterrows():
                ctx = {}
                opp_fpa = fpa.get((r.get("opponent"), pos))
                if opp_fpa:
                    ctx["opponent_fpa"] = opp_fpa
                players.append({
                    "player": r.get("Nickname"),
                    "team": r.get("Team"),
                    "opponent": r.get("opponent"),
                    "game": r.get("game"),
                    "salary": int(r["salary"]),
                    "projected_points": round(float(r["projected_points"]), 2),
                    "floor": _opt(r.get("floor")),
                    "ceiling": _opt(r.get("ceiling")),
                    "value": float(r["value"]),
                    "fppg": _opt(r.get("fppg")),
                    "value_vs_fppg": _opt(r.get("value_vs_fppg")),
                    "tier_rank": int(r["tier_rank"]),
                    "position_rank": int(r["position_rank"]),
                    "injury_flag": r.get("injury_flag") or None,
                    # What he has actually scored. Reading material, not a
                    # band: see dfs_targets.attach_history.
                    "history": ({"games": int(r["hist_games"]),
                                 "low": round(float(r["hist_low"]), 1),
                                 "high": round(float(r["hist_high"]), 1),
                                 "median": round(float(r["hist_median"]), 1)}
                                if pd.notna(r.get("hist_games")) else None),
                    "summary": targets.summarize(r, ctx),
                    "context": ctx or None,
                })
            groups.append({"position": pos, "tier": tier,
                           "band": _band_text(tier, premium_min, value_max),
                           "count": int(len(sel)), "players": players})

    missing_positions = [p for p in wanted
                         if not any(g["position"] == p for g in groups)]

    return {
        "status": "success",
        "season": season,
        "week": week,
        "slate": {"slate_id": slate_row.id, "contest": slate_row.contest,
                  "filename": slate_row.filename,
                  "uploaded_at": slate_row.uploaded_at.isoformat()
                                 if slate_row.uploaded_at else None,
                  "games_in_file": len(all_games),
                  "games_used": chosen or [str(g) for g in all_games],
                  "players": int(len(slate))},
        "slate_note": (
            "A FanDuel week file holds every game, not the main slate. "
            f"This read used all {len(all_games)} of them — pass `games=` to "
            "restrict it, or a lineup can be built around someone who is not "
            "in the contest." if chosen is None else
            f"Restricted to {len(chosen)} game(s) as asked."),
        "provenance": _run_block(db, season, week),
        "ranked_by": "our projected points per $1,000 of salary",
        "context_only": (
            "Opponent grades under each player's `context` did not affect any "
            "ordering here. Our own tests found matchup added nothing over "
            "role and volume, so it is shown for a human to break ties with, "
            "not used to rank."),
        "no_vegas": ("Implied totals and spreads are deliberately absent: a "
                     "line may be a benchmark but never an input."),
        "excluded_injury_flags": list(targets.UNROSTERABLE),
        # A position asked for that came back with no group at all. DEF is the
        # standing case and it is not a data gap: the model does not project
        # defenses, so that slot runs on FanDuel's own FPPG and there is
        # nothing here to tier. Returning ten groups where eleven were asked
        # for, without a word, read as a missing week.
        "positions_requested": list(wanted),
        "positions_absent": missing_positions,
        "positions_absent_note": (
            None if not missing_positions else
            ("No group for " + ", ".join(missing_positions) + ". "
             + ("DEF is not projected by the model at all -- the optimizer "
                "fills that slot from FanDuel's own FPPG on the uploaded "
                "slate, so there is no projection here to rank. "
                if "D" in missing_positions else "")
             + "Anything else absent means the board and the slate share no "
               "player at that position.")),
        "history_note": (
            "Each player's `history` is the lowest, highest and median he has "
            "actually scored this season. It is description, not a band: "
            "measured on 2026 weeks 5-9 (n=617), a low-to-high band covers "
            "71.0% of a player's next result against the model band's 81.7%, "
            "and both ends drift outward all season as the sample grows "
            "rather than as the player changes. It never enters a projection, "
            "a value or a rank." if history else
            "History was not attached (`history=false`)."),
        "unmatched_on_slate": unmatched[:25],
        "unmatched_count": len(unmatched),
        "groups": groups,
    }


def _opt(v):
    return None if v is None or pd.isna(v) else round(float(v), 2)


def _band_text(tier: str, premium_min: int, value_max: int) -> str:
    if tier == "premium":
        return f"${premium_min:,} and up"
    if tier == "value":
        return f"under ${value_max:,}"
    return f"${value_max:,}–${premium_min - 1:,}"
