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
            "player_id": row.player_id,
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


def _consistency():
    """The research module, or a 503 explaining what is missing."""
    try:
        from nfl_projections import consistency

        return consistency
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"nfl_projections.consistency unavailable: {exc}")


def _weekly_frame(db, season: int) -> pd.DataFrame:
    """Every regular-season player-week we hold, in nflverse's own column names.

    Read from our own `player_stats` rather than nflverse directly, so a
    request does not depend on a network fetch -- the loader keeps this table
    current and the consistency read is then a database query like every other
    endpoint here.
    """
    rows = db.query(PlayerStat).filter(PlayerStat.season == season).all()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([
        {c.name: getattr(r, c.name) for c in r.__table__.columns}
        for r in rows])
    if "season_type" in frame.columns:
        frame = frame[frame.season_type.fillna("REG") == "REG"]
    # The package reads nflverse's spelling; our table keeps the API's.
    return frame.rename(columns={"recent_team": "team",
                                 "interceptions": "passing_interceptions"})


@router.get("/consistency")
def dfs_consistency(
    season: Optional[int] = Query(None),
    week: Optional[int] = Query(
        None, description="Only weeks BEFORE this one are counted, so the "
                          "table is usable before a slate rather than after."),
    positions: str = Query("QB,RB,WR,TE"),
    min_games: int = Query(4, ge=2, le=18),
    min_opportunities: float = Query(
        5.0, ge=0, description="Opportunities a game to be ranked. A player "
                               "touching the ball twice a week has a tiny "
                               "spread and is not what consistent means."),
    limit: int = Query(60, ge=1, le=300),
    slate_id: Optional[int] = Query(
        None, description="Join salaries from a saved slate, if one is stored."),
    db: Session = Depends(get_db),
):
    """Who gets the ball steadily — the only kind of consistency that measures.

    Ranked on **volume** steadiness, never on scoring spread, because scoring
    spread is not a property of a player. Split-half on 2025 (210 players, odd
    weeks against even) a player's fantasy-point standard deviation correlates
    +0.111 with his own other half, and **-0.047** once scoring level is
    controlled, CI [-0.181, +0.089]; on 2026 through week 9, -0.096. The same
    players' MEAN self-correlates +0.643, so this is not a shortage of signal
    in the data, it is an absence of the property.

    Raw standard deviation is worse than useless as a ranking, because
    dispersion scales with scoring: it correlates +0.647 with the mean, and
    2025's ten "steadiest" players by that measure averaged 5 to 6 points a
    game. Their scores do not move because they do not score.

    What does measure is usage. Opportunities per game self-correlate +0.943,
    their dispersion +0.734, and +0.412 even after controlling for the volume
    level itself, CI [+0.293, +0.518]. It transmits to the thing actually
    wanted, weakly but measurably: opportunity CV in one half predicts fantasy
    CV in the other at +0.231, CI [+0.099, +0.355].

    One honest caveat in-season: dispersion needs games. Across 2026's nine
    weeks the volume-dispersion reliability falls to +0.070, CI [-0.080,
    +0.217] — four or five games a half cannot measure a spread. Early in a
    season, read `games` before reading the rank.
    """
    consistency = _consistency()
    season = season or get_current_nfl_season()
    weekly = _weekly_frame(db, season)
    if weekly.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No player stats stored for {season}.")

    wanted = [p.strip().upper() for p in positions.split(",") if p.strip()]
    table = consistency.player_consistency(
        weekly, min_games=min_games, through_week=week)
    if table.empty:
        raise HTTPException(
            status_code=404,
            detail=(f"No player has {min_games} games in {season}"
                    + (f" before week {week}" if week else "") + "."))
    table = table[table.position.isin(wanted)]
    ranked = table[table.opp_per_game >= min_opportunities].copy()

    salaries, slate_matcher = {}, None
    if slate_id is not None or week is not None:
        slate = _slate_row(db, season, week, slate_id) if week else None
        if slate is not None:
            try:
                frame = pd.read_csv(io.StringIO(slate.content))
                # The slate is the "board" here: each ranked player is looked up
                # in it by name + team (names.Matcher), not by a bare name.
                from nfl_projections.names import Matcher

                slate_matcher = Matcher(frame, name="Nickname", team="Team",
                                        pos="Position")
                salaries = {i: s for i, s in zip(frame.index, frame["Salary"])}
            except Exception as exc:                  # pragma: no cover
                logger.warning("slate salaries unavailable: %s", exc)

    rows = []
    for _, r in ranked.nlargest(limit, "volume_steadiness").iterrows():
        hit = (slate_matcher.match(r.player_name, r.team, r.position)
               if salaries else None)
        salary = salaries.get(hit) if hit is not None else None
        rows.append({
            "player": r.player_name,
            "position": r.position,
            "team": r.team,
            "games": int(r.games),
            "salary": int(salary) if salary is not None and salary == salary
            else None,
            # Volume: the measurable half.
            "opportunities_per_game": r.opp_per_game,
            "opportunity_low": r.opp_low,
            "opportunity_high": r.opp_high,
            "opportunity_cv": r.opp_cv,
            "volume_steadiness": r.volume_steadiness,
            "steadiness_rank": int(r.steadiness_rank)
            if pd.notna(r.steadiness_rank) else None,
            "position_rank": int(r.position_rank)
            if pd.notna(r.position_rank) else None,
            # Trend: a steady 18 touches reads differently when it was 24 a
            # month ago.
            "opportunities_recent": r.opp_recent,
            "opportunity_trend": None if pd.isna(r.opp_trend) else r.opp_trend,
            # Scoring: description, and never part of the order above.
            "points_per_game": r.points_per_game,
            "points_low": r.points_low,
            "points_high": r.points_high,
            "points_median": r.points_median,
            "points_sd": r.points_sd,
            "points_cv": r.points_cv,
            "points_recent": r.points_recent,
            "hit_rate_10": r.hit_rate_10,
            "hit_rate_15": r.hit_rate_15,
            "summary": consistency.describe(r),
        })

    return {
        "status": "success",
        "season": season,
        "through_week": week,
        "count": len(rows),
        "players_considered": int(len(table)),
        "ranked_by": "volume steadiness — 100 x (1 - the coefficient of "
                     "variation of opportunities per game)",
        "why_not_scoring": (
            "Scoring spread is not a property of a player. Split-half on 2025 "
            "(n=210), a player's fantasy-point SD correlates +0.111 with his "
            "own other half and -0.047 once scoring level is controlled, CI "
            "[-0.181, +0.089]; the same players' mean self-correlates +0.643. "
            "Ranking on it would be ranking on last month's noise, and ranking "
            "on RAW SD is worse still: dispersion scales with scoring (+0.647 "
            "with the mean), so it returns the players who do not score."),
        "what_does_measure": (
            "Opportunities per game self-correlate +0.943 across halves and "
            "their dispersion +0.734, still +0.412 with the volume level "
            "controlled, CI [+0.293, +0.518]. Opportunity CV in one half "
            "predicts fantasy CV in the other at +0.231, CI [+0.099, +0.355] "
            "— the only route to steady scoring that measures."),
        "small_sample_warning": (
            "Dispersion needs games. Over 2026's nine weeks the "
            "volume-dispersion reliability falls to +0.070, CI [-0.080, "
            "+0.217], so early in a season read `games` before the rank."),
        "best_single_predictor": (
            "A player's scoring LEVEL predicts his future scoring consistency "
            "better than any spread does: fantasy mean against next-half "
            "fantasy CV is -0.338 (2026: -0.317) against +0.235 for past CV. "
            "The consistent player is mostly just the good player."),
        "min_opportunities": min_opportunities,
        "data": rows,
    }
