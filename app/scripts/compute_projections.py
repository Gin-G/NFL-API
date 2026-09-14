#!/usr/bin/env python3
"""
Pre-compute weekly fantasy projections and cache them in `player_projections`.

Self-contained: builds the historical dataset fresh from nflreadpy via the
nfl_projections package, trains the model (mean + quantile floor/ceiling), and
projects the requested week THROUGH the end of the regular season. Progress is
tracked in analytics_job_status (job_type="projections") and served via
GET /projections/status.

Two tables come out of every run:

  player_projections          one row per (season, week, player) — the current
                              best answer, replaced each time a week is
                              reprojected. This is what the API and the betting
                              board read.
  player_projection_vintages  the same rows, plus `as_of_week`: how many weeks
                              were complete when they were computed. Never
                              deleted, so a week accumulates one row per vintage
                              and the season ends holding every projection it
                              ever had.

Training happens once per run, so projecting the remaining 17 weeks instead of
1 is nearly free — the extra weeks are the same mean re-scaled by each week's
game environment.

Only the upcoming week is PUBLISHED to player_projections; the rest are
archived only. That keeps this job additive: every number the API and the
betting board already serve is exactly what it was, and the later weeks
accumulate as evidence until the accuracy record says which vintage deserves
to be the live one.

Run:
    python -m scripts.compute_projections --season 2025 --week 3
    python -m scripts.compute_projections            # current season + week

Requires the nfl_projections package (installed from git in the job image):
    pip install "nfl-projections @ git+https://github.com/Gin-G/nfl-data-py.git"
"""

import argparse
import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compute_projections")


def _current_nfl_season() -> int:
    now = datetime.now()
    return now.year if now.month >= 9 else now.year - 1


def _current_week(season: int) -> int:
    """Best-guess upcoming week: the earliest week with a game today or later,
    else the last scheduled week. Falls back to 1 on any error."""
    try:
        import nflreadpy as nfl
        import pandas as pd

        sched = nfl.load_schedules(seasons=[season])
        sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
        sched = sched[sched["game_type"] == "REG"].copy()
        sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
        today = pd.Timestamp.now().normalize()
        upcoming = sched[sched["gameday"] >= today]
        if not upcoming.empty:
            return int(upcoming["week"].min())
        return int(sched["week"].max())
    except Exception as exc:
        logger.warning("Could not determine current week (%s); defaulting to 1", exc)
        return 1


def _update_job(db, job, **kwargs):
    for k, v in kwargs.items():
        setattr(job, k, v)
    job.updated_at = datetime.utcnow()
    db.merge(job)
    db.commit()


def _espn_frames(db, week: int):
    """Build (rosters, depth_charts) frames for the Projector from the ESPN roster sync,
    for seasons nflreadpy hasn't published rosters for yet. Rosters carry gsis player_id
    (for history lookup); depth carries the real ESPN depth rank (1=starter, 2=backup, ...)
    so the projection engine applies its depth-role discount (backup QBs, committee RBs)."""
    import pandas as pd
    from database.models import EspnRoster
    rows = db.query(EspnRoster).filter(
        EspnRoster.status == "active",
        EspnRoster.position.in_(["QB", "RB", "WR", "TE"]),
        EspnRoster.gsis_id.isnot(None),
    ).all()
    rosters = pd.DataFrame([{
        "player_id": r.gsis_id, "player_name": r.full_name, "full_name": r.full_name,
        "position": r.position, "team": r.team, "week": week,
        "status": "ACT",  # Projector skips non-"ACT"; we already filtered to active
    } for r in rows])
    # Null depth_rank (player not on ESPN's depth chart) -> treat as a mid-roster backup,
    # matching nfl_projections.roles._UNKNOWN_RANK (3).
    depth = pd.DataFrame([{
        "team": r.team, "pos_abb": r.position,
        "pos_rank": int(r.depth_rank) if r.depth_rank else 3,
        "player_name": r.full_name, "player_id": r.gsis_id,
    } for r in rows])
    return rosters, depth


def run(db, season: int, week: int, epochs: int, job, end_week: int = None,
        seeds: int = None) -> None:
    from database.models import PlayerProjection
    import nfl_projections
    from nfl_projections import ProjectionService
    from nfl_projections import dataset as nflp_dataset

    _update_job(db, job, status="running", current_season=season,
                current_coach=f"season {season} week {week}")

    logger.info("Building dataset from nflreadpy...")
    df = nflp_dataset.build_dataset(output_path=None)  # build in-memory, don't write a CSV
    # The mean model is a seed ensemble (nfl_projections default 5): averaging
    # several seeds is worth ~0.04 MAE and removes the single-seed lottery, at
    # the cost of training that many networks.
    logger.info("Training model (mean ensemble x%s + quantile, epochs=%d)...",
                seeds if seeds else "default", epochs)
    svc = ProjectionService(dataset=df, quantiles=True, epochs=epochs, n_seeds=seeds)

    # nflreadpy publishes rosters only through the prior season; for a future season
    # (e.g. 2026 preseason) use the nightly ESPN roster sync + rookie draft-capital prior,
    # and derive floor/ceiling from the Monte-Carlo simulator (no current-season form yet).
    max_nflreadpy_season = int(df["season"].max())
    use_espn = season > max_nflreadpy_season
    proj_kwargs = {}
    if use_espn:
        rosters, depth = _espn_frames(db, week)
        if rosters.empty:
            _update_job(db, job, status="failed",
                        error_message="no ESPN roster rows; run the roster sync first")
            logger.error("No ESPN rosters for season %d; run nfl-api-roster-sync first", season)
            return
        logger.info("Using ESPN rosters (%d players) + rookie prior for season %d",
                    len(rosters), season)
        proj_kwargs = dict(rosters=rosters, depth_charts=depth,
                           rookie_fallback=True, use_injuries=False)

    logger.info("Projecting season %d week %d...", season, week)
    base = svc.project(season, week, as_frame=True, **proj_kwargs)
    if base is None or base.empty:
        _update_job(db, job, status="completed", total_entries=0, processed_entries=0)
        logger.warning("No projections produced for %d week %d", season, week)
        return
    model_version = getattr(nfl_projections, "__version__", "unknown")

    # How much current-season football this projection was allowed to see. Stamped
    # on every row written below, and the axis the vintage archive is keyed on.
    as_of = _completed_weeks(season)
    # Capture anything already in player_projections that predates the archive,
    # before the writes below replace it. Must happen before the first _write_week.
    _backfill_vintages(db, season)

    if not use_espn:
        # In-season path (nflreadpy rosters). The base projection is matchup-neutral
        # and reflects form through week `as_of`; applying each remaining week's game
        # environment turns it into an outlook for the rest of the season. Doing that
        # every week is what builds the triangle: 18 weeks projected preseason, 17
        # after week 1, and so on, each vintage kept for comparison.
        #
        # Only the upcoming week is PUBLISHED to player_projections. The later weeks
        # are archived only — see _write_week — so this job adds a record without
        # changing a single number the board or the season endpoint already serves.
        last = end_week if end_week is not None else _final_week(season)
        env_all = _game_environments(season)
        written = archived = 0
        for w in range(week, max(last, week) + 1):
            f = base if w == week else _apply_environment(base, w, env_all)
            if f is None or getattr(f, "empty", False):
                continue        # bye week, or no environment for that week
            if w != week:
                # Re-roll the range for the target week; the mean already moved.
                _apply_simulator(f, df, w)
            n = _write_week(db, f, season, w, model_version, as_of_week=as_of,
                            live=(w == week))
            archived += n
            if w == week:
                written = n
            _update_job(db, job, current_coach=f"season {season} week {w}",
                        processed_entries=archived)
        _update_job(db, job, status="completed", total_entries=written,
                    processed_entries=archived)
        logger.info("Published %d projections for %d week %d; archived %d across "
                    "weeks %d-%d as_of week %d (model %s)",
                    written, season, week, archived, week, last, as_of, model_version)
        return

    # Future-season path: the base projection is matchup-neutral (same every week for a
    # preseason projection), so train/project once and apply EACH week's game environment
    # + simulator to produce a matchup-varying full-season outlook.
    env_all = _game_environments(season)
    budgets = _position_budgets(df)
    share_pred = _predict_shares(df, depth, season)
    games_model, prev_games = _fit_games(df, season)
    total = 0
    for w in range(week, (end_week or week) + 1):
        f = _apply_environment(base, w, env_all)     # per-week env; drops bye teams
        if f is None or f.empty:
            continue
        _apply_roles(f, budgets, games_model, prev_games)  # availability (durability) + team pool
        _apply_shares(f, share_pred)                  # redistribute group total by predicted share
        _apply_simulator(f, df, w)
        total += _write_week(db, f, season, w, model_version, as_of_week=as_of)
        _update_job(db, job, current_coach=f"season {season} week {w}", processed_entries=total)
        logger.info("week %d: wrote %d projections", w, len(f))
    _update_job(db, job, status="completed", total_entries=total, processed_entries=total)
    logger.info("Wrote %d total projections for %d weeks %d-%d (model %s)",
                total, season, week, end_week or week, model_version)


def _write_week(db, frame, season: int, week: int, model_version: str,
                as_of_week: int = 0, live: bool = True) -> int:
    """Archive `frame` as the `as_of_week` vintage, and when `live`, also make it
    the current projection for that week.

    Two destinations, deliberately. `player_projections` is the current best
    answer and is replaced; `player_projection_vintages` is the permanent record
    and is only ever merged, so reprojecting week 10 in week 5 leaves week 10's
    preseason and week-1..4 vintages exactly where they were.

    `live=False` archives without publishing. The in-season run uses it for
    every week after the upcoming one: those extrapolations are worth recording
    and comparing, but they are built without the availability and share models
    (both need the preseason ESPN depth charts), so promoting them over the
    preseason full-season projection would quietly strip the availability
    weighting out of the season-totals endpoint. Which projection actually
    deserves to be live for a distant week is a question the archive is being
    built to answer — see GET /projections/vintages/accuracy — rather than one
    to guess at here.
    """
    from database.models import PlayerProjection, PlayerProjectionVintage
    if live:
        db.query(PlayerProjection).filter(
            PlayerProjection.season == season, PlayerProjection.week == week
        ).delete()
        db.commit()
    written = 0
    for _, r in frame.iterrows():
        pid = str(r.get("player_id") or "").strip()
        if not pid:
            continue
        payload = dict(
            season=season, week=week, player_id=pid,
            player_name=str(r.get("player_name") or ""),
            position=str(r.get("position") or ""),
            team=str(r.get("team") or ""),
            projected_points=_f(r.get("fanduel_fantasy_points")),
            floor=_f(r.get("floor")), median=_f(r.get("projection_median")),
            ceiling=_f(r.get("ceiling")),
            passing_yards=_f(r.get("passing_yards")),
            passing_tds=_f(r.get("passing_tds")),
            passing_interceptions=_f(r.get("passing_interceptions")),
            rushing_yards=_f(r.get("rushing_yards")),
            rushing_tds=_f(r.get("rushing_tds")),
            receiving_yards=_f(r.get("receiving_yards")),
            receptions=_f(r.get("receptions")),
            receiving_tds=_f(r.get("receiving_tds")),
            # 1.0 on the in-season path, which applies no availability discount
            exp_games=_f(r.get("exp_games")) if r.get("exp_games") is not None else 1.0,
            prediction_type=str(r.get("prediction_type") or ""),
            model_version=model_version, computed_at=datetime.utcnow(),
        )
        if live:
            db.merge(PlayerProjection(**payload))
        db.merge(PlayerProjectionVintage(as_of_week=as_of_week, **payload))
        written += 1
    db.commit()
    return written


def _final_week(season: int) -> int:
    """Last scheduled regular-season week (18 in the current format, 17 before).

    Read from the schedule rather than hard-coded so a future format change does
    not silently truncate the projection horizon.
    """
    try:
        import nflreadpy as nfl

        sched = nfl.load_schedules(seasons=[season])
        sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
        return int(sched[sched["game_type"] == "REG"]["week"].max())
    except Exception as exc:
        logger.warning("Could not determine final week (%s); using 18", exc)
        return 18


def _completed_weeks(season: int, when=None) -> int:
    """Completed regular-season weeks of `season` as of `when` (default now).

    This is the `as_of_week` stamp: the amount of current-season football the
    projection was allowed to learn from. A week counts as complete once its
    LAST game has kicked off plus a few hours, so a Wednesday run sees the
    Monday nighter that just finished. Returns 0 preseason, and 0 on any error —
    understating what the model knew is the safe direction for an archive whose
    whole purpose is comparing vintages.
    """
    try:
        import nflreadpy as nfl
        import pandas as pd

        sched = nfl.load_schedules(seasons=[season])
        sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
        sched = sched[sched["game_type"] == "REG"].copy()
        sched["kick"] = pd.to_datetime(sched["gameday"], errors="coerce")
        cutoff = (pd.Timestamp(when) if when is not None
                  else pd.Timestamp.now()) - pd.Timedelta(hours=6)
        # A week is complete once its LAST game is in the books, so take the
        # latest kickoff per week and count the weeks entirely behind us. Using
        # the max (not the min) is what stops a Thursday game from marking the
        # whole week done.
        last = sched.groupby("week")["kick"].max().dropna()
        complete = [int(w) for w, k in last.items() if k < cutoff]
        return max(complete) if complete else 0
    except Exception as exc:
        logger.warning("Could not determine completed weeks (%s); using 0", exc)
        return 0


def _backfill_vintages(db, season: int) -> int:
    """Archive any `player_projections` row that has no vintage row yet.

    Everything written before this table existed is otherwise lost the first
    time its week is reprojected — including the preseason full-season outlook,
    which is the most interesting vintage in the whole archive precisely because
    it is the one built with the least information. Each row's vintage is
    recovered from its own `computed_at`, so the August batch lands at
    as_of_week=0 and a mid-season recompute lands where it belongs.

    Idempotent: only inserts (season, week, player_id, as_of_week) keys that are
    missing, so re-running it never disturbs an existing vintage.
    """
    from database.models import PlayerProjection, PlayerProjectionVintage

    rows = db.query(PlayerProjection).filter(
        PlayerProjection.season == season).all()
    if not rows:
        return 0
    # tuple(), not the Row objects themselves — a Row will not hash-match a
    # plain tuple, and the membership test below would silently never hit.
    have = {tuple(k) for k in db.query(
        PlayerProjectionVintage.week, PlayerProjectionVintage.player_id,
        PlayerProjectionVintage.as_of_week).filter(
            PlayerProjectionVintage.season == season).all()}

    # One schedule lookup per distinct computed_at, not per row.
    asof_cache: dict = {}
    added = 0
    for r in rows:
        stamp = r.computed_at or datetime.utcnow()
        bucket = stamp.replace(minute=0, second=0, microsecond=0)
        if bucket not in asof_cache:
            asof_cache[bucket] = _completed_weeks(season, stamp)
        as_of = asof_cache[bucket]
        if (r.week, r.player_id, as_of) in have:
            continue
        db.add(PlayerProjectionVintage(
            season=r.season, week=r.week, player_id=r.player_id, as_of_week=as_of,
            player_name=r.player_name, position=r.position, team=r.team,
            projected_points=r.projected_points, floor=r.floor,
            median=r.median, ceiling=r.ceiling,
            passing_yards=r.passing_yards, passing_tds=r.passing_tds,
            passing_interceptions=r.passing_interceptions,
            rushing_yards=r.rushing_yards, rushing_tds=r.rushing_tds,
            receiving_yards=r.receiving_yards, receptions=r.receptions,
            receiving_tds=r.receiving_tds, exp_games=r.exp_games,
            prediction_type=r.prediction_type, model_version=r.model_version,
            computed_at=r.computed_at,
        ))
        added += 1
    if added:
        db.commit()
        logger.info("Backfilled %d existing projections into the vintage archive", added)
    return added


def _game_environments(season: int):
    """All-week game-environment table (or None if unavailable)."""
    try:
        from nfl_projections import ratings as nflp_ratings
        return nflp_ratings.game_environments(season)
    except Exception as exc:
        logger.warning("game environment unavailable (%s); skipping", exc)
        return None


def _apply_environment(frame, week: int, env_all):
    """Return a copy of `frame` with each player's mean scaled by their game's
    scoring-environment multiplier for `week`, and players whose team is on BYE that week
    dropped (no game). Shootouts boost both teams; the simulator then builds the
    distribution around the adjusted mean, lifting ceilings in high-total games."""
    if env_all is None:
        return frame.copy()   # copy so later per-week mutations don't compound on `base`
    env = env_all[env_all["week"] == week].set_index("team")["env_mult"].to_dict()
    if not env:
        return frame.copy()
    f = frame[frame["team"].isin(env)].copy()   # drop bye-week teams
    mult = f["team"].map(env).fillna(1.0)
    # scale fantasy points AND the volume/scoring components so they stay consistent
    for col in ("fanduel_fantasy_points", "passing_yards", "passing_tds", "rushing_yards",
                "rushing_tds", "receiving_yards", "receptions", "receiving_tds"):
        if col in f.columns:
            f[col] = f[col].astype(float) * mult
    return f


_ROLE_SCALE_COLS = (
    "fanduel_fantasy_points", "passing_yards", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds", "receiving_yards", "receptions", "receiving_tds",
    "floor", "projection_median", "ceiling",
)


def _position_budgets(history):
    """Per-position team scoring pool for the finite-pool cap (from nfl_projections.roles)."""
    try:
        from nfl_projections import roles
        return roles.position_budgets(history)
    except Exception as exc:
        logger.warning("position budgets unavailable (%s); skipping team cap", exc)
        return None


def _fit_games(history, season: int):
    """Fit the availability model (games.py) + a prior-year games map. (None, None) if
    unavailable — callers then fall back to the flat depth-role games assumption."""
    try:
        from nfl_projections import games as nflp_games
        return nflp_games.fit_games_model(history, max_season=season - 1), \
            nflp_games.prev_games_map(history, season)
    except Exception as exc:
        logger.warning("games model unavailable (%s); using flat role games", exc)
        return None, None


def _apply_roles(frame, budgets, games_model=None, prev_games=None) -> None:
    """Depth-role corrections on the mean (in place), matching nfl_projections.season:
      * availability — scale each player's week by their expected GAMES. With the games
        model (games.py) an established starter's durability (regressed prior-year games)
        replaces the flat ~16.5; otherwise the depth-role baseline (a backup QB ~1-2 games).
      * finite team pool — cap each (team, position) group to a realistic per-game total
        so teammates SHARE (two RBs can't both project as bell cows).
    The per-game role RATE discount is already applied upstream via the ESPN depth rank.
    """
    try:
        from nfl_projections import roles
        from nfl_projections import games as nflp_games
    except Exception as exc:
        logger.warning("roles unavailable (%s); skipping depth-role corrections", exc)
        return
    if "depth_rank" not in frame.columns or "position" not in frame.columns:
        return
    scale_cols = [c for c in _ROLE_SCALE_COLS if c in frame.columns]

    def play_weight(r):
        pos, rank = r.get("position"), r.get("depth_rank")
        if games_model is not None:
            pg = (prev_games or {}).get(r.get("player_id"))
            return nflp_games.expected_games(pos, rank, pg, games_model) / 17.0
        return roles.play_weight(pos, rank)

    pw = frame.apply(play_weight, axis=1)
    for c in scale_cols:
        frame[c] = frame[c].astype(float) * pw
    # Keep the weight: summed over a season's weeks it is expected games played,
    # and without it the stored projection cannot be turned back into a per-game
    # rate downstream (the API used to report games=17 for everyone).
    frame["exp_games"] = pw
    if budgets:
        roles.apply_team_budget(frame, budgets, points_col="fanduel_fantasy_points",
                                group_cols=("team", "position"), scale_cols=scale_cols)


def _predict_shares(history, depth, season: int):
    """Validated share model (nfl_projections.shares): predicted carry/target share per player
    for the season, from ESPN depth ranks + prior-year usage. None if unavailable."""
    try:
        from nfl_projections import shares as nflp_shares
        sroster = depth.rename(columns={"pos_abb": "position", "pos_rank": "depth_rank"})[
            ["player_id", "team", "position", "depth_rank"]].copy()
        sroster = sroster[sroster["player_id"].astype(str).str.len() > 0]
        return nflp_shares.project_shares(sroster, history, season)
    except Exception as exc:
        logger.warning("share model unavailable (%s); skipping share allocation", exc)
        return None


def _apply_shares(frame, share_pred, blend: float = 0.2) -> None:
    """Redistribute each (team, position) group's projected fantasy TOTAL by a light blend of
    rolling-form weight and the validated share model's volume weight (RB = carry+0.5*target,
    WR/TE = target; QB stays on form), conserving the group total. Backtested modest net win
    concentrated on role-change cases (committees / vacated share). Modifies `frame` in place."""
    if share_pred is None or getattr(share_pred, "empty", True):
        return
    if not {"team", "position", "player_id", "fanduel_fantasy_points"} <= set(frame.columns):
        return
    import numpy as np

    sp = share_pred[["player_id", "team", "carry_share", "target_share"]].drop_duplicates(
        ["player_id", "team"])
    m = frame.merge(sp, on=["player_id", "team"], how="left")
    cs = m["carry_share"].fillna(0.0).to_numpy(float)
    ts = m["target_share"].fillna(0.0).to_numpy(float)
    pos = m["position"].to_numpy()
    vw = np.where(pos == "RB", cs + 0.5 * ts, np.where(np.isin(pos, ["WR", "TE"]), ts, np.nan))
    m["_vw"] = vw

    grp = m.groupby(["team", "position"])
    fsum = grp["fanduel_fantasy_points"].transform("sum").to_numpy(float)
    fp = m["fanduel_fantasy_points"].to_numpy(float)
    fw = np.where(fsum > 0, fp / fsum, 0.0)
    vws = grp["_vw"].transform("sum").to_numpy(float)
    use = np.isin(pos, ["RB", "WR", "TE"]) & (vws > 0)
    sw = np.where(use, np.where(vws > 0, np.nan_to_num(vw) / np.where(vws > 0, vws, 1.0), fw), fw)
    blended = blend * sw + (1 - blend) * fw
    scale = np.where(fw > 0, blended / np.where(fw > 0, fw, 1.0), 1.0)

    for col in _ROLE_SCALE_COLS:
        if col in frame.columns:
            frame[col] = frame[col].astype(float).to_numpy() * scale


def _apply_simulator(frame, history, week: int) -> None:
    """Replace floor/median/ceiling with Monte-Carlo simulator distributions (anchored to
    the model mean) for players with enough history — better early-season ranges than the
    quantile model when there's no current-season form. Modifies `frame` in place."""
    try:
        from nfl_projections import simulate as nflp_sim
    except Exception as exc:
        logger.warning("simulator unavailable (%s); keeping model quantiles", exc)
        return
    summ, _ = nflp_sim.project_distributions(frame, history, n_sims=1000, seed=week)
    if summ.empty:
        return
    sm = summ.set_index("player_id")
    n = 0
    for i, r in frame.iterrows():
        pid = r.get("player_id")
        if pid in sm.index:
            frame.at[i, "floor"] = float(sm.loc[pid, "floor"])
            frame.at[i, "projection_median"] = float(sm.loc[pid, "median"])
            frame.at[i, "ceiling"] = float(sm.loc[pid, "ceiling"])
            n += 1
    logger.info("Applied simulator distributions to %d players", n)


def _f(v):
    try:
        import math
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute weekly fantasy projections")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--end-week", type=int, default=None,
                        help="Project --week through --end-week. Defaults to the "
                             "last regular-season week, so a run reprojects the "
                             "whole remaining season; pass the same value as "
                             "--week for a single-week run.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=None,
                        help="Networks in the mean-model seed ensemble "
                             "(default: nfl_projections' 5; 1 for a fast run)")
    args = parser.parse_args()

    season = args.season or _current_nfl_season()
    week = args.week or _current_week(season)
    logger.info("Projections job: season %d week %d", season, week)

    from database.session import engine, SessionLocal
    from database.models import Base, AnalyticsJobStatus, apply_light_migrations

    Base.metadata.create_all(engine)
    apply_light_migrations(engine)
    db = SessionLocal()
    try:
        stuck = db.query(AnalyticsJobStatus).filter(
            AnalyticsJobStatus.status == "running",
            AnalyticsJobStatus.job_type == "projections",
        ).all()
        for s in stuck:
            s.status = "interrupted"
        if stuck:
            db.commit()

        job = AnalyticsJobStatus(
            job_type="projections",
            status="starting",
            started_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            total_entries=0, processed_entries=0, skipped_entries=0, failed_entries=0,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        run(db, season, week, args.epochs, job, end_week=args.end_week,
            seeds=args.seeds)

    except Exception as exc:
        logger.error("Projections job failed: %s", exc, exc_info=True)
        try:
            job.status = "failed"
            job.error_message = str(exc)
            job.updated_at = datetime.utcnow()
            db.merge(job)
            db.commit()
        except Exception:
            pass
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
