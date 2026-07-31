#!/usr/bin/env python3
"""
Pre-compute weekly fantasy projections and cache them in `player_projections`.

Self-contained: builds the historical dataset fresh from nflreadpy via the
nfl_projections package, trains the model (mean + quantile floor/ceiling), and
projects the requested week. Progress is tracked in analytics_job_status
(job_type="projections") and served via GET /projections/status.

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
    (for history lookup); depth is a minimal frame (all starters) — enough for the
    preseason path where injuries/depth aren't used."""
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
    depth = pd.DataFrame([{
        "team": r.team, "pos_abb": r.position, "pos_rank": 1,
        "player_name": r.full_name, "player_id": r.gsis_id,
    } for r in rows])
    return rosters, depth


def run(db, season: int, week: int, epochs: int, job, end_week: int = None) -> None:
    from database.models import PlayerProjection
    import nfl_projections
    from nfl_projections import ProjectionService
    from nfl_projections import dataset as nflp_dataset

    _update_job(db, job, status="running", current_season=season,
                current_coach=f"season {season} week {week}")

    logger.info("Building dataset from nflreadpy...")
    df = nflp_dataset.build_dataset(output_path=None)  # build in-memory, don't write a CSV
    logger.info("Training model (mean + quantile, epochs=%d)...", epochs)
    svc = ProjectionService(dataset=df, quantiles=True, epochs=epochs)

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

    if not use_espn:
        # in-season path (nflreadpy rosters): single matchup-neutral week, as before
        written = _write_week(db, base, season, week, model_version)
        _update_job(db, job, status="completed", total_entries=written, processed_entries=written)
        logger.info("Wrote %d projections for %d week %d (model %s)", written, season, week, model_version)
        return

    # Future-season path: the base projection is matchup-neutral (same every week for a
    # preseason projection), so train/project once and apply EACH week's game environment
    # + simulator to produce a matchup-varying full-season outlook.
    env_all = _game_environments(season)
    total = 0
    for w in range(week, (end_week or week) + 1):
        f = _apply_environment(base, w, env_all)     # per-week env; drops bye teams
        if f is None or f.empty:
            continue
        _apply_simulator(f, df, w)
        total += _write_week(db, f, season, w, model_version)
        _update_job(db, job, current_coach=f"season {season} week {w}", processed_entries=total)
        logger.info("week %d: wrote %d projections", w, len(f))
    _update_job(db, job, status="completed", total_entries=total, processed_entries=total)
    logger.info("Wrote %d total projections for %d weeks %d-%d (model %s)",
                total, season, week, end_week or week, model_version)


def _write_week(db, frame, season: int, week: int, model_version: str) -> int:
    """Replace the season/week rows in player_projections with `frame`."""
    from database.models import PlayerProjection
    db.query(PlayerProjection).filter(
        PlayerProjection.season == season, PlayerProjection.week == week
    ).delete()
    db.commit()
    written = 0
    for _, r in frame.iterrows():
        pid = str(r.get("player_id") or "").strip()
        if not pid:
            continue
        db.merge(PlayerProjection(
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
            prediction_type=str(r.get("prediction_type") or ""),
            model_version=model_version, computed_at=datetime.utcnow(),
        ))
        written += 1
    db.commit()
    return written


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
        return frame
    env = env_all[env_all["week"] == week].set_index("team")["env_mult"].to_dict()
    if not env:
        return frame
    f = frame[frame["team"].isin(env)].copy()   # drop bye-week teams
    mult = f["team"].map(env).fillna(1.0)
    # scale fantasy points AND the volume/scoring components so they stay consistent
    for col in ("fanduel_fantasy_points", "passing_yards", "passing_tds", "rushing_yards",
                "rushing_tds", "receiving_yards", "receptions", "receiving_tds"):
        if col in f.columns:
            f[col] = f[col].astype(float) * mult
    return f


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
                        help="Project --week through --end-week (future-season full-season run)")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    season = args.season or _current_nfl_season()
    week = args.week or _current_week(season)
    logger.info("Projections job: season %d week %d", season, week)

    from database.session import engine, SessionLocal
    from database.models import Base, AnalyticsJobStatus

    Base.metadata.create_all(engine)
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

        run(db, season, week, args.epochs, job, end_week=args.end_week)

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
