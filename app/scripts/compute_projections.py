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


def run(db, season: int, week: int, epochs: int, job) -> None:
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

    logger.info("Projecting season %d week %d...", season, week)
    frame = svc.project(season, week, as_frame=True)
    if frame is None or frame.empty:
        _update_job(db, job, status="completed", total_entries=0, processed_entries=0)
        logger.warning("No projections produced for %d week %d", season, week)
        return

    model_version = getattr(nfl_projections, "__version__", "unknown")
    _update_job(db, job, total_entries=len(frame))

    # Replace any existing rows for this season/week
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
            season=season,
            week=week,
            player_id=pid,
            player_name=str(r.get("player_name") or ""),
            position=str(r.get("position") or ""),
            team=str(r.get("team") or ""),
            projected_points=_f(r.get("fanduel_fantasy_points")),
            floor=_f(r.get("floor")),
            median=_f(r.get("projection_median")),
            ceiling=_f(r.get("ceiling")),
            prediction_type=str(r.get("prediction_type") or ""),
            model_version=model_version,
            computed_at=datetime.utcnow(),
        ))
        written += 1
    db.commit()

    _update_job(db, job, status="completed", processed_entries=written)
    logger.info("Wrote %d projections for %d week %d (model %s)",
                written, season, week, model_version)


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

        run(db, season, week, args.epochs, job)

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
