#!/usr/bin/env python3
"""Score stored projections against what actually happened.

Every backtest number the model has (~4.2 MAE, ~0.05 better than a trailing-5
average) comes from re-running history. This job builds the other kind of
evidence: projections that were computed BEFORE kickoff, scored once the week's
actuals land, accumulating forever in `projection_accuracy`.

Two rules keep the record honest:
  * Rows are frozen. `player_projections` is keyed (season, week, player_id) and
    a later projections run overwrites it, so once a player-week is scored it is
    never rewritten (pass --rescore to override deliberately).
  * Only player-weeks with a real stat line are scored — a player who didn't
    play isn't a 0-point miss, and the backtest excludes them too, so the two
    numbers stay comparable.

Each row also stores the naive trailing-5-game average, so accuracy queries can
report the skill-over-naive margin rather than a bare MAE.

Usage (from app/):
    python -m scripts.score_projections                    # every unscored complete week
    python -m scripts.score_projections --season 2025 --week 3
    python -m scripts.score_projections --season 2025 --rescore
"""
import argparse
import logging
import os
import sys
from datetime import datetime

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("score_projections")

NAIVE_WINDOW = 5   # games in the trailing-average baseline
_CHUNK = 400       # player ids per IN (...) clause; SQLite caps bound parameters


def _fp(stat) -> float:
    from api.utils import fanduel_points

    return fanduel_points({c.name: getattr(stat, c.name)
                           for c in stat.__table__.columns})


def _regular(query, model):
    """Regular-season rows only (season_type is null in some older loads)."""
    from sqlalchemy import or_

    return query.filter(or_(model.season_type == "REG", model.season_type.is_(None)))


def _gamedays(db, season: int, week: int) -> dict:
    """{team: date the game was played} for one week, from the schedule."""
    from database.models import Schedule

    out = {}
    rows = db.query(Schedule).filter(
        Schedule.season == season, Schedule.week == week
    ).all()
    for row in rows:
        if not row.gameday:
            continue
        try:
            day = datetime.strptime(str(row.gameday)[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        out[row.home_team] = day
        out[row.away_team] = day
    return out


def _naive_baselines(db, season: int, week: int, player_ids) -> dict:
    """Trailing-NAIVE_WINDOW-game FanDuel average per player, from games strictly
    before (season, week). Falls back into the prior season so week 1 isn't blank."""
    from sqlalchemy import and_, or_

    from database.models import PlayerStat

    history = {}
    ids = list(player_ids)
    for i in range(0, len(ids), _CHUNK):
        chunk = ids[i:i + _CHUNK]
        q = db.query(PlayerStat).filter(
            PlayerStat.player_id.in_(chunk),
            or_(
                and_(PlayerStat.season == season, PlayerStat.week < week),
                PlayerStat.season == season - 1,
            ),
        )
        for row in _regular(q, PlayerStat).all():
            history.setdefault(row.player_id, []).append(row)

    out = {}
    for player_id, rows in history.items():
        rows.sort(key=lambda r: (r.season, r.week))
        recent = rows[-NAIVE_WINDOW:]
        if recent:
            out[player_id] = sum(_fp(r) for r in recent) / len(recent)
    return out


def score_week(db, season: int, week: int, rescore: bool = False) -> dict:
    """Score one week. Returns a summary dict; commits what it writes."""
    from database.models import PlayerProjection, PlayerStat, ProjectionAccuracy

    projections = db.query(PlayerProjection).filter(
        PlayerProjection.season == season, PlayerProjection.week == week
    ).all()
    if not projections:
        return {"season": season, "week": week, "status": "no_projections", "scored": 0}

    stats_q = db.query(PlayerStat).filter(
        PlayerStat.season == season, PlayerStat.week == week
    )
    actuals = {s.player_id: s for s in _regular(stats_q, PlayerStat).all()}
    if not actuals:
        return {"season": season, "week": week, "status": "no_actuals", "scored": 0}

    existing = {
        pid for (pid,) in db.query(ProjectionAccuracy.player_id).filter(
            ProjectionAccuracy.season == season, ProjectionAccuracy.week == week
        ).all()
    }

    todo = [p for p in projections if rescore or p.player_id not in existing]
    if rescore and existing:
        db.query(ProjectionAccuracy).filter(
            ProjectionAccuracy.season == season, ProjectionAccuracy.week == week
        ).delete()
        db.commit()

    naive = _naive_baselines(db, season, week, {p.player_id for p in todo})
    gamedays = _gamedays(db, season, week)

    scored, no_actual, skipped = 0, 0, len(projections) - len(todo)
    after_kickoff = 0
    for proj in todo:
        stat = actuals.get(proj.player_id)
        if stat is None:
            no_actual += 1        # didn't play: not a miss, not scored
            continue
        if proj.projected_points is None:
            continue

        # The point of this table is that we said it BEFORE the game. A
        # projection recomputed after kickoff (a manual re-run, say) would
        # flatter the record, so leave it out rather than record it.
        gameday = gamedays.get(proj.team)
        if gameday and proj.computed_at and proj.computed_at.date() > gameday:
            after_kickoff += 1
            continue

        actual = _fp(stat)
        error = float(proj.projected_points) - actual
        naive_points = naive.get(proj.player_id)
        in_band = None
        if proj.floor is not None and proj.ceiling is not None:
            in_band = int(proj.floor <= actual <= proj.ceiling)

        db.add(ProjectionAccuracy(
            season=season, week=week, player_id=proj.player_id,
            player_name=proj.player_name, position=proj.position, team=proj.team,
            projected_points=proj.projected_points, floor=proj.floor,
            median=proj.median, ceiling=proj.ceiling,
            actual_points=round(actual, 2),
            error=round(error, 2), abs_error=round(abs(error), 2), in_band=in_band,
            naive_points=round(naive_points, 2) if naive_points is not None else None,
            naive_abs_error=(round(abs(naive_points - actual), 2)
                             if naive_points is not None else None),
            prediction_type=proj.prediction_type, model_version=proj.model_version,
            projected_at=proj.computed_at, scored_at=datetime.utcnow(),
        ))
        scored += 1

    db.commit()

    summary = {
        "season": season, "week": week, "status": "scored", "scored": scored,
        "already_scored": skipped, "projected_but_did_not_play": no_actual,
        "computed_after_kickoff": after_kickoff,
    }
    if scored:
        rows = db.query(ProjectionAccuracy).filter(
            ProjectionAccuracy.season == season, ProjectionAccuracy.week == week
        ).all()
        summary["mae"] = round(sum(r.abs_error for r in rows) / len(rows), 3)
        naive_rows = [r for r in rows if r.naive_abs_error is not None]
        if naive_rows:
            summary["naive_mae"] = round(
                sum(r.naive_abs_error for r in naive_rows) / len(naive_rows), 3)
    return summary


def scorable_weeks(db, season: int):
    """Weeks with both projections and actuals, ordered."""
    from database.models import PlayerProjection, PlayerStat

    projected = {w for (w,) in db.query(PlayerProjection.week).filter(
        PlayerProjection.season == season).distinct().all()}
    played = {w for (w,) in db.query(PlayerStat.week).filter(
        PlayerStat.season == season).distinct().all()}
    return sorted(projected & played)


def run(db, season: int, week: int = None, rescore: bool = False) -> list:
    from database.models import AnalyticsJobStatus

    weeks = [week] if week is not None else scorable_weeks(db, season)
    if not weeks:
        logger.warning("No week of %s has both projections and actuals yet", season)

    job = AnalyticsJobStatus(
        job_type="projection_accuracy", status="running",
        started_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        total_entries=len(weeks), processed_entries=0, skipped_entries=0,
        failed_entries=0, current_season=season,
    )
    db.add(job)
    db.commit()

    summaries = []
    try:
        for wk in weeks:
            summary = score_week(db, season, wk, rescore=rescore)
            summaries.append(summary)
            logger.info("week %s: %s", wk, summary)
            job.processed_entries = (job.processed_entries or 0) + summary["scored"]
            job.skipped_entries = (job.skipped_entries or 0) + summary.get("already_scored", 0)
            job.updated_at = datetime.utcnow()
            db.commit()
        job.status = "completed"
    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)[:500]
        raise
    finally:
        job.updated_at = datetime.utcnow()
        db.commit()

    total = sum(s["scored"] for s in summaries)
    logger.info("Scored %d player-week(s) across %d week(s) of %s",
                total, len(summaries), season)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score stored projections against actual results")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None,
                        help="Single week (default: every week with projections + actuals)")
    parser.add_argument("--rescore", action="store_true",
                        help="Overwrite rows already scored (off by default: the "
                             "record is frozen so re-run projections can't rewrite history)")
    args = parser.parse_args()

    from api.utils import get_current_nfl_season
    from database.models import Base
    from database.session import SessionLocal, engine

    season = args.season or get_current_nfl_season()
    Base.metadata.create_all(engine)   # creates projection_accuracy on first run

    db = SessionLocal()
    try:
        run(db, season, week=args.week, rescore=args.rescore)
    except Exception as exc:
        logger.error("Scoring job failed: %s", exc, exc_info=True)
        raise SystemExit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
