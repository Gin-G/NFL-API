#!/usr/bin/env python3
"""Freeze the week's matchup signals before kickoff, and score last week's.

Run twice a week alongside the projections job. Recording and scoring are in
one script because they are the same loop seen from either end: this week gets
a snapshot, and any earlier week whose games are finished gets its outcomes.

The whole point is that nothing here feeds a projection. See
`nfl_projections.signals` and `database.models.MatchupSignal` for why a
borderline matchup effect gets a prospective record instead of a coefficient.
"""
import logging
import sys
from datetime import datetime

from sqlalchemy import func

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("record_signals")


def _board(db, season, week):
    from database.models import PlayerProjection

    rows = db.query(PlayerProjection).filter(
        PlayerProjection.season == season, PlayerProjection.week == week).all()
    import pandas as pd

    return pd.DataFrame([{
        "season": r.season, "week": r.week, "player_id": r.player_id,
        "player_name": r.player_name, "position": r.position, "team": r.team,
        "projected": r.projected_points,
    } for r in rows if r.position in ("QB", "RB", "WR", "TE")])


def _trenches(season, week):
    """Per-team run/pass matchup, or an empty frame if it cannot be had."""
    try:
        import nflreadpy as nfl
        from nfl_projections import trenches
        from nfl_projections.utils import to_pandas

        pbp = None
        rush, drop = trenches.plays(seasons=[season - 1, season], pbp=pbp)[:2]
        grades = trenches.ratings_through(rush, drop, season, week)
        schedule = to_pandas(nfl.load_schedules(seasons=[season]))
        return trenches.matchups_for_week(grades, schedule, season, week)
    except Exception as exc:                       # pragma: no cover - network
        logger.warning("no trench matchups for %s wk%s (%s)", season, week, exc)
        import pandas as pd
        return pd.DataFrame()


def _coverage(season, week):
    try:
        import nflreadpy as nfl
        from nfl_projections import coverage
        from nfl_projections.utils import to_pandas

        charting = coverage.load_charting([season - 1, season])
        weeks = coverage.corner_weeks(charting)
        grades = coverage.corner_grades(weeks, season, week)
        schedule = to_pandas(nfl.load_schedules(seasons=[season]))
        return coverage.matchups_for_week(coverage.team_coverage(grades),
                                          schedule, season, week)
    except Exception as exc:                       # pragma: no cover - network
        logger.warning("no coverage matchups for %s wk%s (%s)", season, week, exc)
        import pandas as pd
        return pd.DataFrame()


def record(db, season, week):
    """Snapshot one week. Idempotent: a re-run refreshes an unscored week."""
    from nfl_projections import signals
    from database.models import MatchupSignal

    board = _board(db, season, week)
    if board.empty:
        logger.info("no board for %s wk%s yet; nothing to record", season, week)
        return 0
    snap = signals.snapshot(board, _trenches(season, week), _coverage(season, week))

    existing = {s.player_id: s for s in db.query(MatchupSignal).filter_by(
        season=season, week=week).all()}
    written = 0
    for row in snap.itertuples(index=False):
        current = existing.get(row.player_id)
        if current is not None and current.scored_at is not None:
            continue                 # a scored week is history, not a draft
        values = dict(
            player_name=row.player_name, position=row.position, team=row.team,
            opponent=row.opponent, projected=_f(row.projected),
            run_matchup=_f(row.run_matchup), pass_matchup=_f(row.pass_matchup),
            opp_cover_grade=_f(row.opp_cover_grade), run_lean=row.run_lean,
            rosterable=int(bool(row.rosterable)), recorded_at=datetime.utcnow())
        if current is None:
            db.add(MatchupSignal(season=season, week=week,
                                 player_id=row.player_id, **values))
        else:
            for key, value in values.items():
                setattr(current, key, value)
        written += 1
    db.commit()
    logger.info("recorded %d signals for %s wk%s", written, season, week)
    return written


def _f(value):
    try:
        import math
        value = float(value)
        return None if math.isnan(value) else value
    except (TypeError, ValueError):
        return None


def score(db, season, week):
    """Attach outcomes to a recorded week once its games have stats."""
    from database.models import MatchupSignal

    pending = db.query(MatchupSignal).filter_by(season=season, week=week,
                                                scored_at=None).all()
    if not pending:
        return 0
    try:
        import nflreadpy as nfl
        from nfl_projections import scoring
        from nfl_projections.utils import to_pandas

        stats = to_pandas(nfl.load_player_stats(seasons=[season]))
        stats = stats[(stats.get("season_type", "REG") == "REG")
                      & (stats["week"] == week)]
    except Exception as exc:                       # pragma: no cover - network
        logger.warning("cannot score %s wk%s (%s)", season, week, exc)
        return 0
    if stats.empty:
        logger.info("no stats for %s wk%s yet", season, week)
        return 0

    # The package's one FanDuel rule, so a signal's actual sits on the same
    # scale as the board it is measured against.
    points = scoring.fanduel_points_from_stats(stats)
    actual = dict(zip(stats["player_id"], points))

    now, done = datetime.utcnow(), 0
    for row in pending:
        if row.player_id in actual:
            row.actual = float(actual[row.player_id])
            row.scored_at = now
            done += 1
    db.commit()
    logger.info("scored %d of %d signals for %s wk%s", done, len(pending), season, week)
    return done


def main(argv=None):
    from api.utils import get_current_nfl_season
    from database.models import Base, PlayerProjection, apply_light_migrations
    from database.session import SessionLocal, engine

    argv = list(sys.argv[1:] if argv is None else argv)
    Base.metadata.create_all(engine)
    apply_light_migrations(engine)
    db = SessionLocal()
    try:
        season = get_current_nfl_season()
        if "--week" in argv:
            week = int(argv[argv.index("--week") + 1])
        else:
            # The board's own upcoming week, which is the week whose signals
            # are worth freezing -- not a date calculation that can disagree
            # with what has actually been published.
            week = db.query(func.max(PlayerProjection.week)).filter(
                PlayerProjection.season == season).scalar()
            if week is None:
                logger.info("no %s board published; nothing to do", season)
                return 0
            week = int(week)
        record(db, season, week)
        # Everything still unscored, oldest first: a week whose stats arrived
        # late is not a week to forget about.
        for earlier in range(1, week + 1):
            score(db, season, earlier)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
