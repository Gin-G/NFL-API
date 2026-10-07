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
import json
import logging
import os
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
    """Build (rosters, depth_charts) frames for the Projector from the roster sync,
    for seasons nflreadpy hasn't published rosters for yet. Rosters carry gsis player_id
    (for history lookup); depth carries the real depth rank (1=starter, 2=backup, ...)
    so the projection engine applies its depth-role discount (backup QBs, committee RBs).

    Only used on the preseason path, where the sync's rows are ESPN's; in season the
    Projector reads nflverse directly.
    """
    import pandas as pd
    from database.models import CurrentRoster
    rows = db.query(CurrentRoster).filter(
        CurrentRoster.status == "active",
        CurrentRoster.position.in_(["QB", "RB", "WR", "TE"]),
        CurrentRoster.gsis_id.isnot(None),
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


MODEL_CACHE_ENV = "PROJECTION_MODEL_CACHE"


# Weeks of the current season that have to be in the books before the live model
# trains on the current season ALONE. The board is deliberately a this-season
# model (see _training_window); before that threshold there is nothing to train
# on, which is the one case where the old seasons still earn their place.
MIN_CURRENT_WEEKS = int(os.getenv("PROJECTION_MIN_CURRENT_WEEKS", "1"))


# What the historical window is when nfl_projections cannot be asked. The job
# pip-installs the package so it normally can, but the API image deliberately
# does not carry it (TensorFlow), and 0 would mean "every season ever" for the
# preseason fallback and "no shadow model" for the challenger — both silent.
FALLBACK_MIN_SEASON = 2020


def _package_min_season() -> int:
    """nfl_projections' own default training window (the deep one)."""
    try:
        from nfl_projections import config as nflp_config

        return int(nflp_config.TRAINING_MIN_SEASON)
    except Exception as exc:
        logger.warning("Could not read the package training window (%s); using %d",
                       exc, FALLBACK_MIN_SEASON)
        return FALLBACK_MIN_SEASON


def _training_window(df, season: int, as_of: int, min_current_weeks: int = None) -> int:
    """`min_season` for the live model: the current season once it has football
    in it, the historical default before that.

    Owner's call (2026-10-07), made against the measurement rather than because
    of it: a board for this season should be trained on this season. The cost is
    ~0.097 MAE on finished seasons, replicated 2024 and 2025 at five seeds
    (EXPERIMENTS #26) — the reasoning being that a weekly MAE average flatters
    whichever model happened to fit the week, while what the board is FOR is the
    season being played. The deep window is kept as the shadow model so the
    trade keeps being measured on live weeks instead of argued about.

    The exception falls out of the same reasoning: with no current-season games
    there is nothing to train on, so the preseason board still uses history.
    """
    threshold = MIN_CURRENT_WEEKS if min_current_weeks is None else min_current_weeks
    import pandas as pd

    if "season" not in df.columns:
        return _package_min_season()
    current = df[df["season"] == season]
    weeks = pd.to_numeric(current.get("week"), errors="coerce") if len(current) else None
    played = int(weeks.dropna().nunique()) if weeks is not None else 0
    if min(played, as_of) >= threshold:
        return season
    logger.info("Only %d week(s) of %d on hand (need %d); training on %d onward instead",
                min(played, as_of), season, threshold, _package_min_season())
    return _package_min_season()


def _cache_key(season: int, as_of: int, seeds, epochs: int, min_season=None) -> dict:
    """What a cached model has to match to be reusable: the same football.

    The model is a function of the training data, which only changes when
    another week is played — so every run inside one week can share one model,
    and the extra runs cost a dataset build and a projection rather than 40
    minutes of training. Seeds and epochs are in the key because a model trained
    with fewer of either is a different model.
    """
    try:
        import nfl_projections

        version = getattr(nfl_projections, "__version__", "unknown")
    except ImportError:
        version = "unknown"      # the API image has no nfl_projections; the job does

    return {
        "season": season,
        "as_of_week": as_of,
        "seeds": seeds,
        "epochs": epochs,
        "min_season": min_season,
        "package_version": version,
    }


def _cached_model(cache_dir: str, key: dict):
    """``(mean_dir, quantile_dir)`` when the cache holds a model for this exact
    key, else ``(None, None)``."""
    try:
        with open(os.path.join(cache_dir, "manifest.json")) as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None, None
    if {k: manifest.get(k) for k in key} != key:
        return None, None
    mean_dir = os.path.join(cache_dir, "mean")
    quantile_dir = os.path.join(cache_dir, "quantile")
    if not os.path.exists(os.path.join(mean_dir, "metadata.json")):
        return None, None
    if not os.path.exists(os.path.join(quantile_dir, "q_metadata.json")):
        return None, None
    return mean_dir, quantile_dir


def _save_model(cache_dir: str, key: dict, svc) -> None:
    """Cache the trained models, manifest LAST so a half-written cache is never
    mistaken for a usable one. Never fatal: a projection run that cannot write
    its cache is still a good projection run."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        manifest_path = os.path.join(cache_dir, "manifest.json")
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        svc.model.save(os.path.join(cache_dir, "mean"))
        if svc.quantile_model is not None:
            svc.quantile_model.save(os.path.join(cache_dir, "quantile"))
        with open(manifest_path, "w") as f:
            json.dump({**key, "trained_at": datetime.utcnow().isoformat()}, f, indent=2)
        logger.info("Cached the trained model in %s", cache_dir)
    except Exception as exc:
        logger.warning("Could not cache the trained model (%s)", exc)


def run(db, season: int, week: int, epochs: int, job, end_week: int = None,
        seeds: int = None, overwrite_started: bool = False,
        model_cache: str = None, shadow_min_season: int = None) -> None:
    from database.models import PlayerProjection
    import nfl_projections
    from nfl_projections import ProjectionService
    from nfl_projections import dataset as nflp_dataset

    _update_job(db, job, status="running", current_season=season,
                current_coach=f"season {season} week {week}")

    logger.info("Building dataset from nflreadpy...")
    df = nflp_dataset.build_dataset(output_path=None)  # build in-memory, don't write a CSV

    # Before training: a run that lands after the scores but before nflverse's
    # stats would project off an incomplete week and say nothing. Refuse instead;
    # the job runs again the next night.
    if season <= int(df["season"].max()):
        done_week, missing = _teams_missing_stats(df, season)
        if missing:
            msg = (f"week {done_week} has final scores but no nflverse stats yet for "
                   f"{', '.join(missing)}; not projecting off an incomplete week")
            logger.error(msg)
            _update_job(db, job, status="failed", error_message=msg)
            return
    # How much current-season football this projection was allowed to see. Stamped
    # on every row written below, the axis the vintage archive is keyed on, and
    # what decides whether a cached model still fits.
    as_of = _completed_weeks(season)

    # The mean model is a seed ensemble (nfl_projections default 5): averaging
    # several seeds is worth ~0.04 MAE and removes the single-seed lottery, at
    # the cost of training that many networks. Which is why a second run in the
    # same week reuses the first run's model: no new football has been played,
    # so it would train an equivalent model to project fresher injuries and
    # depth charts against.
    cache_dir = model_cache or os.getenv(MODEL_CACHE_ENV)
    live_min_season = _training_window(df, season, as_of)
    logger.info("Live model trains on %d onward", live_min_season)
    key = _cache_key(season, as_of, seeds, epochs, min_season=live_min_season)
    mean_dir, quantile_dir = _cached_model(cache_dir, key) if cache_dir else (None, None)
    if mean_dir:
        logger.info("Reusing the model already trained for %d through week %d (%s)",
                    season, as_of, cache_dir)
        svc = ProjectionService(dataset=df, model_dir=mean_dir,
                                quantile_model_dir=quantile_dir)
    else:
        logger.info("Training model (mean ensemble x%s + quantile, epochs=%d)...",
                    seeds if seeds else "default", epochs)
        svc = ProjectionService(dataset=df, quantiles=True, epochs=epochs, n_seeds=seeds,
                                min_season=live_min_season)
        if cache_dir:
            _save_model(cache_dir, key, svc)

    # nflreadpy publishes rosters only through the prior season; for a future season
    # (e.g. 2026 preseason) use the nightly ESPN roster sync + rookie draft-capital prior,
    # and derive floor/ceiling from the Monte-Carlo simulator (no current-season form yet).
    max_nflreadpy_season = int(df["season"].max())
    use_espn = season > max_nflreadpy_season
    # Both paths keep the draft-capital prior: the network needs two games of
    # history, so without it every first-year player would drop off the board
    # until week 3.
    proj_kwargs = dict(rookie_fallback=True)
    if use_espn:
        rosters, depth = _espn_frames(db, week)
        if rosters.empty:
            _update_job(db, job, status="failed",
                        error_message="no ESPN roster rows; run the roster sync first")
            logger.error("No ESPN rosters for season %d; run nfl-api-roster-sync first", season)
            return
        logger.info("Using ESPN rosters (%d players) + rookie prior for season %d",
                    len(rosters), season)
        proj_kwargs.update(rosters=rosters, depth_charts=depth, use_injuries=False)

    logger.info("Projecting season %d week %d...", season, week)
    base = svc.project(season, week, as_frame=True, **proj_kwargs)
    if base is None or base.empty:
        _update_job(db, job, status="completed", total_entries=0, processed_entries=0)
        logger.warning("No projections produced for %d week %d", season, week)
        return
    model_version = getattr(nfl_projections, "__version__", "unknown")

    # Teams already playing (or done) keep the projection they went in with. The
    # Wednesday-night run lands after a Wednesday kickoff (week 12's GB@LA), and a
    # row computed after kickoff is one projection_accuracy refuses to score.
    started = set() if overwrite_started else _kicked_off_teams(season, week)
    if started:
        logger.info("Leaving week %d projections for %s as they were: already kicked off",
                    week, ", ".join(sorted(started)))
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
        # Progress is counted in WEEKS, not rows: a run writes one week live and
        # archives the rest, so counting rows had /projections/status reporting
        # 1500% complete and gave a progress bar nothing to work with.
        _update_job(db, job, total_entries=max(last, week) - week + 1, processed_entries=0)
        written = archived = weeks_done = 0
        for w in range(week, max(last, week) + 1):
            # The upcoming week included: the base is matchup-neutral, and
            # publishing it bare would ignore the Vegas total that is already
            # posted for exactly that week. The simulator then builds the range
            # around the adjusted mean, the same as the preseason path does.
            f = _apply_environment(base, w, env_all)
            if f is None or getattr(f, "empty", False):
                continue        # bye week, or no environment for that week
            _apply_simulator(f, df, w)
            n = _write_week(db, f, season, w, model_version, as_of_week=as_of,
                            live=(w == week), started=started if w == week else ())
            archived += n
            weeks_done += 1
            if w == week:
                written = n
            _update_job(db, job, current_coach=f"season {season} week {w}",
                        processed_entries=weeks_done)
        # The board is finished HERE. The shadow model publishes nothing, so
        # completing the job first keeps /projections/status (and the dashboard
        # button reading it) about the board rather than about an experiment
        # that runs for another half hour behind it.
        _update_job(db, job, status="completed", processed_entries=weeks_done)
        logger.info("Published %d projections for %d week %d; archived %d across "
                    "weeks %d-%d as_of week %d (model %s)",
                    written, season, week, archived, week, last, as_of, model_version)
        if shadow_min_season and shadow_min_season != live_min_season:
            _run_shadow(db, df, season, week, model_version,
                        min_season=shadow_min_season, seeds=seeds, epochs=epochs,
                        proj_kwargs=proj_kwargs, env_all=env_all, cache_dir=cache_dir,
                        as_of=as_of, started=started)
        return

    # Future-season path: the base projection is matchup-neutral (same every week for a
    # preseason projection), so train/project once and apply EACH week's game environment
    # + simulator to produce a matchup-varying full-season outlook.
    env_all = _game_environments(season)
    budgets = _position_budgets(df)
    share_pred = _predict_shares(df, depth, season)
    games_model, prev_games = _fit_games(df, season)
    _update_job(db, job, total_entries=(end_week or week) - week + 1, processed_entries=0)
    total = weeks_done = 0
    for w in range(week, (end_week or week) + 1):
        f = _apply_environment(base, w, env_all)     # per-week env; drops bye teams
        if f is None or f.empty:
            continue
        _apply_roles(f, budgets, games_model, prev_games)  # availability (durability) + team pool
        _apply_shares(f, share_pred)                  # redistribute group total by predicted share
        _apply_simulator(f, df, w)
        total += _write_week(db, f, season, w, model_version, as_of_week=as_of,
                             started=started if w == week else ())
        weeks_done += 1
        _update_job(db, job, current_coach=f"season {season} week {w}",
                    processed_entries=weeks_done)
        logger.info("week %d: wrote %d projections", w, len(f))
    _update_job(db, job, status="completed", processed_entries=weeks_done)
    logger.info("Wrote %d total projections for %d weeks %d-%d (model %s)",
                total, season, week, end_week or week, model_version)
    if shadow_min_season and shadow_min_season != live_min_season:
        _run_shadow(db, df, season, week, model_version,
                    min_season=shadow_min_season, seeds=seeds, epochs=epochs,
                    proj_kwargs=proj_kwargs, env_all=env_all, cache_dir=cache_dir,
                    as_of=as_of, started=started)


def _run_shadow(db, df, season: int, week: int, model_version: str, *,
                min_season: int, seeds, epochs: int, proj_kwargs: dict,
                env_all, cache_dir: str, as_of: int, started) -> int:
    """Train a challenger on a shallower window, project the same week, park the
    rows in `shadow_projections`.

    Published nowhere. It exists so "would less history have been better" is
    settled on football nobody had seen, the same way the board itself is
    judged — EXPERIMENTS #26/#27 could only ask it of finished seasons. Same
    pipeline as the live projection (same roster/injury inputs, same game
    environment) so the only difference left is the training window.

    Quantiles and the simulator are skipped: the comparison is of the point
    projection, and the simulator is most of the run's remaining time.
    """
    from nfl_projections import ProjectionService
    from database.models import ShadowProjection

    variant = f"min_season_{min_season}"
    logger.info("Shadow run %s: training on %d onward...", variant, min_season)
    try:
        key = _cache_key(season, as_of, seeds, epochs, min_season=min_season)
        shadow_cache = os.path.join(cache_dir, variant) if cache_dir else None
        mean_dir, _ = _cached_model(shadow_cache, key) if shadow_cache else (None, None)
        if mean_dir:
            logger.info("Reusing the cached %s model", variant)
            svc = ProjectionService(dataset=df, model_dir=mean_dir)
        else:
            svc = ProjectionService(dataset=df, quantiles=False, epochs=epochs,
                                    n_seeds=seeds, min_season=min_season)
            if shadow_cache:
                _save_model(shadow_cache, key, svc)
        frame = svc.project(season, week, as_frame=True, **proj_kwargs)
        if frame is None or getattr(frame, "empty", False):
            logger.warning("Shadow run %s produced nothing", variant)
            return 0
        frame = _apply_environment(frame, week, env_all)
    except Exception as exc:
        # A challenger is never worth failing the real projection over.
        logger.warning("Shadow run %s failed (%s); the live projection stands", variant, exc)
        return 0

    config_json = json.dumps({"min_season": min_season, "seeds": seeds,
                              "epochs": epochs, "as_of_week": as_of})
    locked = set(started or ())
    db.query(ShadowProjection).filter(
        ShadowProjection.season == season, ShadowProjection.week == week,
        ShadowProjection.variant == variant).delete(synchronize_session=False)
    written = 0
    for _, r in frame.iterrows():
        pid = str(r.get("player_id") or "").strip()
        if not pid or str(r.get("team") or "") in locked:
            continue
        db.add(ShadowProjection(
            season=season, week=week, player_id=pid, variant=variant,
            player_name=str(r.get("player_name") or ""),
            position=str(r.get("position") or ""),
            team=str(r.get("team") or ""),
            projected_points=_f(r.get("fanduel_fantasy_points")),
            floor=_f(r.get("floor")), ceiling=_f(r.get("ceiling")),
            prediction_type=str(r.get("prediction_type") or ""),
            model_version=model_version, config_json=config_json,
            computed_at=datetime.utcnow(),
        ))
        written += 1
    db.commit()
    logger.info("Shadow run %s: %d projections recorded for week %d", variant, written, week)
    return written


def _write_week(db, frame, season: int, week: int, model_version: str,
                as_of_week: int = 0, live: bool = True, started=()) -> int:
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

    `started` teams are skipped in both tables: their game has kicked off, so the
    rows already there are the last honest pre-game projection.
    """
    from database.models import PlayerProjection, PlayerProjectionVintage
    started = set(started or ())
    if live:
        stale = db.query(PlayerProjection).filter(
            PlayerProjection.season == season, PlayerProjection.week == week)
        if started:
            stale = stale.filter(~PlayerProjection.team.in_(sorted(started)))
        stale.delete(synchronize_session=False)
        db.commit()
    written = 0
    for _, r in frame.iterrows():
        pid = str(r.get("player_id") or "").strip()
        if not pid or str(r.get("team") or "") in started:
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


GAME_LENGTH = 4   # hours from kickoff to a final score, generously


def _kickoff_times(schedule, season: int):
    """Kickoff instants (UTC) for a season's regular-season games.

    nflverse dates a game in `gameday` and times it in `gametime`, US Eastern. A
    missing time counts as midnight — treating a game as earlier than it is,
    which is the safe direction for "has this kicked off".
    """
    import pandas as pd

    games = schedule[(schedule["season"] == season) & (schedule["game_type"] == "REG")].copy()
    kick = pd.to_datetime(games["gameday"].astype(str) + " "
                          + games["gametime"].fillna("00:00").astype(str), errors="coerce")
    games["kick"] = (kick.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
                     .dt.tz_convert("UTC"))
    return games


def _load_schedule(season: int):
    import nflreadpy as nfl

    sched = nfl.load_schedules(seasons=[season])
    return sched.to_pandas() if hasattr(sched, "to_pandas") else sched


def _completed_weeks(season: int, when=None, schedule=None) -> int:
    """Completed regular-season weeks of `season` as of `when` (default now).

    This is the `as_of_week` stamp: the amount of current-season football the
    projection was allowed to learn from. A week is complete once its LAST game
    has been played, so a Tuesday-night run sees the Monday nighter.

    The times matter. Reading `gameday` alone put every game at midnight, which
    marked a week complete on the MORNING of its Monday night game: it stamped
    the archive with football that had not happened, and retrained a model whose
    cached copy was still perfectly good. Returns 0 preseason, and 0 on any
    error — understating what the model knew is the safe direction for an
    archive whose whole purpose is comparing vintages.
    """
    try:
        import pandas as pd

        games = _kickoff_times(schedule if schedule is not None else _load_schedule(season), season)
        now = pd.Timestamp(when) if when is not None else pd.Timestamp.now(tz="UTC")
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        cutoff = now - pd.Timedelta(hours=GAME_LENGTH)
        # The LAST game of a week decides it, which is what stops a Thursday
        # nighter from marking the whole week done.
        last = games.groupby("week")["kick"].max().dropna()
        complete = [int(w) for w, k in last.items() if k < cutoff]
        return max(complete) if complete else 0
    except Exception as exc:
        logger.warning("Could not determine completed weeks (%s); using 0", exc)
        return 0


def _kicked_off_teams(season: int, week: int, now=None, schedule=None) -> set:
    """Teams whose game in `week` has kicked off by `now` (default: now).

    nflverse's gametime is US Eastern. A missing gametime counts as midnight, so a
    game dated today is treated as started: the cost of that is keeping a
    projection a few hours older, where the other way is overwriting a live game.
    """
    import pandas as pd

    try:
        games = _kickoff_times(schedule if schedule is not None else _load_schedule(season), season)
        games = games[games["week"] == week]
        now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        begun = games[games["kick"].notna() & (games["kick"] <= now)]
    except Exception as exc:
        logger.warning("Could not check week %d kickoffs (%s); overwriting all teams",
                       week, exc)
        return set()
    return set(begun["home_team"]) | set(begun["away_team"])


def _teams_missing_stats(df, season: int, schedule=None):
    """``(week, teams)``: teams whose game in the last completed week has a final
    score but no player stats in the dataset. ``(week, [])`` when the week is all
    there; ``(None, [])`` when it can't be checked, which lets the run proceed.

    Only scored games count. A score posts within hours of the final whistle and
    nflverse's stats can trail it by a day, which is the window a run the night
    after Monday Night Football can land in. A cancelled game never gets a score,
    so it can't block every run for a week.
    """
    import pandas as pd

    week = _completed_weeks(season)
    if week < 1 or not {"season", "week", "team"} <= set(df.columns):
        return None, []
    try:
        if schedule is None:
            import nflreadpy as nfl

            schedule = nfl.load_schedules(seasons=[season])
            schedule = schedule.to_pandas() if hasattr(schedule, "to_pandas") else schedule
        games = schedule[(schedule["season"] == season) & (schedule["week"] == week)
                         & (schedule["game_type"] == "REG") & schedule["home_score"].notna()]
    except Exception as exc:
        logger.warning("Could not check week %d stats completeness (%s)", week, exc)
        return None, []
    played = set(games["home_team"]) | set(games["away_team"])
    rows = df[(df["season"] == season) & (pd.to_numeric(df["week"], errors="coerce") == week)]
    return week, sorted(played - set(rows["team"].dropna()))


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
    parser.add_argument("--shadow-min-season", type=int, default=None,
                        help="Train a challenger on this season onward and record it in "
                             "shadow_projections, to be graded against the live model on "
                             "the week itself. Defaults to the historical window the live "
                             "model no longer uses, so the trade stays measured. 0 turns "
                             "it off.")
    parser.add_argument("--model-cache", default=None,
                        help=f"Directory holding the trained model (default: ${MODEL_CACHE_ENV}). "
                             "A run reuses it when no new football has been played since "
                             "it was trained, which is what makes a mid-week refresh quick.")
    parser.add_argument("--overwrite-started", action="store_true",
                        help="Also rewrite teams whose game has kicked off (by default "
                             "they keep their pre-game projection; needed to rerun a "
                             "past week on purpose)")
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
            seeds=args.seeds, overwrite_started=args.overwrite_started,
            model_cache=args.model_cache,
            # The live model is this season's; the challenger is the deep
            # historical window it replaced, so the choice keeps being scored.
            shadow_min_season=(_package_min_season() if args.shadow_min_season is None
                               else args.shadow_min_season or None))

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
