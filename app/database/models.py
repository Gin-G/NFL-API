#!/usr/bin/env python3
"""
SQLAlchemy ORM models for NFL data.
"""

import logging
from datetime import datetime

from sqlalchemy import (Boolean, Column, String, Integer, Float, Text, DateTime,
                        UniqueConstraint, Index)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    team_abbr = Column(String, primary_key=True)
    team_name = Column(String)
    team_conf = Column(String)
    team_division = Column(String)
    team_color = Column(String)
    team_color2 = Column(String)
    team_color3 = Column(String)
    team_color4 = Column(String)
    team_logo_espn = Column(String)
    team_wordmark = Column(String)
    team_conference_logo = Column(String)
    team_league_logo = Column(String)


class Schedule(Base):
    __tablename__ = "schedules"

    game_id = Column(String, primary_key=True)
    season = Column(Integer, index=True)
    week = Column(Integer, index=True)
    game_type = Column(String)
    gameday = Column(String)
    gametime = Column(String)
    home_team = Column(String, index=True)
    away_team = Column(String, index=True)
    home_score = Column(Integer)
    away_score = Column(Integer)
    location = Column(String)
    result = Column(String)
    total = Column(Float)
    overtime = Column(Integer)
    old_game_id = Column(String)
    pfr = Column(String)
    pff_game_id = Column(Float)
    espn = Column(Integer)
    away_rest = Column(Integer)
    home_rest = Column(Integer)
    away_moneyline = Column(Float)
    home_moneyline = Column(Float)
    spread_line = Column(Float)
    away_spread_odds = Column(Float)
    home_spread_odds = Column(Float)
    total_line = Column(Float)
    under_odds = Column(Float)
    over_odds = Column(Float)
    div_game = Column(Integer)
    roof = Column(String)
    surface = Column(String)
    temp = Column(Float)
    wind = Column(Float)
    away_qb_id = Column(String)
    home_qb_id = Column(String)
    away_qb_name = Column(String)
    home_qb_name = Column(String)
    away_coach = Column(String)
    home_coach = Column(String)
    referee = Column(String)
    stadium_id = Column(String)
    stadium = Column(String)


class PlayerRoster(Base):
    __tablename__ = "player_rosters"

    # Composite primary key; merge() treats all PK columns as the identity
    player_id = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    week = Column(Integer, primary_key=True)

    player_name = Column(String)
    player_display_name = Column(String)
    position = Column(String, index=True)
    team = Column(String, index=True)
    height = Column(String)
    weight = Column(Float)
    college = Column(String)
    rookie_year = Column(Integer)
    jersey_number = Column(Float)
    birth_date = Column(String)
    status = Column(String)
    status_description_abbr = Column(String)
    football_name = Column(String)
    esb_id = Column(String)
    gsis_it_id = Column(String)
    smart_id = Column(String)
    entry_year = Column(Integer)
    draft_club = Column(String)
    draft_number = Column(Float)
    years_exp = Column(Integer)
    headshot_url = Column(String)

    __table_args__ = (
        Index("ix_roster_player_season", "player_id", "season"),
        Index("ix_roster_team_season_pos", "team", "season", "position"),
    )


class PlayerStat(Base):
    __tablename__ = "player_stats"

    # Composite primary key on (player_id, season, week)
    player_id = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    week = Column(Integer, primary_key=True)

    player_name = Column(String)
    player_display_name = Column(String)
    position = Column(String, index=True)
    position_group = Column(String)
    recent_team = Column(String, index=True)
    season_type = Column(String)

    # Passing
    completions = Column(Float)
    attempts = Column(Float)
    passing_yards = Column(Float)
    passing_tds = Column(Float)
    interceptions = Column(Float)
    sacks = Column(Float)
    sack_yards = Column(Float)
    sack_fumbles = Column(Float)
    sack_fumbles_lost = Column(Float)
    passing_air_yards = Column(Float)
    passing_yards_after_catch = Column(Float)
    passing_first_downs = Column(Float)
    passing_epa = Column(Float)
    passing_2pt_conversions = Column(Float)
    pacr = Column(Float)
    dakota = Column(Float)

    # Rushing
    carries = Column(Float)
    rushing_yards = Column(Float)
    rushing_tds = Column(Float)
    rushing_fumbles = Column(Float)
    rushing_fumbles_lost = Column(Float)
    rushing_first_downs = Column(Float)
    rushing_epa = Column(Float)
    rushing_2pt_conversions = Column(Float)

    # Receiving
    receptions = Column(Float)
    targets = Column(Float)
    receiving_yards = Column(Float)
    receiving_tds = Column(Float)
    receiving_fumbles = Column(Float)
    receiving_fumbles_lost = Column(Float)
    receiving_air_yards = Column(Float)
    receiving_yards_after_catch = Column(Float)
    receiving_first_downs = Column(Float)
    receiving_epa = Column(Float)
    receiving_2pt_conversions = Column(Float)
    racr = Column(Float)
    target_share = Column(Float)
    air_yards_share = Column(Float)
    wopr = Column(Float)

    # Special teams / other
    special_teams_tds = Column(Float)
    fantasy_points = Column(Float)
    fantasy_points_ppr = Column(Float)

    __table_args__ = (
        Index("ix_stats_player_season", "player_id", "season"),
        Index("ix_stats_team_season_pos", "recent_team", "season", "position"),
    )


class CoachSeasonAnalytics(Base):
    __tablename__ = "coach_season_analytics"

    coach_name     = Column(String, primary_key=True)
    season         = Column(Integer, primary_key=True)
    team           = Column(String, primary_key=True)
    breakdown_json = Column(Text)
    computed_at    = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_coach_analytics_name", "coach_name"),)


class PlayerSeasonAnalytics(Base):
    """Cached PBP-derived analytics for an individual player in a season."""
    __tablename__ = "player_season_analytics"

    player_id      = Column(String, primary_key=True)
    season         = Column(Integer, primary_key=True)
    team           = Column(String, primary_key=True)
    player_name    = Column(String)
    position       = Column(String)
    analytics_json = Column(Text)
    computed_at    = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_player_analytics_name", "player_name"),
        Index("ix_player_analytics_pos_season", "position", "season"),
    )


class DepthChart(Base):
    __tablename__ = "depth_charts"

    player_id            = Column(String,  primary_key=True)
    season               = Column(Integer, primary_key=True)
    week                 = Column(Integer, primary_key=True)
    team                 = Column(String,  primary_key=True)
    position             = Column(String,  primary_key=True)
    depth_chart_position = Column(String)
    depth_team           = Column(Integer)
    full_name            = Column(String)
    game_type            = Column(String)

    __table_args__ = (Index("ix_depth_charts_team_season", "team", "season"),)


class SnapCount(Base):
    __tablename__ = "snap_counts"

    player_id     = Column(String,  primary_key=True)
    season        = Column(Integer, primary_key=True)
    week          = Column(Integer, primary_key=True)
    team          = Column(String,  primary_key=True)
    player_name   = Column(String)
    position      = Column(String)
    offense_snaps = Column(Integer)
    offense_pct   = Column(Float)
    defense_snaps = Column(Integer)
    defense_pct   = Column(Float)
    st_snaps      = Column(Integer)
    st_pct        = Column(Float)

    __table_args__ = (Index("ix_snap_counts_team_season", "team", "season"),)


class PlayerProjection(Base):
    """Weekly fantasy projection for a player: a mean point projection plus a
    floor / median / ceiling range from the quantile model. Populated by the
    scripts.compute_projections job (which uses the nfl_projections package)."""
    __tablename__ = "player_projections"

    season           = Column(Integer, primary_key=True)
    week             = Column(Integer, primary_key=True)
    player_id        = Column(String,  primary_key=True)
    player_name      = Column(String)
    position         = Column(String)
    team             = Column(String)
    projected_points = Column(Float)   # mean projection (fanduel points)
    floor            = Column(Float)   # 10th percentile
    median           = Column(Float)   # 50th percentile
    ceiling          = Column(Float)   # 90th percentile
    # projected component stats (from the multi-output model / rookie prior)
    passing_yards        = Column(Float)
    passing_tds          = Column(Float)
    passing_interceptions = Column(Float)
    rushing_yards        = Column(Float)
    rushing_tds          = Column(Float)
    receiving_yards      = Column(Float)
    receptions           = Column(Float)
    receiving_tds        = Column(Float)
    # Expected fraction of THIS week the player is available for — the
    # availability weight already multiplied into projected_points and every
    # component above. Summed across a season's rows it is the player's expected
    # GAMES PLAYED, which is what makes projected_points/exp_games an on-field
    # rate rather than a rate diluted by the weeks he is projected to miss.
    # Null on rows written before this column existed, and on the in-season path
    # where no availability discount is applied.
    exp_games        = Column(Float, nullable=True)
    prediction_type  = Column(String)  # veteran_ml / rookie_ml / injured_out
    model_version    = Column(String)
    computed_at      = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_player_projections_week", "season", "week"),
        Index("ix_player_projections_pos", "position", "season", "week"),
    )


class PlayerProjectionVintage(Base):
    """Every projection ever made for a player-week, kept forever.

    `player_projections` holds one row per (season, week, player) — the CURRENT
    best projection, overwritten each time that week is reprojected. That is the
    right thing for "what do we think about week 10", and the wrong thing for
    "what did we think about week 10 back in August, and did more data help".

    This table is the second thing. The extra key is `as_of_week`: how many
    regular-season weeks were complete when the projection was computed. Week 10
    therefore accumulates a row per vintage —

        as_of_week=0   the preseason full-season outlook
        as_of_week=1   reprojected once week 1 was in the books
        ...
        as_of_week=9   the last projection before week 10 kicks off

    so the season ends with a triangle of 18+17+...+1 = 171 projections per
    player, and "does a projection get better as the season feeds it" becomes a
    query rather than an argument. Rows are never deleted, only re-merged in
    place if the same vintage is recomputed.
    """
    __tablename__ = "player_projection_vintages"

    season           = Column(Integer, primary_key=True)
    week             = Column(Integer, primary_key=True)
    player_id        = Column(String,  primary_key=True)
    # Completed regular-season weeks of the SAME season behind this projection.
    # 0 means preseason: no current-season football had been played.
    as_of_week       = Column(Integer, primary_key=True)
    player_name      = Column(String)
    position         = Column(String)
    team             = Column(String)
    projected_points = Column(Float)
    floor            = Column(Float)
    median           = Column(Float)
    ceiling          = Column(Float)
    passing_yards        = Column(Float)
    passing_tds          = Column(Float)
    passing_interceptions = Column(Float)
    rushing_yards        = Column(Float)
    rushing_tds          = Column(Float)
    receiving_yards      = Column(Float)
    receptions           = Column(Float)
    receiving_tds        = Column(Float)
    exp_games        = Column(Float, nullable=True)
    prediction_type  = Column(String)
    model_version    = Column(String)
    computed_at      = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        # The trajectory query: one player-week, every vintage, in order.
        Index("ix_ppv_player_week", "season", "week", "player_id", "as_of_week"),
        # The cohort query: everything produced by one run.
        Index("ix_ppv_asof", "season", "as_of_week"),
        Index("ix_ppv_pos", "position", "season", "week"),
    )


class ExternalSnapshot(Base):
    """One hand-exported third-party matchup table, kept verbatim.

    The first source is FantasyGuru's weekly "smash" tables, exported by a
    subscriber. **Nothing fetches these.** The source's terms are personal use
    only, so the only way a row gets here is a person uploading a CSV, and
    there is deliberately no fetcher anywhere in either repo — see
    `nfl_projections.external`.

    The bytes are stored as uploaded, the same choice `fanduel_slates` makes
    and for the same reason: the file is a few kilobytes, and keeping it means
    the source changing a column next month is something to re-read rather
    than something already lost to today's schema. The shredded rows in the
    two tables below are derived from it and can be rebuilt.

    Keyed (source, season, week, kind) so a re-upload REPLACES: a corrected
    export should overwrite, not accumulate. `content_hash` says whether it
    actually changed.
    """
    __tablename__ = "external_snapshots"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    source       = Column(String)        # fantasyguru
    kind         = Column(String)        # wr_coverage | team_lines | line_matchups
    season       = Column(Integer)
    week         = Column(Integer)
    filename     = Column(String)
    content      = Column(Text)          # the CSV exactly as uploaded
    content_hash = Column(String)        # sha256
    row_count    = Column(Integer)       # data rows in the file
    parsed_rows  = Column(Integer)       # rows that shredded cleanly
    # Names and teams the mapping could not resolve, comma-joined. Not an
    # error: it is the list to fix, and it belongs with the upload that
    # produced it rather than in a log nobody reads.
    unmapped     = Column(Text, nullable=True)
    uploaded_at  = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("source", "season", "week", "kind",
                         name="uq_external_snapshot"),
        Index("ix_external_snapshots_week", "season", "week"),
    )


class ExternalTeamLine(Base):
    """A third-party offensive-line / defensive-front rating, one per team-week.

    Compared against /trenches rather than merged with it. The two measure
    overlapping but different things: across week 9 2026 their O-line ranks
    Spearman 0.66 against our run_block and only 0.42 against pass_block, and
    their D-front 0.73 against run_stop and 0.22 against pass_rush -- so their
    ratings are substantially a run-game measure. Their own O-line and D-front
    also correlate 0.6 with each other, which is what team strength leaking
    into both would look like.
    """
    __tablename__ = "external_team_lines"

    source          = Column(String,  primary_key=True)
    season          = Column(Integer, primary_key=True)
    week            = Column(Integer, primary_key=True)
    team            = Column(String,  primary_key=True)   # nflverse abbreviation
    team_raw        = Column(String)      # the source's own spelling
    offensive_line  = Column(Float)
    defensive_front = Column(Float)
    snapshot_id     = Column(Integer, index=True)
    uploaded_at     = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_external_team_lines_week", "season", "week"),)


class ExternalWrMatchup(Base):
    """A third-party corner-vs-receiver pairing for one week.

    The reason this table is worth keeping at all is `shadow` and the pairing
    itself. Our own coverage work closed the WR matchup question three times
    over -- EXPERIMENTS.md "Per-corner coverage grades" has wide receiver at
    -0.0032 MAE, actively harmful with the interval clear of zero -- but every
    one of those tests used a TEAM-level secondary, because nflverse publishes
    no assignment. Which corner a receiver actually draws, and whether he is
    shadowed, is the one column here that is not a second opinion about
    something we already measure.

    `shell`/`man_zone` use nflverse's own vocabulary where the source's label
    maps onto it, so the two are directly comparable. "Match" does not map and
    keeps its own value rather than being forced into a shell it is not.

    Keyed on the source's own rank, which is unique within an export, so a
    re-upload replaces cleanly.
    """
    __tablename__ = "external_wr_matchups"

    source         = Column(String,  primary_key=True)
    season         = Column(Integer, primary_key=True)
    week           = Column(Integer, primary_key=True)
    fg_rank        = Column(Integer, primary_key=True)

    defender       = Column(String)
    defender_id    = Column(String, index=True)   # gsis_id, null if unresolved
    defender_team  = Column(String)
    # exact | team | name | team_fuzzy | null. A fuzzy match is a one-letter
    # spelling difference resolved inside one roster; it is recorded rather
    # than passed off as exact.
    defender_match = Column(String)
    receiver       = Column(String)
    receiver_id    = Column(String, index=True)
    receiver_team  = Column(String)
    receiver_match = Column(String)

    coverage_type  = Column(String)      # the source's raw label
    shell_pct      = Column(Float)
    shell          = Column(String)      # nflverse vocabulary, or MATCH
    man_zone       = Column(String)      # MAN_COVERAGE | ZONE_COVERAGE
    shadow         = Column(Boolean, default=False)
    advantage      = Column(Float)       # higher = better for the receiver

    snapshot_id    = Column(Integer, index=True)
    uploaded_at    = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_external_wr_week", "season", "week"),
        Index("ix_external_wr_receiver", "receiver_id", "season", "week"),
    )


class ProjectionRun(Base):
    """One row per projection run: what the board was allowed to see.

    The projections themselves cannot answer this. A week where every club
    filed a clean injury report and a week where nobody had filed yet produce
    the same rows — no overrides either way — and the depth chart a run read is
    overwritten by the next nflverse publish within the day. So a board two
    weeks old can be argued about but not checked.

    It matters because the conditions differ *systematically*, not randomly.
    The projections cron runs Tuesday and Wednesday nights; clubs file their
    first injury report on Wednesday. The Tuesday run therefore can never see
    one and the Wednesday run always can, and the two publish to the same
    table. Week 9 shipped with Baltimore's injured starting quarterback
    projected for 8.3 points and the whole passing game priced off him, which
    looked like a model failure and was a run-condition failure.

    Written by scripts.compute_projections at publish time and served by
    GET /projections/runs, plus the `provenance` block on GET /projections.
    Append-only: a run is a historical fact, so re-projecting a week adds a row
    rather than replacing one.
    """
    __tablename__ = "projection_runs"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    season        = Column(Integer)
    # The week PUBLISHED to player_projections. The same run archives every
    # later week to the vintage table, which is keyed on as_of_week instead.
    week          = Column(Integer)
    as_of_week    = Column(Integer)      # completed weeks of football behind it
    computed_at   = Column(DateTime, default=datetime.utcnow)
    job_id        = Column(Integer, nullable=True)   # analytics_job_status.id

    # -- the model
    model_version       = Column(String)
    model_reused        = Column(Boolean, default=False)  # cache hit, no retrain
    training_min_season = Column(Integer, nullable=True)
    seeds               = Column(Integer, nullable=True)
    epochs              = Column(Integer, nullable=True)

    # -- what the injury sources offered for THIS week
    injury_source             = Column(String)   # nflverse | sportradar | disabled
    # Whether any club had filed for `week` yet. The flag the Tuesday run needs.
    injury_report_available   = Column(Boolean)
    injury_report_rows        = Column(Integer)  # report rows for `week`
    injury_report_latest_week = Column(Integer, nullable=True)  # newest week on file
    injury_report_players     = Column(Integer)  # designations parsed out of it
    ir_players                = Column(Integer)  # from roster status, not the report
    players_zeroed            = Column(Integer)  # OUT/doubtful, both sources
    backups_elevated          = Column(Integer)
    questionable              = Column(Integer)

    # -- rosters and depth charts
    roster_source     = Column(String)           # nflverse | espn
    roster_rows       = Column(Integer)
    depth_chart_as_of = Column(DateTime, nullable=True)  # the snapshot in force

    # -- what came out
    projections_published = Column(Integer)
    weeks_archived        = Column(Integer)
    # Teams already kicked off, whose rows the run left alone.
    teams_frozen          = Column(String, nullable=True)

    __table_args__ = (
        # Both lookups are season-leading, so these cover the bare-season
        # listing too; no single-column index is needed beside them.
        Index("ix_projection_runs_season_week", "season", "week"),
        Index("ix_projection_runs_season_asof", "season", "as_of_week"),
    )


class ShadowProjection(Base):
    """A projection from an ALTERNATIVE model, published nowhere, kept to be
    graded against the live one on the same week.

    The board is settled by prospective accuracy rather than by backtests
    (projection_accuracy), and a backtest cannot answer whether a different
    training window would have done better on football nobody had seen. So the
    weekly job trains the challenger too, projects the same week with the same
    pipeline, and parks the rows here. Nothing reads them for the board; the
    comparison is a join against projection_accuracy, which already holds the
    frozen actuals for exactly those player-weeks.

    `variant` names the challenger — "min_season_2025" for a model trained on
    last season and this one — so several can run at once, and `config_json`
    records what it was so a result stays interpretable after the code moves on.
    """
    __tablename__ = "shadow_projections"

    season           = Column(Integer, primary_key=True)
    week             = Column(Integer, primary_key=True)
    player_id        = Column(String,  primary_key=True)
    variant          = Column(String,  primary_key=True)
    player_name      = Column(String)
    position         = Column(String)
    team             = Column(String)
    projected_points = Column(Float)
    floor            = Column(Float)
    ceiling          = Column(Float)
    prediction_type  = Column(String)
    model_version    = Column(String)
    config_json      = Column(Text)        # {"min_season": 2025, "seeds": 5, ...}
    computed_at      = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_shadow_variant", "season", "variant", "week"),
    )


class ProjectionAccuracy(Base):
    """A projection scored against what actually happened — the prospective,
    truly out-of-sample accuracy record, accruing one week at a time.

    Written by the scripts.score_projections job once a week's actuals land.
    Rows are FROZEN: the job never rewrites an existing (season, week,
    player_id), because `player_projections` is keyed the same way and a later
    projections run would otherwise let hindsight rewrite history.

    `naive_points` is the honest denominator from the model's experiment ledger
    — the player's trailing-5-game average, which the full model beats by only
    ~0.05 MAE in backtests. Storing it per row lets every accuracy query report
    the skill-over-naive margin instead of a bare absolute MAE."""
    __tablename__ = "projection_accuracy"

    season           = Column(Integer, primary_key=True)
    week             = Column(Integer, primary_key=True)
    player_id        = Column(String,  primary_key=True)
    player_name      = Column(String)
    position         = Column(String)
    team             = Column(String)

    projected_points = Column(Float)   # what we said (mean projection)
    floor            = Column(Float)
    median           = Column(Float)
    ceiling          = Column(Float)
    actual_points    = Column(Float)   # FanDuel points actually scored
    error            = Column(Float)   # projected - actual (sign = bias direction)
    abs_error        = Column(Float)
    in_band          = Column(Integer) # 1 when floor <= actual <= ceiling (band coverage)

    naive_points     = Column(Float)   # trailing-5-game average FanDuel points
    naive_abs_error  = Column(Float)

    prediction_type  = Column(String)  # veteran_ml / rookie_ml / injured_out
    model_version    = Column(String)
    projected_at     = Column(DateTime)  # the projection's computed_at (predates the game)
    scored_at        = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_projection_accuracy_week", "season", "week"),
        Index("ix_projection_accuracy_pos", "position", "season"),
        Index("ix_projection_accuracy_player", "player_id"),
    )


class AnalyticsJobStatus(Base):
    """Tracks progress of long-running analytics pre-computation jobs."""
    __tablename__ = "analytics_job_status"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    job_type          = Column(String, default="coach_analytics")
    status            = Column(String)           # running | completed | failed | interrupted
    started_at        = Column(DateTime)
    updated_at        = Column(DateTime)
    total_entries     = Column(Integer, default=0)
    processed_entries = Column(Integer, default=0)  # newly computed this run
    skipped_entries   = Column(Integer, default=0)  # already cached, skipped
    failed_entries    = Column(Integer, default=0)
    current_season    = Column(Integer, nullable=True)
    current_coach     = Column(String, nullable=True)
    error_message     = Column(Text, nullable=True)


class CurrentRoster(Base):
    """Who is on which team right now, replaced by every roster sync.

    nflverse's weekly rosters once the season is under way, ESPN's live API
    before it exists for the new year (`source` says which). A cache of an
    upstream snapshot, never the system of record — so the sync rebuilds it
    wholesale and `updated_at` is published, since a caller weighing it against
    other sources should be able to age it rather than assume it is current.
    """
    __tablename__ = "current_roster"

    # Surrogate key: the table is rebuilt each run, and neither natural id is
    # dependably present — nflverse carries no espn_id for some veterans, ESPN no
    # gsis_id for some rookies.
    id          = Column(Integer, primary_key=True, autoincrement=True)
    gsis_id     = Column(String, index=True)   # nflverse id (for joins); may be null
    espn_id     = Column(String, index=True)   # may be null
    full_name   = Column(String)
    position    = Column(String, index=True)
    team        = Column(String, index=True)   # standard nflverse abbreviation
    # active | injured_reserve | practice_squad | suspended | exempt | cut | retired
    status      = Column(String, index=True)
    raw_status  = Column(String)               # the source's own code, e.g. nflverse ACT/INA/RES
    jersey      = Column(String)
    age         = Column(Integer, nullable=True)
    experience  = Column(Integer, nullable=True)
    depth_rank  = Column(Integer, nullable=True)  # 1=starter, 2=backup, ... latest depth chart
    source      = Column(String)               # nflverse | espn
    week        = Column(Integer, nullable=True)  # the roster week nflverse published
    updated_at  = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_current_roster_team_pos", "team", "position"),)


class FanduelSlate(Base):
    """A FanDuel slate CSV, kept so it is uploaded once instead of every build.

    FanDuel publishes salaries nowhere fetchable (see api/optimizer.py for the
    search), so the contest's own player list is the input -- and
    re-downloading it for every lineup build is the only manual step left in
    the whole chain. The file is stored verbatim rather than shredded into
    rows: it is ~100 KB, the optimizer reads it with ``read_csv`` either way,
    and keeping the bytes means FanDuel changing a column later is something to
    re-read rather than something already lost to today's schema.

    Keyed (season, week, slate_key) so a re-upload REPLACES: prices move during
    the week and the newest download is the one that matches the contest.
    `slate_key` is "classic" or "showdown:AWAY@HOME", because Showdown is a
    different contest with its own file -- one per game, so a week can hold
    several.
    """
    __tablename__ = "fanduel_slates"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    season       = Column(Integer, index=True)
    week         = Column(Integer, index=True)
    contest      = Column(String)        # classic | showdown
    slate_key    = Column(String)        # classic | showdown:AWAY@HOME
    filename     = Column(String)
    content      = Column(Text)          # the CSV as uploaded
    content_hash = Column(String)        # sha256: shows whether a re-upload changed anything
    row_count    = Column(Integer)
    teams        = Column(String)        # comma-joined, to show what the file covers
    uploaded_at  = Column(DateTime, default=datetime.utcnow)
    used_at      = Column(DateTime)      # the last build that read it

    __table_args__ = (
        UniqueConstraint("season", "week", "slate_key", name="uq_fanduel_slate"),
        Index("ix_fanduel_slate_week", "season", "week"),
    )


class LineupBuild(Base):
    """One lineup-build request, with the settings that produced it.

    The settings are stored and not just the lineups, because the honest
    baselines can only be rebuilt from the same pool: what FanDuel's own FPPG
    would have picked under that cap, and what the best lineup on the slate
    turned out to be. Both are computed after the games by
    scripts.score_lineups, which needs the cap, the slate window and the
    exclusions to reconstruct what the optimizer was choosing from.
    """
    __tablename__ = "lineup_builds"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    season               = Column(Integer, index=True)
    week                 = Column(Integer, index=True)
    # fanduel_slates.id, deliberately without a foreign key: a slate can be
    # deleted and the record of what was built from it should survive that.
    slate_id             = Column(Integer, index=True)
    contest              = Column(String)   # classic | showdown
    slate                = Column(String)   # all | main | sunday | primetime | single_game
    objective            = Column(String)   # mean | median | floor | ceiling
    salary_cap           = Column(Integer)
    max_usage_percentage = Column(Integer)
    exclude_players      = Column(Text)     # comma-joined, as sent
    num_requested        = Column(Integer)
    num_built            = Column(Integer)
    built_at             = Column(DateTime, default=datetime.utcnow)

    # The correlation rules in force, because a build is not reproducible
    # without them and two builds under different rules are not comparable.
    max_per_team         = Column(Integer)  # at most this many of the nine from one team
    stack_qb             = Column(Integer)  # WR/TE required from the QB's team
    rb_with_qb           = Column(Integer)  # 0 = no RB from the QB's team
    # Points charged per percent of projected ownership. Recorded because it
    # trades expected points for a tilt against the field, so a build's score
    # cannot be read without knowing whether it paid that.
    ownership_weight     = Column(Float)

    # WHEN the board it read was computed. 2026 week 7 was built twice, once
    # against a board that turned out to be the preseason projection still
    # sitting in the table because the in-season run had not fired yet, and
    # nothing recorded which was which. Without this the lineups cannot be
    # told apart after the fact.
    projection_computed_at = Column(DateTime)

    # The player pool in force, as JSON, and whether this build was told what
    # had already been entered. 2026 week 9 entered 45 lineups and held 31
    # distinct ones, because six builds each re-solved the same slate from
    # scratch; without these two a repeat cannot be told from a choice.
    player_rules         = Column(Text)     # {"Nickname": {"min": x, "max": y}}
    deduped_against      = Column(Integer)  # lineups already entered that it avoided
    duplicates_skipped   = Column(Integer)  # solved lineups dropped as already entered
    pool_exposure        = Column(Integer)  # 1 = percentages counted the whole week

    # Filled in once the week has actuals, by scripts.score_lineups.
    fppg_points          = Column(Float)    # FanDuel's own FPPG lineup, actually scored
    fppg_projected       = Column(Float)    # ... and what FPPG said it would score
    best_points          = Column(Float)    # the best lineup the slate held, in hindsight
    scored_at            = Column(DateTime)

    __table_args__ = (Index("ix_lineup_build_week", "season", "week"),)


class SlatePlayerRule(Base):
    """What the player pool says about one player on one slate.

    One row per player the user has an opinion about -- locked in, faded out,
    or held to a share of the lineups -- so the opinion outlives the request
    that set it and every later build obeys it without being told again. A
    player with no row is unconstrained, which is why this table stays small:
    it holds decisions, not the slate.

    `min_exposure` and `max_exposure` are percentages of the lineups, and they
    are one dial: 0/0 is an exclusion, 100/100 is a lock, 40/40 is "about four
    in ten". A floor is the thing the global `max_usage_percentage` could never
    express -- it caps everyone equally, and can neither require a player nor
    name one.

    Keyed (slate_id, nickname) rather than (season, week): prices and the
    player list both move during the week, a re-upload REPLACES the slate, and
    a rule set against Wednesday's file should not silently attach itself to
    Saturday's different contest. `nickname` is FanDuel's own spelling, because
    that is what the optimizer matches on.
    """
    __tablename__ = "slate_player_rules"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    slate_id     = Column(Integer, index=True)
    season       = Column(Integer, index=True)
    week         = Column(Integer, index=True)
    nickname     = Column(String, index=True)
    min_exposure = Column(Float)            # null = no floor
    max_exposure = Column(Float)            # null = no ceiling
    note         = Column(Text)             # why, in the user's own words
    updated_at   = Column(DateTime, default=datetime.utcnow,
                          onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("slate_id", "nickname", name="uq_slate_player_rule"),
        Index("ix_slate_player_rule_week", "season", "week"),
    )


class LineupEntry(Base):
    """One built lineup, and what it went on to score.

    `actual_points` is the sum over the players that could be scored, with
    `unscored_players` saying how many could not be -- a defense on a week
    nflverse team stats were unavailable, or a player who did not play. A
    lineup with anything unscored is not comparable to a contest result, so the
    count is stored beside the total rather than folded into it.
    """
    __tablename__ = "lineup_entries"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    build_id         = Column(Integer, index=True)
    lineup_index     = Column(Integer)     # 1 = the most expensive, as returned
    salary           = Column(Integer)
    projected_points = Column(Float)
    actual_points    = Column(Float)
    scored_players   = Column(Integer)
    unscored_players = Column(Integer)
    scored_at        = Column(DateTime)

    __table_args__ = (Index("ix_lineup_entry_build", "build_id", "lineup_index"),)


class LineupPlayer(Base):
    """A player in a built lineup, with his projection frozen at build time."""
    __tablename__ = "lineup_players"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    lineup_id        = Column(Integer, index=True)
    slot             = Column(String)      # the CSV's Roster Position
    # The CSV's Position. Kept because it is what identifies a DEFENSE at
    # scoring time: defenses have no player_id and no stat line, and in a
    # Showdown lineup the slot is "MVP"/"AnyFLEX" and names no position at all.
    position         = Column(String)
    player_id        = Column(String, index=True)  # null for DEF, and when the name join missed
    # FanDuel's own id, "<contest>-<player>". Kept because it is the only thing
    # their bulk-entry upload accepts, and it cannot be reconstructed from a
    # name: the contest half differs per contest.
    fanduel_id       = Column(String)
    player_name      = Column(String)
    team             = Column(String)
    salary           = Column(Integer)
    projected_points = Column(Float)
    from_model       = Column(Integer)     # 0 = FanDuel's FPPG (defenses, join misses)
    actual_points    = Column(Float)

    __table_args__ = (Index("ix_lineup_player_lineup", "lineup_id"),)


class MatchupSignal(Base):
    """What the matchup signals said about a week, frozen before kickoff.

    Every matchup idea in the projection repo's EXPERIMENTS.md was judged by
    fitting it to history, and the few that did not come back null came back
    borderline -- where there is no honest way to separate a real effect from
    the best of sixteen looks at one dataset. The trench report sits exactly
    there for running backs: a +0.166 correlation with the board's error whose
    bootstrap interval touches zero, but which pointed the right way in five of
    five testable weeks for a pooled +4.98 points.

    So this table is the only thing that can settle it: the signals recorded
    before each slate, scored afterwards, accumulating a sample nobody has
    looked at. Nothing here adjusts a projection, and a week whose answer is
    "no" is as valuable as the others.
    """
    __tablename__ = "matchup_signals"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    season          = Column(Integer, index=True)
    week            = Column(Integer, index=True)
    player_id       = Column(String, index=True)
    player_name     = Column(String)
    position        = Column(String)
    team            = Column(String)
    opponent        = Column(String)
    projected       = Column(Float)
    run_matchup     = Column(Float)
    pass_matchup    = Column(Float)
    opp_cover_grade = Column(Float)
    run_lean        = Column(String)
    rosterable      = Column(Integer)
    actual          = Column(Float)
    recorded_at     = Column(DateTime, default=datetime.utcnow)
    scored_at       = Column(DateTime)

    __table_args__ = (
        UniqueConstraint("season", "week", "player_id", name="uq_matchup_signal"),
    )


def apply_light_migrations(engine) -> None:
    """Idempotent additive schema fixups for columns added after a table already
    exists (no Alembic here; ``create_all`` only creates missing tables, never alters
    existing ones). Adds any missing column via ``ALTER TABLE ... ADD COLUMN`` — the
    ``ADD COLUMN <name> <type>`` form is supported by both SQLite and Postgres."""
    from sqlalchemy import inspect, text

    # espn_roster became current_roster when nflverse took over as the source: a
    # rebuilt-nightly cache with a different primary key, so it is dropped rather
    # than migrated, and the next sync fills the new table.
    with engine.begin() as conn:
        try:
            if inspect(engine).has_table("espn_roster"):
                conn.execute(text("DROP TABLE espn_roster"))
        except Exception as exc:                       # pragma: no cover - best effort
            logger.warning("Could not drop the old espn_roster table: %s", exc)

    wanted = {("player_projections", "exp_games"): "FLOAT",
              ("lineup_builds", "max_per_team"): "INTEGER",
              ("lineup_builds", "stack_qb"): "INTEGER",
              ("lineup_builds", "rb_with_qb"): "INTEGER",
              ("lineup_builds", "ownership_weight"): "FLOAT",
              ("lineup_players", "fanduel_id"): "VARCHAR",
              ("matchup_signals", "scored_at"): "TIMESTAMP",
              ("lineup_builds", "projection_computed_at"): "TIMESTAMP",
              ("lineup_builds", "player_rules"): "TEXT",
              ("lineup_builds", "deduped_against"): "INTEGER",
              ("lineup_builds", "duplicates_skipped"): "INTEGER",
              ("lineup_builds", "pool_exposure"): "INTEGER"}
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    for (table, col), coltype in wanted.items():
        if table not in existing_tables:
            continue  # create_all will make it fresh with the column
        cols = {c["name"] for c in insp.get_columns(table)}
        if col in cols:
            continue
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}"))
