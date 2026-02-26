#!/usr/bin/env python3
"""
SQLAlchemy ORM models for NFL data.
"""

from datetime import datetime

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, UniqueConstraint, Index
from sqlalchemy.orm import DeclarativeBase


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
