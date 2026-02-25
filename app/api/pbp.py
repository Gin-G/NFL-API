#!/usr/bin/env python3
"""
Play-By-Play API Router
Exposes raw PBP records stored in the play_by_play table (loaded from nflreadpy).

The table is created dynamically by pandas to_sql on first load — no ORM model.
All queries use raw SQL via SQLAlchemy text().
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

_NO_DATA = {"status": "no_data", "total_plays": 0, "data": []}


@router.get("/")
async def get_pbp(
    game_id: Optional[str] = Query(None, description="Return all plays for a specific game"),
    season: Optional[int] = Query(None, description="Season year, e.g. 2024"),
    week: Optional[int] = Query(None, description="Week number (1-22)"),
    team: Optional[str] = Query(None, description="Possession team abbreviation, e.g. KC"),
    def_team: Optional[str] = Query(None, description="Defending team abbreviation"),
    play_type: Optional[str] = Query(None, description="play_type value: pass, run, punt, kickoff, etc."),
    down: Optional[int] = Query(None, description="Down (1-4)"),
    limit: Optional[int] = Query(500, description="Max plays to return when game_id not specified (max 500)"),
    db: Session = Depends(get_db),
):
    """
    Fetch play-by-play records.

    When `game_id` is provided, returns all plays for that game (no cap).
    Otherwise applies filters and caps at `limit` (max 500).
    """
    try:
        conditions: list[str] = []
        params: dict = {}

        if game_id:
            conditions.append("game_id = :game_id")
            params["game_id"] = game_id
        if season is not None:
            conditions.append("season = :season")
            params["season"] = season
        if week is not None:
            conditions.append("week = :week")
            params["week"] = week
        if team:
            conditions.append("posteam = :team")
            params["team"] = team.upper()
        if def_team:
            conditions.append("defteam = :def_team")
            params["def_team"] = def_team.upper()
        if play_type:
            conditions.append("play_type = :play_type")
            params["play_type"] = play_type.lower()
        if down is not None:
            conditions.append("down = :down")
            params["down"] = down

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        # No row cap when fetching a specific game; otherwise cap at 500
        cap = "" if game_id else f"LIMIT {min(int(limit or 500), 500)}"

        sql = f"SELECT * FROM play_by_play {where} ORDER BY game_id, play_id {cap}"
        rows = db.execute(text(sql), params).mappings().all()
        data = [dict(r) for r in rows]

        return {
            "status": "success",
            "total_plays": len(data),
            "data": data,
        }

    except Exception as exc:
        logger.warning("PBP query failed (table may not exist yet): %s", exc)
        return _NO_DATA
