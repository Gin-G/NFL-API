#!/usr/bin/env python3
"""
Chat API Router
LLM-powered chatbot using Claude tool-use to answer NFL data questions.

Tool execution queries the database first; falls back to nflreadpy when the
DB has no data for the requested criteria.
"""

import os
import json
import logging
from typing import List

import nflreadpy as nfl
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .utils import clean_data_for_json, get_current_nfl_season, _to_pandas, _orm_to_dict
from database.session import get_db
from database.models import PlayerStat, PlayerRoster, Schedule, Team as TeamModel

logger = logging.getLogger(__name__)
router = APIRouter()

MODEL = os.getenv("CHAT_MODEL", "claude-sonnet-4-6")
MAX_ITERATIONS = 5
MAX_TOKENS = 2048

SYSTEM_PROMPT = (
    "You are an NFL analytics assistant. "
    "You MUST use the provided tools for every factual answer — "
    "never use your training knowledge for statistics, rosters, scores, or records. "
    "If a tool returns no data, say so honestly. "
    "Be concise. Format numbers clearly (e.g., '1,234 yards')."
)

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_player_stats",
        "description": (
            "Fetch NFL player statistics. Returns weekly rows; omit week for all weeks. "
            "Use sort_by to rank (e.g. receiving_yards desc) and limit to control result size."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "season": {
                    "type": "integer",
                    "description": "Season year, e.g. 2024",
                },
                "week": {
                    "type": "integer",
                    "description": "1-18 regular season, 19-22 playoffs",
                },
                "position": {
                    "type": "string",
                    "description": "QB, WR, RB, TE, etc.",
                },
                "team": {
                    "type": "string",
                    "description": "Team abbreviation, e.g. KC",
                },
                "sort_by": {
                    "type": "string",
                    "description": "Column to sort by, e.g. receiving_yards",
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort direction (default: desc)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max records to return (default 20, max 50)",
                },
            },
        },
    },
    {
        "name": "get_player_roster",
        "description": (
            "Fetch NFL player roster information including positions, teams, "
            "and physical attributes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "season": {
                    "type": "integer",
                    "description": "Season year, e.g. 2024",
                },
                "week": {
                    "type": "integer",
                    "description": "Specific roster week",
                },
                "team": {
                    "type": "string",
                    "description": "Team abbreviation, e.g. KC",
                },
                "position": {
                    "type": "string",
                    "description": "Player position filter, e.g. QB",
                },
            },
        },
    },
    {
        "name": "get_schedules",
        "description": (
            "Fetch NFL game schedules and results. "
            "Includes scores, teams, dates, and stadium info."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "season": {
                    "type": "integer",
                    "description": "Season year, e.g. 2024",
                },
                "week": {
                    "type": "integer",
                    "description": "Specific week (1-22)",
                },
                "team": {
                    "type": "string",
                    "description": "Team abbreviation to filter by",
                },
            },
        },
    },
    {
        "name": "get_teams",
        "description": (
            "Fetch all NFL teams with their abbreviations, names, "
            "conferences, and divisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_coaches",
        "description": "Fetch NFL head coach records by season with win/loss records.",
        "input_schema": {
            "type": "object",
            "properties": {
                "years": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of season years, e.g. [2024]",
                },
            },
        },
    },
]


# ── Internal normalisation helpers (mirrors players.py) ───────────────────────

def _normalize_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "team": "recent_team",
        "passing_interceptions": "interceptions",
    })


def _normalize_roster_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
    if "player_display_name" not in df.columns:
        df = df.copy()
        df["player_display_name"] = df.get("player_name")
    if "height" in df.columns and df["height"].dtype != object:
        df = df.copy()
        df["height"] = df["height"].apply(
            lambda h: f"{int(h)//12}-{int(h)%12}" if pd.notna(h) else None
        )
    return df


# ── Tool execution ────────────────────────────────────────────────────────────

def _execute_tool(name: str, inputs: dict, db: Session = None):
    """Execute a named tool call and return JSON-serialisable results."""
    try:
        if name == "get_player_stats":
            season = inputs.get("season") or get_current_nfl_season()
            limit = min(int(inputs.get("limit") or 20), 50)

            if db is not None:
                q = db.query(PlayerStat).filter(PlayerStat.season == season)
                if inputs.get("position"):
                    q = q.filter(PlayerStat.position == inputs["position"].upper())
                if inputs.get("team"):
                    q = q.filter(PlayerStat.recent_team == inputs["team"].upper())
                if inputs.get("week") is not None:
                    q = q.filter(PlayerStat.week == inputs["week"])
                sort_by = inputs.get("sort_by", "fantasy_points")
                col = getattr(PlayerStat, sort_by, None)
                if col is not None:
                    asc = inputs.get("sort_order", "desc") == "asc"
                    q = q.order_by(col.asc() if asc else col.desc())
                rows = q.limit(limit).all()
                if rows:
                    return [_orm_to_dict(r) for r in rows]

            # Fallback: nflreadpy
            stats = _normalize_stats_df(_to_pandas(nfl.load_player_stats(seasons=[season])))
            if inputs.get("position"):
                stats = stats[stats["position"] == inputs["position"].upper()]
            if inputs.get("team"):
                stats = stats[stats["recent_team"] == inputs["team"].upper()]
            if inputs.get("week") is not None:
                stats = stats[stats["week"] == inputs["week"]]
            sort_by = inputs.get("sort_by")
            if sort_by and sort_by in stats.columns:
                ascending = inputs.get("sort_order", "desc") == "asc"
                stats = stats.sort_values(sort_by, ascending=ascending)
            return clean_data_for_json(stats.head(limit))

        elif name == "get_player_roster":
            season = inputs.get("season") or get_current_nfl_season()

            if db is not None:
                q = db.query(PlayerRoster).filter(PlayerRoster.season == season)
                if inputs.get("week") is not None:
                    q = q.filter(PlayerRoster.week == inputs["week"])
                if inputs.get("team"):
                    q = q.filter(PlayerRoster.team == inputs["team"].upper())
                if inputs.get("position"):
                    q = q.filter(PlayerRoster.position == inputs["position"].upper())
                rows = q.limit(50).all()
                if rows:
                    if inputs.get("week") is None:
                        # Deduplicate to latest week per player
                        seen: dict = {}
                        for r in rows:
                            existing = seen.get(r.player_id)
                            if existing is None or r.week > existing.week:
                                seen[r.player_id] = r
                        rows = list(seen.values())[:50]
                    return [_orm_to_dict(r) for r in rows]

            # Fallback: nflreadpy
            rosters = _normalize_roster_df(_to_pandas(nfl.load_rosters_weekly(seasons=[season])))
            if inputs.get("week") is not None:
                rosters = rosters[rosters["week"] == inputs["week"]]
            else:
                rosters = rosters.sort_values("week").drop_duplicates(
                    subset=["player_id"], keep="last"
                )
            if inputs.get("team"):
                rosters = rosters[rosters["team"] == inputs["team"].upper()]
            if inputs.get("position"):
                rosters = rosters[rosters["position"] == inputs["position"].upper()]
            return clean_data_for_json(rosters.head(50))

        elif name == "get_schedules":
            season = inputs.get("season") or get_current_nfl_season()

            if db is not None:
                q = db.query(Schedule).filter(Schedule.season == season)
                if inputs.get("week") is not None:
                    q = q.filter(Schedule.week == inputs["week"])
                if inputs.get("team"):
                    t = inputs["team"].upper()
                    q = q.filter((Schedule.home_team == t) | (Schedule.away_team == t))
                rows = q.limit(50).all()
                if rows:
                    return [_orm_to_dict(r) for r in rows]

            # Fallback: nflreadpy
            schedules = _to_pandas(nfl.load_schedules(seasons=[season]))
            if inputs.get("week") is not None:
                schedules = schedules[schedules["week"] == inputs["week"]]
            if inputs.get("team"):
                team = inputs["team"].upper()
                schedules = schedules[
                    (schedules["home_team"] == team) | (schedules["away_team"] == team)
                ]
            return clean_data_for_json(schedules.head(50))

        elif name == "get_teams":
            if db is not None:
                rows = db.query(TeamModel).all()
                if rows:
                    return [_orm_to_dict(r) for r in rows]

            # Fallback: nflreadpy
            teams = _to_pandas(nfl.load_teams())
            return clean_data_for_json(teams)

        elif name == "get_coaches":
            from .utils import get_coaching_analytics
            years = inputs.get("years") or [get_current_nfl_season()]
            analytics = get_coaching_analytics(years)
            coaches = analytics.get_available_coaches()
            result = []
            for coach in coaches[:30]:
                seasons = []
                for (c, s), data in analytics.coaching_data.items():
                    if c == coach:
                        games = data["games"]
                        wins = sum(1 for g in games if g["result"] == "W")
                        losses = sum(1 for g in games if g["result"] == "L")
                        total = wins + losses
                        seasons.append({
                            "season": s,
                            "teams": list(data["teams"]),
                            "wins": wins,
                            "losses": losses,
                            "win_pct": round(wins / total * 100, 1) if total > 0 else 0,
                        })
                result.append({
                    "name": coach,
                    "seasons": sorted(seasons, key=lambda x: x["season"]),
                })
            return result

        else:
            return {"error": f"Unknown tool: {name}"}

    except Exception as exc:
        logger.error("Tool %s failed: %s", name, exc)
        return {"error": str(exc)}


# ── Request / Response models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str
    history: List[ChatMessage]


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """LLM-powered chat endpoint. Requires ANTHROPIC_API_KEY environment variable."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Chat is not available: ANTHROPIC_API_KEY not configured.",
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Chat is not available: anthropic package not installed.",
        )

    # Build message list from history + new user message
    messages = [{"role": m.role, "content": m.content} for m in request.history]
    messages.append({"role": "user", "content": request.message})

    try:
        for _ in range(MAX_ITERATIONS):
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if resp.stop_reason == "end_turn":
                text = "".join(b.text for b in resp.content if b.type == "text")
                messages.append({"role": "assistant", "content": text})
                history = [
                    ChatMessage(role=m["role"], content=m["content"])
                    for m in messages
                    if isinstance(m.get("content"), str)
                ]
                return ChatResponse(response=text, history=history)

            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                tool_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(
                            _execute_tool(block.name, block.input, db=db), default=str
                        ),
                    }
                    for block in resp.content
                    if block.type == "tool_use"
                ]
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        raise HTTPException(
            status_code=500,
            detail="Chat loop did not produce a final response.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Anthropic API error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Chat service error: {exc}",
        )
