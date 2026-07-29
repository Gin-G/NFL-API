"""Current-roster endpoint, backed by the nightly ESPN sync (`espn_roster` table)."""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database.session import get_db
from database.models import EspnRoster

logger = logging.getLogger(__name__)
router = APIRouter()


def _row(r: EspnRoster) -> dict:
    return {
        "espn_id": r.espn_id, "gsis_id": r.gsis_id, "player_name": r.full_name,
        "position": r.position, "team": r.team, "status": r.status,
        "jersey": r.jersey, "age": r.age, "experience": r.experience,
    }


@router.get("/{team}")
async def get_team_roster(
    team: str,
    position: Optional[str] = Query(None, description="Filter to QB/RB/WR/TE/..."),
    status: Optional[str] = Query("active", description="active | injured_reserve | practice_squad | suspended | all"),
    db: Session = Depends(get_db),
):
    """Current roster for a team (from the nightly ESPN sync)."""
    q = db.query(EspnRoster).filter(EspnRoster.team == team.upper())
    if position:
        q = q.filter(EspnRoster.position == position.upper())
    if status and status != "all":
        q = q.filter(EspnRoster.status == status)
    rows = q.all()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No roster for {team} (sync may not have run)")
    return {"status": "success", "team": team.upper(), "count": len(rows),
            "data": [_row(r) for r in rows]}


@router.get("/player/{gsis_id}")
async def get_player_team(gsis_id: str, db: Session = Depends(get_db)):
    """Current team/status for a player by gsis_id."""
    r = db.query(EspnRoster).filter(EspnRoster.gsis_id == gsis_id).first()
    if r is None:
        raise HTTPException(status_code=404, detail=f"{gsis_id} not on a current roster")
    return {"status": "success", "data": _row(r)}
