"""Current-roster endpoints, backed by the nightly roster sync (`current_roster`).

nflverse's weekly rosters in season, ESPN's live API before nflverse publishes the
new year — `source` on every row says which, and `updated_at` says when.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database.session import get_db
from database.models import CurrentRoster

logger = logging.getLogger(__name__)
router = APIRouter()


def _row(r: CurrentRoster) -> dict:
    return {
        "espn_id": r.espn_id, "gsis_id": r.gsis_id, "player_name": r.full_name,
        "position": r.position, "team": r.team, "status": r.status,
        "jersey": r.jersey, "age": r.age, "experience": r.experience,
        # Where he sits on his team's depth chart, 1 being the starter. The one
        # field here that says whether a player will be on the field, and the
        # only current answer for anyone a roster line alone cannot settle: a
        # veteran who is employed but third on the chart is not a starter.
        "depth_rank": r.depth_rank,
        # Which upstream this row came from, and the code it used. nflverse's own
        # ACT/INA/RES is kept because the mapping above deliberately flattens it
        # (INA is a gameday designation, not roster standing).
        "source": r.source,
        "raw_status": r.raw_status,
        # When the sync behind this row ran. Exposed because the table is a
        # snapshot rather than a live read, and a caller weighing it against
        # other sources should be able to see how old it is instead of assuming
        # it is current.
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    }


@router.get("/{team}")
async def get_team_roster(
    team: str,
    position: Optional[str] = Query(None, description="Filter to QB/RB/WR/TE/..."),
    status: Optional[str] = Query("active", description="active | injured_reserve | practice_squad | suspended | all"),
    db: Session = Depends(get_db),
):
    """Current roster for a team, as of the last roster sync."""
    q = db.query(CurrentRoster).filter(CurrentRoster.team == team.upper())
    if position:
        q = q.filter(CurrentRoster.position == position.upper())
    if status and status != "all":
        q = q.filter(CurrentRoster.status == status)
    rows = q.all()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No roster for {team} (sync may not have run)")
    return {"status": "success", "team": team.upper(), "count": len(rows),
            "data": [_row(r) for r in rows]}


@router.get("/player/{gsis_id}")
async def get_player_team(gsis_id: str, db: Session = Depends(get_db)):
    """Current team/status for a player by gsis_id."""
    r = db.query(CurrentRoster).filter(CurrentRoster.gsis_id == gsis_id).first()
    if r is None:
        raise HTTPException(status_code=404, detail=f"{gsis_id} not on a current roster")
    return {"status": "success", "data": _row(r)}
