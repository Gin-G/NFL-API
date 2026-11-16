"""A player ruled out after the board was built: zeroed, and his backups
promoted on the stored board, without rerunning the model.

2026 week 10: Kenneth Walker was ruled out, the weekend refresh never ran, and
Emmett Johnson stayed a 6.0-point backup in every build. He scored 35.1.
"""
import io
from datetime import datetime

import pandas as pd
import pytest

from database.models import FanduelSlate, PlayerProjection, SlatePlayerRule

pytest.importorskip("nfl_projections.roles")

from api.optimizer import _apply_scratches                          # noqa: E402

SEASON, WEEK = 2026, 10


def _kc(depth_rank=None):
    return pd.DataFrame({
        "Nickname": ["Kenneth Walker", "Emmett Johnson", "Jaydn Ott", "Rashee Rice"],
        "Position": ["RB", "RB", "RB", "WR"],
        "Team": ["KC", "KC", "KC", "KC"],
        "fanduel_fantasy_points": [20.4, 6.0, 1.3, 10.5],
        "floor": [6.2, 1.0, 0.2, 2.1], "ceiling": [38.8, 12.6, 3.0, 21.4],
        "depth_rank": depth_rank if depth_rank is not None else [None] * 4,
    })


class TestTheArithmetic:
    def test_the_rb1_out_promotes_the_rb2_into_the_job(self):
        from nfl_projections import roles
        out, notes = _apply_scratches(_kc(), ["Kenneth Walker"])
        mult = roles.promotion_multiplier("RB", 2, (1,))
        assert out.loc[0, "fanduel_fantasy_points"] == 0.0
        assert out.loc[1, "fanduel_fantasy_points"] == pytest.approx(6.0 * mult)
        assert out.loc[1, "ceiling"] == pytest.approx(12.6 * mult)
        assert mult == pytest.approx(1.25 * 1.42)          # the RB2-into-starter factor
        assert "Kenneth Walker" in notes["Emmett Johnson"]
        assert notes["Kenneth Walker"] == "ruled out"

    def test_the_third_back_moves_up_one_and_receivers_are_untouched(self):
        out, _ = _apply_scratches(_kc(), ["Kenneth Walker"])
        assert out.loc[2, "fanduel_fantasy_points"] == pytest.approx(1.3 * 0.80 / 0.55)
        assert out.loc[3, "fanduel_fantasy_points"] == 10.5

    def test_a_stored_depth_rank_wins_over_the_board_order(self):
        # The chart says Ott is the RB2, whatever the board's order.
        out, notes = _apply_scratches(_kc(depth_rank=[1, 3, 2, 1]), ["Kenneth Walker"])
        assert out.loc[2, "fanduel_fantasy_points"] > 1.3 * 1.25 * 1.42 - 1e-9
        assert out.loc[1, "fanduel_fantasy_points"] == pytest.approx(6.0 * 0.80 / 0.55)

    def test_a_backup_out_promotes_nobody_above_him(self):
        out, notes = _apply_scratches(_kc(), ["Jaydn Ott"])
        assert out.loc[0, "fanduel_fantasy_points"] == 20.4
        assert out.loc[1, "fanduel_fantasy_points"] == 6.0
        assert set(notes) == {"Jaydn Ott"}

    def test_nothing_out_changes_nothing(self):
        frame = _kc()
        out, notes = _apply_scratches(frame, [])
        assert out is frame and notes == {}


SLATE = [("Kenneth Walker", "RB", 9800, "KC"), ("Emmett Johnson", "RB", 5000, "KC"),
         ("Jaydn Ott", "RB", 4500, "KC")]


@pytest.fixture
def week(db_session):
    for model in (SlatePlayerRule, FanduelSlate, PlayerProjection):
        db_session.query(model).delete()
    for pid, (name, pos, _, team), pts in zip(("w", "j", "o"), SLATE, (20.4, 6.0, 1.3)):
        db_session.add(PlayerProjection(
            season=SEASON, week=WEEK, player_id=pid, player_name=name, position=pos,
            team=team, projected_points=pts, floor=pts * .3, median=pts, ceiling=pts * 2,
            prediction_type="veteran_ml", model_version="t", computed_at=datetime.utcnow()))
    lines = ["Id,Nickname,Position,Roster Position,Salary,FPPG,Team,Injury Indicator"]
    lines += [f"{i},{n},{p},{p},{s},10,{t}," for i, (n, p, s, t) in enumerate(SLATE)]
    db_session.add(FanduelSlate(season=SEASON, week=WEEK, contest="classic",
                                content="\n".join(lines), uploaded_at=datetime.utcnow()))
    db_session.commit()
    yield


def test_the_pool_shows_the_scratch_and_the_promotion(client, db_session, week):
    r = client.post("/optimizer/pool", params={"season": SEASON, "week": WEEK},
                    json=[{"nickname": "Kenneth Walker", "rule": "out"}])
    assert r.status_code == 200, r.text
    assert r.json()["rules"]["Kenneth Walker"]["kind"] == "out"
    rows = {p["nickname"]: p for p in client.get(
        "/optimizer/pool", params={"season": SEASON, "week": WEEK,
                                   "history": False}).json()["data"]}
    assert rows["Kenneth Walker"]["projected_points"] == 0.0
    assert rows["Emmett Johnson"]["projected_points"] == pytest.approx(6.0 * 1.25 * 1.42, rel=1e-3)
    assert "ruled out" in rows["Emmett Johnson"]["scratch"]
