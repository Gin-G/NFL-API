"""One player reaching the projection twice must not kill the run.

2026-11-13: the Friday weekend refresh died with "cannot convert the series to
<class 'float'>" before writing a row, which is what `float(sm.loc[pid, ...])`
raises when a player_id appears twice. Wednesday's board stood through the
injury news the refresh exists to pick up.
"""
import pandas as pd
import pytest

from scripts import compute_projections as cp


def _frame():
    return pd.DataFrame({
        "player_id": ["00-1", "00-1", "00-2"],
        "player_name": ["James Cook", "James Cook III", "Josh Allen"],
        "team": ["BUF", "BUF", "BUF"], "position": ["RB", "RB", "QB"],
        "fanduel_fantasy_points": [9.3, 18.0, 18.2],
        "floor": [1.0, 2.0, 3.0], "projection_median": [8.0, 16.0, 17.0],
        "ceiling": [18.0, 30.0, 32.0],
    })


def test_keeps_one_row_per_player_id():
    out = cp._one_row_per_player(_frame())
    assert list(out.player_id) == ["00-1", "00-2"]
    assert out.player_name.iloc[0] == "James Cook"


def test_leaves_a_clean_frame_alone():
    f = _frame().iloc[1:]
    assert cp._one_row_per_player(f) is f


def test_tolerates_missing_ids_and_empty_frames():
    f = _frame()
    f.loc[2, "player_id"] = None
    f = pd.concat([f, f.iloc[[2]]], ignore_index=True)
    out = cp._one_row_per_player(f)
    assert out.player_id.isna().sum() == 2      # no id is not a duplicate
    assert cp._one_row_per_player(pd.DataFrame()).empty
    assert cp._one_row_per_player(None) is None


def test_simulator_survives_a_doubled_player(monkeypatch):
    """The exact failure: the simulator's summary repeats an id."""
    nflp_sim = pytest.importorskip("nfl_projections.simulate")
    summ = pd.DataFrame({"player_id": ["00-1", "00-1", "00-2"],
                         "floor": [2.0, 2.5, 4.0], "median": [9.0, 9.5, 17.0],
                         "ceiling": [20.0, 21.0, 33.0]})
    monkeypatch.setattr(nflp_sim, "project_distributions",
                        lambda frame, history, n_sims, seed: (summ, None))
    frame = _frame()
    cp._apply_simulator(frame, pd.DataFrame(), 10)
    assert frame.loc[2, "floor"] == 4.0
    assert frame.loc[0, "floor"] == 2.0
