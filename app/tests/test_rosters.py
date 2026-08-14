"""The current-roster endpoint, and the depth chart it carries."""

from datetime import datetime

import pytest

from database.models import EspnRoster


@pytest.fixture
def raiders(db_session):
    """A quarterback room where the depth chart says more than the status does."""
    rows = [
        EspnRoster(
            espn_id="1", gsis_id="00-0000001", full_name="Kirk Cousins",
            position="QB", team="LV", status="active", depth_rank=1,
            updated_at=datetime(2026, 8, 4, 14, 0, 0),
        ),
        EspnRoster(
            espn_id="2", gsis_id="00-0041562", full_name="Fernando Mendoza",
            position="QB", team="LV", status="active", depth_rank=2,
            updated_at=datetime(2026, 8, 4, 14, 0, 0),
        ),
        EspnRoster(
            espn_id="3", gsis_id="00-0000003", full_name="Practice Squadder",
            position="QB", team="LV", status="practice_squad", depth_rank=None,
            updated_at=datetime(2026, 8, 4, 14, 0, 0),
        ),
    ]
    db_session.add_all(rows)
    db_session.commit()
    return rows


def test_a_team_roster_carries_the_depth_chart(client, raiders):
    body = client.get("/rosters/LV").json()

    by_name = {r["player_name"]: r for r in body["data"]}
    assert by_name["Kirk Cousins"]["depth_rank"] == 1
    assert by_name["Fernando Mendoza"]["depth_rank"] == 2


def test_the_snapshot_says_when_it_was_taken(client, raiders):
    """This table is a snapshot, so a caller has to be able to age it."""
    body = client.get("/rosters/LV").json()

    assert body["data"][0]["updated_at"].startswith("2026-08-04T14:00:00")


def test_a_player_lookup_carries_it_too(client, raiders):
    body = client.get("/rosters/player/00-0041562").json()

    assert body["data"]["depth_rank"] == 2
    assert body["data"]["team"] == "LV"


def test_the_active_filter_still_holds(client, raiders):
    body = client.get("/rosters/LV?status=active").json()

    assert {r["player_name"] for r in body["data"]} == {"Kirk Cousins", "Fernando Mendoza"}


def test_a_player_with_no_rank_reports_none_rather_than_dropping_out(client, raiders):
    body = client.get("/rosters/LV?status=all").json()

    squadder = next(r for r in body["data"] if r["player_name"] == "Practice Squadder")
    assert squadder["depth_rank"] is None


def test_an_unsynced_team_says_so(client, raiders):
    assert client.get("/rosters/GB").status_code == 404
