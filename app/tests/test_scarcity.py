#!/usr/bin/env python3
"""Tests for the positional scarcity endpoints.

The point of these numbers is that a projection means different things at
different positions, so the fixture builds a league where that is true by
construction: quarterbacks score most, tight ends least, and each position has
more players than it has starting slots so replacement level exists.
"""
import pytest

from database.models import PlayerStat

pytest.importorskip("nfl_projections.scarcity",
                    reason="scarcity maths comes from nfl_projections (installed --no-deps)")

SEASONS = (2023, 2024)
# position -> (players, best player's per-game yards, decline per rank)
SHAPE = {"QB": (18, 330.0, 11.0), "RB": (30, 150.0, 4.0),
         "WR": (42, 130.0, 3.0), "TE": (20, 80.0, 2.5)}


@pytest.fixture(autouse=True)
def _stats(db_session):
    from api import scarcity

    scarcity._cache.clear()
    db_session.query(PlayerStat).delete()
    rows = []
    for season in SEASONS:
        for pos, (count, top, decay) in SHAPE.items():
            for rank in range(count):
                yards = max(top - decay * rank, 5.0)
                for week in range(1, 13):
                    stat = PlayerStat(
                        player_id=f"{pos}{rank}", season=season, week=week,
                        player_display_name=f"{pos} {rank}", position=pos,
                        recent_team="AAA", season_type="REG")
                    if pos == "QB":
                        stat.passing_yards = yards
                        stat.passing_tds = 1.5
                    elif pos == "RB":
                        stat.rushing_yards = yards
                        stat.receptions = 2.0
                    else:
                        stat.receiving_yards = yards
                        stat.receptions = 5.0
                    rows.append(stat)
    db_session.add_all(rows)
    db_session.commit()
    yield
    db_session.query(PlayerStat).delete()
    db_session.commit()
    scarcity._cache.clear()


class TestReplacement:
    def test_replacement_sits_below_the_last_starter(self, client):
        body = client.get("/scarcity/replacement?seasons=2").json()
        assert body["seasons"] == list(SEASONS)
        for row in body["pooled"]:
            assert row["replacement_ppg"] < row["starter_ppg"]

    def test_quarterback_replacement_beats_tight_end_replacement(self, client):
        """The whole reason the endpoint exists: the same projection is not
        worth the same at both positions."""
        pooled = {r["position"]: r for r in
                  client.get("/scarcity/replacement?seasons=2").json()["pooled"]}
        assert pooled["QB"]["replacement_ppg"] > pooled["TE"]["replacement_ppg"]

    def test_publishes_the_slots_and_bar_it_used(self, client):
        body = client.get("/scarcity/replacement?seasons=2").json()
        assert body["starter_slots"]["QB"] == 12
        assert body["usable_bar"] > 0
        assert body["scoring"] == "FanDuel half-PPR"

    def test_no_stats_is_a_404(self, client, db_session):
        from api import scarcity
        db_session.query(PlayerStat).delete()
        db_session.commit()
        scarcity._cache.clear()
        assert client.get("/scarcity/replacement").status_code == 404


class TestUsable:
    def test_rates_are_shares(self, client):
        body = client.get("/scarcity/usable?seasons=2").json()
        for row in body["pooled"] + body["per_season"]:
            assert 0.0 <= row["usable_rate"] <= 1.0

    def test_a_higher_bar_cannot_raise_the_rate(self, client):
        low = {r["position"]: r["usable_rate"] for r in
               client.get("/scarcity/usable?seasons=2&bar=10").json()["pooled"]}
        high = {r["position"]: r["usable_rate"] for r in
                client.get("/scarcity/usable?seasons=2&bar=25").json()["pooled"]}
        for pos in low:
            assert high[pos] <= low[pos]

    def test_the_flat_bar_favours_quarterbacks(self, client):
        """Not a bug in the bar — the finding that justifies a positional one."""
        pooled = {r["position"]: r["usable_rate"] for r in
                  client.get("/scarcity/usable?seasons=2").json()["pooled"]}
        assert pooled["QB"] > pooled["TE"]


class TestComposition:
    def test_counts_add_up_to_each_cut(self, client):
        body = client.get("/scarcity/composition?seasons=2").json()
        totals = {}
        for row in body["per_season"]:
            totals[(row["season"], row["top_n"])] = \
                totals.get((row["season"], row["top_n"]), 0) + row["players"]
        for (season, top_n), total in totals.items():
            assert total == top_n

    def test_reports_what_it_ranked_on(self, client):
        body = client.get("/scarcity/composition?seasons=2").json()
        assert body["ranked_on"] == "season total points"
        assert {r["top_n"] for r in body["pooled"]} == {25, 50, 100}
