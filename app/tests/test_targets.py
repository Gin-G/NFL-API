"""Players to Target: expert picks beside our board, in agree / skeptical piles."""
import io
from datetime import datetime

import pytest

from database.models import ExpertPick, FanduelSlate, PlayerProjection, PlayerStat

pytest.importorskip("nfl_projections.names", reason="matcher comes from nfl_projections")

from api import targets                                               # noqa: E402

SEASON, WEEK = 2026, 10


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in (ExpertPick, FanduelSlate, PlayerProjection, PlayerStat):
        db_session.query(model).delete()
    db_session.commit()
    yield


def _board(db):
    """Eighteen RBs and ten TEs ahead of the two we test, so both sit outside
    the depth (16 RBs, 8 TEs) by points."""
    rows = [(f"rb{i:02d}", f"Back Number{i}", "RB", "XXX", 25.0 - i) for i in range(18)]
    rows += [(f"te{i:02d}", f"Tight End{i}", "TE", "YYY", 11.0 - i * .5) for i in range(10)]
    rows += [("cook", "James Cook", "RB", "BUF", 7.0),        # rank 19 by points
             ("pitts", "Kyle Pitts", "TE", "ATL", 2.5),
             ("bowers", "Brock Bowers", "TE", "LV", 12.6)]
    for pid, name, pos, team, pts in rows:
        db.add(PlayerProjection(season=SEASON, week=WEEK, player_id=pid,
                                player_name=name, position=pos, team=team,
                                projected_points=pts, floor=pts * .4, median=pts,
                                ceiling=pts * 1.9, prediction_type="veteran_ml",
                                model_version="t", computed_at=datetime.utcnow()))
    db.commit()


def _slate(db, rows):
    lines = ["Id,Nickname,Position,Roster Position,Salary,FPPG,Team,Injury Indicator"]
    lines += [f"{i},{n},{p},{p},{s},10,{t}," for i, (n, p, s, t) in enumerate(rows)]
    db.add(FanduelSlate(season=SEASON, week=WEEK, contest="classic",
                        content="\n".join(lines), uploaded_at=datetime.utcnow()))
    db.commit()


PICKS = """player,team,position,tags,article,author,salary_fd
James Cook III,BUF,RB,Cash + GPP,RB Coach,Ted Schuster,8400
Kyle Pitts Sr.,ATL,TE,GPP,TE Coach,Jorge Pucks,5100
Brock Bowers,LV,TE,cash;gpp,TE Coach,Jorge Pucks,8000
Nobody Here,ARI,WR,punt,WR Coach,Tyler Buecher,4300
"""


def _upload(client, text=PICKS):
    return client.post("/targets/picks", data={"season": SEASON, "week": WEEK},
                       files={"file": ("picks.csv", io.BytesIO(text.encode()), "text/csv")})


class TestParsing:
    def test_tags_are_normalised_and_typos_refused(self):
        assert targets._tags("Cash + GPP") == "cash;gpp"
        assert targets._tags("favorite cash") == "favorite_cash"
        with pytest.raises(ValueError):
            targets._tags("cahs")

    def test_a_position_outside_the_four_is_refused(self):
        with pytest.raises(ValueError, match="position"):
            targets.parse_picks("player,position\nRavens,DST\n")


class TestVerdict:
    def test_points_or_value_inside_the_depth_is_agreement(self):
        assert targets.verdict("RB", 5, None)[0] == "agree"
        assert targets.verdict("RB", 30, 4)[0] == "agree"     # cheap, good value

    def test_outside_both_is_skepticism(self):
        pile, why = targets.verdict("TE", 36, 20)
        assert pile == "skeptical" and "outside our top 8" in why


class TestThePage:
    def test_upload_matches_suffixed_names_to_the_board(self, client, db_session):
        _board(db_session)
        body = _upload(client).json()
        assert body["stored"] == 4
        assert body["unmatched"] == ["Nobody Here"]
        ids = {p.player_name: p.player_id for p in db_session.query(ExpertPick)}
        assert ids["James Cook III"] == "cook" and ids["Kyle Pitts Sr."] == "pitts"

    def test_piles(self, client, db_session):
        _board(db_session)
        _upload(client)
        body = client.get("/targets/", params={"season": SEASON, "week": WEEK}).json()
        piles = {k: [r["player"] for r in body[k]] for k in ("agree", "skeptical", "no_board")}
        assert piles["agree"] == ["Brock Bowers"]
        assert sorted(piles["skeptical"]) == ["James Cook III", "Kyle Pitts Sr."]
        assert piles["no_board"] == ["Nobody Here"]
        cook = body["skeptical"][[r["player"] for r in body["skeptical"]].index("James Cook III")]
        assert cook["proj_rank"] == 19 and cook["tags"] == ["cash", "gpp"]

    def test_value_on_the_slate_can_earn_agreement(self, client, db_session):
        _board(db_session)
        # Cook at a punt price is our best value at RB.
        _slate(db_session, [("James Cook", "RB", 2000, "BUF"),
                            ("Back Number0", "RB", 9000, "XXX")])
        _upload(client)
        body = client.get("/targets/", params={"season": SEASON, "week": WEEK}).json()
        cook = next(r for r in body["agree"] if r["player_id"] == "cook")
        assert cook["value_rank"] == 1 and cook["salary"] == 2000

    def test_reupload_of_an_article_replaces_it(self, client, db_session):
        _board(db_session)
        _upload(client)
        _upload(client, "player,team,position,tags,article\nBrock Bowers,LV,TE,gpp,TE Coach\n")
        names = sorted(p.player_name for p in db_session.query(ExpertPick))
        assert names == ["Brock Bowers", "James Cook III", "Nobody Here"]

    def test_after_the_games_each_pile_is_scored_against_the_board(self, client, db_session):
        _board(db_session)
        _upload(client)
        db_session.add(PlayerStat(player_id="cook", season=SEASON, week=WEEK,
                                  player_name="J.Cook", player_display_name="James Cook",
                                  position="RB", recent_team="BUF", season_type="REG",
                                  rushing_yards=120.0, rushing_tds=1.0))
        db_session.commit()
        body = client.get("/targets/", params={"season": SEASON, "week": WEEK}).json()
        cook = next(r for r in body["skeptical"] if r["player_id"] == "cook")
        assert cook["actual"] == pytest.approx(21.0)
        assert cook["vs_board"] == pytest.approx(14.0)
        assert body["scored"]["skeptical"]["n"] == 1
        assert body["scored"]["skeptical"]["beat_board"] == 1

    def test_an_empty_week(self, client):
        body = client.get("/targets/", params={"season": SEASON, "week": 3}).json()
        assert body["count"] == 0 and body["agree"] == []


class TestThePoolShowsThem:
    def test_tags_by_player_id_merged_across_articles(self, client, db_session):
        from api.optimizer import _expert_tags
        _board(db_session)
        _upload(client)
        _upload(client, "player,team,position,tags,article\n"
                        "James Cook,BUF,RB,cash;favorite_cash,Cash Breakdown\n")
        tags = _expert_tags(db_session, SEASON, WEEK)
        assert tags["cook"]["tags"] == ["cash", "gpp", "favorite_cash"]
        assert tags["cook"]["articles"] == ["RB Coach", "Cash Breakdown"]
        assert "nobody" not in str(tags)            # unmatched picks carry no id
