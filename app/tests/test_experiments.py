"""The experiment runner: guarded start, readable results."""
import json
import os
import sys
import textwrap

import pytest

from database.models import ExperimentRun


@pytest.fixture(autouse=True)
def _clean(db_session, monkeypatch):
    db_session.query(ExperimentRun).delete()
    db_session.commit()
    monkeypatch.setenv("EXPERIMENT_TOKEN", "s3cret")
    monkeypatch.setenv("EXPERIMENT_TOKEN_FILE", "/nonexistent/token")
    import api.experiments as ex
    monkeypatch.setattr(ex, "TOKEN_FILE", "/nonexistent/token")
    yield


@pytest.fixture
def started(monkeypatch):
    calls = []
    from api import k8s_jobs

    def fake(cronjob, **kw):
        calls.append((cronjob, kw))
        return f"{cronjob}-r1"
    monkeypatch.setattr(k8s_jobs, "start_job_from_cronjob", fake)
    return calls


def _post(client, body, token="s3cret"):
    headers = {"X-Experiment-Token": token} if token else {}
    return client.post("/experiments/run", json=body, headers=headers)


class TestGate:
    def test_no_token_configured_refuses(self, client, monkeypatch, started):
        monkeypatch.delenv("EXPERIMENT_TOKEN")
        assert _post(client, {"name": "selection_tax"}).status_code == 503
        assert started == []

    def test_wrong_or_missing_token(self, client, started):
        assert _post(client, {"name": "selection_tax"}, token="nope").status_code == 401
        assert _post(client, {"name": "selection_tax"}, token=None).status_code == 401
        assert started == []

    @pytest.mark.parametrize("body", [
        {"name": "../etc/passwd"},
        {"name": "Selection_Tax"},
        {"name": "x", "env": {"DATABASE_URL": "postgres://evil"}},
        {"name": "x", "env": {"PGHOST": "evil"}},
        {"name": "x", "env": {"PYTHONPATH": "/tmp"}},
        {"name": "x", "env": {"SEEDS": "3; rm -rf /"}},
        {"name": "x", "env": {"lower": "1"}},
        {"name": "x", "ref": "main; curl"},
    ])
    def test_refused_inputs(self, client, started, body):
        assert _post(client, body).status_code == 400
        assert started == []


def test_the_mounted_file_wins_and_is_read_per_request(client, monkeypatch, tmp_path,
                                                       started):
    import api.experiments as ex
    f = tmp_path / "token"
    f.write_text("from-file\n")
    monkeypatch.setattr(ex, "TOKEN_FILE", str(f))
    monkeypatch.delenv("EXPERIMENT_TOKEN")
    assert _post(client, {"name": "selection_tax"}, token="from-file").status_code == 200
    f.write_text("rotated")
    assert _post(client, {"name": "lineup_depth"}, token="from-file").status_code == 401


class TestRun:
    def test_starts_a_job_with_the_settings(self, client, db_session, started):
        body = _post(client, {"name": "train_window_expanding",
                              "env": {"SEEDS": 3, "FIRST_WEEK": "15"}}).json()
        run = body["run"]
        assert run["status"] == "queued" and run["job_name"] == "nfl-api-experiments-r1"
        cronjob, kw = started[0]
        assert cronjob == "nfl-api-experiments"
        assert kw["env"]["EXPERIMENT_NAME"] == "train_window_expanding"
        assert json.loads(kw["env"]["EXPERIMENT_ENV"]) == {"SEEDS": "3", "FIRST_WEEK": "15"}

    def test_one_at_a_time(self, client, started):
        assert _post(client, {"name": "selection_tax"}).status_code == 200
        r = _post(client, {"name": "lineup_depth"})
        assert r.status_code == 409 and "one experiment at a time" in r.json()["detail"]

    def test_a_failed_start_is_recorded_and_frees_the_queue(self, client, db_session,
                                                            monkeypatch):
        from api import k8s_jobs

        def boom(*a, **k):
            raise k8s_jobs.JobStartError("no room")
        monkeypatch.setattr(k8s_jobs, "start_job_from_cronjob", boom)
        assert _post(client, {"name": "selection_tax"}).status_code == 503
        run = db_session.query(ExperimentRun).one()
        assert run.status == "failed" and "no room" in run.error

    def test_runs_are_readable(self, client, db_session, started):
        _post(client, {"name": "selection_tax"})
        run = db_session.query(ExperimentRun).one()
        run.status, run.log = "completed", "line 1\nline 2\nline 3"
        db_session.commit()
        listed = client.get("/experiments/runs").json()["runs"]
        assert listed[0]["name"] == "selection_tax" and "log" not in listed[0]
        one = client.get(f"/experiments/runs/{run.id}", params={"tail": 2}).json()["run"]
        assert one["log"] == "line 2\nline 3"


class TestRunner:
    def test_streams_output_and_records_the_result(self, db_session, monkeypatch, tmp_path):
        """The job side, end to end, with the download replaced by a local tree."""
        from scripts import run_experiment as rx

        run = ExperimentRun(name="hello", env="{}", status="queued")
        db_session.add(run)
        db_session.commit()
        src = tmp_path / "src"
        (src / "experiments").mkdir(parents=True)
        (src / "experiments" / "hello.py").write_text(textwrap.dedent("""
            import os
            print("Epoch 1/60")
            print("seeds", os.environ["SEEDS"])
            print("done")
        """))
        monkeypatch.setattr(rx, "checkout", lambda ref, workdir: str(src))
        monkeypatch.setattr(rx, "_commit_sha", lambda ref: "abc123")
        monkeypatch.setattr(rx, "_session", lambda: db_session)
        monkeypatch.setattr(db_session, "close", lambda: None)
        code = rx.run(run.id, "hello", "main", {"SEEDS": "3"})
        db_session.refresh(run)
        assert code == 0 and run.status == "completed" and run.exit_code == 0
        assert run.log == "seeds 3\ndone"          # progress noise dropped
        assert run.git_ref == "main@abc123"

    def test_a_missing_script_fails_cleanly(self, db_session, monkeypatch, tmp_path):
        from scripts import run_experiment as rx

        run = ExperimentRun(name="nope", env="{}", status="queued")
        db_session.add(run)
        db_session.commit()
        (tmp_path / "experiments").mkdir()
        monkeypatch.setattr(rx, "checkout", lambda ref, workdir: str(tmp_path))
        monkeypatch.setattr(rx, "_commit_sha", lambda ref: None)
        with pytest.raises(FileNotFoundError):
            rx.run(run.id, "nope", "main", {})
