"""Run nfl-data-py experiment scripts on the cluster, and read the results.

The experiments that train models outgrew the workstation: on 2026-11-13 two
were killed for low memory mid-run. The cluster already runs the projection
job on AVX nodes, so an experiment runs there the same way -- a Job created
from a suspended CronJob template (helm: experiments-cronjob.yaml) -- and its
output lands in `experiment_runs`, readable over HTTPS.

Guard rails, because this starts compute on shared nodes:

* **A token.** `X-Experiment-Token` must equal `EXPERIMENT_TOKEN`, which comes
  from its own secret. Without one configured, every POST is refused.
* **Only scripts from the repo.** `name` is a bare identifier; the job runs
  `experiments/<name>.py` from nfl-data-py's main branch and nothing else.
* **Plain settings only.** `env` keys are UPPER_SNAKE, values short and
  printable, and nothing that could redirect the job (paths, the database,
  the interpreter) is accepted.
* **One at a time.** A second request while one is queued or running is
  refused; the production projection runs share these nodes.
"""
import hmac
import json
import os
import re
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.models import ExperimentRun
from database.session import get_db

router = APIRouter()

CRONJOB = os.getenv("EXPERIMENT_CRONJOB", "nfl-api-experiments")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,39}$")
VALUE_RE = re.compile(r"^[A-Za-z0-9_.,:-]{0,64}$")
FORBIDDEN_KEYS = {"PATH", "PYTHONPATH", "PYTHONHOME", "DATABASE_URL", "HOME",
                  "EXPERIMENT_TOKEN", "EXPERIMENT_RUN_ID", "EXPERIMENT_NAME",
                  "EXPERIMENT_ENV", "EXPERIMENT_REF"}
FORBIDDEN_PREFIXES = ("PG", "LD_", "PIP_", "KUBERNETES", "PYTHON")
# A run that never reports back (pod could not schedule, node lost) stops
# blocking the queue after this long.
STALE_AFTER = timedelta(hours=8)


class RunRequest(BaseModel):
    name: str = Field(..., description="Script in nfl-data-py/experiments, without .py")
    env: dict = Field(default_factory=dict, description="Settings the script reads")
    ref: str = Field("main", description="Branch of nfl-data-py to run")


def validate_request(req: RunRequest) -> dict:
    """The env to pass, or a 400 saying what was refused."""
    if not NAME_RE.match(req.name):
        raise HTTPException(400, "name must be a script name like train_window_expanding")
    if not re.match(r"^[A-Za-z0-9_.-]{1,64}$", req.ref):
        raise HTTPException(400, "ref must be a branch name")
    env = {}
    for key, value in (req.env or {}).items():
        value = str(value)
        if (not KEY_RE.match(key) or key in FORBIDDEN_KEYS
                or key.startswith(FORBIDDEN_PREFIXES)):
            raise HTTPException(400, f"setting {key!r} is not allowed")
        if not VALUE_RE.match(value):
            raise HTTPException(400, f"value for {key} must be short and plain")
        env[key] = value
    return env


def _authorised(token: Optional[str]) -> None:
    expected = os.getenv("EXPERIMENT_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Experiments are not configured: no EXPERIMENT_TOKEN.")
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(401, "Wrong or missing X-Experiment-Token.")


def _busy(db) -> Optional[ExperimentRun]:
    cutoff = datetime.utcnow() - STALE_AFTER
    return (db.query(ExperimentRun)
            .filter(ExperimentRun.status.in_(["queued", "running"]),
                    ExperimentRun.requested_at >= cutoff)
            .order_by(ExperimentRun.id.desc()).first())


def _row(r: ExperimentRun, with_log=False) -> dict:
    out = {"id": r.id, "name": r.name, "env": json.loads(r.env or "{}"),
           "status": r.status, "job_name": r.job_name, "exit_code": r.exit_code,
           "git_ref": r.git_ref, "error": r.error,
           "requested_at": r.requested_at, "started_at": r.started_at,
           "finished_at": r.finished_at}
    if with_log:
        out["log"] = r.log
    return out


@router.post("/run")
def run_experiment(req: RunRequest, db: Session = Depends(get_db),
                   x_experiment_token: Optional[str] = Header(None)):
    """Start one experiment on the cluster. Returns the run id to poll."""
    _authorised(x_experiment_token)
    env = validate_request(req)
    busy = _busy(db)
    if busy is not None:
        raise HTTPException(409, f"Run {busy.id} ({busy.name}) is still {busy.status}; "
                                 "one experiment at a time.")
    run = ExperimentRun(name=req.name, env=json.dumps(env), status="queued",
                        requested_at=datetime.utcnow())
    db.add(run)
    db.commit()
    from .k8s_jobs import JobStartError, start_job_from_cronjob
    try:
        run.job_name = start_job_from_cronjob(
            CRONJOB, suffix=f"r{run.id}",
            env={"EXPERIMENT_RUN_ID": run.id, "EXPERIMENT_NAME": req.name,
                 "EXPERIMENT_REF": req.ref, "EXPERIMENT_ENV": json.dumps(env)},
            labels={"experiment": req.name.replace("_", "-")[:63]})
    except JobStartError as exc:
        run.status, run.error = "failed", f"could not start: {exc}"
        db.commit()
        raise HTTPException(503, run.error)
    db.commit()
    return {"status": "success", "run": _row(run)}


@router.get("/runs")
def list_runs(limit: int = Query(25, le=200), name: Optional[str] = None,
              db: Session = Depends(get_db)):
    q = db.query(ExperimentRun)
    if name:
        q = q.filter(ExperimentRun.name == name)
    rows = q.order_by(ExperimentRun.id.desc()).limit(limit).all()
    return {"status": "success", "runs": [_row(r) for r in rows]}


@router.get("/runs/{run_id}")
def get_run(run_id: int, tail: int = Query(0, ge=0, description="Last N lines only"),
            db: Session = Depends(get_db)):
    r = db.query(ExperimentRun).filter(ExperimentRun.id == run_id).one_or_none()
    if r is None:
        raise HTTPException(404, f"No run {run_id}.")
    out = _row(r, with_log=True)
    if tail and out["log"]:
        out["log"] = "\n".join(out["log"].splitlines()[-tail:])
    return {"status": "success", "run": out}
