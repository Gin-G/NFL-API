#!/usr/bin/env python3
"""Run one nfl-data-py experiment inside the cluster and record its output.

Started by POST /experiments/run as a Job from the suspended experiments
CronJob (helm: experiments-cronjob.yaml). It:

  1. downloads nfl-data-py at EXPERIMENT_REF (a branch) as GitHub's tarball,
     installs it, and notes the commit it got;
  2. points the checkout's `experiments/_cache` and `data/` at the experiments
     volume, so caches and the built dataset survive between runs;
  3. runs `experiments/<EXPERIMENT_NAME>.py` with the requested settings;
  4. streams the output into `experiment_runs.log` (a tail, progress-bar
     noise dropped) and records the exit code.

Everything a person needs is then in GET /experiments/runs/<id>.
"""
import collections
import json
import logging
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_experiment")

REPO = "Gin-G/nfl-data-py"
VOLUME = os.getenv("EXPERIMENT_VOLUME", "/experiments")
KEEP_LINES = 4000
FLUSH_SECONDS = 30
# TensorFlow and tqdm progress output: thousands of lines that say nothing.
NOISE = re.compile(r"(^Epoch \d+/\d+)|( - \d+m?s/step)|(━)|(it/s\])|(s/it\])|"
                   r"(^\d+/\d+ \[)|(oneDNN)|(cuda)|(absl::)|(TF_ENABLE_ONEDNN)")


def _session():
    from database.session import SessionLocal
    return SessionLocal()


def _update(run_id, **fields):
    from database.models import ExperimentRun
    db = _session()
    try:
        run = db.query(ExperimentRun).filter(ExperimentRun.id == run_id).one_or_none()
        if run is None:
            return
        for k, v in fields.items():
            setattr(run, k, v)
        db.commit()
    finally:
        db.close()


def _commit_sha(ref):
    try:
        req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/commits/{ref}",
                                     headers={"Accept": "application/vnd.github.sha"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read().decode().strip()[:40]
    except Exception:
        return None


def checkout(ref, workdir):
    url = f"https://github.com/{REPO}/archive/refs/heads/{ref}.tar.gz"
    path = os.path.join(workdir, "src.tar.gz")
    urllib.request.urlretrieve(url, path)
    with tarfile.open(path) as tar:
        root = tar.getnames()[0].split("/")[0]
        tar.extractall(workdir, filter="data") if sys.version_info >= (3, 12) \
            else tar.extractall(workdir)
    src = os.path.join(workdir, root)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                    "--quiet", src], check=True)
    for sub, link in (("cache", os.path.join(src, "experiments", "_cache")),
                      ("data", os.path.join(src, "data"))):
        target = os.path.join(VOLUME, sub)
        os.makedirs(target, exist_ok=True)
        if os.path.islink(link) or os.path.isfile(link):
            os.remove(link)
        elif os.path.isdir(link):
            # A checked-in directory would hide the volume; move its files over.
            for f in os.listdir(link):
                dest = os.path.join(target, f)
                if not os.path.exists(dest):
                    os.rename(os.path.join(link, f), dest)
            os.rmdir(link)
        os.symlink(target, link)
    return src


def run(run_id, name, ref, extra_env):
    tail = collections.deque(maxlen=KEEP_LINES)
    _update(run_id, status="running", started_at=datetime.utcnow(),
            git_ref=f"{ref}@{_commit_sha(ref) or '?'}")
    with tempfile.TemporaryDirectory() as workdir:
        src = checkout(ref, workdir)
        script = os.path.join(src, "experiments", f"{name}.py")
        if not os.path.isfile(script):
            raise FileNotFoundError(f"experiments/{name}.py is not in {ref}")
        env = dict(os.environ, **extra_env, PYTHONPATH=src, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen([sys.executable, "-u", script], cwd=src, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        last = time.monotonic()
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)           # still visible in the pod's log
            if NOISE.search(line):
                continue
            tail.append(line)
            if time.monotonic() - last > FLUSH_SECONDS:
                _update(run_id, log="\n".join(tail))
                last = time.monotonic()
        code = proc.wait()
    _update(run_id, log="\n".join(tail), exit_code=code,
            status="completed" if code == 0 else "failed",
            finished_at=datetime.utcnow(),
            error=None if code == 0 else f"exited with code {code}")
    return code


def main():
    run_id = int(os.environ["EXPERIMENT_RUN_ID"])
    name = os.environ["EXPERIMENT_NAME"]
    ref = os.environ.get("EXPERIMENT_REF", "main")
    extra = json.loads(os.environ.get("EXPERIMENT_ENV") or "{}")
    try:
        sys.exit(run(run_id, name, ref, extra))
    except Exception as exc:
        logger.exception("experiment %s failed", name)
        _update(run_id, status="failed", finished_at=datetime.utcnow(),
                error=f"{type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
