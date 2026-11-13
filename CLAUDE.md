# NFL-API — working notes

FastAPI service at **nfl-api.nickknows.net**, plus the React dashboard in
`frontend/`. Serves the projections, the FanDuel optimizer, the DFS tools and
the ratings/trench/coverage reports. This file holds the things that are not
inferable from reading the code.

## Running the tests

```bash
cd app
NP=/home/gin-g/miniforge3/pkgs/numpy-1.26.4-py310hb13e2d6_0/lib/python3.10/site-packages
PYTHONPATH=.:/home/gin-g/Documents/Code/nfl-data-py:$NP \
  /home/gin-g/miniforge3/bin/python -m pytest tests/ -q
```

The sibling `nfl-data-py` checkout on `PYTHONPATH` is required — without it
~50 tests error with `ModuleNotFoundError: nfl_projections` and it looks like a
code break. ~766 tests, ~30 s.

### The test harness commits, and the engine is session-scoped

`conftest._test_engine` is one in-memory SQLite for the whole session, and the
endpoints **commit**, so rows outlive each test's rollback. This has caused two
separate debugging hunts:

- Recorded lineup builds accumulated across tests until dedupe found no
  distinct lineup left and builds came back empty.
- `PlayerStat` rows collided on insert between test classes.

If a test fails only when run alongside others, suspect this first.
`test_optimizer` and `test_dfs` each carry an autouse fixture that deletes the
tables they own — add to that pattern rather than inventing a new one.

**`PlayerStat`'s primary key is composite: `(player_id, season, week)`.** A
fixture that gives each week its own `player_id` makes every week a separate
"player" with one game, which silently defeats any min-games filter.

## Database and migrations

There is **no Alembic**. `create_all` adds missing *tables* but never missing
*columns* — a new column must also be listed in `apply_light_migrations` in
`database/models.py`, or it exists locally and not in production.

Read production data through the HTTPS API; that is the supported path and it
is usually faster than the alternatives. Note the exported `KUBECONFIG` env var
fails auth, so pass `~/.kube/config` explicitly for any `kubectl get`/`logs`.

## Deploying

Push to `main` runs tests → builds the image → commits an image-tag bump to
`helm/nfl-api/values.yaml` → **ArgoCD picks that up on its own poll cycle**.
End to end is ~9 minutes, and the roll can lag the green build by several more.

Because CI commits that bump, local `main` is usually one commit behind after a
deploy: `git fetch` and rebase onto `origin/main` before the next push.

`app/Dockerfile` installs `nfl-projections` from the GitHub **main tarball**
(the image has no git), and so does the CI test job. A change spanning both
repos must therefore be **pushed to nfl-data-py first**, or this build tests
and ships against the old package.

The image pins `pandas==2.1.4`, which is why pandas-3 behaviour differences
surface in nfl-data-py's CI but never in production here.

### `k8s/` is not deployed by anything

Only `helm/nfl-api/` is synced by ArgoCD. The manifests in `k8s/` (the Tue/Wed
projections cron, the accuracy and roster-sync crons, the model-cache claim)
exist in the cluster only if someone ran `kubectl apply` on them, and a change
there does nothing until someone does again. The weekend projection refresh
sat in `k8s/` for two weeks without ever running for exactly this reason. A new
scheduled job belongs in `helm/nfl-api/templates/`.

### The smoke-test job

Cloudflare bot-challenges the GitHub runner, so the job often cannot reach the
API at all. The wait step separates that case (reports `reachable=false`, skips
its endpoint checks, exits after ~1 minute) from a real failure (no response,
or HTTP 200 with the wrong status).

**A green smoke test does not prove the new image is serving.** It proves
something healthy answered, and the old pod answers identically. Check the
deployed image tag to confirm a roll.

## The frontend

`frontend/` is Vite + React + Tailwind + Recharts, built by the Dockerfile's
first stage into `app/static/dashboard/`. That directory is **not tracked** —
never hand-edit the built bundle.

`npm run dev` proxies to production by default. To work against an endpoint
that is not deployed yet:

```bash
VITE_API_TARGET=http://127.0.0.1:8000 npm run dev
```

Every API prefix lives in one list in `vite.config.ts`. A missing entry means
that page silently fails in dev while working in production.

**Render a chart before shipping it.** Writing one unseen has already shipped
overlapping, illegible labels.

## Conventions worth keeping

- Endpoints say *why*, not just *what*. Several responses carry the measured
  reliability of the thing they return, and refusals name the cause rather than
  the symptom. That is deliberate — match it.
- Soft-fail context, hard-fail the answer: an unavailable side-car (ownership,
  opportunity, FPA) logs and omits itself, while a broken primary lookup
  reports the exception rather than reading as "no data for this week".
- The board does not project defenses; that slot uses FanDuel's own FPPG from
  the uploaded slate. Anything listing positions should say so rather than
  quietly omitting DEF.
- Commit and push **only when asked**.

## Experiments run on the cluster, not the workstation

Heavy nfl-data-py experiments (anything that trains) run as Jobs:
`POST /experiments/run` with header `X-Experiment-Token` and a body like
`{"name": "train_window_expanding", "env": {"SEEDS": "3"}}` starts
`experiments/<name>.py` from nfl-data-py main on an AVX node
(`helm/nfl-api/templates/experiments-cronjob.yaml`, suspended, template only).
Read the result at `GET /experiments/runs/<id>` (`?tail=N` for the end of the
log). One run at a time; caches and the built dataset live on the
`nfl-api-experiments` volume. The token comes from OpenBao
(`nfl-api` / `experiment_token`); without it the endpoint refuses.
