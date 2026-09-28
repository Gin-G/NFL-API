#!/usr/bin/env python3
"""Start the projections job from inside the API pod.

The API image has no TensorFlow on purpose — it would triple the image for code
that serves cached rows — so the API cannot compute a projection itself. What it
can do is ask Kubernetes to run the same job the CronJob runs, which installs
nfl_projections at runtime and writes to the same tables.

The pod's ServiceAccount is allowed to read that one CronJob and create Jobs in
its own namespace, and nothing else (k8s/projections-rbac.yaml).
"""
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

CRONJOB_NAME = os.getenv("PROJECTION_CRONJOB", "nfl-api-projections")
_NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"


class JobStartError(RuntimeError):
    """Kubernetes would not start the job (not in a cluster, RBAC, API down)."""


def current_namespace() -> str:
    """The namespace this pod runs in."""
    try:
        with open(_NAMESPACE_FILE) as f:
            return f.read().strip() or "nfl-api"
    except OSError:
        return os.getenv("POD_NAMESPACE", "nfl-api")


def active_projection_jobs(cronjob: str = None, namespace: str = None) -> list:
    """Projection Jobs that have not finished, as ``{"name", "ready"}``.

    The database says a run is going only once the pod has installed its
    dependencies and written a status row, which is minutes after the Job
    exists. Asking Kubernetes closes that window.

    ``ready`` separates a pod doing the work from one the scheduler has nowhere
    to put: this cluster ran out of room for 37 hours in September and a queued
    refresh looked, from the dashboard, exactly like a running one. An empty
    list on any failure — the caller still has the database to fall back on.
    """
    cronjob = cronjob or CRONJOB_NAME
    namespace = namespace or current_namespace()
    try:
        from kubernetes import client, config

        config.load_incluster_config()
        jobs = client.BatchV1Api().list_namespaced_job(
            namespace, label_selector=f"app={cronjob}").items
    except Exception as exc:
        logger.debug("Could not list projection jobs (%s)", exc)
        return []
    out = []
    for j in jobs:
        if (j.status.active or 0) <= 0:
            continue
        ready = j.status.ready
        # `ready` is None on clusters that do not report it; assume it is running
        # rather than claim a queue that may not exist.
        out.append({"name": j.metadata.name, "ready": True if ready is None else ready > 0})
    return out


def start_projection_job(cronjob: str = None, namespace: str = None) -> str:
    """Create a one-off Job from the projections CronJob; return its name.

    The Job is the CronJob's own template, unmodified: same image, same command,
    same model cache. A mid-week run therefore reuses the model the last run
    trained and spends its time on the projection instead — see
    scripts.compute_projections.
    """
    cronjob = cronjob or CRONJOB_NAME
    namespace = namespace or current_namespace()
    try:
        from kubernetes import client, config
    except ImportError as exc:                      # pragma: no cover - import guard
        raise JobStartError(f"kubernetes client not installed: {exc}") from exc

    try:
        config.load_incluster_config()
    except Exception as exc:
        raise JobStartError(f"not running in a cluster: {exc}") from exc

    batch = client.BatchV1Api()
    try:
        template = batch.read_namespaced_cron_job(cronjob, namespace).spec.job_template
    except Exception as exc:
        raise JobStartError(f"could not read cronjob {cronjob}: {exc}") from exc

    name = f"{cronjob}-manual-{datetime.utcnow():%Y%m%d-%H%M%S}"
    labels = dict(getattr(template.metadata, "labels", None) or {})
    labels["triggered-by"] = "api"
    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=name, namespace=namespace, labels=labels,
            annotations={"cronjob.kubernetes.io/instantiate": "manual"},
        ),
        spec=template.spec,
    )
    try:
        batch.create_namespaced_job(namespace, job)
    except Exception as exc:
        raise JobStartError(f"could not create job: {exc}") from exc
    logger.info("Started projections job %s in %s", name, namespace)
    return name
