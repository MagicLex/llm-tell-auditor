"""Register the I job `audit-dossiers` as a Hopsworks PYTHON job.

Points appPath straight at the FUSE-staged script (same HopsFS, one source of
truth, no copy) and pins the `tell-audit` env (pandas-training-pipeline +
pylatexenc, sklearn 1.8.0 matching the model pickle). Idempotent: if the job
exists, just update its config.

Run:  hops job run audit-dossiers --args "--per-category 8 --skip-existing"
"""
from __future__ import annotations

from pathlib import Path

import hopsworks

JOB_NAME = "audit-dossiers"
ENV_NAME = "tell-audit"

# this repo lives on the FUSE mount; derive the HopsFS path from the file location
# so nothing is hardcoded to a project or username.
_rel = str(Path(__file__).resolve()).split("/hopsfs/", 1)[1].rsplit("/", 1)[0]


def main() -> None:
    project = hopsworks.login()
    ja = project.get_job_api()
    app_path = f"hdfs:///Projects/{project.name}/{_rel}/audit_job.py"

    cfg = ja.get_configuration("PYTHON")
    cfg["appPath"] = app_path
    cfg["environmentName"] = ENV_NAME
    cfg["resourceConfig"]["memory"] = 4096  # sklearn + pylatexenc + pandas headroom

    job = ja.get_job(JOB_NAME)
    if job is None:
        job = ja.create_job(JOB_NAME, cfg)
        print(f"created job {job.name} on {ENV_NAME}", flush=True)
    else:
        job.config = cfg
        job.save()
        print(f"updated job {job.name}", flush=True)
    print(f"appPath={app_path}", flush=True)


if __name__ == "__main__":
    main()
