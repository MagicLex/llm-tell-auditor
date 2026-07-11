"""Deploy the LLM Tell Auditor as a custom (FastAPI) Hopsworks app.

Thin client on stock python-app-pipeline: fastapi/uvicorn uv-installed at pod
start, no clone (nothing heavy loads in this pod, it only reads paper_dossiers).
Redeploy uses the recovery sequence (stop, purge lingering k8s deployment,
drain, stop zombie executions, settle) since app.stop() returns before the
execution actually dies.
"""
import subprocess
import time
from pathlib import Path

import hopsworks

APP_NAME = "tellauditor"
# tell-audit has sklearn 1.8.0 (matches the model pickle) + pylatexenc + anthropic,
# so the app can score live and write feedback in-pod; only the web server is
# uv-installed at start.
ENV_NAME = "tell-audit"

_here = Path(__file__).resolve()
rel = str(_here).split("/hopsfs/", 1)[1]
APP_PATH = str(Path(rel).parent / "server.py")
SERVER = f"/hopsfs/{rel.rsplit('/', 1)[0]}/server.py"
# fastapi/uvicorn/python-multipart live in the tell-audit env (no `uv` in it to
# install at start), so the entrypoint just runs the server.
ENTRYPOINT = f'bash -lc "exec python {SERVER}"'


def _pods():
    out = subprocess.run(["kubectl", "get", "pods"], capture_output=True, text=True).stdout
    return [l.split()[0] for l in out.splitlines() if APP_NAME in l]


def _purge_k8s():
    out = subprocess.run(["kubectl", "get", "deployment"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if APP_NAME in line:
            name = line.split()[0]
            subprocess.run(["kubectl", "delete", "deployment", name], capture_output=True)
            print(f"purged k8s deployment {name}", flush=True)
    for _ in range(60):
        if not _pods():
            return
        time.sleep(5)
    raise RuntimeError("app pods refused to drain")


def _stop_zombies(project):
    job = project.get_job_api().get_job(APP_NAME)
    if job is None:
        return
    for ex in job.get_executions() or []:
        if ex.final_status in ("UNDEFINED", None):
            try:
                ex.stop()
                print(f"stopped zombie execution {ex.id}", flush=True)
            except Exception:
                pass


def _create(apps):
    return apps.create_app(
        name=APP_NAME, app_path=APP_PATH, app_kind="CUSTOM",
        entrypoint_command=ENTRYPOINT, app_port=8000,
        environment=ENV_NAME, memory=4096, cores=1.0,
        description="LLM Tell Auditor -- browse audited arXiv preprints and audit any "
                    "text live for LLM writing tells, with plain-language feedback. "
                    "Signal, not verdict.")


def main():
    project = hopsworks.login()
    apps = project.get_app_api()
    print(f"app_path={APP_PATH} env={ENV_NAME}", flush=True)
    app = apps.get_app(APP_NAME)
    # env is fixed at create time, so to (re)deploy on a possibly-changed env we
    # tear the app down and recreate it, draining k8s in between.
    if app is not None:
        try:
            app.stop()
        except Exception:
            pass
        _purge_k8s()
        _stop_zombies(project)
        try:
            app.delete()
        except Exception as e:
            print(f"delete: {e}", flush=True)
        for _ in range(24):
            if apps.get_app(APP_NAME) is None:
                break
            time.sleep(5)
        time.sleep(10)
    app = _create(apps)
    app.run(await_serving=True)
    print(f"URL: {app.app_url}", flush=True)


if __name__ == "__main__":
    main()
