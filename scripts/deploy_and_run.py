#!/usr/bin/env python3
"""Despliega el paquete S4.5 a un UC Volume, importa el notebook y lo ejecuta."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cmd(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args[:4]), "..." if len(args) > 4 else "")
    result = subprocess.run(args, check=False, text=True, capture_output=True)
    if check and result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "sin detalle del CLI"
        raise RuntimeError(f"comando Databricks falló ({result.returncode}): {message}")
    return result


def dbx(profile: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return cmd(["databricks", *args, "-p", profile], check=check)


def ensure_schema(profile: str, catalog: str, schema: str) -> None:
    got = dbx(profile, "schemas", "get", f"{catalog}.{schema}", check=False)
    if got.returncode != 0:
        dbx(
            profile, "schemas", "create", schema, catalog,
            "--comment", "Laboratorio S4.5 RAG: 24 documentos y 70 preguntas",
        )


def ensure_volume(profile: str, catalog: str, schema: str, volume: str) -> None:
    full = f"{catalog}.{schema}.{volume}"
    got = dbx(profile, "volumes", "read", full, check=False)
    if got.returncode != 0:
        dbx(
            profile, "volumes", "create", catalog, schema, volume, "MANAGED",
            "--comment", "Paquete autocontenido del laboratorio S4.5",
        )


def upload(profile: str, source: Path, target: str) -> None:
    dbx(profile, "fs", "cp", str(source), target, "--recursive", "--overwrite")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="webinar-aws-v2")
    ap.add_argument("--catalog", default="neptuno_manuel_arguelles")
    ap.add_argument("--schema", default="rag_s4_5")
    ap.add_argument("--volume", default="laboratorio")
    ap.add_argument(
        "--workspace-path",
        default="/Shared/curso-databricks-ai-engineer/S4.5-laboratorio-rag-24-documentos",
    )
    ap.add_argument("--timeout-minutes", type=int, default=40)
    args = ap.parse_args()

    ensure_schema(args.profile, args.catalog, args.schema)
    ensure_volume(args.profile, args.catalog, args.schema, args.volume)
    volume_uri = f"dbfs:/Volumes/{args.catalog}/{args.schema}/{args.volume}"

    # Los dos grupos se cargan planos. La clasificación sigue en el manifiesto y en el repo.
    upload(args.profile, ROOT / "corpus/pdf/base", f"{volume_uri}/corpus/pdf")
    upload(args.profile, ROOT / "corpus/pdf/distractores", f"{volume_uri}/corpus/pdf")
    dbx(
        args.profile, "fs", "cp", str(ROOT / "corpus/manifest.yml"),
        f"{volume_uri}/corpus/manifest.yml", "--overwrite",
    )
    upload(args.profile, ROOT / "data", f"{volume_uri}/data")
    upload(args.profile, ROOT / "src", f"{volume_uri}/src")

    notebook = ROOT / "notebooks/S4.5-laboratorio-rag-24-documentos.py"
    dbx(
        args.profile, "workspace", "import", args.workspace_path,
        "--file", str(notebook), "--format", "SOURCE", "--language", "PYTHON", "--overwrite",
    )

    payload = {
        "run_name": "S4.5 RAG 24 documentos · validacion E2E",
        "timeout_seconds": args.timeout_minutes * 60,
        "tasks": [
            {
                "task_key": "validar_s4_5",
                "environment_key": "default",
                "notebook_task": {
                    "notebook_path": args.workspace_path,
                    "source": "WORKSPACE",
                    "base_parameters": {
                        "catalogo": args.catalog,
                        "schema": args.schema,
                        "volume": args.volume,
                        "k": "8",
                    },
                },
            }
        ],
        "environments": [{"environment_key": "default", "spec": {"client": "3"}}],
    }
    submit = dbx(
        args.profile, "jobs", "submit", "--json", json.dumps(payload),
        "--no-wait", "-o", "json",
    )
    run_id = json.loads(submit.stdout)["run_id"]
    deadline = time.time() + args.timeout_minutes * 60
    run = None
    while time.time() < deadline:
        out = dbx(args.profile, "jobs", "get-run", str(run_id), "-o", "json")
        run = json.loads(out.stdout)
        life = run.get("state", {}).get("life_cycle_state")
        print(f"run {run_id}: {life}")
        if life in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR", "BLOCKED"}:
            break
        time.sleep(10)
    if run is None:
        raise RuntimeError("no se pudo consultar la corrida")
    result_state = run.get("state", {}).get("result_state")
    task_run_id = run["tasks"][0]["run_id"]
    output = dbx(args.profile, "jobs", "get-run-output", str(task_run_id), "-o", "json")
    output_json = json.loads(output.stdout)
    raw_result = output_json.get("notebook_output", {}).get("result", "{}")
    try:
        notebook_summary = json.loads(raw_result)
    except json.JSONDecodeError:
        notebook_summary = {"raw": raw_result}

    remote_files = dbx(
        args.profile, "fs", "ls", f"{volume_uri}/corpus/pdf", "--absolute", "-o", "json",
    )
    file_entries = json.loads(remote_files.stdout)
    pdf_count = sum(
        1
        for item in file_entries
        if item.get("is_directory") is False and item.get("name", "").lower().endswith(".pdf")
    )

    report = {
        "state": result_state,
        "run_id": run_id,
        "task_run_id": task_run_id,
        "run_page_url": run.get("run_page_url"),
        "workspace_path": args.workspace_path,
        "volume_uri": volume_uri,
        "volume_pdf_count": pdf_count,
        "notebook_summary": notebook_summary,
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "databricks-run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if result_state != "SUCCESS" or pdf_count != 24:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
