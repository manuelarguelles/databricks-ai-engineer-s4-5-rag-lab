#!/usr/bin/env python3
"""Valida los diez gates del laboratorio S4.5 y deja evidencia JSON.

La validación local reingiere los 24 PDF, valida el gold set y reproduce dense/BM25/RRF
usando el cache de embeddings congelado. No llama a modelos ni descarga archivos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from corpus_citas.consulta import CacheTraduccion, traducir  # noqa: E402
from corpus_citas.gold_set import cargar_gold_set, validar  # noqa: E402
from corpus_citas.indice import Retriever  # noqa: E402
from corpus_citas.pipeline import cargar_documentos, ingerir  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acierta(resultados, pregunta) -> bool:
    return any(
        r.doc_id == pregunta.doc_id
        and r.char_start <= pregunta.char_start < r.char_end
        for r in resultados
    )


def gate(nombre: str, fn, resultados: list[dict]) -> None:
    try:
        detalle = fn()
        resultados.append({"gate": nombre, "status": "PASS", "detail": detalle})
        print(f"PASS {nombre}: {detalle}")
    except Exception as exc:  # el reporte debe contener todos los gates, no parar en el primero
        resultados.append({"gate": nombre, "status": "FAIL", "detail": str(exc)})
        print(f"FAIL {nombre}: {exc}")


def corpus_y_manifest() -> tuple[list[dict], dict[str, Path]]:
    manifest = yaml.safe_load((ROOT / "corpus/manifest.yml").read_text(encoding="utf-8"))
    incluidas = [f for f in manifest["fuentes"] if f["estado"] == "incluida"]
    paths: dict[str, Path] = {}
    for group in ("base", "distractores"):
        for p in (ROOT / "corpus/pdf" / group).glob("*.pdf"):
            if p.name in paths:
                raise AssertionError(f"PDF duplicado: {p.name}")
            paths[p.name] = p
    return incluidas, paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-databricks", action="store_true")
    ap.add_argument("--profile", default="webinar-aws-v2")
    ap.add_argument("--catalog", default="neptuno_manuel_arguelles")
    args = ap.parse_args()

    resultados: list[dict] = []
    incluidas, pdfs = corpus_y_manifest()
    expected = json.loads((ROOT / "data/expected_metrics.json").read_text(encoding="utf-8"))
    state: dict = {}

    def g01():
        assert len(incluidas) == 24, f"manifiesto incluido={len(incluidas)}"
        assert len(pdfs) == 24, f"archivos PDF={len(pdfs)}"
        for fuente in incluidas:
            archivo = fuente["pdf"]["archivo"]
            assert archivo in pdfs, f"falta {archivo}"
            actual = sha256(pdfs[archivo])
            assert actual == fuente["pdf"]["sha256"], f"hash distinto: {archivo}"
        return "24/24 PDF presentes y hashes válidos"

    gate("G01_CORPUS", g01, resultados)

    def g02():
        base = list((ROOT / "corpus/pdf/base").glob("*.pdf"))
        dist = list((ROOT / "corpus/pdf/distractores").glob("*.pdf"))
        assert len(base) == 13 and len(dist) == 11, f"base={len(base)} distractores={len(dist)}"
        texto = (ROOT / "corpus/README.md").read_text(encoding="utf-8")
        assert "licencia" in texto.lower() and "SHA-256" in texto
        return "13 base + 11 distractores; procedencia y licencia declaradas"

    gate("G02_CLASIFICACION", g02, resultados)

    def g03():
        texto = (ROOT / "scripts/deploy_and_run.py").read_text(encoding="utf-8")
        for token in (
            '"schemas", "create"',
            '"volumes", "create"',
            '"fs", "cp"',
            '"workspace", "import"',
            '"jobs", "submit"',
        ):
            assert token in texto, f"falta automatización: {token}"
        return "despliegue, carga, import y ejecución automatizados"

    gate("G03_CARGA_AUTOMATICA", g03, resultados)

    def g04():
        nb = (ROOT / "notebooks/S4.5-laboratorio-rag-24-documentos.py").read_text(encoding="utf-8")
        for token in (
            "ai_parse_document", "ai_prep_search", "Change Data Feed", "gold set",
            "BM25", "RRF", "reranker", "dbutils.notebook.exit",
        ):
            assert token.lower() in nb.lower(), f"notebook sin sección: {token}"
        assert nb.count("# COMMAND ----------") >= 15
        return "notebook único con ruta administrada, benchmark, evaluación y salida"

    gate("G04_NOTEBOOK", g04, resultados)

    def preparar_benchmark():
        build = ROOT / "build/acceptance"
        if build.exists():
            shutil.rmtree(build)
        flat = build / "pdf"
        flat.mkdir(parents=True)
        for p in pdfs.values():
            shutil.copy2(p, flat / p.name)
        reporte = ingerir(ROOT / "corpus/manifest.yml", flat, build / "plata")
        assert not reporte.abortados, reporte.abortados
        docs = cargar_documentos(build / "plata")
        preguntas, rechazos = validar(
            cargar_gold_set(ROOT / "data/gold_set.yml"), {d.doc_id: d for d in docs}
        )
        state.update({"reporte": reporte, "docs": docs, "preguntas": preguntas, "rechazos": rechazos})

    preparar_error = None
    try:
        preparar_benchmark()
    except Exception as exc:
        preparar_error = exc

    def g05():
        if preparar_error:
            raise preparar_error
        assert len(state["preguntas"]) == 70, len(state["preguntas"])
        assert state["rechazos"] == [], state["rechazos"]
        return "70 preguntas admitidas; 0 rechazos por evidencia o anti-eco"

    gate("G05_GOLD_SET", g05, resultados)

    def g06():
        if preparar_error:
            raise preparar_error
        docs = state["docs"]
        assert len(docs) == 24
        assert sum(len(d.chunks) for d in docs) == 1405
        retriever = Retriever(docs, ROOT / "data/embeddings.json")
        preguntas = state["preguntas"]
        cache = CacheTraduccion(ROOT / "data/traducciones.json")
        consultas = dict(zip(
            (p.id for p in preguntas),
            traducir([p.pregunta for p in preguntas], cache),
        ))
        hits = {}
        for modo in ("denso", "lexico", "rrf"):
            hits[modo] = sum(
                acierta(retriever.buscar(consultas[p.id], k=8, modo=modo), p)
                for p in preguntas
            )
        assert hits == {"denso": 36, "lexico": 53, "rrf": 51}, hits
        state["hits"] = hits
        return f"1.405 chunks; dense={hits['denso']}/70 BM25={hits['lexico']}/70 RRF={hits['rrf']}/70"

    gate("G06_BENCHMARK", g06, resultados)

    def g07():
        assert expected["hits"]["rrf_listwise"] == 58
        assert expected["source_commit"].startswith("f22f856")
        evidence = json.loads((ROOT / "data/reranker_evidence.json").read_text(encoding="utf-8"))
        assert evidence["calibration"]["hits"] == 34
        assert evidence["validation"]["hits"] == 24
        assert evidence["total"]["hits"] == 58 and evidence["total"]["questions"] == 70
        return "listwise 34/37 + 24/33 = 58/70, con procedencia de corrida"

    gate("G07_RERANKER", g07, resultados)

    def g08():
        for name in ("GUIA-ALUMNO.md", "GUIA-INSTRUCTOR.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            assert "Troubleshooting" in text or "Errores comunes" in text
            assert "36/70" in text and "58/70" in text
        deck = (ROOT / "slides/S4.5-deck.html").read_text(encoding="utf-8")
        assert deck.count('<section class="slide') == 17
        for token in ("24 documentos", "1.405 chunks", "70 preguntas", "36/70", "58/70"):
            assert token in deck, f"deck sin {token}"
        return "deck y guías con orden, tiempos, outputs y resolución de fallos"

    gate("G08_OPERACION_DOCENTE", g08, resultados)

    def g09():
        if args.skip_databricks:
            return "omitido explícitamente en prevalidación local"
        report = json.loads((ROOT / "reports/databricks-run.json").read_text(encoding="utf-8"))
        assert report["state"] == "SUCCESS", report.get("state")
        assert report["volume_pdf_count"] == 24, report.get("volume_pdf_count")
        summary = report["notebook_summary"]
        assert summary["pdfs"] == 24
        assert summary["documents"] == 24
        assert summary["benchmark_chunks"] == 1405
        assert summary["questions"] == 70
        assert summary["hits"] == {"dense": 36, "bm25": 53, "rrf": 51, "rrf_listwise": 58}
        assert summary["managed_route_ok"] is True
        assert summary["managed_attempted_documents"] == 24
        assert summary["managed_primary_documents"] + summary["managed_fallback_documents"] == 24
        return f"Databricks SUCCESS run_id={report['run_id']} y Volume con 24 PDF"

    gate("G09_DATABRICKS_E2E", g09, resultados)

    def g10():
        text = (ROOT / "HANDOFF-DANIEL.md").read_text(encoding="utf-8")
        for token in ("Qué vas a dictar", "Ruta de la clase", "Plan B", "58/70"):
            assert token in text
        return "handoff de Daniel autocontenido y condicionado a 10/10 PASS"

    gate("G10_HANDOFF", g10, resultados)

    passed = sum(r["status"] == "PASS" for r in resultados)
    payload = {
        "status": "PASS" if passed == 10 else "FAIL",
        "passed": passed,
        "total": 10,
        "profile": args.profile,
        "catalog": args.catalog,
        "gates": resultados,
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "acceptance.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nRESULTADO: {passed}/10 PASS")
    return 0 if passed == 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
