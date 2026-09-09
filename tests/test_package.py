import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_corpus_tiene_24_pdf_y_todos_verifican():
    manifest = yaml.safe_load((ROOT / "corpus/manifest.yml").read_text(encoding="utf-8"))
    incluidas = [f for f in manifest["fuentes"] if f["estado"] == "incluida"]
    pdfs = list((ROOT / "corpus/pdf").glob("*/*.pdf"))
    assert len(incluidas) == len(pdfs) == 24
    por_nombre = {p.name: p for p in pdfs}
    for fuente in incluidas:
        meta = fuente["pdf"]
        assert hashlib.sha256(por_nombre[meta["archivo"]].read_bytes()).hexdigest() == meta["sha256"]


def test_corpus_separa_base_y_distractores():
    assert len(list((ROOT / "corpus/pdf/base").glob("*.pdf"))) == 13
    assert len(list((ROOT / "corpus/pdf/distractores").glob("*.pdf"))) == 11


def test_gold_set_tiene_70_preguntas():
    data = yaml.safe_load((ROOT / "data/gold_set.yml").read_text(encoding="utf-8"))
    assert len(data["preguntas"]) == 70
    assert len({p["id"] for p in data["preguntas"]}) == 70


def test_metricas_congeladas_tienen_procedencia():
    data = json.loads((ROOT / "data/expected_metrics.json").read_text(encoding="utf-8"))
    assert data["chunks"] == 1405
    assert data["hits"] == {"dense": 36, "bm25": 53, "rrf": 51, "rrf_listwise": 58}
    assert len(data["source_commit"]) == 40


def test_material_docente_y_handoff_existen():
    for name in (
        "README.md", "CRITERIOS-ACEPTACION.md", "GUIA-ALUMNO.md",
        "GUIA-INSTRUCTOR.md", "HANDOFF-DANIEL.md",
        "notebooks/S4.5-laboratorio-rag-24-documentos.py",
    ):
        assert (ROOT / name).stat().st_size > 500

