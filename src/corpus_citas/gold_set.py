"""Gold set: preguntas con su evidencia literal, y el gate que las admite.

Dos gates, los dos deterministas y sin LLM:

**Gate de evidencia.** El `span` declarado tiene que verificar con `Documento.verificar()`.
Es exactamente el mismo `in` con el que después se mide el sistema: una pregunta cuyo span
no verifica mediría el gold set, no el agente.

**Gate anti-eco.** Se rechaza la pregunta que repite su propia evidencia. Una pregunta
escrita copiando el span se recupera sola y es un test que no puede fallar — el mismo pecado
que ya apareció cuatro veces en los bancos de este proyecto.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .vista import construir_vista

# Por encima de este solapamiento léxico, la pregunta está citando su propia respuesta.
MAX_ECO = 0.60
MINIMO_PALABRAS_PREGUNTA = 6

VACIAS = {
    "que", "qué", "cual", "cuál", "como", "cómo", "cuando", "cuándo", "donde", "dónde",
    "quien", "quién", "por", "para", "de", "del", "la", "el", "los", "las", "un", "una",
    "y", "o", "en", "con", "sin", "se", "su", "sus", "es", "son", "al", "a", "lo", "le",
    "the", "of", "and", "in", "to", "for", "a", "an", "is", "are", "on", "with",
}

PALABRA = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Pregunta:
    id: str
    doc_id: str
    tipo: str
    pregunta: str
    span: str
    char_start: int | None = None
    char_end: int | None = None
    pagina: int | None = None


class GoldSetInvalido(Exception):
    """Una pregunta no se puede usar para medir."""


def _terminos(texto: str) -> set[str]:
    return {
        p for p in PALABRA.findall(construir_vista(texto).texto)
        if len(p) > 2 and p not in VACIAS
    }


def eco(pregunta: str, span: str) -> float:
    """Fracción de los términos de la pregunta que ya están en su propia evidencia."""
    terminos = _terminos(pregunta)
    if not terminos:
        return 1.0
    return len(terminos & _terminos(span)) / len(terminos)


def cargar_gold_set(path: str | Path) -> list[Pregunta]:
    datos = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [Pregunta(**p) for p in datos.get("preguntas", [])]


def validar(preguntas: list[Pregunta], documentos: dict) -> tuple[list[Pregunta], list[str]]:
    """Devuelve (admitidas con sus offsets, motivos de rechazo)."""
    admitidas: list[Pregunta] = []
    rechazos: list[str] = []
    vistos: set[str] = set()

    for p in preguntas:
        if p.id in vistos:
            rechazos.append(f"{p.id}: id duplicado")
            continue
        vistos.add(p.id)

        if len(PALABRA.findall(p.pregunta)) < MINIMO_PALABRAS_PREGUNTA:
            rechazos.append(f"{p.id}: la pregunta es demasiado corta para ser una pregunta")
            continue

        doc = documentos.get(p.doc_id)
        if doc is None:
            rechazos.append(f"{p.id}: el documento {p.doc_id} no está en el corpus")
            continue

        cita = doc.verificar(p.span)
        if cita is None:
            rechazos.append(
                f"{p.id}: el span no verifica contra {p.doc_id} — no existe como texto "
                f"contiguo del documento"
            )
            continue

        e = eco(p.pregunta, p.span)
        if e > MAX_ECO:
            rechazos.append(
                f"{p.id}: eco léxico {e:.0%} (máximo {MAX_ECO:.0%}) — la pregunta repite "
                f"su propia evidencia y se recuperaría sola"
            )
            continue

        admitidas.append(
            Pregunta(
                id=p.id, doc_id=p.doc_id, tipo=p.tipo, pregunta=p.pregunta, span=p.span,
                char_start=cita.char_start, char_end=cita.char_end, pagina=cita.pagina,
            )
        )

    return admitidas, rechazos
