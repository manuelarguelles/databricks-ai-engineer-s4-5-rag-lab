"""Qué cuenta como acierto de recuperación. Una sola definición para todos los bancos.

Estaba copiada en tres scripts. Una métrica duplicada es una métrica que puede divergir sin
que nadie lo note: basta con arreglar el criterio en un banco y no en el otro para que dos
números del mismo README dejen de ser comparables.

**Acierto = algún chunk recuperado CONTIENE el span esperado.** No alcanza con traer el
documento correcto: el span es lo que sostiene la cita, y un documento de 60 páginas
recuperado «bien» sin el pasaje no le sirve al modelo para citar.
"""

from __future__ import annotations


def contiene(resultado, pregunta) -> bool:
    return (resultado.doc_id == pregunta.doc_id
            and resultado.char_start <= pregunta.char_start < resultado.char_end)


def acierta(resultados, pregunta) -> bool:
    return any(contiene(r, pregunta) for r in resultados)


def posicion(resultados, pregunta) -> int | None:
    """Posición 1-indexada del primer chunk que contiene el span, o None."""
    for i, r in enumerate(resultados, start=1):
        if contiene(r, pregunta):
            return i
    return None


def rr(resultados, pregunta) -> float:
    """Reciprocal rank de una pregunta: 0 si no se recuperó."""
    p = posicion(resultados, pregunta)
    return 1 / p if p else 0.0
