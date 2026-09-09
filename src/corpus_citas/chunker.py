"""Particiona el crudo en chunks citables.

Dos decisiones medidas viven acá:

**1. Los spans particionan** — sin solape. Medido sobre 10 documentos y 71 oraciones de
ground truth, el overlap entre chunks daba recall@5 32/71 contra 26/71 sin overlap... pero
la misma ganancia se consigue **sin** solapar los spans:

    sin overlap                      26/71   partición ✅
    overlap 15%                      32/71   partición ❌ (la cita puede caer en otro chunk)
    span limpio + vector con contexto 32/71  partición ✅   <-- elegido

**2. El span es para citar; el vector es para encontrar.** No son el mismo texto. El overlap
y la expansión-a-vecinos parecen resolver lo mismo y no: la expansión repara el contexto de
la *respuesta* y ocurre **después** de recuperar, así que no puede rescatar un chunk que
nunca se recuperó. El contexto del vector actúa **antes**, en la recuperación.

Este módulo **no conoce páginas**. La página se deriva del offset, en `Documento`.
"""

from __future__ import annotations

from dataclasses import dataclass

# El techo de 2048 chars es de `cohere.embed-multilingual-v3` y aplica a **lo que se
# embebe**, no al chunk. La validacion adversarial mostro que `texto_para_embeber` llegaba
# a 2355 chars (+15%): el proveedor lo trunca por el final, que es justo donde vive el texto
# citable del chunk. Por eso el chunk se dimensiona para que chunk + contexto entren enteros.
TECHO_EMBEBIDO = 2048
CONTEXTO = 307            # 15% del techo: cuanto se extiende hacia atras el vector
TECHO = TECHO_EMBEBIDO - CONTEXTO   # 1741 chars de chunk

# Nota: el banco que eligio "span limpio + vector con contexto" ya media asi — su funcion
# `embeber()` recortaba con `t[:2048]`. O sea que el 32/71 de recall se obtuvo con este
# presupuesto efectivo, no con 2355 chars.


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    char_start: int
    char_end: int
    emb_start: int
    text: str
    eje: str | None


def _cortes(crudo: str, bloques: list[int]) -> list[tuple[int, int]]:
    """Agrupa bloques hasta TECHO. Un bloque que solo ya excede el techo se parte."""
    if not crudo:
        return []

    # `bloques` puede venir de un JSON de oro: un offset pasado del final generaba
    # chunks enteramente fuera del crudo (los atrapaba la invariante 2, pero tarde).
    bloques = [b for b in bloques if 0 <= b < len(crudo)]
    limites = sorted(set(bloques + [len(crudo)]))
    if limites[0] != 0:
        limites = [0] + limites

    spans: list[tuple[int, int]] = []
    inicio = 0
    for siguiente in limites[1:]:
        if siguiente - inicio < TECHO:
            continue  # todavia entra otro bloque
        if siguiente - inicio == TECHO:
            spans.append((inicio, siguiente))
            inicio = siguiente
            continue
        # Se paso del techo: se cierra en el limite anterior si dejaria un chunk no vacio.
        anterior = max((l for l in limites if inicio < l < siguiente), default=inicio)
        if anterior > inicio:
            spans.append((inicio, anterior))
            inicio = anterior
        # Lo que queda del bloque puede seguir excediendo el techo: se parte por longitud.
        while siguiente - inicio > TECHO:
            spans.append((inicio, inicio + TECHO))
            inicio += TECHO

    if inicio < len(crudo):
        spans.append((inicio, len(crudo)))
    return spans


def chunkear(doc_id: str, crudo: str, bloques: list[int], eje: str | None) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc_id}:{inicio}-{fin}",
            doc_id=doc_id,
            char_start=inicio,
            char_end=fin,
            emb_start=max(0, inicio - CONTEXTO),
            text=crudo[inicio:fin],
            eje=eje,
        )
        for inicio, fin in _cortes(crudo, bloques)
    ]


def texto_para_embeber(chunk: Chunk, crudo: str) -> str:
    """El chunk más su contexto previo. Es lo que se vectoriza, no lo que se cita."""
    return crudo[chunk.emb_start:chunk.char_end]
