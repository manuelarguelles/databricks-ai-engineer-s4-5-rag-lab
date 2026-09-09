"""Extracción del texto crudo canónico. **El único módulo que importa PyMuPDF.**

Orden de lectura **por bloques y por columna**, no por renglón: los papers son a dos columnas
y leer por coordenada Y entrelaza las dos, fabricando oraciones que nunca existieron. Medido:
ordenar por columna baja los cortes raros un 33-53%.

El crudo sale **tal cual** del extractor: con sus guiones de corte, sus saltos y su tipografía.

La clasificación de fronteras vive en `fronteras.py`, que es donde se decide qué costuras se
pueden cruzar. Acá sólo se recoge la geometría que esa decisión necesita.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from .fronteras import Bloque, clasificar

EXTRACTOR = "pymupdf-bloques@1.27.2.3"

TIPO_TEXTO = 0  # get_text("blocks") marca los bloques de imagen con 1


@dataclass(frozen=True)
class Extraccion:
    crudo: str
    paginas: list[tuple[int, int]]  # (offset donde arranca, nº de página, 1-based)
    bloques: list[int]              # offsets donde arranca cada bloque de texto
    fronteras: list[int]            # subconjunto de `bloques`: los cruces prohibidos


def _cuerpo_por_bloque(page) -> dict[int, float]:
    """Tamaño de fuente dominante de cada bloque, por número de bloque.

    El registro tipográfico separa el cuerpo del mobiliario editorial (running head, nota
    al pie, copyright) cuando la geometría no alcanza. Se toma el tamaño que cubre más
    caracteres, no el promedio: un superíndice no debe mover el valor del párrafo.
    """
    cuerpos: dict[int, dict[float, int]] = {}
    for bloque in page.get_text("dict").get("blocks", []):
        numero = bloque.get("number")
        if numero is None:
            continue
        conteo = cuerpos.setdefault(numero, {})
        for linea in bloque.get("lines", []):
            for span in linea.get("spans", []):
                tamano = round(span.get("size", 0.0), 1)
                conteo[tamano] = conteo.get(tamano, 0) + len(span.get("text", ""))
    return {
        n: max(c, key=c.get) if c else 0.0
        for n, c in cuerpos.items()
    }


def extraer(path: str | Path) -> Extraccion:
    doc = fitz.open(str(path))
    partes: list[str] = []
    paginas: list[tuple[int, int]] = []
    detalle: list[Bloque] = []
    alto_pagina: dict[int, float] = {}
    pos = 0

    try:
        for numero, page in enumerate(doc, start=1):
            alto_pagina[numero] = page.rect.height
            ancho = page.rect.width
            crudos = [b for b in page.get_text("blocks") if b[6] == TIPO_TEXTO]
            cuerpos = _cuerpo_por_bloque(page)

            # Sólo se registra la página si aporta texto. Antes se registraban todas y
            # después se acotaban al largo del crudo, lo cual no eliminaba la página
            # fantasma: la reubicaba sobre texto real y colapsaba las siguientes, así que
            # `pagina(offset)` devolvía un número equivocado.
            if any(b[4].strip() for b in crudos):
                paginas.append((pos, numero))
            # Columna primero (izquierda entera, luego derecha), despues y, despues x.
            crudos.sort(key=lambda b: (0 if b[0] < ancho / 2 else 1, round(b[1], 1), b[0]))

            for b in crudos:
                texto = b[4]
                if not texto.strip():
                    continue
                detalle.append(
                    Bloque(
                        offset=pos,
                        texto=texto,
                        pagina=numero,
                        columna=0 if b[0] < ancho / 2 else 1,
                        bbox=(b[0], b[1], b[2], b[3]),
                        cuerpo=cuerpos.get(b[5], 0.0),
                    )
                )
                partes.append(texto)
                pos += len(texto) + 1  # el "\n" con el que se unen

        crudo = "\n".join(partes)

        return Extraccion(
            crudo=crudo,
            paginas=paginas,
            bloques=[b.offset for b in detalle],
            fronteras=clasificar(detalle, alto_pagina),
        )
    finally:
        doc.close()
