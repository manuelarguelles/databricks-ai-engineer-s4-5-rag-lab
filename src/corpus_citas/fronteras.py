"""Clasificación de fronteras entre bloques: ¿el texto continúa, o son piezas distintas?

Una frontera es **dura** cuando el texto de un lado no es continuación del otro por más que
lo parezca al colapsar los blancos. Cruzar una dura fabrica evidencia.

El primer intento usó sólo distancia vertical (hueco > 18 pt). Medido, dejaba pasar 271
soldaduras: **una tabla tiene el mismo interlineado que un párrafo**, así que ningún umbral de
distancia separa una celda de un renglón. Y de paso convertía en duras las costuras de columna
y de página, que sí son contiguas en la lectura, generando 55 falsos rechazos.

Este módulo mide **estructura**, no distancia:

    periférico   folio, encabezado de página: vive en el margen y no pertenece al cuerpo
    fila         hay ≥3 bloques a su misma altura ⇒ es una celda de tabla
    referencias  todo lo que sigue al encabezado «References» son entradas independientes

Cruzar de columna o de página **sí se permite** — el texto fluye — siempre que no se pase por
un bloque periférico, que es lo que realmente ensuciaba esas costuras.
"""

from __future__ import annotations

import re

MARGEN = 0.075        # fracción de la altura de página que se considera margen
BANDA_Y = 6.0         # pt de tolerancia para decir que dos bloques están «a la misma altura»
MINIMO_EN_FILA = 3    # ≥3 bloques a la misma altura ⇒ tabla (dos columnas de texto dan 2)
# Calibrado con `scripts/banco_fronteras.py` sobre los 13 documentos, mirando LAS DOS tasas:
#     pt    soldaduras aceptadas    cruces legítimos rechazados
#      6            5%                        82%
#     10           13%                        68%
#     14           27%                        15%   <-- elegido: el codo
#     18           36%                        15%
#     24           41%                        13%
# 14 domina a 18 (mismo falso rechazo, 9 puntos menos de soldaduras) y bajar a 10 cambia
# 14 puntos de soldadura por 53 de falso rechazo.
SALTO_MAXIMO = 14.0
TOLERANCIA_Y = 3.0    # pt de retroceso tolerado (redondeos de layout)
LARGO_PERIFERICO = 90  # un folio o un running head es corto
# Barrido sobre los 13 documentos (soldaduras / cruces rechazados):
#   0,6 pt -> 19% / 22%      1,6 pt -> 24% / 20%
#   1,0 pt -> 20% / 22%      2,5 pt -> 26% / 20%
# El umbral casi no mueve el falso rechazo, así que se toma el que más soldaduras cierra.
DELTA_CUERPO = 0.6

ENCABEZADO_REFS = re.compile(
    r"^\s*(references|bibliography|referencias|works cited)\s*$", re.I
)


class Bloque:
    """Un bloque de texto con lo que hace falta para clasificar sus fronteras."""

    __slots__ = ("offset", "texto", "pagina", "columna", "x0", "y0", "x1", "y1",
                 "cuerpo", "periferico", "en_fila", "en_refs")

    def __init__(self, offset, texto, pagina, columna, bbox, cuerpo=0.0):
        self.offset = offset
        self.texto = texto
        self.pagina = pagina
        self.columna = columna
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.cuerpo = cuerpo          # tamaño de fuente dominante, en pt
        self.periferico = False
        self.en_fila = False
        self.en_refs = False


def marcar_perifericos(bloques: list[Bloque], alto_pagina: dict[int, float]) -> None:
    """Folio y running head: cortos y pegados al borde superior o inferior."""
    for b in bloques:
        alto = alto_pagina.get(b.pagina, 792.0)
        cerca_del_borde = b.y0 < alto * MARGEN or b.y1 > alto * (1 - MARGEN)
        if cerca_del_borde and len(b.texto.strip()) <= LARGO_PERIFERICO:
            b.periferico = True


def marcar_filas(bloques: list[Bloque]) -> None:
    """Una celda de tabla tiene vecinos a su misma altura; un párrafo no.

    Se exigen ≥3 en la banda porque un paper a dos columnas siempre tiene 2 bloques a la
    misma altura y eso NO es una tabla.
    """
    por_pagina: dict[int, list[Bloque]] = {}
    for b in bloques:
        por_pagina.setdefault(b.pagina, []).append(b)

    for pagina in por_pagina.values():
        for b in pagina:
            vecinos = sum(
                1 for o in pagina
                if o is not b and abs(o.y0 - b.y0) < BANDA_Y and abs(o.x0 - b.x0) > 15
            )
            if vecinos + 1 >= MINIMO_EN_FILA:
                b.en_fila = True


def marcar_referencias(bloques: list[Bloque]) -> None:
    """Tras «References», cada entrada es independiente: soldarlas inventa una cita."""
    visto = False
    for b in bloques:
        if not visto and ENCABEZADO_REFS.match(b.texto.strip()[:40]):
            visto = True
        b.en_refs = visto


def es_dura(a: Bloque, b: Bloque) -> bool:
    """¿Cruzar de `a` a `b` fabrica adyacencia?"""
    if a.periferico or b.periferico:
        return True                       # el folio y el encabezado no son cuerpo
    if a.en_fila or b.en_fila:
        return True                       # celdas de tabla: nada se solda con nada
    if a.en_refs or b.en_refs:
        return True                       # entradas de bibliografía independientes

    # El registro tipográfico distingue lo que la geometría no. El running head, la nota
    # al pie, el pie de figura y la línea de copyright caían en blanda porque están a
    # 13-17 pt del cuerpo — por debajo del umbral. Pero tienen otro cuerpo de letra.
    # Medido: 381 costuras con Δcuerpo ≥ 1 pt, 272 de ellas aceptaban una cita fabricada.
    if a.cuerpo and b.cuerpo and abs(a.cuerpo - b.cuerpo) >= DELTA_CUERPO:
        return True

    # PROBADA Y DESCARTADA: «rangos horizontales disjuntos ⇒ dura». Cerraba 40 casos de
    # celdas y encabezados rotados, pero costaba 3 puntos de falso rechazo para ganar 1 de
    # soldadura (20%/19% -> 19%/22%). Mal negocio; queda anotada por si el corpus cambia.

    if a.pagina != b.pagina or a.columna != b.columna:
        return False                      # el texto fluye; ya se filtró lo periférico
    if b.y0 < a.y1 - TOLERANCIA_Y:
        return True                       # el bloque siguiente está ARRIBA
    if b.y0 - a.y1 > SALTO_MAXIMO:
        return True                       # figura, tabla o cambio de sección en el medio
    return False


def clasificar(bloques: list[Bloque], alto_pagina: dict[int, float]) -> list[int]:
    """Devuelve los offsets de las fronteras duras."""
    marcar_perifericos(bloques, alto_pagina)
    marcar_filas(bloques)
    marcar_referencias(bloques)
    return [b.offset for a, b in zip(bloques, bloques[1:]) if es_dura(a, b)]
