"""El documento ensamblado y su API de consulta.

`verificar()` es el corazón del proyecto: **la única función que puede afirmar que una cita
existe**. Devuelve la evidencia — el span en el crudo con su página — o `None`.

El camino cruza las tres capas una sola vez:
    se busca en la VISTA → se traduce el span al CRUDO con el mapa → se resuelve la página.

🚨 **Corrección tras la validación adversarial (16-ago-2026).** `extraer.py` une bloques y
páginas con `"\\n"`, y la vista colapsa ese salto a un espacio: dos bloques que no son
contiguos en la lectura real quedaban **soldados**. Medido, **151 de 168** citas fabricadas a
través de una costura de columna o de página eran aceptadas, con span y página.

Por eso el documento persiste los offsets de bloque y de **frontera dura**, y `verificar()`
rechaza todo span que cruce una dura. Cuáles lo son se decide en `fronteras.py`, con la
geometría y la tipografía de la página: rechazar *todo* cruce de bloque se midió primero y
dejaba pasar sólo 1 de cada 10 citas legítimas.

El criterio se calibra con `scripts/banco_fronteras.py`, que mide **las dos tasas** —
soldaduras aceptadas y cruces legítimos rechazados. El canario no sirve para eso: sus
oraciones viven dentro de un bloque, así que no mira donde este criterio cobra su precio.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field, replace

from .chunker import Chunk, chunkear
from .contrato import CONTRATO
from .extraer import EXTRACTOR, Extraccion
from .vista import NORMALIZACION, Vista, construir_vista, normalizar_consulta

NOMBRE_VALIDO = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class Cita:
    doc_id: str
    texto: str          # el texto del CRUDO, no el de la consulta
    char_start: int
    char_end: int
    pagina: int
    pagina_fin: int     # distinta de `pagina` si el span cruza páginas


@dataclass(frozen=True)
class Documento:
    doc_id: str
    sha256: str
    crudo: str
    paginas: list[tuple[int, int]]
    vista: Vista
    bloques: list[int] = field(default_factory=list)
    fronteras: list[int] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    eje: str | None = None
    extractor: str = EXTRACTOR
    normalizacion: str = NORMALIZACION
    contrato: int = CONTRATO

    # -- construccion -------------------------------------------------------------------

    @classmethod
    def desde_extraccion(
        cls, doc_id: str, sha256: str, extraccion: Extraccion, eje: str | None = None
    ) -> Documento:
        if not NOMBRE_VALIDO.match(doc_id):
            # Un doc_id con "/" o ".." escribe fuera de oro/ y `cargar_documentos` no lo ve:
            # el reporte diria "procesado" y el documento no existiria para nadie.
            raise ValueError(
                f"doc_id inválido: {doc_id!r} (sólo letras, números, punto, guión y _)"
            )
        vista = construir_vista(extraccion.crudo)
        chunks = chunkear(doc_id, extraccion.crudo, extraccion.bloques, eje)
        return cls(
            doc_id=doc_id,
            sha256=sha256,
            crudo=extraccion.crudo,
            paginas=extraccion.paginas,
            vista=vista,
            bloques=list(extraccion.bloques),
            fronteras=list(extraccion.fronteras),
            chunks=chunks,
            eje=eje,
        )

    # -- consulta -----------------------------------------------------------------------

    def pagina(self, offset: int) -> int:
        """La página es una función del offset, no un campo del chunk."""
        if not 0 <= offset < len(self.crudo):
            raise IndexError(f"offset {offset} fuera del crudo ({len(self.crudo)})")
        offsets = [o for o, _ in self.paginas]
        i = bisect.bisect_right(offsets, offset) - 1
        return self.paginas[max(i, 0)][1]

    def a_crudo(self, inicio_vista: int, fin_vista: int) -> tuple[int, int]:
        """Traduce un span del espacio de la vista al del crudo."""
        mapa = self.vista.mapa
        if not (0 <= inicio_vista < fin_vista <= len(mapa)):
            raise IndexError(f"span de vista inválido: {inicio_vista}..{fin_vista}")
        inicio = mapa[inicio_vista]
        # El fin es exclusivo: se toma el origen del ultimo caracter incluido y se avanza
        # hasta cubrirlo entero (una ligadura ocupa 1 char del crudo y varios de la vista).
        fin = mapa[fin_vista - 1] + 1
        return inicio, max(fin, inicio + 1)

    def cruza_frontera(self, inicio: int, fin: int) -> bool:
        """¿El span atraviesa una frontera DURA entre bloques?

        Duras son las que rompen la contigüidad de lectura: cambio de página, cambio de
        columna, retroceso vertical o hueco grande. Las blandas (un párrafo partido en dos
        bloques que siguen uno debajo del otro) se pueden cruzar: rechazarlas también
        dejaba pasar sólo 1 de cada 10 citas legítimas.
        """
        i = bisect.bisect_right(self.fronteras, inicio)
        return i < len(self.fronteras) and self.fronteras[i] < fin

    def verificar(self, cita: str, cerca_de: int | None = None) -> Cita | None:
        """Devuelve la evidencia o `None`. El único juez es `in`.

        `cerca_de` es el offset del pasaje donde se espera encontrarla. Sin él, se devolvía
        **la primera ocurrencia del documento**, que con texto repetido (un encabezado, un
        pie de figura, una fórmula de bibliografía) apunta a otro lugar: medido, 59 casos en
        el corpus y **38 con una página distinta** de la del pasaje que el modelo leyó.
        Reportar una página equivocada es una cita falsificable.
        """
        aguja = normalizar_consulta(cita)
        if len(aguja) < 20:
            return None  # una cita muy corta matchea por azar; no es evidencia

        candidatas: list[tuple[int, int]] = []
        desde = 0
        while True:
            i = self.vista.texto.find(aguja, desde)
            if i < 0:
                break
            inicio, fin = self.a_crudo(i, i + len(aguja))
            if not self.cruza_frontera(inicio, fin):
                candidatas.append((inicio, fin))
                if cerca_de is None:
                    break          # sin pista, la primera válida es la respuesta
            desde = i + 1

        if not candidatas:
            return None
        inicio, fin = (
            min(candidatas, key=lambda s: abs(s[0] - cerca_de))
            if cerca_de is not None else candidatas[0]
        )
        if True:
                return Cita(
                    doc_id=self.doc_id,
                    texto=self.crudo[inicio:fin],
                    char_start=inicio,
                    char_end=fin,
                    pagina=self.pagina(inicio),
                    pagina_fin=self.pagina(min(fin - 1, len(self.crudo) - 1)),
                )

    # -- helpers para los tests de deteccion --------------------------------------------

    def con_mapa(self, mapa: list[int]) -> Documento:
        return replace(self, vista=Vista(texto=self.vista.texto, mapa=mapa))

    def con_paginas(self, paginas: list[tuple[int, int]]) -> Documento:
        return replace(self, paginas=paginas)

    def con_crudo(self, crudo: str) -> Documento:
        return replace(self, crudo=crudo, vista=construir_vista(crudo))
