"""Canario: mide el falso rechazo contra ground truth **externo**.

Las siete invariantes garantizan que un chunk es subcadena del crudo. **No garantizan que el
crudo sea fiel al PDF.** Ese hueco es el que cerró tres defectos a producción con todo en
verde: los gates miraban el fuente y no lo publicado.

El juez son los abstracts **oficiales** publicados por arXiv y Crossref: texto real de cada
documento que ningún componente de este pipeline produjo. Se mide qué fracción de sus
oraciones NO logra verificar contra el crudo ingerido.

⚠️ **La línea base de 15,5% es un piso pesimista, no la tasa real del sistema.** El abstract
publicado no siempre coincide con el del PDF (versiones distintas, JATS de Crossref,
correcciones editoriales). Sirve como **detector de regresión**, no como métrica de calidad:
lo que importa es que no empeore, no su valor absoluto.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

LINEA_BASE = 0.155
TOLERANCIA = 0.20  # por encima de esto, es regresión y no ruido

MINIMO_ORACION = 45


MINIMO_ORACIONES = 60  # por debajo de esto el canario no tiene poder estadístico


@dataclass
class Canario:
    oraciones: int = 0
    verifican: int = 0
    por_documento: dict[str, tuple[int, int]] = field(default_factory=dict)
    sin_ground_truth: list[str] = field(default_factory=list)

    @property
    def falso_rechazo(self) -> float | None:
        """`None` cuando no midió nada.

        Antes devolvía `0.0`, que el reporte imprimía como «0.0% de falso rechazo» — se leía
        como una mejora espectacular cuando en realidad el juez estaba desconectado. Dos
        caminos reales llegaban ahí: renombrar los ids del catálogo (el ground truth se
        indexa por id) o excluir todas las fuentes. Ambos terminaban en `exit 0`.
        """
        return 1 - (self.verifican / self.oraciones) if self.oraciones else None

    @property
    def midio(self) -> bool:
        return self.oraciones >= MINIMO_ORACIONES

    @property
    def hay_regresion(self) -> bool:
        """No medir es una regresión: un juez desconectado no es un juez satisfecho."""
        if not self.midio:
            return True
        return self.falso_rechazo > TOLERANCIA

    @property
    def motivo(self) -> str | None:
        if not self.midio:
            return (
                f"el canario sólo evaluó {self.oraciones} oraciones (mínimo "
                f"{MINIMO_ORACIONES}): el juez está desconectado, no satisfecho"
            )
        if self.falso_rechazo > TOLERANCIA:
            return f"falso rechazo {self.falso_rechazo:.1%} sobre una base de {LINEA_BASE:.1%}"
        return None

    def a_dict(self) -> dict:
        return {
            "oraciones": self.oraciones,
            "verifican": self.verifican,
            "falso_rechazo": round(self.falso_rechazo, 4) if self.midio else None,
            "midio": self.midio,
            "sin_ground_truth": self.sin_ground_truth,
        }


def cargar_ground_truth(path: str | Path) -> dict[str, str]:
    """Las claves del JSON son nombres de archivo; el doc_id es su prefijo."""
    crudo = json.loads(Path(path).read_text(encoding="utf-8"))
    return {archivo.split("_", 1)[0]: abstract for archivo, abstract in crudo.items()}


def oraciones_de(abstract: str) -> list[str]:
    return [
        s.strip() for s in re.split(r"(?<=[.!?]) +", abstract)
        if len(s.strip()) > MINIMO_ORACION
    ]


def medir_canario(documentos, ground_truth: dict[str, str]) -> Canario:
    c = Canario()
    for doc in documentos:
        abstract = ground_truth.get(doc.doc_id)
        if not abstract:
            # Saltear en silencio dejaba 3 de 13 documentos sin auditar y sin que
            # apareciera en ningún lado. Ahora se nombran.
            c.sin_ground_truth.append(doc.doc_id)
            continue
        oraciones = oraciones_de(abstract)
        ok = sum(1 for o in oraciones if doc.verificar(o) is not None)
        c.oraciones += len(oraciones)
        c.verifican += ok
        c.por_documento[doc.doc_id] = (ok, len(oraciones))
    return c
