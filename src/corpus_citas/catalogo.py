"""Catálogo del corpus — el único módulo que interpreta `corpus.yml`.

`corpus.yml` es la FUENTE del pipeline. Notion es procedencia, no fuente: el importador
corre aparte y su salida es un diff que un humano aprueba. Así la ingesta corre offline y
no depende de que una API externa siga en pie.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

VERSION_SOPORTADA = 1
ESTADOS = {"incluida", "excluida", "sin-oa"}


@dataclass(frozen=True)
class Fuente:
    id: str
    titulo: str
    estado: str
    eje: str | None = None
    doi: str | None = None
    archivo: str | None = None
    sha256: str | None = None
    motivo: str | None = None


class CatalogoInvalido(ValueError):
    """El catálogo no se puede usar como contrato."""


def cargar_catalogo(path: str | Path) -> list[Fuente]:
    datos = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    version = datos.get("version")
    if version != VERSION_SOPORTADA:
        raise CatalogoInvalido(
            f"version {version!r} no soportada (esperada: {VERSION_SOPORTADA})"
        )

    fuentes: list[Fuente] = []
    vistos: set[str] = set()

    for cruda in datos.get("fuentes") or []:
        fid = str(cruda.get("id", ""))
        if not fid:
            raise CatalogoInvalido("hay una fuente sin id")
        if fid in vistos:
            raise CatalogoInvalido(f"id duplicado: {fid!r}")
        vistos.add(fid)

        estado = cruda.get("estado")
        if estado not in ESTADOS:
            raise CatalogoInvalido(
                f"fuente {fid}: estado {estado!r} desconocido (válidos: {sorted(ESTADOS)})"
            )

        pdf = cruda.get("pdf") or {}
        fuente = Fuente(
            id=fid,
            titulo=cruda.get("titulo", ""),
            estado=estado,
            eje=cruda.get("eje"),
            doi=cruda.get("doi"),
            archivo=pdf.get("archivo"),
            sha256=pdf.get("sha256"),
            motivo=cruda.get("motivo"),
        )

        # Una exclusion sin motivo escrito es una decision que nadie puede explicar despues.
        if fuente.estado == "excluida" and not (fuente.motivo or "").strip():
            raise CatalogoInvalido(f"fuente {fid}: excluida sin `motivo`")

        # Sin hash no hay clausula que hacer cumplir y los offsets quedan indefendibles.
        if fuente.estado == "incluida":
            if not fuente.archivo:
                raise CatalogoInvalido(f"fuente {fid}: incluida sin `pdf.archivo`")
            if not fuente.sha256:
                raise CatalogoInvalido(f"fuente {fid}: incluida sin `pdf.sha256`")

        fuentes.append(fuente)

    return fuentes


def fuentes_incluidas(fuentes: list[Fuente]) -> list[Fuente]:
    return [f for f in fuentes if f.estado == "incluida"]
