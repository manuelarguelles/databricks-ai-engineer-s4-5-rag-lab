"""Capa bronce — el PDF, intacto, y la cláusula que lo hace cumplir.

La verdad vive en S3; el espejo local se reconcilia por `sha256`. Nada de este módulo
modifica un byte del PDF, y nada más del pipeline sabe de dónde vino el archivo.

El hash no es higiene: es la invariante 6. Si el PDF cambió, todos los offsets de ese
documento son ficción, y el pipeline tiene que **fallar ruidosamente** en vez de recalcular
en silencio sobre un texto distinto.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .catalogo import Fuente

BLOQUE = 65536


class BronceInvalido(Exception):
    """El PDF no se puede usar como base de offsets."""


def sha256_de(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for bloque in iter(lambda: fh.read(BLOQUE), b""):
            h.update(bloque)
    return h.hexdigest()


def verificar_bronce(fuente: Fuente, dir_pdf: str | Path) -> Path:
    """Devuelve la ruta del PDF verificado, o levanta. Gate 1."""
    if not fuente.archivo or not fuente.sha256:
        raise BronceInvalido(f"fuente {fuente.id}: sin PDF asociado")

    path = Path(dir_pdf) / fuente.archivo
    if not path.exists():
        raise BronceInvalido(f"fuente {fuente.id}: {path} no existe")

    real = sha256_de(path)
    if real != fuente.sha256:
        raise BronceInvalido(
            f"fuente {fuente.id}: sha256 no coincide "
            f"(catálogo {fuente.sha256[:12]}…, archivo {real[:12]}…) — "
            f"los offsets de este documento serían inválidos"
        )
    return path
