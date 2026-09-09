"""Traza por consulta: qué se recuperó, qué se citó, qué se rechazó y cuánto costó.

Un agente sin traza es una caja que a veces acierta. La regla acá es la misma que en el
reporte de ingesta: **si algo se descartó, tiene que estar escrito**. Un pasaje bloqueado por
el guardrail, un chunk recuperado que nadie citó o una referencia inválida son justamente lo
que hace falta para entender una respuesta mala — y lo primero que se pierde si no se registra.

La traza es un JSON por consulta, apendeable a un `.jsonl`. Sin dependencias de plataforma:
lo que se quiera hacer después con CloudWatch o con un panel se hace leyendo este archivo.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Precios de LISTA de `amazon.nova-2-lite-v1:0`, en USD por millon de tokens.
# 🚨 Los valores anteriores (0.06 / 0.24) eran los de `nova-lite-v1`, otro modelo:
# la traza subestimaba el costo ~5x en entrada y ~10x en salida, y el comentario decia
# "medidos en la cuenta" cuando no lo estaban. Un numero auditable que no lo era.
USD_POR_MILLON_ENTRADA = 0.30
USD_POR_MILLON_SALIDA = 2.50
USD_POR_MILLON_EMBEDDING = 0.10   # cohere.embed-multilingual-v3
FUENTE_PRECIOS = "lista pública de Amazon Bedrock, ago-2026 — no medido contra factura"


@dataclass
class Etapa:
    nombre: str
    ms: float
    detalle: dict = field(default_factory=dict)


@dataclass
class Traza:
    consulta: str
    consulta_traducida: str | None = None
    etapas: list[Etapa] = field(default_factory=list)
    recuperados: list[dict] = field(default_factory=list)
    citados: list[dict] = field(default_factory=list)
    rechazos: list[str] = field(default_factory=list)
    pasajes_bloqueados: list[str] = field(default_factory=list)
    tokens_entrada: int = 0
    tokens_salida: int = 0
    tokens_embedding: int = 0
    permitido: bool = True
    respuesta: str = ""

    @property
    def ms_total(self) -> float:
        return round(sum(e.ms for e in self.etapas), 1)

    @property
    def usd(self) -> float:
        return round(
            self.tokens_entrada / 1e6 * USD_POR_MILLON_ENTRADA
            + self.tokens_salida / 1e6 * USD_POR_MILLON_SALIDA
            + self.tokens_embedding / 1e6 * USD_POR_MILLON_EMBEDDING,
            6,
        )

    @property
    def recuperados_sin_citar(self) -> int:
        """Cuántos pasajes se trajeron y no se usaron: mide el desperdicio del retrieval."""
        usados = {(c["doc_id"], c["char_start"]) for c in self.citados}
        return sum(
            1 for r in self.recuperados
            if not any(
                d == r["doc_id"] and r["char_start"] <= s < r["char_end"] for d, s in usados
            )
        )

    @contextmanager
    def etapa(self, nombre: str, **detalle):
        inicio = time.perf_counter()
        try:
            yield
        finally:
            self.etapas.append(
                Etapa(nombre, round((time.perf_counter() - inicio) * 1000, 1), detalle)
            )

    def a_dict(self) -> dict:
        d = asdict(self)
        d["ms_total"] = self.ms_total
        d["usd"] = self.usd
        d["recuperados_sin_citar"] = self.recuperados_sin_citar
        return d

    def anexar(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.a_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    def resumen(self) -> str:
        etapas = " · ".join(f"{e.nombre} {e.ms:.0f}ms" for e in self.etapas)
        estado = "permitida" if self.permitido else "BLOQUEADA"
        return (
            f"[{estado}] {self.ms_total:.0f}ms  US$ {self.usd:.6f}  "
            f"{len(self.recuperados)} recuperados / {len(self.citados)} citados "
            f"({self.recuperados_sin_citar} sin usar)  |  {etapas}"
            + (f"\n  rechazos: {'; '.join(self.rechazos)}" if self.rechazos else "")
        )
