"""Carril A — retrieval construido a mano: denso + léxico, fusionados con RRF.

Tres piezas, y la razón de cada una:

**Denso** (`cohere.embed-multilingual-v3`). El corpus está en inglés y las consultas en
español: hace falta un embedder multilingüe, y es además el único que esta cuenta puede
invocar. Se embebe `crudo[emb_start:char_end]` — el chunk **más su contexto previo**, que es
la decisión medida en `bench_overlap.py` (recall@5 26→32 de 71).

**Léxico** (BM25 propio, sin dependencias). Un identificador como `INC-2023-Q4-011`, un DOI o
un nombre propio no tienen vecindario semántico: el denso los pierde y el léxico los encuentra.

**RRF** para fusionar. Suma `1/(k + posición)` de cada lista, así que no hace falta calibrar
escalas entre un coseno y un puntaje BM25 — sólo importa el orden.

Los embeddings se cachean en disco por `sha256` del texto: sin eso, cada corrida del banco
vuelve a pagar la API y los números dejan de ser reproducibles offline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MODELO_EMBEDDING = "cohere.embed-multilingual-v3"
REGION = "ca-central-1"
LOTE = 96          # máximo de textos por llamada, medido
TOPE_CHARS = 2048  # techo duro del modelo, medido

K_RRF = 60         # constante estándar del Reciprocal Rank Fusion
K1, B = 1.5, 0.75  # parámetros BM25 por defecto

PALABRA = re.compile(r"\w+", re.UNICODE)


def tokenizar(texto: str) -> list[str]:
    return PALABRA.findall(texto.lower())


@dataclass(frozen=True)
class Resultado:
    chunk_id: str
    doc_id: str
    char_start: int
    char_end: int
    puntaje: float


# ── denso ────────────────────────────────────────────────────────────────────────────

class CacheEmbeddings:
    """Cachea por hash del texto. Sin esto la reproducibilidad depende de la API."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.datos: dict[str, list[float]] = {}
        if self.path.exists():
            self.datos = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def clave(texto: str, tipo: str) -> str:
        return hashlib.sha256(f"{tipo}\x00{texto}".encode()).hexdigest()[:32]

    def faltantes(self, textos: list[str], tipo: str) -> list[str]:
        return [t for t in textos if self.clave(t, tipo) not in self.datos]

    def guardar(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.datos, sort_keys=True), encoding="utf-8")


def embeber(textos: list[str], tipo: str, cache: CacheEmbeddings) -> list[list[float]]:
    """`tipo` es `search_document` o `search_query`: Cohere los embebe distinto."""
    faltan = cache.faltantes(textos, tipo)
    if faltan:
        import boto3

        os.environ.setdefault("AWS_PROFILE", "tyv")
        br = boto3.client("bedrock-runtime", region_name=REGION)
        unicos = list(dict.fromkeys(faltan))
        for i in range(0, len(unicos), LOTE):
            lote = [t[:TOPE_CHARS] for t in unicos[i:i + LOTE]]
            r = br.invoke_model(
                modelId=MODELO_EMBEDDING,
                body=json.dumps({"texts": lote, "input_type": tipo}),
            )
            vectores = json.loads(r["body"].read())["embeddings"]
            for texto, v in zip(unicos[i:i + LOTE], vectores):
                cache.datos[cache.clave(texto, tipo)] = v
        cache.guardar()
    return [cache.datos[cache.clave(t, tipo)] for t in textos]


def _normalizar(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class IndiceDenso:
    def __init__(self, chunks, vectores):
        self.chunks = chunks
        self.vectores = [_normalizar(v) for v in vectores]

    def buscar(self, vector_consulta: list[float], k: int) -> list[tuple[int, float]]:
        q = _normalizar(vector_consulta)
        puntajes = [(i, sum(a * b for a, b in zip(q, v)))
                    for i, v in enumerate(self.vectores)]
        puntajes.sort(key=lambda p: -p[1])
        return puntajes[:k]


# ── léxico ───────────────────────────────────────────────────────────────────────────

class IndiceBM25:
    """BM25 propio. Un DOI o un identificador no tienen vecindario semántico."""

    def __init__(self, chunks, textos: list[str]):
        self.chunks = chunks
        self.docs = [Counter(tokenizar(t)) for t in textos]
        self.largos = [sum(d.values()) for d in self.docs]
        self.promedio = sum(self.largos) / max(len(self.largos), 1)
        self.df: Counter = Counter()
        for d in self.docs:
            self.df.update(d.keys())
        self.n = len(self.docs)

    def _idf(self, termino: str) -> float:
        df = self.df.get(termino, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def buscar(self, consulta: str, k: int) -> list[tuple[int, float]]:
        terminos = tokenizar(consulta)
        puntajes = []
        for i, doc in enumerate(self.docs):
            total = 0.0
            for t in terminos:
                f = doc.get(t)
                if not f:
                    continue
                denom = f + K1 * (1 - B + B * self.largos[i] / max(self.promedio, 1))
                total += self._idf(t) * f * (K1 + 1) / denom
            if total:
                puntajes.append((i, total))
        puntajes.sort(key=lambda p: -p[1])
        return puntajes[:k]


# ── fusión ───────────────────────────────────────────────────────────────────────────

def rrf(listas: list[list[tuple[int, float]]], k: int) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: sólo importa el orden, no la escala de cada puntaje."""
    acumulado: dict[int, float] = {}
    for lista in listas:
        for posicion, (i, _) in enumerate(lista):
            acumulado[i] = acumulado.get(i, 0.0) + 1.0 / (K_RRF + posicion + 1)
    orden = sorted(acumulado.items(), key=lambda p: (-p[1], p[0]))
    return orden[:k]


class Retriever:
    """Denso + léxico fusionados. `modo` permite medir cada pierna por separado."""

    def __init__(self, documentos, cache_path):
        from .chunker import texto_para_embeber

        self.chunks = []
        self.textos_emb = []
        for doc in documentos:
            for c in doc.chunks:
                self.chunks.append(c)
                self.textos_emb.append(texto_para_embeber(c, doc.crudo))

        self.cache = CacheEmbeddings(cache_path)
        self.denso = IndiceDenso(
            self.chunks, embeber(self.textos_emb, "search_document", self.cache)
        )
        self.lexico = IndiceBM25(self.chunks, self.textos_emb)

    def buscar(self, consulta: str, k: int = 8, modo: str = "rrf",
               reranker=None, documentos: dict | None = None,
               n_candidatos: int = 20) -> list[Resultado]:
        """Con `reranker`, la búsqueda es de dos etapas: se recuperan `n_candidatos` y se
        reordenan a `k`. El techo del reranker es el recall@n_candidatos de esta etapa:
        no puede rescatar lo que nunca se trajo.
        """
        if reranker is not None:
            if documentos is None:
                raise ValueError("el reranker necesita los documentos para leer los pasajes")
            candidatos = self.buscar(consulta, k=max(n_candidatos, k), modo=modo)
            return reranker.reordenar(consulta, candidatos, documentos, k)

        if modo in ("denso", "rrf"):
            vector = embeber([consulta], "search_query", self.cache)[0]
            d = self.denso.buscar(vector, k * 4)
        if modo in ("lexico", "rrf"):
            l = self.lexico.buscar(consulta, k * 4)

        if modo == "denso":
            top = d[:k]
        elif modo == "lexico":
            top = l[:k]
        else:
            top = rrf([d, l], k)

        return [
            Resultado(
                chunk_id=self.chunks[i].chunk_id,
                doc_id=self.chunks[i].doc_id,
                char_start=self.chunks[i].char_start,
                char_end=self.chunks[i].char_end,
                puntaje=p,
            )
            for i, p in top
        ]
