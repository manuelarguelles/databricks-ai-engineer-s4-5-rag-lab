"""Carril B — el mismo agente, con el almacén vectorial gestionado por AWS.

Los carriles A y B son **el experimento**, no alternativas: el mismo corpus, el mismo gold
set y el mismo modelo, construido dos veces. La pregunta no es cuál gana sino **qué se
paga y qué se pierde** al delegar.

🚨 **Qué quedó afuera y por qué — importa para leer el resultado.**

El carril B «completo» era Knowledge Bases gestionado: Bedrock ingiere de S3, chunkea,
embebe, indexa y responde con `RetrieveAndGenerate`. **No se pudo construir, y el motivo
cambió respecto de lo que decía el repo.** Estaba anotado como una decisión de gasto
(OpenSearch Serverless con piso de cientos de USD/mes). Al ir a hacerlo aparecieron dos
hechos que la corrigen:

1. Ese piso **ya no es universal**: las colecciones NextGen de OpenSearch Serverless (GA
   28-may-2026) escalan a cero y se cobran por OCU-hora usada. La premisa de la decisión
   anterior venció.
2. El bloqueo real es **IAM, no dinero** — y después resultó ser **más chico todavía**:
   `terraform-tyv` **sí tiene `iam:CreateRole`**, pero su política lo limita a
   `arn:aws:iam::*:role/tyv-*`, y el rol que la consola de Bedrock autogenera se llama
   `AmazonBedrockExecutionRoleForKnowledgeBase_*`. **No falta un permiso: falta un prefijo.**
   Verificado creando los dos roles: `tyv-bedrock-kb-prueba` ✅, el otro nombre ❌.

Así que el carril B se construyó donde sí se podía llegar sin pedirle nada a nadie:
**S3 Vectors como almacén gestionado**, con los mismos vectores del carril A. Eso aísla una
variable —el motor de búsqueda— y deja las otras fijas, que es la única forma de que la
comparación signifique algo.

**Lo que esta variante NO mide, y hay que decirlo:** no mide el chunking de Bedrock ni sus
citas. La pérdida de trazabilidad que el proyecto quería cuantificar es mayor en el B
completo, porque ahí el chunk lo decide el servicio y el offset original no vuelve.

**Lo que sí mide, y es la mitad más interesante:** S3 Vectors **no hace búsqueda híbrida**
—es semántico y nada más (documentado por AWS). En el carril A la pierna léxica sola
recupera 79% contra 60% de la densa. Si eso se sostiene, el carril B hereda el techo de la
pierna densa, y la comparación deja de ser sobre latencia y costo para ser sobre **qué
recupera cada uno**.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from .chunker import texto_para_embeber
from .indice import Resultado, embeber

REGION = "ca-central-1"
BUCKET = "nix-vectores-citas-836064768526"
INDICE = "chunks"
DIMENSION = 1024        # cohere.embed-multilingual-v3
LOTE_PUT = 200          # vectores por llamada


def cliente(region: str = REGION):
    import boto3

    os.environ.setdefault("AWS_PROFILE", "tyv")
    return boto3.client("s3vectors", region_name=region)


def asegurar_indice(bucket: str = BUCKET, indice: str = INDICE, region: str = REGION) -> None:
    sv = cliente(region)
    try:
        sv.create_vector_bucket(vectorBucketName=bucket)
    except Exception:
        pass  # ya existe: crear es idempotente para lo que necesitamos
    try:
        sv.create_index(vectorBucketName=bucket, indexName=indice, dataType="float32",
                        dimension=DIMENSION, distanceMetric="cosine")
    except Exception:
        pass


@dataclass
class RetrieverS3Vectors:
    """Misma interfaz que el `Retriever` del carril A, distinto motor.

    ⚠️ Los vectores que se suben son **exactamente los del carril A**: mismo modelo, mismo
    texto embebido, mismo cache. Si se re-embebiera acá, la diferencia medida sería una
    mezcla de «otro motor» y «otros vectores», y no se podría atribuir a nada.
    """

    documentos: list
    cache: object
    bucket: str = BUCKET
    indice: str = INDICE
    region: str = REGION
    chunks: list = field(default_factory=list)

    def __post_init__(self):
        self.chunks = [c for d in self.documentos for c in d.chunks]
        self._por_clave = {self._clave(c): c for c in self.chunks}

    @staticmethod
    def _clave(chunk) -> str:
        # La clave del vector es el chunk_id, así el offset vuelve del almacén sin
        # depender de que la metadata sobreviva: es la trazabilidad mínima.
        return chunk.chunk_id.replace(":", "_")

    def indexar(self, verbose: bool = True) -> int:
        asegurar_indice(self.bucket, self.indice, self.region)
        sv = cliente(self.region)
        por_doc = {d.doc_id: d for d in self.documentos}
        textos = [texto_para_embeber(c, por_doc[c.doc_id].crudo) for c in self.chunks]
        vectores = embeber(textos, "search_document", self.cache)

        subidos = 0
        for i in range(0, len(self.chunks), LOTE_PUT):
            lote = [
                {"key": self._clave(c),
                 "data": {"float32": [float(x) for x in v]},
                 "metadata": {"doc_id": c.doc_id,
                              "char_start": c.char_start,
                              "char_end": c.char_end}}
                for c, v in zip(self.chunks[i:i + LOTE_PUT], vectores[i:i + LOTE_PUT])
            ]
            sv.put_vectors(vectorBucketName=self.bucket, indexName=self.indice, vectors=lote)
            subidos += len(lote)
            if verbose:
                print(f"  subidos {subidos}/{len(self.chunks)}", end="\r")
        if verbose:
            print()
        return subidos

    def buscar(self, consulta: str, k: int = 8, modo: str = "denso", **_) -> list[Resultado]:
        """`modo` se acepta y se ignora: S3 Vectors **no tiene** pierna léxica ni híbrida.

        Aceptar el parámetro y no hacer nada sería mentir en la interfaz, así que si se
        pide un modo que este motor no puede dar, se levanta el error en vez de devolver
        silenciosamente otra cosa — el banco compararía peras con manzanas sin saberlo.
        """
        if modo not in ("denso", "semantico"):
            raise ValueError(
                f"S3 Vectors sólo hace búsqueda semántica; se pidió modo={modo!r}. "
                "La ausencia de híbrida es un resultado del experimento, no un bug."
            )
        vector = embeber([consulta], "search_query", self.cache)[0]
        sv = cliente(self.region)
        r = sv.query_vectors(
            vectorBucketName=self.bucket, indexName=self.indice,
            queryVector={"float32": [float(x) for x in vector]},
            topK=k, returnMetadata=True, returnDistance=True,
        )
        salida = []
        for v in r.get("vectors", []):
            chunk = self._por_clave.get(v["key"])
            meta = v.get("metadata") or {}
            doc_id = chunk.doc_id if chunk else meta.get("doc_id")
            if doc_id is None:
                continue
            salida.append(Resultado(
                chunk_id=chunk.chunk_id if chunk else v["key"],
                doc_id=doc_id,
                char_start=int(chunk.char_start if chunk else meta["char_start"]),
                char_end=int(chunk.char_end if chunk else meta["char_end"]),
                # `distance` es coseno: 0 es idéntico. Se invierte para que, como en el
                # carril A, más puntaje sea más parecido y los bancos no se confundan.
                puntaje=1.0 - float(v.get("distance", 0.0)),
            ))
        return salida


def medir_latencia(retriever, consultas: list[str], k: int = 8) -> dict:
    """p50 y p95 de la consulta, que es donde un almacén sobre S3 se paga."""
    tiempos = []
    for c in consultas:
        t0 = time.perf_counter()
        retriever.buscar(c, k=k)
        tiempos.append(time.perf_counter() - t0)
    tiempos.sort()
    return {"n": len(tiempos),
            "p50": tiempos[len(tiempos) // 2] if tiempos else 0.0,
            "p95": tiempos[int(0.95 * (len(tiempos) - 1))] if tiempos else 0.0,
            "media": sum(tiempos) / max(len(tiempos), 1)}
