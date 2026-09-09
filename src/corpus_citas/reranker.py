"""Reordenamiento del top-N recuperado. La brecha entre recall@8 (76%) y recall@20 (87%)
dice que **el material está y el problema es de orden**: 8 preguntas de 70 tienen su
evidencia entre las posiciones 9 y 20, donde el modelo nunca la ve.

Un reranker no puede recuperar lo que el retriever no trajo: su techo es el recall@N de la
primera etapa. Por eso acá el número que importa no es «cuánto sube» sino **cuánto de la
brecha cierra**, y se reporta contra ese techo.

🚨 **El rerank gestionado de Bedrock no se pudo usar.** `amazon.rerank-v1:0` y
`cohere.rerank-v3-5:0` existen en `ca-central-1` y en `us-west-2` pero devuelven
`ThrottlingException` en las dos regiones tras cuatro reintentos con espera: son el cuarto y
quinto modelo con cuota efectiva en 0 en esta cuenta (ver `PENDIENTES.md`). El reranker es
propio por bloqueo de cuota, no por decisión de diseño.

Cuatro candidatos, y la razón de cada uno:

    maxsim      El chunk se puntúa por su MEJOR oración, no por su promedio. Un chunk de
                1741 chars diluye una oración muy pertinente entre veinte que no lo son;
                el vector del chunk mide el tema, no la respuesta. Determinista, sin LLM.

    listwise    Una sola llamada al modelo con los N pasajes numerados, que devuelve el
                orden. Barato (1 invocación) pero el modelo ve todo junto y puede sesgarse
                por posición.

    pointwise   Una llamada por pasaje con una escala de 0 a 3. Caro (N invocaciones) pero
                cada juicio es independiente del orden en que llegaron.

    lexico_max  Fusión que le da a la pierna léxica el peso que su recall justifica. No es
                un reranker sino el control barato: si un LLM no le gana a esto, no paga.

⚠️ Ninguno se elige mirando las 70 preguntas. Ver `scripts/evaluar_reranker.py`: la
elección se hace en la mitad de calibración y el número que se publica sale de la otra
mitad, que el candidato ganador nunca vio.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .indice import Resultado, embeber

MODELO = "us.amazon.nova-2-lite-v1:0"
REGION = "ca-central-1"

# Una oración de menos de 40 chars no sostiene una cita (mismo umbral que `agente.py`) y
# como unidad de puntaje es ruido: "See Table 3." matchea cualquier cosa.
MINIMO_ORACION = 40
FIN_ORACION = re.compile(r"(?<=[.!?])\s+")


def oraciones_de(texto: str) -> list[str]:
    """Parte un pasaje en oraciones puntuables. No calcula offsets: acá no se cita."""
    partes = FIN_ORACION.split(texto)
    return [p.strip() for p in partes if len(p.strip()) >= MINIMO_ORACION]


# ── determinista ─────────────────────────────────────────────────────────────────────

@dataclass
class RerankerMaxSim:
    """Puntúa el chunk por su oración más parecida a la consulta.

    Reusa el mismo cache de embeddings que el índice: las oraciones se embeben una vez y
    quedan en disco, así que la corrida es reproducible sin la API.
    """

    cache: object
    nombre: str = "maxsim"

    def reordenar(self, consulta: str, resultados: list[Resultado], documentos: dict,
                  k: int) -> list[Resultado]:
        if not resultados:
            return []
        # Todas las oraciones de todos los pasajes, en una sola tanda: el lote de Cohere es
        # de 96 textos y llamar por pasaje multiplicaría por veinte las invocaciones.
        indice: list[tuple[int, str]] = []
        for i, r in enumerate(resultados):
            crudo = documentos[r.doc_id].crudo
            for o in oraciones_de(crudo[r.char_start:r.char_end]) or [crudo[r.char_start:r.char_end]]:
                indice.append((i, o))
        if not indice:
            return resultados[:k]

        textos = [o for _, o in indice]
        vectores = embeber(textos, "search_document", self.cache)
        q = embeber([consulta], "search_query", self.cache)[0]
        qn = _norm(q)

        mejor: dict[int, float] = {}
        for (i, _), v in zip(indice, vectores):
            s = _coseno(qn, _norm(v))
            if s > mejor.get(i, -2.0):
                mejor[i] = s
        orden = sorted(range(len(resultados)), key=lambda i: -mejor.get(i, -2.0))
        return [_con_puntaje(resultados[i], mejor.get(i, 0.0)) for i in orden[:k]]


@dataclass
class RerankerLexicoMax:
    """Control barato: reordena por el puntaje BM25 del pasaje, sin llamar a nada.

    Existe porque la pierna léxica sola (79%) le gana a RRF (76%) en k=8 — deuda técnica D3.
    Si un reranker con LLM no le gana a esto, no justifica ni su costo ni su latencia.
    """

    lexico: object
    nombre: str = "lexico_max"

    def reordenar(self, consulta: str, resultados: list[Resultado], documentos: dict,
                  k: int) -> list[Resultado]:
        por_chunk = {}
        for i, p in self.lexico.buscar(consulta, k=10_000):
            por_chunk[self.lexico.chunks[i].chunk_id] = p
        orden = sorted(resultados, key=lambda r: -por_chunk.get(r.chunk_id, 0.0))
        return [_con_puntaje(r, por_chunk.get(r.chunk_id, 0.0)) for r in orden[:k]]


def _norm(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def _coseno(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _con_puntaje(r: Resultado, p: float) -> Resultado:
    return Resultado(r.chunk_id, r.doc_id, r.char_start, r.char_end, float(p))


# ── con modelo ───────────────────────────────────────────────────────────────────────

PROMPT_LISTWISE = """You rank passages by how well they ANSWER a question.

Question: {consulta}

PASSAGES
{pasajes}

Reply with a JSON object and nothing else:
{{"ranking": [<passage numbers, most relevant first>]}}

Include every passage number exactly once. A passage that merely mentions the topic ranks
below one that states the answer."""

PROMPT_POINTWISE = """Rate how well the passage answers the question.

3 = states the answer directly
2 = contains the information needed to answer
1 = same topic, does not answer
0 = unrelated

Question: {consulta}

PASSAGE
{pasaje}

Reply with a JSON object and nothing else: {{"score": <0-3>}}"""

TOPE_PASAJE = 1200   # lo que se le muestra al reranker de cada pasaje


def _invocar(prompt: str, max_tokens: int = 300) -> tuple[str, int, int]:
    import boto3

    os.environ.setdefault("AWS_PROFILE", "tyv")
    br = boto3.client("bedrock-runtime", region_name=REGION)
    r = br.converse(
        modelId=MODELO,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
    )
    uso = r.get("usage", {})
    return (r["output"]["message"]["content"][0]["text"],
            uso.get("inputTokens", 0), uso.get("outputTokens", 0))


def _json_de(salida: str) -> dict | None:
    for cand in (salida, salida[salida.find("{"): salida.rfind("}") + 1]):
        try:
            d = json.loads(cand.strip().strip("`").removeprefix("json").strip())
            if isinstance(d, dict):
                return d
        except (json.JSONDecodeError, ValueError):
            continue
    return None


@dataclass
class RerankerListwise:
    """Un solo viaje al modelo con los N pasajes numerados.

    ⚠️ Si la salida no es un ranking usable **se devuelve el orden original**, no un orden
    arbitrario: un reranker caído tiene que degradar al baseline, no por debajo.
    """

    nombre: str = "listwise"
    uso: dict = None

    def __post_init__(self):
        self.uso = {"entrada": 0, "salida": 0, "llamadas": 0, "fallos": 0}

    def reordenar(self, consulta: str, resultados: list[Resultado], documentos: dict,
                  k: int) -> list[Resultado]:
        if not resultados:
            return []
        pasajes = "\n\n".join(
            f"[{i + 1}] {documentos[r.doc_id].crudo[r.char_start:r.char_end][:TOPE_PASAJE]}"
            for i, r in enumerate(resultados)
        )
        try:
            salida, e, s = _invocar(
                PROMPT_LISTWISE.format(consulta=consulta, pasajes=pasajes),
                max_tokens=200,
            )
            self.uso["entrada"] += e
            self.uso["salida"] += s
            self.uso["llamadas"] += 1
        except Exception:
            self.uso["fallos"] += 1
            return resultados[:k]

        datos = _json_de(salida) or {}
        crudo_ranking = datos.get("ranking")
        if not isinstance(crudo_ranking, list):
            self.uso["fallos"] += 1
            return resultados[:k]

        vistos, orden = set(), []
        for n in crudo_ranking:
            if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(resultados) \
                    and n not in vistos:
                vistos.add(n)
                orden.append(n - 1)
        # Los que el modelo omitió conservan su orden original detrás: omitir no es
        # descartar, y perder un pasaje que el retriever sí trajo sería un retroceso.
        orden += [i for i in range(len(resultados)) if (i + 1) not in vistos]
        if not orden:
            self.uso["fallos"] += 1
            return resultados[:k]
        return [_con_puntaje(resultados[i], 1.0 / (p + 1)) for p, i in enumerate(orden[:k])]


@dataclass
class RerankerPointwise:
    """Una llamada por pasaje. Independiente del orden de llegada, y N veces más caro.

    Los empates se rompen por la posición original: sin eso el orden de un empate depende
    del orden de iteración y la medición deja de ser reproducible.
    """

    nombre: str = "pointwise"
    uso: dict = None

    def __post_init__(self):
        self.uso = {"entrada": 0, "salida": 0, "llamadas": 0, "fallos": 0}

    def reordenar(self, consulta: str, resultados: list[Resultado], documentos: dict,
                  k: int) -> list[Resultado]:
        puntajes: list[float] = []
        for r in resultados:
            texto = documentos[r.doc_id].crudo[r.char_start:r.char_end][:TOPE_PASAJE]
            try:
                salida, e, s = _invocar(
                    PROMPT_POINTWISE.format(consulta=consulta, pasaje=texto), max_tokens=30
                )
                self.uso["entrada"] += e
                self.uso["salida"] += s
                self.uso["llamadas"] += 1
                d = _json_de(salida) or {}
                v = d.get("score")
                puntajes.append(float(v) if isinstance(v, (int, float)) else -1.0)
            except Exception:
                self.uso["fallos"] += 1
                puntajes.append(-1.0)
        orden = sorted(range(len(resultados)), key=lambda i: (-puntajes[i], i))
        return [_con_puntaje(resultados[i], puntajes[i]) for i in orden[:k]]


CANDIDATOS = ("maxsim", "listwise", "pointwise", "lexico_max")


def construir(nombre: str, retriever):
    if nombre == "maxsim":
        return RerankerMaxSim(cache=retriever.cache)
    if nombre == "lexico_max":
        return RerankerLexicoMax(lexico=retriever.lexico)
    if nombre == "listwise":
        return RerankerListwise()
    if nombre == "pointwise":
        return RerankerPointwise()
    raise ValueError(f"reranker desconocido: {nombre!r}")
