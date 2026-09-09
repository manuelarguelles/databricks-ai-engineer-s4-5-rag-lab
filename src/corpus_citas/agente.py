"""El agente: responde citando, y el código verifica la cita.

Dos modos, implementados a propósito para poder **compararlos con un número**:

    COPIA    el modelo escribe la cita literal en su respuesta. El código la verifica con
             `Documento.verificar()`. Puede fallar: copiar 200 caracteres sin equivocarse
             es una tarea de transcripción, y los modelos parafrasean.

    SEÑALA   las oraciones de los pasajes se numeran, el modelo responde con los NÚMEROS,
             y **el código extrae el texto**. La cita es exacta por construcción; lo que
             puede fallar es que señale un número inexistente o que no respalde.

🔑 El segundo es el patrón del proyecto: *el modelo señala, el código extrae y verifica.* El
primero está para medir cuánto se gana, no para usarse.

⚠️ Con el modo SEÑALA la «tasa de citas verificadas» es trivialmente 100%, así que deja de
ser informativa y se reemplaza por **tasa de referencias inválidas** (el modelo inventó un
número). Una métrica que no puede bajar no mide nada.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

MODELO = "us.amazon.nova-2-lite-v1:0"
REGION = "ca-central-1"

FIN_ORACION = re.compile(r"(?<=[.!?])\s+")
MINIMO_ORACION = 40


@dataclass
class Oracion:
    numero: int
    doc_id: str
    char_start: int
    char_end: int
    texto: str


@dataclass
class Respuesta:
    texto: str
    citas: list = field(default_factory=list)       # objetos Cita verificados
    invalidas: list[str] = field(default_factory=list)
    contexto: list[Oracion] = field(default_factory=list)
    error: str | None = None
    tokens_entrada: int = 0
    tokens_salida: int = 0


def numerar_oraciones(documentos: dict, resultados) -> list[Oracion]:
    """Parte los pasajes recuperados en oraciones citables, con su offset real.

    El offset se calcula sobre el crudo, no sobre el texto que ve el modelo: así la cita
    que el código extrae apunta al documento y no a la ventana de contexto.
    """
    oraciones: list[Oracion] = []
    numero = 1
    for r in resultados:
        doc = documentos[r.doc_id]
        trozo = doc.crudo[r.char_start:r.char_end]
        # `split` descartaba el separador y el bucle avanzaba `largo + 1`, asumiendo que
        # siempre medía un carácter. Con ".\n\n" o ".  " —o sea, en cada costura de bloque—
        # el offset se desincronizaba y el error se ACUMULABA: 990 de 3232 oraciones del
        # corpus quedaban con un span que no era el suyo. `finditer` usa el largo real.
        pos = r.char_start
        for m in FIN_ORACION.finditer(trozo):
            largo = m.start() - (pos - r.char_start)
            if largo >= MINIMO_ORACION and not doc.cruza_frontera(pos, pos + largo):
                oraciones.append(
                    Oracion(numero, r.doc_id, pos, pos + largo,
                            doc.crudo[pos:pos + largo].strip())
                )
                numero += 1
            pos = r.char_start + m.end()
        cola = r.char_end - pos
        if cola >= MINIMO_ORACION and not doc.cruza_frontera(pos, r.char_end):
            oraciones.append(
                Oracion(numero, r.doc_id, pos, r.char_end,
                        doc.crudo[pos:r.char_end].strip())
            )
            numero += 1
    return oraciones


PROMPT_SEÑALA = """Answer the question using ONLY the numbered sentences below.

Reply with a JSON object and nothing else:
{{"answer": "<your answer, in Spanish>", "sentences": [<numbers you used>]}}

Cite by NUMBER only. Never copy or rewrite the sentence text.
If the sentences do not contain the answer, reply with an empty "sentences" list and say so.

SENTENCES
{contexto}

QUESTION: {pregunta}"""

PROMPT_COPIA = """Answer the question using ONLY the passages below.

Reply with a JSON object and nothing else:
{{"answer": "<your answer, in Spanish>", "quotes": ["<exact quote from the passages>"]}}

Each quote must be copied VERBATIM from the passages, character by character.

PASSAGES
{contexto}

QUESTION: {pregunta}"""


def _invocar(prompt: str) -> tuple[str, int, int]:
    import boto3

    os.environ.setdefault("AWS_PROFILE", "tyv")
    br = boto3.client("bedrock-runtime", region_name=REGION)
    r = br.converse(
        modelId=MODELO,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 600, "temperature": 0.0},
    )
    uso = r.get("usage", {})
    return (
        r["output"]["message"]["content"][0]["text"],
        uso.get("inputTokens", 0),
        uso.get("outputTokens", 0),
    )


def _json_de(salida: str) -> dict | None:
    """El modelo suele envolver el JSON en texto o en un bloque de código."""
    for candidato in (salida, salida[salida.find("{"): salida.rfind("}") + 1]):
        try:
            datos = json.loads(candidato.strip().strip("`").removeprefix("json").strip())
            if isinstance(datos, dict):
                return datos
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def responder(pregunta: str, resultados, documentos: dict, modo: str = "señala") -> Respuesta:
    oraciones = numerar_oraciones(documentos, resultados)
    if not oraciones:
        return Respuesta(texto="", error="no hay pasajes citables para esta consulta")

    if modo == "señala":
        contexto = "\n".join(f"[{o.numero}] {o.texto}" for o in oraciones)
        prompt = PROMPT_SEÑALA.format(contexto=contexto, pregunta=pregunta)
    else:
        contexto = "\n\n".join(o.texto for o in oraciones)
        prompt = PROMPT_COPIA.format(contexto=contexto, pregunta=pregunta)

    try:
        salida, entrada, generados = _invocar(prompt)
    except Exception as e:
        return Respuesta(texto="", contexto=oraciones, error=f"{type(e).__name__}: {e}")

    datos = _json_de(salida)
    if datos is None:
        return Respuesta(texto=salida[:400], contexto=oraciones,
                         tokens_entrada=entrada, tokens_salida=generados,
                         error="la salida del modelo no es JSON")

    respuesta = Respuesta(texto=str(datos.get("answer", "")), contexto=oraciones,
                          tokens_entrada=entrada, tokens_salida=generados)
    por_numero = {o.numero: o for o in oraciones}

    if modo == "señala":
        for n in datos.get("sentences") or []:
            valido = isinstance(n, int) and not isinstance(n, bool)
            o = por_numero.get(n if valido else -1)
            if o is None:
                respuesta.invalidas.append(f"oración {n} no existe")
                continue
            # El codigo extrae: se verifica contra el documento, no contra el prompt.
            # La pista evita que la cita se resuelva a otra ocurrencia del mismo
            # texto en otra página del documento.
            cita = documentos[o.doc_id].verificar(o.texto, cerca_de=o.char_start)
            if cita is None:
                respuesta.invalidas.append(f"oración {n} no verifica contra {o.doc_id}")
            else:
                respuesta.citas.append(cita)
    else:
        for q in datos.get("quotes") or []:
            texto = str(q)
            encontrada = None
            for doc_id in {o.doc_id for o in oraciones}:
                encontrada = documentos[doc_id].verificar(texto)
                if encontrada is not None:
                    break
            if encontrada is None:
                respuesta.invalidas.append(f"cita no verifica: {texto[:70]!r}")
            else:
                respuesta.citas.append(encontrada)

    return respuesta
