"""P5 · Jueces con modelo para el guardrail de salida, **aditivos** sobre lo determinista.

El guardrail determinista ya verifica lo que se puede verificar sin opinión: que la cita
exista en el documento, que no haya fuga de PII y que las cifras de la respuesta aparezcan
en alguna cita. Lo que no puede es leer: una respuesta puede **contradecir su propia
evidencia** y pasar los tres controles. Ese hueco estaba declarado como deuda D1.

**El orden importa y no es estético.** Lo determinista corre primero porque es gratis,
instantáneo y no se equivoca; los jueces corren después y sólo sobre lo que sobrevivió. Un
juez que corre primero paga latencia y tokens para revisar respuestas que un `in` iba a
tirar igual.

Tres jueces, uno por clase de error que el determinista no ve:

    sustento      ¿la respuesta se SIGUE de las citas? Es el hueco central: la cita existe,
                  es literal, y aun así puede no decir lo que la respuesta afirma.
    atribucion    ¿le adjudica a un autor algo que el pasaje adjudica a otro? La auditoría
                  ya encontró una cita real que se comió el apellido y dejó el año colgado:
                  así se fabrica una atribución falsa sin inventar una sola palabra.
    completitud   ¿omite una condición del pasaje que cambia la respuesta? Un «salvo que»
                  perdido convierte una recomendación condicional en una regla.

🚨 **Un juez es un componente que falla, no un oráculo.** Por eso:

- **Falla cerrada o abierta, pero declarado.** Si la llamada al modelo se cae, el juez
  devuelve `None` — ni aprueba ni rechaza — y `evaluar_con_jueces` lo reporta como
  `indeterminado`. Contar un error de red como «aprobado» es exactamente el defecto de
  «valor por defecto que se lee como buen resultado» que este repo ya cometió tres veces.
- **Cada juez tiene sus DOS tasas medidas** en `scripts/evaluar_jueces.py`, contra casos
  que no escribió quien escribió el juez (D44). Un juez sin su falso rechazo medido no se
  enciende: sube la seguridad en el papel y baja el servicio en la realidad.
- **Se pueden encender de a uno.** `evaluar_con_jueces(..., jueces=("sustento",))`.

⚠️ El juez corre sobre **Nova**, el mismo modelo que genera. Un modelo que se juzga a sí
mismo tiende a aprobarse: el número que mide eso es el de detección sobre las respuestas
mutadas, y ahí no puede esconderse. Cuando Claude se destrabe en esta cuenta, el juez
debería mudarse a otro modelo — el módulo toma el modelo por parámetro justamente para eso.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

MODELO = "us.amazon.nova-2-lite-v1:0"
REGION = "ca-central-1"

PROMPTS = {
    "sustento": """You check whether an ANSWER is supported by its EVIDENCE.

Answer "supported": false if the answer states anything the evidence does not support,
contradicts the evidence, or reverses its meaning (for example by negating it).
Answer "supported": true only if every claim in the answer follows from the evidence.

EVIDENCE
{evidencia}

ANSWER
{respuesta}

Reply with a JSON object and nothing else:
{{"supported": <true|false>, "reason": "<short reason>"}}""",

    "atribucion": """You check whether an ANSWER attributes statements to the right source.

Answer "correct": false if the answer credits a claim to an author, paper, model, tool or
organisation that the evidence credits to a different one, or names a source the evidence
never names.
Answer "correct": true if every attribution in the answer matches the evidence, or if the
answer attributes nothing to anyone.

EVIDENCE
{evidencia}

ANSWER
{respuesta}

Reply with a JSON object and nothing else:
{{"correct": <true|false>, "reason": "<short reason>"}}""",

    "completitud": """You check whether an ANSWER drops a condition that changes its meaning.

Answer "complete": false if the evidence states a condition, exception, limit or scope
("unless", "only when", "in this dataset", "for this model") that the answer presents as
unconditional.
Answer "complete": true if the answer keeps the conditions the evidence attaches, or if the
evidence attaches none.

EVIDENCE
{evidencia}

ANSWER
{respuesta}

Reply with a JSON object and nothing else:
{{"complete": <true|false>, "reason": "<short reason>"}}""",
}

# La clave del JSON cuyo `true` significa «la respuesta está bien» para cada juez.
CLAVE_OK = {"sustento": "supported", "atribucion": "correct", "completitud": "complete"}

JUECES = tuple(PROMPTS)

# 🔑 Los que pagaron su entrada, medidos con `scripts/evaluar_jueces.py --n 40`:
#
#   juez          atrapa lo roto   rechaza lo bueno   veredicto
#   sustento      39/39 (100%)     1/39 (3%)          ✅ encendido
#   atribucion     6/6  (100%)     1/39 (3%)          ✅ encendido, con n chico declarado
#   completitud    0/0  (vacío)    17/39 (44%)        ❌ apagado
#
# `completitud` queda afuera por el falso rechazo, no por falta de detección: rechaza casi
# la mitad de las respuestas correctas. Encenderlo cortaría el servicio a la mitad a cambio
# de una seguridad que nadie midió. **«Cuantos más, mejor» es falso** si cada juez no paga
# su entrada por separado.
#
# ⚠️ `atribucion` se midió sobre **6** casos rotos: es la evidencia que la mutación mecánica
# pudo generar sobre respuestas reales, y no alcanza para afirmar 100%. Se enciende porque
# su falso rechazo es bajo y el error que ataca es el más caro del proyecto (la atribución
# falsa), pero el número de detección necesita más casos antes de publicarse solo.
JUECES_APROBADOS = ("sustento", "atribucion")

TOPE_EVIDENCIA = 4000


@dataclass
class Dictamen:
    juez: str
    aprueba: bool | None          # None = no se pudo determinar
    motivo: str = ""
    tokens_entrada: int = 0
    tokens_salida: int = 0


@dataclass
class VeredictoJueces:
    permitido: bool
    dictamenes: list[Dictamen] = field(default_factory=list)
    indeterminados: list[str] = field(default_factory=list)
    motivos: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> tuple[int, int]:
        return (sum(d.tokens_entrada for d in self.dictamenes),
                sum(d.tokens_salida for d in self.dictamenes))


def _invocar(prompt: str, modelo: str, region: str) -> tuple[str, int, int]:
    import boto3

    os.environ.setdefault("AWS_PROFILE", "tyv")
    br = boto3.client("bedrock-runtime", region_name=region)
    r = br.converse(
        modelId=modelo,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 200, "temperature": 0.0},
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


def evidencia_de(citas) -> str:
    """El texto de las citas verificadas, que es lo único contra lo que se juzga.

    No se le pasa al juez el pasaje recuperado entero: la pregunta es si la respuesta se
    sigue de **lo que citó**, no de lo que tenía a mano y no citó.
    """
    trozos = [getattr(c, "texto", str(c)) for c in citas]
    return "\n\n".join(trozos)[:TOPE_EVIDENCIA]


def juzgar(nombre: str, respuesta_texto: str, evidencia: str,
           modelo: str = MODELO, region: str = REGION) -> Dictamen:
    if nombre not in PROMPTS:
        raise ValueError(f"juez desconocido: {nombre!r}")
    if not evidencia.strip():
        # Sin evidencia no hay nada contra qué juzgar. Es rechazo, no indeterminación:
        # el determinista ya debería haberla frenado y si llegó acá, algo se saltó.
        return Dictamen(nombre, False, "no hay evidencia contra la cual juzgar")
    prompt = PROMPTS[nombre].format(evidencia=evidencia, respuesta=respuesta_texto)
    try:
        salida, entrada, generados = _invocar(prompt, modelo, region)
    except Exception as e:
        return Dictamen(nombre, None, f"el juez no respondió: {type(e).__name__}")

    datos = _json_de(salida)
    if datos is None or not isinstance(datos.get(CLAVE_OK[nombre]), bool):
        # Una salida que no es el JSON pedido NO se interpreta: un `"false" in salida`
        # convierte «I could not decide» en un rechazo y la métrica se vuelve ficción.
        return Dictamen(nombre, None, "el juez no devolvió el JSON pedido",
                        entrada, generados)
    return Dictamen(nombre, bool(datos[CLAVE_OK[nombre]]),
                    str(datos.get("reason", ""))[:200], entrada, generados)


def evaluar_con_jueces(respuesta, jueces: tuple[str, ...] = JUECES,
                       modelo: str = MODELO, region: str = REGION,
                       bloquear_indeterminado: bool = False) -> VeredictoJueces:
    """Corre los jueces sobre una respuesta que YA pasó el guardrail determinista.

    `bloquear_indeterminado` decide qué hacer cuando un juez no pudo expedirse: por defecto
    no bloquea (un modelo caído no puede tirar el servicio), pero queda registrado y el
    banco lo cuenta aparte. En un despliegue donde la cita es contractual, se enciende.
    """
    v = VeredictoJueces(permitido=True)
    evidencia = evidencia_de(getattr(respuesta, "citas", []))
    texto = getattr(respuesta, "texto", str(respuesta))
    for nombre in jueces:
        d = juzgar(nombre, texto, evidencia, modelo, region)
        v.dictamenes.append(d)
        if d.aprueba is False:
            v.permitido = False
            v.motivos.append(f"juez de {nombre}: {d.motivo or 'rechazó la respuesta'}")
        elif d.aprueba is None:
            v.indeterminados.append(nombre)
            if bloquear_indeterminado:
                v.permitido = False
                v.motivos.append(f"juez de {nombre} indeterminado: {d.motivo}")
    return v
