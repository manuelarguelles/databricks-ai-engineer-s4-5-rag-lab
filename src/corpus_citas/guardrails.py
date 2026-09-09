"""Guardrails de entrada y de salida.

**Entrada.** El corpus son PDFs bajados de internet, así que la inyección de prompt no es
hipotética: un documento puede traer texto que le dé órdenes al modelo.

⚠️ **El caso difícil está dentro del propio corpus.** El documento 35 es un paper *sobre*
ataques de inyección: su texto contiene, citadas, las mismas frases que un atacante usaría.
Un detector ingenuo lo bloquea entero. Por eso el criterio pesa la **forma imperativa dirigida
al modelo** y se desactiva ante señales de que el pasaje *describe* ataques.

🚨 **Corrección tras la segunda validación adversarial.** Esa excepción académica era un
**override global**: bastaba agregar `"See Table 3."` a un ataque para desactivar el guardrail
entero. Confirmado en vivo — la inyección pasó, el modelo recomendó un candidato y la salida se
aprobó con 9 citas. Ahora la excepción es **local**: sólo cuenta lo que está en una ventana
alrededor del imperativo.

**Salida.** Se verifica que haya evidencia, que no haya fuga de PII, y que las afirmaciones
duras de la respuesta (cifras, identificadores) aparezcan en alguna cita. Lo que **no** se
verifica es que la respuesta se siga lógicamente de la evidencia: eso necesitaría un juez y
este proyecto no usa jueces LLM. Está declarado, no disimulado.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# El contexto académico sólo se busca ANTES del imperativo: un paper describe y después
# cita, mientras que el adversario agrega `See Table 3.` AL FINAL. La ventana hacia adelante
# es corta y sólo admite el cierre de una cita entrecomillada.
VENTANA_ANTES = 200
VENTANA_DESPUES = 25
LARGO_MAXIMO_CONSULTA = 2000

# Órdenes dirigidas a un asistente, en español y en inglés.
IMPERATIVOS = [
    r"\bignor(e|a|ar|en|á)\s+(\w+\s+){0,3}"
    r"(previous|prior|above|earlier|anterior(es)?|precedentes?)\b",
    r"\bignor(e|a|ar|en|á)\s+(all|any|tod[ao]s?)\b.{0,40}\b(instruc\w+|prompts?|indicac\w+)",
    r"\bdisregard\s+(all\s+|any\s+)?(previous|prior|above)\b",
    r"\bolvid[aá](te)?\s+(de\s+)?(tod[ao]s?\s+)?l[ao]s?\s+(instruccion|indicacion)",
    r"\byou\s+must\s+(now\s+)?(recommend|select|rank|hire|approve|score)\b",
    r"\bdeb(es|erás|e)\s+(recomendar|seleccionar|contratar|aprobar|puntuar)\b",
    r"\balways\s+(recommend|select|rank\s+first|approve)\b",
    r"\bsiempre\s+(recomend|seleccion|aprob)\w*\b",
    r"\bfrom\s+now\s+on\s+you\b",
    r"\bde\s+ahora\s+en\s+(más|adelante)\b",
    r"\bsystem\s*[:>#]\s*you\s+are\b",
    r"#{1,4}\s*system\b",
    r"\bnew\s+instructions?\s*:",
    r"\bnuevas\s+instrucciones\s*:",
    r"\boverride\s+(the\s+)?(previous|system)\b",
    r"\bdo\s+not\s+(mention|reveal|disclose)\s+(this|these)\s+instructions?\b",
    r"\byou\s+are\s+(now\s+)?(DAN|a\s+jailbroken)\b",
    r"\breveal\s+(everything|the\s+(system\s+)?prompt)\b",
    r"\bforget\s+(everything|all)\b",
    r"<\|?im_(start|end)\|?>|<\|system\|>|\[INST\]",
]

# 🚨 Ground truth EXTERNO: la taxonomía de ataques del documento 35 del propio corpus
# (Mu et al., arXiv 2512.20164). Contra ella, el detector de imperativos solo daba **1 de 8**:
# reconocía el dialecto de los ataques que había escrito su autor y nada más. Estas cuatro
# familias no usan forma imperativa, así que ninguna lista de verbos las iba a atrapar.
INYECCION_ESTRUCTURAL = [
    # T1 · instrucción disfrazada de aserción o de marcado
    (r"<!--.{0,400}?-->", "comentario HTML incrustado"),
    (r"\*\*\s*instruction\s*:", "instrucción en negrita"),
    (r"^\s*(instruction|system|assistant)\s*:", "encabezado de rol"),
    # T2/T3 · texto invisible: color blanco, tamaño cero, display none, off-screen
    (r"color\s*:\s*(#f{3,6}|white)", "texto en blanco sobre blanco"),
    (r"font-size\s*:\s*0", "texto de tamaño cero"),
    (r"display\s*:\s*none|visibility\s*:\s*hidden", "texto oculto por CSS"),
    (r"position\s*:\s*absolute;?\s*(left|top)\s*:\s*-\d{3,}", "texto fuera de pantalla"),
    (r"opacity\s*:\s*0(\.0+)?\b", "texto transparente"),
    # T2/T3 · campos fabricados con nombre de metadato
    (r"\[\s*hidden[_\s-]?\w+\s*:", "campo oculto fabricado"),
    (r"\[\s*(system|instruction|note[_\s-]?to[_\s-]?(ai|llm|model))\s*:", "campo de sistema"),
    # T4 · manipulación del puesto, no del candidato
    (r"\b(updated|revised|new)\s+(requirements?|job\s+(posting|description))\s*:",
     "requisitos del puesto reescritos"),
    (r"\bno\s+experience\s+required\b.{0,60}\b(entry|junior)\b", "requisitos rebajados"),
]

# Señales de que el texto DESCRIBE ataques en vez de ejecutarlos. Se buscan sólo en la
# ventana alrededor del imperativo: globalmente eran un interruptor de apagado.
ACADEMICAS = [
    # `such as` a secas es demasiado común en prosa: se exige que hable de ataques.
    r"\b(taxonomy|defen[cs]e|mitigat\w+|benchmark|we\s+(propose|present|show|evaluate)|"
    r"this\s+(paper|study|work|section))\b",
    r"\b(for\s+example|e\.g\.|such\s+as|illustrat\w+|instance)\b[^.]{0,60}"
    r"\b(attack\w*|inject\w+|adversarial|prompts?)\b",
    r"\b(figure|table|section)\s+\d+\b",
    r"\bin:\s|\bproceedings\b|\bworkshop\b|\barxiv\b|\bdoi\b",
]

PII = [
    (r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "correo"),
    (r"\+?\d{1,3}[\s.-]?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}\b", "teléfono"),
]

# Identificadores académicos que NO son PII. Sin esto, `arXiv:2305.12345` o un DOI
# activaban el patrón de teléfono y **bloqueaban una respuesta legítima** en un agente
# cuyo dominio son justamente los papers.
NO_ES_PII = re.compile(
    r"(?:arxiv:\s*\d{4}\.\d{4,5}(v\d+)?|10\.\d{4,9}/[-._;()/:\w]+|isbn[\s:-]*[\d-]{10,17}"
    r"|issn[\s:-]*[\d-]{8,9}|https?://\S+)",
    re.I,
)

CIFRA = re.compile(r"\b\d+(?:[.,]\d+)?\s*%|\b\d{2,}(?:[.,]\d+)?\b")


def _sin_identificadores(texto: str) -> str:
    return NO_ES_PII.sub(" ", texto)


def _desofuscar(texto: str) -> str:
    """Homóglifos, soft hyphens y zero-width: un ataque no se salva por cómo se escribe."""
    limpio = unicodedata.normalize("NFKC", texto)
    for basura in ("­", "​", "‌", "‍", "﻿"):
        limpio = limpio.replace(basura, "")
    # Cirílicas y griegas que se ven como latinas.
    tabla = str.maketrans({
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i",
        "ѕ": "s", "ԁ": "d", "ո": "n", "α": "a", "ο": "o", "ρ": "p", "ε": "e", "ι": "i",
    })
    return limpio.translate(tabla).lower()


@dataclass
class Veredicto:
    permitido: bool
    motivos: list[str] = field(default_factory=list)
    pasajes_bloqueados: list[str] = field(default_factory=list)


# Una entrada de bibliografía tiene forma reconocible y su TÍTULO puede ser la frase del
# ataque: «Perez, F., Ribeiro, I.: Ignore previous prompt…». Se evalúa sobre el texto
# entero porque la forma es del registro, no de una ventana.
REFERENCIA = re.compile(
    r"^\s*[A-ZÁÉÍÓÚÑ][\w'’-]+,\s*[A-Z]\.", re.M
)
MARCADOR_PUBLICACION = re.compile(
    r"\bin:\s|\bproceedings\b|\bworkshop\b|\barxiv\b|\bdoi\b|https?://", re.I
)

# El imperativo entre comillas es la forma canónica de CITAR un ataque, no de lanzarlo.
COMILLAS = "\"'“”‘’«»"


def _es_referencia(texto: str) -> bool:
    return bool(REFERENCIA.search(texto)) and bool(MARCADOR_PUBLICACION.search(texto))


def _entrecomillado(texto: str, inicio: int, fin: int) -> bool:
    antes = texto[max(0, inicio - 3):inicio]
    despues = texto[fin:fin + 40]
    return any(c in antes for c in COMILLAS) and any(c in despues for c in COMILLAS)


def inyeccion_estructural(texto: str) -> list[str]:
    """Ataques que NO usan forma imperativa: texto oculto, campos fabricados, marcado.

    Estos no se citan en prosa académica —un paper describe el mecanismo, no incrusta el
    CSS— así que no llevan excepción por contexto.
    """
    bajo = texto.lower()
    return [nombre for patron, nombre in INYECCION_ESTRUCTURAL
            if re.search(patron, bajo, re.S | re.M)]


def parece_inyeccion(texto: str) -> bool:
    """Órdenes al modelo, salvo que ESA orden esté citada o descrita."""
    if inyeccion_estructural(texto):
        return True
    if _es_referencia(texto):
        return False
    bajo = _desofuscar(texto)
    for patron in IMPERATIVOS:
        for m in re.finditer(patron, bajo):
            if _entrecomillado(bajo, m.start(), m.end()):
                continue
            desde = max(0, m.start() - VENTANA_ANTES)
            ventana = bajo[desde:m.end() + VENTANA_DESPUES]
            if not any(re.search(p, ventana) for p in ACADEMICAS):
                return True
    return False


def guardrail_entrada(consulta: str, pasajes: list[str] | None = None) -> Veredicto:
    v = Veredicto(permitido=True)

    if len(consulta.strip()) < 8:
        return Veredicto(False, ["la consulta está vacía o es demasiado corta"])
    if len(consulta) > LARGO_MAXIMO_CONSULTA:
        return Veredicto(False, [f"la consulta supera {LARGO_MAXIMO_CONSULTA} caracteres"])
    if parece_inyeccion(consulta):
        return Veredicto(False, ["la consulta contiene instrucciones dirigidas al modelo"])

    limpia = _sin_identificadores(consulta)
    for patron, clase in PII:
        if re.search(patron, limpia):
            v.motivos.append(f"la consulta contiene {clase}: se pseudonimiza antes de enviar")

    # Los pasajes del corpus no bloquean la consulta: se descartan los sospechosos y se
    # sigue con el resto. Bloquear la respuesta entera le daría al atacante un modo de
    # negar el servicio con sólo publicar un PDF.
    for i, p in enumerate(pasajes or []):
        if parece_inyeccion(p):
            v.pasajes_bloqueados.append(f"pasaje {i}")

    # El contexto ENSAMBLADO también se revisa: una inyección partida en dos pasajes pasa
    # cada chequeo por separado y se reconstituye en el prompt.
    if pasajes and len(pasajes) > 1 and parece_inyeccion("\n".join(pasajes)):
        v.motivos.append("el contexto ensamblado contiene una instrucción partida en pasajes")
    return v


def pseudonimizar(texto: str) -> tuple[str, dict[str, str]]:
    """Reemplaza PII por marcadores y devuelve el mapa para rehidratar."""
    mapa: dict[str, str] = {}
    salida = texto
    protegidos = {m.group(0) for m in NO_ES_PII.finditer(texto)}
    for patron, clase in PII:
        for encontrado in sorted(set(re.findall(patron, _sin_identificadores(salida))),
                                 key=len, reverse=True):
            if not encontrado.strip() or any(encontrado in p for p in protegidos):
                continue
            marcador = f"[{clase.upper()}_{len(mapa)}]"
            mapa[marcador] = encontrado
            salida = salida.replace(encontrado, marcador)
    return salida, mapa


def rehidratar(texto: str, mapa: dict[str, str]) -> str:
    for marcador, valor in mapa.items():
        texto = texto.replace(marcador, valor)
    return texto


def cifras_sin_respaldo(respuesta_texto: str, citas) -> list[str]:
    """Cifras de la respuesta que no aparecen en ninguna cita.

    No prueba que la respuesta se siga de la evidencia —eso necesitaría un juez— pero sí
    caza la clase más dura: un porcentaje o un identificador que el modelo inventó.
    """
    respaldo = " ".join(getattr(c, "texto", "") for c in citas)
    return [
        c for c in set(CIFRA.findall(_sin_identificadores(respuesta_texto)))
        if c.strip() and c.strip().rstrip("%").strip() not in respaldo
    ]


def guardrail_salida(respuesta) -> Veredicto:
    """Una afirmación sin evidencia no sale."""
    v = Veredicto(permitido=True)
    if respuesta.error:
        return Veredicto(False, [f"el agente falló: {respuesta.error}"])
    if not respuesta.texto.strip():
        return Veredicto(False, ["la respuesta está vacía"])
    if not respuesta.citas:
        return Veredicto(False, ["la respuesta no trae ninguna cita verificada"])

    limpia = _sin_identificadores(respuesta.texto)
    fugas = [clase for patron, clase in PII if re.search(patron, limpia)]
    if fugas:
        return Veredicto(False, [f"la respuesta filtra {', '.join(fugas)}"])

    sueltas = cifras_sin_respaldo(respuesta.texto, respuesta.citas)
    if sueltas:
        return Veredicto(
            False,
            [f"la respuesta afirma cifras que no están en ninguna cita: {sueltas[:4]}"],
        )
    if respuesta.invalidas:
        v.motivos.append(f"{len(respuesta.invalidas)} referencia(s) inválida(s), descartadas")
    return v
