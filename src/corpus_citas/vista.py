"""Vista normalizada del crudo, con mapa de índices de vuelta al crudo.

**Política (b) media**, elegida midiendo las dos tasas sobre 10 documentos con ground truth
externo (abstracts oficiales de arXiv/Crossref):

    política            falso rechazo   match espurio
    (a) mínima              18,3%           0/30
    (b) media               15,5%           0/30    <-- domina a (a)
    (c) agresiva             9,9%          17/30    <-- descartada

La agresiva bajaba más el falso rechazo pero aceptaba **9 de 10 adyacencias fabricadas**.

⚠️ **Corrección tras la validación adversarial (16-ago-2026).** El docstring anterior afirmaba
que la (b) «no fabrica adyacencia». Era falso por dos vías distintas, ambas demostradas sobre
PDFs reales del corpus:

1. **Colapsar blancos suelda bloques.** `extraer.py` une bloques y páginas con `"\\n"`, y
   colapsarlo produce un espacio: dos bloques distantes quedan contiguos en la vista. **No se
   arregla acá** — se arregla rechazando en `Documento.verificar()` todo span que cruce una
   frontera de bloque, para lo cual el documento ahora persiste sus offsets.
2. **La des-hifenación saltaba blancos sin tope**, así que un guión de fin de renglón podía
   unir texto separado por una página entera (`state-` + `Machine` -> `stateMachine`). Se
   corrige acá: **un solo salto de renglón**.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

NORMALIZACION = "b-media@2"  # @2: des-hifenación acotada + tabla de caracteres ampliada

LIGADURAS = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "æ": "ae", "œ": "oe",
}

TIPOGRAFICOS = {
    # comillas
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    # toda la familia de guiones baja a "-". Sin U+2011 (non-breaking hyphen) el título
    # del propio doc 02 no verificaba al tipearlo con guión ASCII.
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",
    "­": "",       # soft hyphen
    # espacios que `.isspace()` no siempre trata igual
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", "　": " ",
    "​": "", "‌": "", "‍": "", "⁠": "",
    # el BOM no es espacio para Python: sobrevivía y partía la vista en dos
    "﻿": "",
    # controles que PyMuPDF deja sueltos en el crudo
    "": "", "": "", "": "", "": "",
}

# Cuántos blancos puede saltar la des-hifenación: exactamente un fin de renglón.
# Sin este tope, un guión unía a través de una página en blanco (401 chars saltados).
MAX_BLANCOS_HIFEN = 3


@dataclass(frozen=True)
class Vista:
    texto: str
    mapa: list[int]  # mapa[i] = posicion en el crudo del caracter i de la vista


def _es_letra(c: str) -> bool:
    return c.isalpha()


def _salto_de_renglon_simple(crudo: str, i: int) -> int | None:
    """Devuelve la posición tras un ÚNICO salto de renglón, o None si hay más.

    `i` apunta al primer carácter blanco después del guión.
    """
    j = i
    saltos = 0
    while j < len(crudo) and crudo[j] in "\r\n \t":
        if crudo[j] == "\n":
            saltos += 1
            if saltos > 1:
                return None          # línea en blanco o frontera de bloque: no unir
        if j - i >= MAX_BLANCOS_HIFEN:
            return None              # demasiada distancia para ser un corte de renglón
        j += 1
    return j if saltos == 1 else None


def construir_vista(crudo: str) -> Vista:
    """Una sola pasada carácter por carácter: cada char emitido registra su origen."""
    salida: list[str] = []
    mapa: list[int] = []
    i = 0
    n = len(crudo)
    ultimo_fue_blanco = False

    while i < n:
        c = crudo[i]

        # Guion de corte al final de renglon: "scree-\nning" -> "screening".
        # Exige letra antes y despues, y UN SOLO salto de renglon en el medio.
        if c == "-" and i + 1 < n and crudo[i + 1] in "\r\n":
            j = _salto_de_renglon_simple(crudo, i + 1)
            if j is not None and j < n and _es_letra(crudo[j]) and salida and \
                    _es_letra(salida[-1]):
                i = j
                continue

        if c in TIPOGRAFICOS:
            reemplazo = TIPOGRAFICOS[c]
            if reemplazo == "":
                i += 1
                continue
            c = reemplazo

        if c in LIGADURAS:
            # Una ligadura ocupa 1 char en el crudo y varios en la vista: todos apuntan
            # al mismo origen, y el mapa sigue siendo monotono no decreciente.
            for ch in LIGADURAS[c].lower():
                salida.append(ch)
                mapa.append(i)
            ultimo_fue_blanco = False
            i += 1
            continue

        if c.isspace():
            if not ultimo_fue_blanco and salida:
                salida.append(" ")
                mapa.append(i)
                ultimo_fue_blanco = True
            i += 1
            continue

        for ch in c.lower():
            salida.append(ch)
            mapa.append(i)
        ultimo_fue_blanco = False
        i += 1

    while salida and salida[-1] == " ":
        salida.pop()
        mapa.pop()

    return Vista(texto="".join(salida), mapa=mapa)


def normalizar_consulta(texto: str) -> str:
    """Lleva una cita del usuario al mismo espacio que la vista."""
    return construir_vista(texto).texto


def sin_tildes(texto: str) -> str:
    """Sólo para diagnóstico. NO se usa en la política (b): quitar tildes colapsa
    `más` con `mas`, y el corpus tiene fuentes en español."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
