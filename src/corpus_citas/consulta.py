"""Traducción de la consulta al idioma del corpus, antes de recuperar.

Medido: con la pregunta en español, BM25 recupera **2 de 70** y arrastra a RRF por debajo del
denso solo (39% contra 49%). Con el span literal en inglés, el mismo índice llega a **89%**.
El índice está sano; lo que falla es que la consulta y el corpus no comparten ni un token.

Traducir la consulta es plomería, no razonamiento: por eso corre en **Nova**, que es el modelo
disponible en esta cuenta, y no requiere que Claude se destrabe.

⚠️ La traducción es un componente más que puede fallar, así que:
  - se cachea por hash (determinismo y costo);
  - si la llamada falla, **se devuelve la consulta original** en vez de romper la corrida;
  - el prompt pide sólo la traducción, y se descarta cualquier cosa que venga con preámbulo.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

MODELO = "us.amazon.nova-2-lite-v1:0"
REGION = "ca-central-1"

PROMPT = (
    "Translate the following question into English. It will be used as a search query over "
    "a corpus of academic papers about AI-assisted hiring. Keep technical terms and proper "
    "nouns as they are. Reply with the translation only, no preamble, no quotes.\n\n"
    "Question: {pregunta}"
)


class CacheTraduccion:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.datos: dict[str, str] = {}
        if self.path.exists():
            self.datos = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def clave(texto: str) -> str:
        return hashlib.sha256(texto.encode()).hexdigest()[:32]

    def guardar(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.datos, ensure_ascii=False, sort_keys=True, indent=1),
            encoding="utf-8",
        )


def _limpiar(salida: str, original: str) -> str:
    """Descarta preámbulos y comillas; si no parece una traducción, devuelve el original."""
    texto = salida.strip().strip('"').strip()
    texto = re.sub(r"^(here (is|'s) the translation:?|translation:?)\s*", "", texto,
                   flags=re.I).strip()
    primera = texto.split("\n")[0].strip()
    if not primera or len(primera) > 4 * len(original) + 80:
        return original
    return primera


# Uso acumulado de la última tanda: la traducción es otra invocación a Nova por consulta
# y no se estaba sumando al costo de la traza.
ULTIMO_USO = {"entrada": 0, "salida": 0}


def traducir(preguntas: list[str], cache: CacheTraduccion) -> list[str]:
    ULTIMO_USO["entrada"] = ULTIMO_USO["salida"] = 0
    faltan = [p for p in preguntas if cache.clave(p) not in cache.datos]
    if faltan:
        import boto3

        os.environ.setdefault("AWS_PROFILE", "tyv")
        br = boto3.client("bedrock-runtime", region_name=REGION)
        for pregunta in dict.fromkeys(faltan):
            try:
                r = br.converse(
                    modelId=MODELO,
                    messages=[{"role": "user",
                               "content": [{"text": PROMPT.format(pregunta=pregunta)}]}],
                    inferenceConfig={"maxTokens": 200, "temperature": 0.0},
                )
                salida = r["output"]["message"]["content"][0]["text"]
                uso = r.get("usage", {})
                ULTIMO_USO["entrada"] += uso.get("inputTokens", 0)
                ULTIMO_USO["salida"] += uso.get("outputTokens", 0)
                cache.datos[cache.clave(pregunta)] = _limpiar(salida, pregunta)
            except Exception:
                # Una traducción caída no puede voltear la corrida: se sigue con el original
                # y el recall lo va a mostrar.
                cache.datos[cache.clave(pregunta)] = pregunta
        cache.guardar()
    return [cache.datos[cache.clave(p)] for p in preguntas]
