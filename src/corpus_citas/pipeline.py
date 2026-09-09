"""Orquestación de la ingesta: catálogo → bronce → plata → oro, con sus cinco gates.

Política de errores, declarada y no improvisada:

    gate 1 · bronce      sha256, el PDF abre, páginas > 0     💥 aborta el documento
    gate 2 · extracción  cobertura y cortes raros              ⚠️ marca y sigue
    gate 3 · vista       mapa monótono y del largo correcto    💥 aborta
    gate 4 · contrato    las 7 invariantes                     💥 aborta
    gate 5 · deriva      cambió alguna de las tres versiones   ⚠️ marca el oro obsoleto

El gate 2 marca en vez de abortar porque es **heurística**: una página de figuras extrae poco
texto legítimamente. Abortar por heurística termina con alguien subiendo el umbral hasta que
deje de molestar. Los gates 1, 3 y 4 abortan porque son deterministas.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from .bronce import BronceInvalido, verificar_bronce
from .catalogo import cargar_catalogo, fuentes_incluidas
from .chunker import CONTEXTO, TECHO
from .contrato import CONTRATO, ViolacionContrato, verificar_contrato
from .documento import Documento
from .extraer import EXTRACTOR, extraer
from .reporte import Reporte
from .vista import NORMALIZACION

# Umbral medido: `pymupdf-bloques` dio 0,73-1,87 cortes raros por mil chars, y los
# extractores con orden de lectura roto, 2,34-4,52. El umbral separa los dos grupos.
UMBRAL_CORTES = 3.0
MINIMO_CHARS_PAGINA = 20

CORTE_RARO = re.compile(r"[a-z,;]\n[A-Z][a-z]")


def _cortes_por_mil(crudo: str) -> float:
    return 1000 * len(CORTE_RARO.findall(crudo)) / max(len(crudo), 1)


def _paginas_vacias(doc: Documento) -> int:
    limites = [o for o, _ in doc.paginas] + [len(doc.crudo)]
    return sum(
        1 for i in range(len(doc.paginas))
        if len(doc.crudo[limites[i]:limites[i + 1]].strip()) < MINIMO_CHARS_PAGINA
    )


def gate_extraccion(doc: Documento) -> list[str]:
    """Devuelve los motivos de sospecha. Nunca aborta: es heurística."""
    motivos = []
    vacias = _paginas_vacias(doc)
    if vacias:
        motivos.append(f"{vacias} página(s) con menos de {MINIMO_CHARS_PAGINA} chars")
    cortes = _cortes_por_mil(doc.crudo)
    if cortes > UMBRAL_CORTES:
        motivos.append(
            f"{cortes:.2f} cortes raros por mil chars (umbral {UMBRAL_CORTES}): "
            f"posible orden de lectura roto"
        )
    if len(doc.crudo) < 2000:
        motivos.append(f"sólo {len(doc.crudo)} chars extraídos")
    return motivos


def gate_vista(doc: Documento) -> None:
    """Gate 3. Aborta: un mapa roto traduce spans a lugares equivocados."""
    mapa = doc.vista.mapa
    if len(mapa) != len(doc.vista.texto):
        raise ViolacionContrato(
            f"gate 3 · {doc.doc_id}: mapa de {len(mapa)} para una vista de "
            f"{len(doc.vista.texto)}"
        )
    if any(a > b for a, b in zip(mapa, mapa[1:])):
        raise ViolacionContrato(f"gate 3 · {doc.doc_id}: el mapa retrocede")


def versiones_actuales() -> dict[str, str | int]:
    return {
        "extractor": EXTRACTOR,
        "normalizacion": NORMALIZACION,
        "contrato": CONTRATO,
        "techo_chunk": TECHO,
        "contexto_vector": CONTEXTO,
    }


def gate_deriva(dir_salida: Path) -> str | None:
    """Gate 5. Compara contra las versiones de la corrida anterior, si la hay."""
    marca = dir_salida / "versiones.json"
    if not marca.exists():
        return None
    previas = json.loads(marca.read_text(encoding="utf-8"))
    actuales = versiones_actuales()
    cambios = [k for k in actuales if previas.get(k) != actuales[k]]
    if not cambios:
        return None
    return (
        f"cambió {', '.join(cambios)} respecto de la corrida anterior: el oro previo y las "
        f"preguntas del gold set que lo referencian quedan obsoletos"
    )


def _escribir_documento(doc: Documento, dir_salida: Path, subdir: str = "oro") -> None:
    """Sin timestamps ni ids aleatorios: mismo bronce + mismas versiones ⇒ bytes idénticos."""
    destino = dir_salida / subdir / f"{doc.doc_id}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(
            {
                "doc_id": doc.doc_id,
                "sha256": doc.sha256,
                "eje": doc.eje,
                "extractor": doc.extractor,
                "normalizacion": doc.normalizacion,
                "contrato": doc.contrato,
                "paginas": doc.paginas,
                "bloques": doc.bloques,
                "fronteras": doc.fronteras,
                "crudo": doc.crudo,
                "vista": doc.vista.texto,
                "mapa": doc.vista.mapa,
                "chunks": [asdict(c) for c in doc.chunks],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=1,
        ),
        encoding="utf-8",
    )


class OroInvalido(Exception):
    """Un artefacto de oro no se puede usar como corpus."""


CLAVES_ORO = {
    "doc_id", "sha256", "eje", "extractor", "normalizacion", "contrato",
    "paginas", "bloques", "fronteras", "crudo", "vista", "mapa", "chunks",
}


def cargar_documentos(dir_salida, catalogo: dict[str, str] | None = None):
    """Lee el oro validando lo no derivable y **recomputando lo derivable**.

    Los gates protegían la escritura y no la lectura, y todo consumidor entra por acá sin
    re-ingerir. La vista, el mapa y los chunks son **derivados** de `(crudo, bloques)`:
    persistirlos crea una segunda fuente de verdad que habría que policiar, y policiarla es
    re-correr el contrato. Recomputarlos hace que no puedan discrepar por construcción.

    Lo que sí se valida es lo que no se puede derivar: que estén todas las claves, que las
    versiones sean las vigentes, y que el documento pertenezca al catálogo con su hash.

    ⚠️ Hueco irreducible y declarado: un `crudo` o un `fronteras` adulterados no se detectan
    leyendo el JSON — cualquier checksum lo escribiría el mismo que escribió el dato. La
    única clausura es re-extraer del PDF, y eso ya es la ingesta. **`oro/` merece el mismo
    nivel de confianza que el código fuente.**
    """
    from .chunker import chunkear
    from .documento import NOMBRE_VALIDO, Documento
    from .vista import construir_vista

    actuales = versiones_actuales()
    docs = []
    # `**/*.json` y no `*.json`: un doc_id con "/" escribía en un subdirectorio y el glob
    # plano lo saltaba en silencio.
    for path in sorted((Path(dir_salida) / "oro").glob("**/*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))

        faltan = CLAVES_ORO - d.keys()
        if faltan:
            raise OroInvalido(f"{path.name}: faltan las claves {sorted(faltan)}")

        difs = {
            k: (d[k], actuales[k])
            for k in ("extractor", "normalizacion", "contrato")
            if d[k] != actuales[k]
        }
        if difs:
            raise OroInvalido(
                f"{path.name}: producido con {difs} — el oro rancio y el nuevo no se mezclan"
            )

        if not NOMBRE_VALIDO.match(d["doc_id"]) or path.stem != d["doc_id"]:
            raise OroInvalido(f"{path.name}: el nombre del archivo no es su doc_id")

        if catalogo is not None:
            if d["doc_id"] not in catalogo:
                raise OroInvalido(f"{path.name}: oro huérfano, no está en el catálogo")
            if catalogo[d["doc_id"]] != d["sha256"]:
                raise OroInvalido(f"{path.name}: el PDF cambió desde que se ingirió")

        docs.append(
            Documento(
                doc_id=d["doc_id"],
                sha256=d["sha256"],
                crudo=d["crudo"],
                paginas=[tuple(p) for p in d["paginas"]],
                vista=construir_vista(d["crudo"]),            # derivado: se recomputa
                bloques=d["bloques"],
                fronteras=d["fronteras"],
                chunks=chunkear(d["doc_id"], d["crudo"], d["bloques"], d["eje"]),
                eje=d["eje"],
                extractor=d["extractor"],
                normalizacion=d["normalizacion"],
                contrato=d["contrato"],
            )
        )
    return docs


def ingerir(corpus_yml, dir_pdf, dir_salida) -> Reporte:
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    reporte = Reporte(versiones=versiones_actuales())

    deriva = gate_deriva(dir_salida)
    if deriva:
        reporte.marcados.append(("__oro__", deriva))

    # El catálogo se lee ANTES de tocar nada: un `corpus.yml` inválido no puede dejarnos
    # sin oro viejo y sin oro nuevo.
    fuentes = cargar_catalogo(corpus_yml)

    # El oro se regenera entero, pero en un directorio aparte que reemplaza al bueno sólo
    # si la corrida produjo algo. Borrar primero dejaba un caso mudo y caro: un catálogo
    # con todo excluido borraba los 13 documentos y el reporte salía enteramente verde.
    oro = dir_salida / "oro"
    nuevo = dir_salida / "oro.nuevo"
    if nuevo.exists():
        shutil.rmtree(nuevo)
    nuevo.mkdir(parents=True)
    for f in fuentes:
        if f.estado != "incluida":
            reporte.omitidos.append((f.id, f.motivo or f"estado: {f.estado}"))

    for fuente in fuentes_incluidas(fuentes):
        try:
            path = verificar_bronce(fuente, dir_pdf)          # gate 1
        except BronceInvalido as e:
            reporte.abortados.append((fuente.id, str(e)))
            continue

        try:
            extraccion = extraer(path)
            if not extraccion.paginas:
                raise ValueError("el PDF no declara páginas")
            doc = Documento.desde_extraccion(
                fuente.id, fuente.sha256, extraccion, fuente.eje
            )
            # El gate 2 corre ANTES de los que abortan: si un documento se cae, sus
            # señales heurísticas son justo las que más se necesitan en el reporte.
            for motivo in gate_extraccion(doc):                # gate 2 (marca)
                reporte.marcados.append((fuente.id, motivo))
            gate_vista(doc)                                    # gate 3
            verificar_contrato(doc, doc.chunks)                # gate 4
        except Exception as e:
            # `except (ViolacionContrato, ValueError)` dejaba escapar FileDataError y
            # EmptyFileError de PyMuPDF (heredan de RuntimeError): un PDF ilegible mataba
            # la corrida entera y dejaba un oro parcial SIN reporte ni versiones.json,
            # indistinguible de uno completo.
            reporte.abortados.append((fuente.id, f"{type(e).__name__}: {e}"))
            continue

        _escribir_documento(doc, dir_salida, subdir="oro.nuevo")
        reporte.procesados.append(fuente.id)
        reporte.chunks += len(doc.chunks)
        reporte.caracteres += len(doc.crudo)

    # Reemplazo atómico. Sólo si hubo oro nuevo: una corrida que no procesó nada deja el
    # oro anterior intacto y lo dice en el reporte.
    if reporte.procesados:
        if oro.exists():
            reporte.regenerados = len(list(oro.glob("*.json")))
            shutil.rmtree(oro)
        nuevo.rename(oro)
        (dir_salida / "versiones.json").write_text(
            json.dumps(versiones_actuales(), sort_keys=True, indent=1), encoding="utf-8"
        )
    else:
        shutil.rmtree(nuevo)
        if oro.exists() and list(oro.glob("*.json")):
            reporte.marcados.append((
                "__oro__",
                "la corrida no produjo ningún documento: se conserva el oro anterior, que "
                "puede no corresponder al catálogo actual",
            ))
    return reporte
