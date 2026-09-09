"""Las siete invariantes. **Esto es lo que hace que sea un contrato y no un esquema.**

Un esquema describe la forma; un contrato es lo que falla cuando se rompe. Estas
comprobaciones corren en **cada ingesta**, no sólo en CI: un chunk que las viola no se
escribe.

La 1 y la 3 juntas dan la propiedad que el proyecto persigue: **todo el paper es citable y
toda cita apunta a texto que existe**.
"""

from __future__ import annotations

from .chunker import Chunk

CONTRATO = 1


class ViolacionContrato(Exception):
    """Un artefacto rompe el contrato. El pipeline no lo escribe."""


def verificar_contrato(doc, chunks: list[Chunk]) -> None:
    """Levanta en la primera violación, con el número de invariante en el mensaje."""
    crudo = doc.crudo
    n = len(crudo)

    # --- Invariante 2: spans posibles -------------------------------------------------
    for c in chunks:
        if not (0 <= c.char_start < c.char_end <= n):
            raise ViolacionContrato(
                f"invariante 2 · {c.chunk_id}: span fuera de rango "
                f"({c.char_start}..{c.char_end} sobre un crudo de {n})"
            )

    # --- Invariante 1: el texto ES el span --------------------------------------------
    for c in chunks:
        if crudo[c.char_start:c.char_end] != c.text:
            raise ViolacionContrato(
                f"invariante 1 · {c.chunk_id}: `text` no coincide con su span — "
                f"la cita que salga de este chunk sería ficticia"
            )

    # --- Invariante 7: el contexto del vector sólo va hacia atrás ----------------------
    for c in chunks:
        if not (0 <= c.emb_start <= c.char_start):
            raise ViolacionContrato(
                f"invariante 7 · {c.chunk_id}: emb_start={c.emb_start} > "
                f"char_start={c.char_start}: el vector invade el chunk siguiente"
            )

    # --- Invariante 3: partición del crudo, sin huecos ni solapes ---------------------
    if chunks:
        ordenados = sorted(chunks, key=lambda c: c.char_start)
        if ordenados[0].char_start != 0:
            raise ViolacionContrato(
                f"invariante 3: el primer chunk arranca en {ordenados[0].char_start}, "
                f"no en 0 — hay texto inalcanzable al principio"
            )
        for a, b in zip(ordenados, ordenados[1:]):
            if a.char_end != b.char_start:
                clase = "hueco" if a.char_end < b.char_start else "solape"
                raise ViolacionContrato(
                    f"invariante 3 · {clase} entre {a.chunk_id} y {b.chunk_id} "
                    f"({a.char_end} vs {b.char_start})"
                )
        if ordenados[-1].char_end != n:
            raise ViolacionContrato(
                f"invariante 3: el último chunk termina en {ordenados[-1].char_end} "
                f"y el crudo tiene {n} — hay texto inalcanzable al final"
            )
    elif n:
        raise ViolacionContrato("invariante 3: hay crudo pero ningún chunk lo cubre")

    # --- Invariante 4: el mapa de la vista ---------------------------------------------
    mapa = doc.vista.mapa
    if len(mapa) != len(doc.vista.texto):
        raise ViolacionContrato(
            f"invariante 4: el mapa tiene {len(mapa)} entradas y la vista "
            f"{len(doc.vista.texto)} caracteres"
        )
    for i, (a, b) in enumerate(zip(mapa, mapa[1:])):
        if a > b:
            raise ViolacionContrato(
                f"invariante 4: el mapa retrocede en {i} ({a} > {b}) — "
                f"traducir un span con él fabricaría adyacencia"
            )
    if mapa and not (0 <= mapa[0] and mapa[-1] < n):
        raise ViolacionContrato("invariante 4: el mapa apunta fuera del crudo")

    # --- Invariante 5: toda posición tiene página ---------------------------------------
    if not doc.paginas:
        raise ViolacionContrato("invariante 5: el documento no declara páginas")
    if doc.paginas[0][0] != 0:
        raise ViolacionContrato(
            f"invariante 5: la primera página arranca en {doc.paginas[0][0]}, no en 0: "
            f"los offsets iniciales quedarían sin página"
        )
    offsets = [o for o, _ in doc.paginas]
    if offsets != sorted(offsets):
        raise ViolacionContrato("invariante 5: los offsets de página no están ordenados")
    fuera = [o for o in offsets if not 0 <= o < max(n, 1)]
    if fuera:
        raise ViolacionContrato(
            f"invariante 5: offsets de página fuera del crudo: {fuera[:3]} "
            f"(crudo de {n}) — esas páginas no son alcanzables"
        )

    # --- Invariante 8: el mapa dice la verdad ------------------------------------------
    # La 4 sólo mira largo, monotonía y rango: un mapa constante las cumple las tres y
    # sigue siendo una mentira completa (una cita de 31 chars respaldada por 1 char de
    # evidencia). Esta ata cada posición de la vista al carácter del crudo que la originó.
    from .vista import construir_vista

    vista_recomputada = construir_vista(crudo)
    if vista_recomputada.texto != doc.vista.texto:
        raise ViolacionContrato(
            "invariante 8: la vista no se reproduce desde el crudo con la política vigente"
        )
    if vista_recomputada.mapa != mapa:
        raise ViolacionContrato(
            "invariante 8: el mapa no es el que produce la política sobre este crudo — "
            "traducir un span con él devolvería evidencia que no corresponde"
        )

    # --- Invariante 9: lo que se embebe entra en el techo del proveedor ----------------
    from .chunker import TECHO_EMBEBIDO

    for c in chunks:
        largo = c.char_end - c.emb_start
        if largo > TECHO_EMBEBIDO:
            raise ViolacionContrato(
                f"invariante 9 · {c.chunk_id}: el texto a embeber son {largo} chars y el "
                f"techo es {TECHO_EMBEBIDO} — el proveedor lo truncaría por el final, "
                f"que es donde vive el texto citable"
            )

    # --- Invariante 10: los bloques y las fronteras son coherentes ---------------------
    # `fronteras` pasó a ser un control de seguridad —es el único freno contra las citas
    # fabricadas— y ninguna invariante lo miraba. Un artefacto de oro sin la clave, o con
    # la lista vacía, revertía el arreglo entero con el contrato en verde.
    if not doc.bloques or doc.bloques[0] != 0:
        raise ViolacionContrato(
            "invariante 10: `bloques` vacío o sin arrancar en 0 — sin bloques no hay "
            "fronteras, y sin fronteras la verificación de citas queda desarmada"
        )
    if doc.bloques != sorted(doc.bloques):
        raise ViolacionContrato("invariante 10: `bloques` no está ordenado")
    fuera_b = [b for b in doc.bloques if not 0 <= b < n]
    if fuera_b:
        raise ViolacionContrato(f"invariante 10: bloques fuera del crudo: {fuera_b[:3]}")
    if not set(doc.fronteras) <= set(doc.bloques):
        raise ViolacionContrato(
            "invariante 10: hay fronteras que no son offsets de bloque"
        )
    if doc.fronteras != sorted(doc.fronteras):
        raise ViolacionContrato("invariante 10: `fronteras` no está ordenado")

    # La invariante 6 (sha256) se hace cumplir en el gate 1, antes de leer el PDF.
