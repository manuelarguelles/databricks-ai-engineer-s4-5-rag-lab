"""El agente completo: guardrail → traducción → retrieval → respuesta → guardrail → traza.

Es el punto de entrada único del sistema. Cada etapa deja su marca en la traza, incluidas
las que descartan cosas: un pasaje bloqueado, un chunk que nadie citó, una referencia
inválida. Sin eso, una respuesta mala no se puede explicar.

    from corpus_citas.e2e import Agente
    agente = Agente(documentos, cache_embeddings, cache_traduccion)
    resultado = agente.responder("¿Qué recomienda el estudio sobre supervisión humana?")
    print(resultado.respuesta)      # o el motivo del bloqueo
    print(resultado.traza.resumen())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .agente import responder
from .consulta import ULTIMO_USO, CacheTraduccion, traducir
from .guardrails import guardrail_entrada, guardrail_salida, pseudonimizar, rehidratar
from .indice import Retriever
from .jueces import evaluar_con_jueces
from .reranker import construir
from .traza import Traza


@dataclass
class Resultado:
    respuesta: str
    citas: list = field(default_factory=list)
    permitido: bool = True
    motivos: list[str] = field(default_factory=list)
    traza: Traza | None = None


class Agente:
    """`reranker` enciende la segunda etapa de recuperación. Medido sobre el gold set:
    recall@8 de 53/70 a 60/70, MRR de 0.46 a 0.69, a costa de ~1,4 s y una invocación más
    por consulta. Se elige por parámetro y no por constante porque el que paga la latencia
    es quien despliega, no quien programa.
    """

    def __init__(self, documentos, cache_embeddings, cache_traduccion, k: int = 8,
                 reranker: str | None = None, n_candidatos: int = 20,
                 jueces: tuple[str, ...] = ()):
        self.docs = {d.doc_id: d for d in documentos}
        self.k = k
        self.retriever = Retriever(documentos, cache_embeddings)
        self.cache_traduccion = CacheTraduccion(cache_traduccion)
        self.n_candidatos = n_candidatos
        self.reranker = construir(reranker, self.retriever) if reranker else None
        self.jueces = tuple(jueces)
        self._uso_previo = {"entrada": 0, "salida": 0}

    def responder(self, consulta: str, traza_path: str | Path | None = None) -> Resultado:
        t = Traza(consulta=consulta)

        # 1 · guardrail de entrada ------------------------------------------------------
        with t.etapa("guardrail_entrada"):
            v = guardrail_entrada(consulta)
        if not v.permitido:
            t.permitido = False
            t.rechazos.extend(v.motivos)
            if traza_path:
                t.anexar(traza_path)
            return Resultado("", permitido=False, motivos=v.motivos, traza=t)

        # La PII de la consulta no viaja al modelo; se rehidrata al final.
        consulta_segura, mapa_pii = pseudonimizar(consulta)
        if mapa_pii:
            t.rechazos.append(f"{len(mapa_pii)} dato(s) personales pseudonimizados")

        # 2 · traducción ----------------------------------------------------------------
        with t.etapa("traduccion"):
            t.consulta_traducida = traducir([consulta_segura], self.cache_traduccion)[0]
        t.tokens_entrada += ULTIMO_USO["entrada"]
        t.tokens_salida += ULTIMO_USO["salida"]

        # 3 · retrieval -----------------------------------------------------------------
        with t.etapa("retrieval", k=self.k, modo="rrf",
                     reranker=getattr(self.reranker, "nombre", None)):
            resultados = self.retriever.buscar(
                t.consulta_traducida, k=self.k, modo="rrf",
                reranker=self.reranker, documentos=self.docs,
                n_candidatos=self.n_candidatos,
            )
        if self.reranker is not None:
            # El reranker ACUMULA su uso entre consultas (el banco lo necesita así), y la
            # traza es por consulta: se anota el delta. Vaciar el contador acá rompería
            # el banco; ignorarlo escondería una invocación entera del costo.
            uso = getattr(self.reranker, "uso", None) or {}
            t.tokens_entrada += uso.get("entrada", 0) - self._uso_previo["entrada"]
            t.tokens_salida += uso.get("salida", 0) - self._uso_previo["salida"]
            self._uso_previo = {"entrada": uso.get("entrada", 0),
                                "salida": uso.get("salida", 0)}
        t.recuperados = [
            {"chunk_id": r.chunk_id, "doc_id": r.doc_id, "char_start": r.char_start,
             "char_end": r.char_end, "puntaje": round(r.puntaje, 5)}
            for r in resultados
        ]

        # 4 · los pasajes con inyección se descartan, no bloquean la consulta -----------
        with t.etapa("guardrail_pasajes"):
            textos = [self.docs[r.doc_id].crudo[r.char_start:r.char_end] for r in resultados]
            vp = guardrail_entrada(consulta_segura, pasajes=textos)
            if vp.pasajes_bloqueados:
                indices = {int(p.split()[-1]) for p in vp.pasajes_bloqueados}
                t.pasajes_bloqueados = [resultados[i].chunk_id for i in indices]
                resultados = [r for i, r in enumerate(resultados) if i not in indices]

        if not resultados:
            motivos = ["todos los pasajes recuperados fueron descartados por el guardrail"]
            t.permitido, t.rechazos = False, t.rechazos + motivos
            if traza_path:
                t.anexar(traza_path)
            return Resultado("", permitido=False, motivos=motivos, traza=t)

        # 5 · respuesta: el modelo señala, el código extrae ------------------------------
        with t.etapa("generacion", modo="señala"):
            r = responder(consulta_segura, resultados, self.docs, modo="señala")
        t.respuesta = r.texto
        t.tokens_entrada += r.tokens_entrada
        t.tokens_salida += r.tokens_salida
        t.citados = [
            {"doc_id": c.doc_id, "char_start": c.char_start, "char_end": c.char_end,
             "pagina": c.pagina, "texto": c.texto[:200]}
            for c in r.citas
        ]
        t.rechazos.extend(r.invalidas)

        # 6 · guardrail de salida ---------------------------------------------------------
        # Se rehidrata ANTES de revisar: chequear la fuga de PII sobre el texto del que la
        # PII fue removida a propósito es revisar el control con el control puesto.
        r.texto = rehidratar(r.texto, mapa_pii)
        with t.etapa("guardrail_salida"):
            vs = guardrail_salida(r)
        t.permitido = vs.permitido
        if not vs.permitido:
            t.rechazos.extend(vs.motivos)
            if traza_path:
                t.anexar(traza_path)
            return Resultado("", permitido=False, motivos=vs.motivos, traza=t)

        # 7 · jueces con modelo, ADITIVOS y sólo sobre lo que sobrevivió ------------------
        # Corren después del determinista a propósito: revisar con un modelo una respuesta
        # que un `in` iba a tirar igual es pagar latencia y tokens por nada.
        if self.jueces:
            with t.etapa("jueces", cuales=list(self.jueces)):
                vj = evaluar_con_jueces(r, jueces=self.jueces)
            entrada, salida = vj.tokens
            t.tokens_entrada += entrada
            t.tokens_salida += salida
            if vj.indeterminados:
                t.rechazos.append(
                    f"jueces sin veredicto: {', '.join(vj.indeterminados)}"
                )
            if not vj.permitido:
                t.permitido = False
                t.rechazos.extend(vj.motivos)
                if traza_path:
                    t.anexar(traza_path)
                return Resultado("", permitido=False, motivos=vj.motivos, traza=t)

        t.respuesta = r.texto
        if traza_path:
            t.anexar(traza_path)
        return Resultado(r.texto, citas=r.citas, motivos=vs.motivos, traza=t)
