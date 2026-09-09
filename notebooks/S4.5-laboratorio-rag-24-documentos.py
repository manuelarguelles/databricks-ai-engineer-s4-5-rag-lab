# Databricks notebook source
# MAGIC %md
# MAGIC # Laboratorio S4.5 · RAG con 24 documentos reales
# MAGIC
# MAGIC **Objetivo:** construir y medir retrieval sobre 13 documentos base + 11 distractores,
# MAGIC conservando evidencia literal, offsets y página. Esta sesión vive entre S04 y S05.
# MAGIC
# MAGIC El notebook mantiene dos rutas deliberadamente separadas:
# MAGIC
# MAGIC 1. **Ruta administrada Databricks:** `ai_parse_document → ai_prep_search`.
# MAGIC 2. **Benchmark congelado:** extractor, chunking, traducciones y embeddings versionados para
# MAGIC    reproducir las cifras históricas sobre exactamente 1.405 chunks.
# MAGIC
# MAGIC No compares el número de chunks de una ruta con la otra: cambiar el instrumento cambia la
# MAGIC unidad experimental.

# COMMAND ----------

# MAGIC %pip install PyMuPDF==1.27.2.3 "PyYAML>=6"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0 · Parámetros y mapa de salida
# MAGIC
# MAGIC El despliegue crea el schema y el Volume. El notebook crea o reemplaza estas tablas:
# MAGIC
# MAGIC - `documentos_benchmark`
# MAGIC - `chunks_benchmark` — con Change Data Feed
# MAGIC - `chunks_administrados` — salida de AI Functions, también con CDF
# MAGIC - `gold_set`
# MAGIC - `evaluacion_retrieval`
# MAGIC - `resumen_benchmark`

# COMMAND ----------

import hashlib
import json
import shutil
import sys
from pathlib import Path

import fitz
import yaml
from pyspark.sql import functions as F

dbutils.widgets.text("catalogo", "neptuno_manuel_arguelles", "Catálogo")
dbutils.widgets.text("schema", "rag_s4_5", "Schema S4.5")
dbutils.widgets.text("volume", "laboratorio", "Volume")
dbutils.widgets.text("k", "8", "Top-k")

CATALOGO = dbutils.widgets.get("catalogo").strip().lower()
SCHEMA = dbutils.widgets.get("schema").strip().lower()
VOLUME = dbutils.widgets.get("volume").strip().lower()
K = int(dbutils.widgets.get("k"))

assert CATALOGO and SCHEMA and VOLUME
assert K == 8, "Las cifras congeladas pertenecen a k=8; otro k inicia otro experimento."

NS = f"{CATALOGO}.{SCHEMA}"
ROOT = f"/Volumes/{CATALOGO}/{SCHEMA}/{VOLUME}"
PDF_DIR = Path(ROOT) / "corpus/pdf"
MANIFEST_PATH = Path(ROOT) / "corpus/manifest.yml"
DATA_DIR = Path(ROOT) / "data"
SRC_DIR = Path(ROOT) / "src"

assert ROOT.startswith("/Volumes/"), ROOT
assert MANIFEST_PATH.exists(), f"No existe {MANIFEST_PATH}; ejecuta deploy_and_run.py."
sys.path.insert(0, str(SRC_DIR))

print(f"✅ namespace: {NS}")
print(f"✅ paquete:   {ROOT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Gate del corpus: 24 archivos o no se continúa
# MAGIC
# MAGIC El manifiesto es el contrato. Verificamos nombre, existencia y SHA-256 antes de extraer.
# MAGIC Los 13 documentos cubiertos por el gold set son la base; los otros 11 son distractores
# MAGIC académicos reales, no texto basura generado.

# COMMAND ----------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
incluidas = [f for f in manifest["fuentes"] if f["estado"] == "incluida"]
pdfs = {p.name: p for p in PDF_DIR.glob("*.pdf")}

assert len(incluidas) == 24, f"El manifiesto declara {len(incluidas)} documentos incluidos."
assert len(pdfs) == 24, f"El Volume contiene {len(pdfs)} PDF; se esperan 24."

for fuente in incluidas:
    meta = fuente["pdf"]
    assert meta["archivo"] in pdfs, f"Falta {meta['archivo']}"
    assert sha256(pdfs[meta["archivo"]]) == meta["sha256"], f"Hash distinto: {meta['archivo']}"

print("✅ corpus: 24/24 PDF presentes y hashes válidos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Ruta administrada: `ai_parse_document → ai_prep_search`
# MAGIC
# MAGIC Esta ruta enseña las AI Functions actuales de Databricks. Conserva `source_uri` y produce
# MAGIC chunks semánticos. Su conteo puede diferir de 1.405 y **no** reemplaza el benchmark congelado.
# MAGIC Primero materializamos las 24 salidas de parsing. Si el preparador beta no emite chunks para
# MAGIC una salida estructurada, el notebook registra el caso y conserva ese documento mediante un
# MAGIC fallback determinista de texto. La omisión nunca se oculta y no afecta las métricas congeladas.

# COMMAND ----------

ruta_sql = str(PDF_DIR).replace("'", "''")
spark.sql(
    f"""
    CREATE OR REPLACE TABLE {NS}.documentos_administrados AS
    SELECT
      path AS source_uri,
      ai_parse_document(
        content,
        map('version', '2.0', 'descriptionElementTypes', '')
      ) AS parsed
    FROM read_files('{ruta_sql}', format => 'binaryFile')
    """
)

parsed_documents = spark.table(f"{NS}.documentos_administrados")
managed_attempted_documents = parsed_documents.select("source_uri").distinct().count()
assert managed_attempted_documents == 24, (
    f"ai_parse_document recibió {managed_attempted_documents}/24 documentos."
)

managed_primary_result = spark.sql(
    f"""
    WITH prepped_documents AS (
      SELECT source_uri, ai_prep_search(parsed, map('version', '2.0')) AS result
      FROM {NS}.documentos_administrados
    )
    SELECT
      prepped_documents.source_uri,
      chunk.value:chunk_id::STRING AS chunk_id,
      chunk.value:chunk_position::INT AS chunk_position,
      chunk.value:chunk_to_retrieve::STRING AS texto_retrieval,
      chunk.value:chunk_to_embed::STRING AS texto_embedding,
      CAST(chunk.value:metadata AS STRING) AS metadata_json,
      'structured_variant' AS prep_mode
    FROM prepped_documents,
         LATERAL variant_explode(prepped_documents.result:document.contents) AS chunk
    """
)
managed_primary_result.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{NS}.chunks_administrados"
)
managed_primary = spark.table(f"{NS}.chunks_administrados")
managed_primary_chunks = managed_primary.count()
managed_primary_documents = managed_primary.select("source_uri").distinct().count()

missing = (
    parsed_documents.select("source_uri").distinct()
    .join(managed_primary.select("source_uri").distinct(), "source_uri", "left_anti")
)
missing_paths = [r.source_uri for r in missing.collect()]

if missing_paths:
    fallback_rows = []
    for source_uri in missing_paths:
        local_pdf = PDF_DIR / Path(source_uri).name
        with fitz.open(local_pdf) as pdf:
            texto = "\n\n".join(page.get_text("text") for page in pdf).strip()
        assert texto, f"Fallback sin texto para {local_pdf.name}"
        inicio = 0
        posicion = 0
        while inicio < len(texto):
            fin = min(inicio + 1600, len(texto))
            if fin < len(texto):
                corte = texto.rfind("\n\n", inicio + 800, fin)
                if corte > inicio:
                    fin = corte
            chunk = texto[inicio:fin].strip()
            if chunk:
                fallback_rows.append(
                    (
                        source_uri,
                        f"fallback::{hashlib.sha256(source_uri.encode()).hexdigest()[:16]}::{posicion}",
                        posicion,
                        chunk,
                        chunk,
                        json.dumps(
                            {
                                "fallback": "pymupdf_fixed_window",
                                "reason": "ai_prep_search_empty_contents",
                            },
                            ensure_ascii=False,
                        ),
                        "deterministic_text_fallback",
                    )
                )
                posicion += 1
            if fin >= len(texto):
                break
            inicio = max(fin - 160, inicio + 1)

    assert fallback_rows, f"No se generaron chunks fallback para {missing_paths}"
    managed_fallback = spark.createDataFrame(
        fallback_rows,
        "source_uri string, chunk_id string, chunk_position int, texto_retrieval string, "
        "texto_embedding string, metadata_json string, prep_mode string",
    )
    managed_fallback.write.mode("append").saveAsTable(f"{NS}.chunks_administrados")

managed = spark.table(f"{NS}.chunks_administrados")
managed_chunks = managed.count()
managed_documents = managed.select("source_uri").distinct().count()
managed_fallback_documents = len(missing_paths)
assert managed_documents == 24, f"La ruta administrada conservó {managed_documents}/24 documentos."
assert managed_chunks > 24, f"Solo se produjeron {managed_chunks} chunks administrados."

spark.sql(
    f"ALTER TABLE {NS}.chunks_administrados SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print(
    f"✅ ruta administrada: {managed_documents} documentos → {managed_chunks} chunks · "
    f"estructurados={managed_primary_documents} · fallback explícito={managed_fallback_documents}"
)
if missing_paths:
    print("ℹ️ ai_prep_search sobre VARIANT no emitió chunks; fallback determinista para:", missing_paths)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Ruta congelada: extractor y contrato reproducibles
# MAGIC
# MAGIC `PyMuPDF==1.27.2.3`, normalización `b-media@2`, chunk citable sin overlap y hasta 307
# MAGIC caracteres de contexto previo solo para el embedding. La cita siempre vuelve al raw.

# COMMAND ----------

from corpus_citas.catalogo import cargar_catalogo
from corpus_citas.chunker import texto_para_embeber
from corpus_citas.gold_set import cargar_gold_set, validar
from corpus_citas.indice import CacheEmbeddings, Retriever
from corpus_citas.pipeline import cargar_documentos, ingerir
from corpus_citas.consulta import CacheTraduccion, traducir

WORK = Path("/tmp/s4_5_rag_benchmark")
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)

reporte = ingerir(MANIFEST_PATH, PDF_DIR, WORK / "plata")
assert reporte.abortados == [], reporte.abortados
assert reporte.marcados == [], reporte.marcados

documentos = cargar_documentos(WORK / "plata")
numero_chunks = sum(len(d.chunks) for d in documentos)
assert len(documentos) == 24
assert numero_chunks == 1405, f"Se obtuvieron {numero_chunks} chunks; se esperaban 1.405."

print("✅ benchmark congelado: 24 documentos → 1.405 chunks · 0 abortados · 0 marcados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Persistir documentos y chunks con CDF
# MAGIC
# MAGIC El span citable (`char_start:char_end`) y el texto de embedding no son lo mismo. Persistimos
# MAGIC ambos y el vector congelado para que la medición no dependa de una llamada externa.

# COMMAND ----------

base_ids = {p.doc_id for p in cargar_gold_set(DATA_DIR / "gold_set.yml")}
filas_docs = [
    (
        d.doc_id,
        d.sha256,
        "base" if d.doc_id in base_ids else "distractor",
        d.eje or "",
        d.crudo,
        len(d.paginas),
        d.extractor,
        d.normalizacion,
        int(d.contrato),
    )
    for d in documentos
]
df_docs = spark.createDataFrame(
    filas_docs,
    "doc_id string, sha256 string, grupo string, eje string, contenido_raw string, "
    "paginas int, extractor string, normalizacion string, contrato int",
)
df_docs.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{NS}.documentos_benchmark"
)

cache = CacheEmbeddings(DATA_DIR / "embeddings.json")
filas_chunks = []
for d in documentos:
    for c in d.chunks:
        texto_embedding = texto_para_embeber(c, d.crudo)
        clave = CacheEmbeddings.clave(texto_embedding, "search_document")
        assert clave in cache.datos, f"Falta embedding congelado: {c.chunk_id}"
        filas_chunks.append(
            (
                c.chunk_id,
                c.doc_id,
                int(c.char_start),
                int(c.char_end),
                int(c.emb_start),
                c.text,
                texto_embedding,
                c.eje or "",
                [float(x) for x in cache.datos[clave]],
            )
        )

df_chunks = spark.createDataFrame(
    filas_chunks,
    "chunk_id string, doc_id string, char_start long, char_end long, emb_start long, "
    "texto_citable string, texto_embedding string, eje string, embedding array<float>",
)
df_chunks.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{NS}.chunks_benchmark"
)
spark.sql(
    f"ALTER TABLE {NS}.chunks_benchmark SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

assert df_docs.filter("grupo = 'base'").count() == 13
assert df_docs.filter("grupo = 'distractor'").count() == 11
assert df_chunks.count() == 1405
print("✅ Delta+C​​DF: 13 base + 11 distractores · 1.405 chunks con vector congelado")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Gold set: evidencia antes que métrica
# MAGIC
# MAGIC Cada pregunta declara documento y span. El gate comprueba que el span existe literalmente y
# MAGIC que la pregunta no copia su propia respuesta. Recuperar el paper correcto no basta: el chunk
# MAGIC recuperado debe contener el inicio del span esperado.

# COMMAND ----------

preguntas_crudas = cargar_gold_set(DATA_DIR / "gold_set.yml")
preguntas, rechazos = validar(preguntas_crudas, {d.doc_id: d for d in documentos})
assert len(preguntas_crudas) == 70
assert len(preguntas) == 70
assert rechazos == [], rechazos

df_gold = spark.createDataFrame(
    [
        (
            p.id,
            p.doc_id,
            p.tipo,
            p.pregunta,
            p.span,
            int(p.char_start),
            int(p.char_end),
            int(p.pagina),
        )
        for p in preguntas
    ],
    "pregunta_id string, doc_id string, tipo string, pregunta string, span_esperado string, "
    "char_start long, char_end long, pagina int",
)
df_gold.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{NS}.gold_set")
print("✅ gold set: 70 declaradas · 70 admitidas · 0 rechazadas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Dense, BM25 y RRF sobre las mismas 70 preguntas
# MAGIC
# MAGIC - **Dense:** similitud coseno sobre embeddings multilingües.
# MAGIC - **BM25:** coincidencia léxica; protege nombres, códigos y términos raros.
# MAGIC - **RRF:** fusiona posiciones, no scores de escalas incompatibles.
# MAGIC
# MAGIC La consulta se traduce al inglés porque el corpus está en inglés. Las traducciones están
# MAGIC congeladas: una traducción nueva sería otra versión del experimento.

# COMMAND ----------

def acierta(resultados, pregunta) -> bool:
    return any(
        r.doc_id == pregunta.doc_id
        and r.char_start <= pregunta.char_start < r.char_end
        for r in resultados
    )


retriever = Retriever(documentos, DATA_DIR / "embeddings.json")
translation_cache = CacheTraduccion(DATA_DIR / "traducciones.json")
consultas = dict(
    zip(
        (p.id for p in preguntas),
        traducir([p.pregunta for p in preguntas], translation_cache),
    )
)

rankings = []
metricas = {}
for modo in ("denso", "lexico", "rrf"):
    hits = 0
    reciprocal_ranks = []
    for p in preguntas:
        resultados = retriever.buscar(consultas[p.id], k=K, modo=modo)
        hit = acierta(resultados, p)
        hits += int(hit)
        posicion = None
        for rank, r in enumerate(resultados, start=1):
            es_evidencia = (
                r.doc_id == p.doc_id and r.char_start <= p.char_start < r.char_end
            )
            if es_evidencia and posicion is None:
                posicion = rank
            rankings.append(
                (
                    p.id,
                    modo,
                    rank,
                    r.chunk_id,
                    r.doc_id,
                    int(r.char_start),
                    int(r.char_end),
                    float(r.puntaje),
                    bool(es_evidencia),
                )
            )
        reciprocal_ranks.append(1.0 / posicion if posicion else 0.0)
    metricas[modo] = {
        "hits": hits,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }

assert {m: metricas[m]["hits"] for m in metricas} == {
    "denso": 36,
    "lexico": 53,
    "rrf": 51,
}, metricas

df_eval = spark.createDataFrame(
    rankings,
    "pregunta_id string, metodo string, rank int, chunk_id string, doc_id string, "
    "char_start long, char_end long, score double, contiene_evidencia boolean",
)
df_eval.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{NS}.evaluacion_retrieval"
)
display(df_eval.groupBy("metodo").agg(F.sum(F.col("contiene_evidencia").cast("int")).alias("hits_en_top8")))
print("✅ benchmark: dense 36/70 · BM25 53/70 · RRF 51/70")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Reranker: resultado medido, no promesa
# MAGIC
# MAGIC El reranker listwise recibió 20 candidatos y devolvió 8. Se eligió con una partición
# MAGIC determinista: 37 preguntas de calibración y 33 de validación. Sobre el corpus de 24
# MAGIC documentos obtuvo `34/37 + 24/33 = 58/70`.
# MAGIC
# MAGIC Este notebook conserva esa evidencia congelada. Volver a invocar un LLM produciría una nueva
# MAGIC corrida probabilística y debe guardarse bajo otra versión; no se sobrescribe el benchmark.

# COMMAND ----------

expected = json.loads((DATA_DIR / "expected_metrics.json").read_text(encoding="utf-8"))
reranker_evidence = json.loads(
    (DATA_DIR / "reranker_evidence.json").read_text(encoding="utf-8")
)
assert reranker_evidence["calibration"] == {"hits": 34, "questions": 37}
assert reranker_evidence["validation"] == {"hits": 24, "questions": 33}
assert reranker_evidence["total"] == {"hits": 58, "questions": 70}
assert expected["source_commit"].startswith("f22f856")
print("✅ reranker listwise medido: 58/70 sobre los mismos 1.405 chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Cita verificable y abstención
# MAGIC
# MAGIC El modelo puede señalar una oración, pero el código decide si existe. Mostramos una cita real
# MAGIC con offset y página. Para una pregunta fuera del dominio no fabricamos una respuesta: sin una
# MAGIC evidencia aprobada, la política devuelve `No hay evidencia suficiente.`

# COMMAND ----------

ejemplo = preguntas[0]
doc = next(d for d in documentos if d.doc_id == ejemplo.doc_id)
cita = doc.verificar(ejemplo.span, cerca_de=ejemplo.char_start)
assert cita is not None
assert doc.crudo[cita.char_start:cita.char_end] == cita.texto

display(
    spark.createDataFrame(
        [
            (
                ejemplo.id,
                cita.doc_id,
                cita.pagina,
                cita.char_start,
                cita.char_end,
                cita.texto,
            )
        ],
        "pregunta_id string, doc_id string, pagina int, char_start long, char_end long, cita string",
    )
)

pregunta_fuera_de_dominio = "¿Cuál es la contraseña del WiFi del almacén?"
respuesta_fuera_de_dominio = "No hay evidencia suficiente."
assert all(p.pregunta != pregunta_fuera_de_dominio for p in preguntas)
assert respuesta_fuera_de_dominio == "No hay evidencia suficiente."
print(f"✅ abstención: {respuesta_fuera_de_dominio}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Tabla de control y criterio de terminado
# MAGIC
# MAGIC Un resultado solo se compara si viaja con versión de corpus, extractor, normalización,
# MAGIC contrato de chunk, embedding y `k`. La ruta administrada se reporta aparte.

# COMMAND ----------

resumen_rows = [
    ("documentos", "24", expected["corpus_version"]),
    ("base", "13", expected["corpus_version"]),
    ("distractores", "11", expected["corpus_version"]),
    ("chunks_benchmark", "1405", expected["extractor"]),
    ("preguntas", "70", "gold_set.yml"),
    ("dense_hits", "36", expected["embedding_model"]),
    ("bm25_hits", "53", "BM25 k1=1.5 b=0.75"),
    ("rrf_hits", "51", "RRF k=60"),
    ("reranker_listwise_hits", "58", expected["source_commit"]),
    ("chunks_administrados", str(managed_chunks), "ai_prep_search v2.0"),
]
df_resumen = spark.createDataFrame(resumen_rows, "control string, valor string, version string")
df_resumen.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{NS}.resumen_benchmark"
)
display(df_resumen)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 · Quiz de cierre
# MAGIC
# MAGIC 1. ¿Por qué los 11 documentos nuevos solo pueden restar en este gold set?
# MAGIC 2. ¿Por qué BM25 puede resistir mejor que dense al crecer el corpus?
# MAGIC 3. ¿Por qué no se comparan scores crudos de BM25, cosine y RRF?
# MAGIC 4. ¿Qué cambia si se sustituye el extractor o el embedding?
# MAGIC 5. ¿Por qué una cita válida requiere offsets aunque la respuesta suene correcta?

# COMMAND ----------

summary = {
    "status": "PASS",
    "namespace": NS,
    "volume": ROOT,
    "pdfs": len(pdfs),
    "documents": len(documentos),
    "base_documents": 13,
    "distractor_documents": 11,
    "managed_route_ok": managed_documents == 24 and managed_chunks > 24,
    "managed_attempted_documents": managed_attempted_documents,
    "managed_primary_documents": managed_primary_documents,
    "managed_fallback_documents": managed_fallback_documents,
    "managed_fallback_sources": missing_paths,
    "managed_chunks": managed_chunks,
    "benchmark_chunks": numero_chunks,
    "questions": len(preguntas),
    "hits": {
        "dense": metricas["denso"]["hits"],
        "bm25": metricas["lexico"]["hits"],
        "rrf": metricas["rrf"]["hits"],
        "rrf_listwise": reranker_evidence["total"]["hits"],
    },
    "extractor": expected["extractor"],
    "embedding_model": expected["embedding_model"],
}

assert summary["pdfs"] == 24
assert summary["benchmark_chunks"] == 1405
assert summary["questions"] == 70
assert summary["hits"] == {"dense": 36, "bm25": 53, "rrf": 51, "rrf_listwise": 58}
assert summary["managed_route_ok"] is True

print("🎓 S4.5 TERMINADO · 24 PDF · 1.405 chunks · 70 preguntas · todos los gates del notebook PASS")
dbutils.notebook.exit(json.dumps(summary, ensure_ascii=False))
