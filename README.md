# Laboratorio S4.5 · RAG con 24 documentos y evidencia verificable

Laboratorio independiente entre S04 y S05 del curso **Databricks AI Engineer**. Parte de un
corpus real de papers sobre selección asistida y lleva al alumno, paso a paso, desde los PDF
hasta una evaluación de retrieval reproducible.

> Estado validado el 2026-09-09: **10/10 gates PASS** · Databricks run
> `148807986461078` en `SUCCESS` · 24/24 PDF.

## Resultado de aprendizaje

Al terminar, el alumno puede:

1. verificar un corpus por manifiesto y SHA-256;
2. cargar archivos no estructurados a un Unity Catalog Volume;
3. comparar la ruta administrada `ai_parse_document → ai_prep_search` con una ruta de benchmark
   cuyo extractor y chunking están versionados;
4. construir tablas Delta con CDF para documentos, chunks, gold set y resultados;
5. medir dense, BM25 y RRF sobre las mismas 70 preguntas;
6. interpretar qué ocurre al agregar 11 documentos distractores;
7. explicar por qué retrieval, reranking y generación se evalúan por separado;
8. citar por offset y abstenerse cuando no hay evidencia.

## Lo que contiene

- `corpus/pdf/base/`: 13 PDFs cubiertos por el gold set.
- `corpus/pdf/distractores/`: 11 PDFs adicionales para la prueba de ruido.
- `corpus/manifest.yml`: procedencia, URL y SHA-256 de cada documento.
- `data/gold_set.yml`: 70 preguntas manuales y sus spans esperados.
- `data/embeddings.json`: vectores Cohere v3 congelados para reproducir el benchmark sin red.
- `src/corpus_citas/`: extractor, contrato del chunk, BM25, dense y RRF auditados.
- `notebooks/S4.5-laboratorio-rag-24-documentos.py`: notebook Databricks principal.
- `slides/S4.5-deck.html`: deck autocontenido de 17 láminas, navegable e imprimible.
- `scripts/deploy_and_run.py`: crea schema/Volume, sube el paquete, importa y ejecuta el notebook.
- `scripts/validate_package.py`: verifica los diez gates de aceptación.
- `GUIA-ALUMNO.md`, `GUIA-INSTRUCTOR.md` y `HANDOFF-DANIEL.md`.

Los 24 PDF están versionados como archivos completos dentro de `corpus/pdf/`; el consumidor no
necesita descargarlos de Internet. Sus hashes y fuentes se verifican antes de ejecutar el pipeline.
La licencia del código no modifica los derechos de los documentos académicos de terceros; consulta
`THIRD-PARTY-NOTICES.md`.

## Dos rutas, sin mezclar métricas

| Ruta | Propósito | Unidad comparable |
|---|---|---|
| Administrada Databricks | aprender `ai_parse_document` y `ai_prep_search` sobre los 24 PDF | conteo y muestra de chunks administrados |
| Benchmark congelado | reproducir la medición histórica con extractor, chunking y embeddings versionados | hits/70 y MRR |

La ruta administrada puede producir un número de chunks distinto. Eso es esperado: cambiar el
extractor o el chunker cambia la unidad experimental. Las cifras `36/70`, `53/70`, `51/70` y
`58/70` pertenecen al benchmark congelado de 1.405 chunks.

En la corrida de aceptación, la ruta operativa produjo 597 chunks: 548 para 23 fuentes mediante
`ai_prep_search` sobre VARIANT y 49 para una fuente mediante fallback determinista etiquetado.
La excepción, el archivo afectado y su aislamiento respecto del benchmark están documentados en
`docs/VALIDACION-2026-09-09.md`.

## Ejecución en Databricks

Prerequisitos: CLI Databricks autenticada, perfil válido y permisos para crear schema/Volume en
el catálogo elegido.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python scripts/validate_package.py --skip-databricks
.venv/bin/python scripts/deploy_and_run.py \
  --profile webinar-aws-v2 \
  --catalog neptuno_manuel_arguelles
.venv/bin/python scripts/validate_package.py \
  --profile webinar-aws-v2 \
  --catalog neptuno_manuel_arguelles
```

Valores por defecto del despliegue:

- schema: `rag_s4_5`;
- Volume: `laboratorio`;
- notebook: `/Shared/curso-databricks-ai-engineer/S4.5-laboratorio-rag-24-documentos`.

## Veredicto de terminado

El laboratorio solo está terminado cuando `reports/acceptance.json` marca los diez gates como
`PASS` y `reports/databricks-run.json` contiene una corrida `SUCCESS` con 24 documentos,
1.405 chunks de benchmark y 70 preguntas admitidas. Consulta `CRITERIOS-ACEPTACION.md`.

## Licencia

Código y material original: MIT. Los PDF y demás obras de terceros quedan excluidos de esa
licencia y conservan sus términos originales.
