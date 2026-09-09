# Validación final · 2026-09-09

## Veredicto

**10/10 gates PASS** sobre el paquete local y una ejecución real en Databricks serverless.

## Corrida de aceptación

- Perfil: `webinar-aws-v2`
- Catálogo/schema: `neptuno_manuel_arguelles.rag_s4_5`
- Volume: `dbfs:/Volumes/neptuno_manuel_arguelles/rag_s4_5/laboratorio`
- Notebook: `/Shared/curso-databricks-ai-engineer/S4.5-laboratorio-rag-24-documentos`
- Run ID: `148807986461078`
- Task run ID: `1020316134958189`
- Estado: `SUCCESS`
- URL: <https://dbc-0410b264-20c7.cloud.databricks.com/?o=7474657121564806#job/519768905811060/run/148807986461078>

## Controles observados

| Control | Resultado |
|---|---:|
| PDF en Volume | 24 |
| Documentos base / distractores | 13 / 11 |
| Intentos `ai_parse_document` | 24 |
| Fuentes con chunks VARIANT de `ai_prep_search` | 23 |
| Fuentes con fallback determinista etiquetado | 1 |
| Chunks de la ruta operativa | 597 |
| Chunks del benchmark congelado | 1.405 |
| Preguntas admitidas / rechazadas | 70 / 0 |
| Dense | 36/70 |
| BM25 | 53/70 |
| RRF | 51/70 |
| RRF + reranker listwise medido | 58/70 |

## Excepción administrada observada

`ai_prep_search` no emitió `document.contents` para
`37_explainable-use-of-foundation-models-for-job-hiring.pdf`, aunque `ai_parse_document` sí
produjo una salida sin `error_status`. Para no reducir silenciosamente el corpus, el notebook crea
49 ventanas de texto con PyMuPDF, las etiqueta `deterministic_text_fallback` y conserva la fuente
en `chunks_administrados`. Este fallback no participa en las métricas congeladas: el benchmark
reingiere los 24 PDF con su contrato versionado.

## Evidencia automatizada

- `reports/databricks-run.json`: salida estructurada de la corrida y conteo del Volume.
- `reports/acceptance.json`: detalle de G01–G10.
- `pytest`: `5 passed`.
- Comando final: `.venv/bin/python scripts/validate_package.py`.

