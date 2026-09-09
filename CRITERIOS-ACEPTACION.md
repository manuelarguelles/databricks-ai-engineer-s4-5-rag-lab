# Criterios de aceptación · Laboratorio S4.5

| Gate | Criterio verificable | Evidencia |
|---|---|---|
| G01 · Corpus | Hay exactamente 24 PDF y todos coinciden con el SHA-256 del manifiesto. | `validate_package.py`, `corpus/manifest.yml` |
| G02 · Base y distractores | El corpus está separado en 13 base + 11 distractores y se declara su procedencia. | `corpus/README.md` |
| G03 · Carga automática | Un comando crea schema/Volume y sube corpus, data y código sin pasos manuales. | `scripts/deploy_and_run.py` |
| G04 · Notebook principal | Un único notebook explica y ejecuta inventario, parsing, chunking, tablas, retrieval, eval y cierre. | `notebooks/S4.5-laboratorio-rag-24-documentos.py` |
| G05 · Gold set | Las 70 preguntas pasan evidencia literal y anti-eco; ninguna queda rechazada. | tabla `gold_set`, reporte de corrida |
| G06 · Benchmark | Sobre 1.405 chunks se reproducen dense 36/70, BM25 53/70 y RRF 51/70. | tabla `evaluacion_retrieval` |
| G07 · Reranker | Se conserva la medición listwise 58/70, con procedencia y sin confundirla con el baseline de 13 docs. | `data/expected_metrics.json` |
| G08 · Operación docente | Hay deck, instrucciones, outputs esperados, tiempos y troubleshooting de endpoint, permisos, catálogo y parsing. | `slides/S4.5-deck.html`, guías de alumno/instructor |
| G09 · Databricks E2E | El notebook corre desde un schema limpio y termina `SUCCESS`; el Volume contiene 24 PDF. | `reports/databricks-run.json` |
| G10 · Handoff | Daniel recibe una ruta autocontenida: objetivo, agenda, comandos, checkpoints, contingencias y cierre. | `HANDOFF-DANIEL.md` |

## Reglas de rechazo

- Un PDF faltante o con hash distinto bloquea la ejecución.
- Un gold set que no tenga exactamente 70 preguntas bloquea la evaluación.
- Una métrica sin versión de corpus, extractor, chunker y embedding no se compara.
- La ruta administrada no puede presentarse como reproducción del benchmark congelado.
- Un run verde con el Volume vacío no cuenta como ejecución E2E.
- El handoff no se entrega si cualquiera de G01–G10 está en `FAIL`.
