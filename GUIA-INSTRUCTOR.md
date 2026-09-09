# Guía del instructor · S4.5

Duración recomendada: **3 horas reloj**, cuatro bloques de 45 minutos y dos pausas cortas.

## Preparación previa · 20–30 min

1. Ejecuta `scripts/deploy_and_run.py` con el perfil y catálogo de clase.
2. Abre `reports/databricks-run.json` y confirma `SUCCESS`.
3. Comprueba que el resumen contiene 24 PDF, 1.405 chunks y 70 preguntas.
4. Abre el notebook desplegado y deja el primer bloque visible.
5. No compartas credenciales ni el bucket de origen; el corpus ya vive en el Volume gobernado.

## Resultados no negociables

La clase solo se declara lista cuando la corrida reproduce estos controles: `24` PDF válidos,
`13 + 11` documentos, `1.405` chunks congelados, `70/70` preguntas admitidas, dense `36/70`,
BM25 `53/70`, RRF `51/70` y reranker listwise medido `58/70`. El conteo de chunks de la ruta
administrada se informa por separado: no tiene que ser 1.405.

## Bloque 1 · Del PDF al contrato · 45 min

- **0–8:** plantea el problema: una respuesta correcta sin evidencia rastreable no es auditable.
- **8–18:** recorre manifiesto, hash, procedencia y separación 13+11.
- **18–32:** ejecuta inventario y `ai_parse_document → ai_prep_search`.
- **32–40:** compara parsing administrado con extractor versionado.
- **40–45:** checkpoint: ¿por qué un cambio de extractor invalida offsets?

## Bloque 2 · Chunks y gold set · 45 min

- **0–15:** explica raw, vista normalizada, mapa y frontera dura.
- **15–28:** confirma 1.405 chunks y CDF.
- **28–38:** valida las 70 preguntas y enseña un span con página.
- **38–45:** actividad: encontrar una pregunta que recupera el documento correcto pero pierde el span.

## Bloque 3 · Dense, BM25 y RRF · 45 min

- **0–12:** analogía: dense busca significado; BM25 busca vocabulario raro; RRF combina posiciones.
- **12–30:** ejecuta el benchmark y revela resultados solo después de recoger predicciones.
- **30–38:** contrasta 13→24 documentos: dense 42→36; BM25 53→53; RRF 53→51.
- **38–45:** discute por qué “más carriles” no garantiza más señal.

## Bloque 4 · Reranking, cita y decisión · 45 min

- **0–15:** explica retrieve 20 → rerank 8 y el techo alcanzable.
- **15–25:** presenta la medición listwise 58/70 y su partición calibración/validación.
- **25–35:** demuestra una consulta y verifica la evidencia por offset/página.
- **35–42:** quiz de cierre.
- **42–45:** puente a S05: el retriever pasa a ser una tool gobernada, no una respuesta libre.

## Troubleshooting

| Síntoma | Diagnóstico | Acción |
|---|---|---|
| `PERMISSION_DENIED` al crear schema/Volume | faltan privilegios UC | usar un catálogo propio o pedir `CREATE_SCHEMA`/`CREATE_VOLUME` |
| Volume muestra menos de 24 PDF | carga incompleta | volver a ejecutar `deploy_and_run.py`; la carga es idempotente |
| Hash distinto | archivo corrupto o sustituido | no seguir; restaurar el PDF desde este repo privado |
| `ai_parse_document` no existe | función no habilitada en la región/runtime | enseñar la ruta congelada y registrar la contingencia |
| `ai_prep_search` no produce chunks | formato/ruta o función no habilitada | inspeccionar una sola ruta y el resultado de `ai_parse_document` |
| `ai_prep_search` omite una salida VARIANT | limitación beta sobre ese layout | conservar el evento; el notebook aplica y etiqueta `deterministic_text_fallback` para esa fuente |
| Falta un embedding congelado | cache y corpus no pertenecen a la misma versión | no llamar otra API en silencio; redeploy del paquete |
| Métricas distintas | cambió corpus, versiones, traducciones o `k` | comparar el encabezado experimental antes del resultado |
| Reranker live varía | componente probabilístico | conservar la medición congelada y etiquetar la nueva corrida |

## Contingencia de tiempo

Si la ruta administrada tarda, no canceles el benchmark: salta a la ruta congelada, que no depende
de llamadas de embedding. La clase debe conservar corpus, gold set, dense/BM25/RRF y criterio de cita.
