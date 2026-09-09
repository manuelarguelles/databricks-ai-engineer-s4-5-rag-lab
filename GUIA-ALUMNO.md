# Guía del alumno · S4.5

## Antes de clase

No necesitas descargar ni buscar documentos. El instructor despliega el paquete y te entrega el
catálogo, schema y Volume. Solo necesitas acceso al workspace y serverless compute.

## Orden de trabajo

1. **Contrato del corpus.** Comprueba 24 archivos y sus hashes. Si falla uno, no continúes.
2. **Ruta administrada.** Ejecuta `ai_parse_document` y `ai_prep_search`; observa que el chunking
   administrado no tiene por qué producir 1.405 chunks.
3. **Ruta congelada.** Reingiere los mismos PDF con el extractor versionado y confirma 1.405 chunks.
4. **Gold set.** Valida las 70 preguntas y localiza su span literal y página.
5. **Retrieval.** Compara dense, BM25 y RRF con `k=8`.
6. **Ruido.** Explica por qué dense cae más que BM25 cuando entran los distractores.
7. **Reranking.** Lee la medición listwise y separa el resultado histórico de una demostración live.
8. **Cierre.** Guarda tablas y responde el quiz sin mirar las respuestas.

## Resultados esperados

| Control | Esperado |
|---|---:|
| PDF válidos | 24 |
| Base / distractores | 13 / 11 |
| Chunks congelados | 1.405 |
| Preguntas admitidas | 70 |
| Dense | 36/70 |
| BM25 | 53/70 |
| RRF | 51/70 |
| Listwise medido | 58/70 |

## Errores comunes

- **Comparar scores crudos:** BM25, cosine y RRF no comparten escala. Compara hits/70 y MRR.
- **Confundir documento con evidencia:** recuperar el paper correcto no basta; el chunk debe contener
  el span esperado.
- **Tomar 1.405 como verdad universal:** ese conteo pertenece a un extractor y chunker versionados.
- **Reindexar con otro embedding y conservar las cifras:** al cambiar el modelo, empieza una nueva
  corrida y etiqueta sus resultados.
- **Usar el gold set para escoger todas las variantes:** separa calibración de validación.

## Entregable del alumno

Una captura o export de la tabla resumen, más cuatro respuestas breves:

1. ¿Por qué los 11 documentos nuevos son distractores en este benchmark?
2. ¿Por qué BM25 resistió mejor que dense?
3. ¿Qué invalida una comparación entre dos corridas?
4. ¿Qué diferencia hay entre una cita literal y una respuesta simplemente plausible?

