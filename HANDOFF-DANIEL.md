# Handoff para Daniel · Laboratorio S4.5

## Qué vas a dictar

Una sesión práctica independiente entre S04 y S05: **RAG con 24 documentos reales, 11
distractores y evaluación sobre 70 preguntas**. El foco no es construir un chatbot bonito; es
demostrar cuándo la evidencia llega, cuándo se pierde y cómo se mide.

## Estado requerido antes de compartir

No dictar si `reports/acceptance.json` no muestra `10/10 PASS` o si la corrida Databricks no está
en `SUCCESS`.

**Estado entregado:** `10/10 PASS`, corrida `148807986461078` en `SUCCESS`, validada el
2026-09-09. Evidencia completa en `docs/VALIDACION-2026-09-09.md`.

## Acceso

- Workspace y perfil usados para validación: `webinar-aws-v2`.
- Notebook: `/Shared/curso-databricks-ai-engineer/S4.5-laboratorio-rag-24-documentos`.
- Catálogo de validación: `neptuno_manuel_arguelles`.
- Schema: `rag_s4_5`.
- Volume: `laboratorio`.

## Ruta de la clase

1. Abre el notebook y explica la promesa de la sesión.
2. Ejecuta inventario y hash: deben aparecer `24 = 13 + 11`.
3. Muestra la ruta administrada de PDF a chunks.
   Señala que produjo 548 chunks para 23 fuentes y que la fuente 37 requirió 49 chunks con
   `deterministic_text_fallback`; no escondas la limitación beta observada.
4. Ejecuta la ruta congelada y confirma `1.405` chunks.
5. Valida el gold set: `70 admitidas, 0 rechazadas`.
6. Pide predicciones antes de revelar dense/BM25/RRF.
7. Ejecuta el benchmark y muestra `36/70`, `53/70`, `51/70`.
8. Explica la medición listwise `58/70` sin presentarla como una corrida determinista nueva.
9. Cierra con cita por página/offset y abstención.

## Frases que evitan confusiones

- “Los 11 nuevos documentos son distractores solo porque no tienen preguntas propias en este gold set.”
- “El conteo de chunks pertenece a una versión de extractor y contrato; no es una constante de los PDF.”
- “BM25, cosine y RRF ordenan; hits/70 permite comparar.”
- “El modelo propone relevancia; el código verifica la evidencia.”

## Plan B

Si `ai_parse_document` falla, continúa con la ruta congelada y declara la contingencia. Si el
benchmark congelado falla, detén la clase: significa que corpus, cache o contrato no corresponden a
la misma versión.
