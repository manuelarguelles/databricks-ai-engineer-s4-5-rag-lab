# Corpus del laboratorio

## Composición

- `pdf/base/`: **13 documentos** para los que existe evidencia en el gold set.
- `pdf/distractores/`: **11 documentos** agregados después para medir degradación por ruido.
- Total: **24 PDF**.

Los distractores son documentos académicos relacionados con el dominio. No son basura artificial:
compiten semánticamente con el corpus base y por eso constituyen una prueba más difícil.

## Integridad y procedencia

`manifest.yml` es el contrato. Para cada fuente incluida registra identificador, título, eje,
URL/DOI, nombre de archivo y SHA-256. Un archivo que no coincida con su hash no entra al pipeline.

Los PDF se obtuvieron de fuentes de acceso abierto o copias públicamente accesibles registradas en
el manifiesto original del proyecto de investigación. Este repositorio es **privado y de uso
docente**. Antes de convertirlo en público hay que revisar la licencia de distribución de cada
documento; una URL públicamente accesible no equivale automáticamente a permiso de redistribución.

## Por qué el gold set cubre solo 13 documentos

Las 70 preguntas se etiquetaron manualmente antes de ampliar el corpus. Los 11 documentos nuevos
solo pueden restar aciertos en esta medición: actúan como distractores. Por eso la caída observada
es un techo del daño, no una estimación equilibrada del valor de los documentos nuevos.

