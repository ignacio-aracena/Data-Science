# CLAUDE.md — Análisis de datos | Riesgo de crédito

Guía para trabajar en este TP. El enunciado completo y fiel está en `consigna.md`.

## Qué es

Tercer TP (Semana 3, entregado). Análisis de la base de solicitudes de préstamo de un banco alemán (German Credit), para más adelante modelar el criterio del banco al aceptar o rechazar una solicitud.

## Datos

En `data/`:
- `Base_Clientes_Alemanes.xlsx` — la base de solicitudes.
- `german_clean.docx` — decodificador de la base: qué significa cada variable codificada. Es clave para interpretar las columnas.
- `Caso_credit_risk.pdf` — un trabajo de referencia sobre un dataset similar, para familiarizarse con variables y métricas.

## Entregable

Presentación (pdf) más el código (Colab o link al repo).

## Cómo encararlo

- El reto del dataset: la muestra es chica para la cantidad de variables, así que muchas serán poco representativas (bajo tamaño muestral) o poco relevantes (bajo information value, baja correlación con la decisión).
- Conviene apoyarse en selección de variables (information value, correlaciones) antes de modelar.
- Como en los otros TPs de análisis: justificar cada técnica (para qué, qué se espera, qué dio).

## Estado

Done (Semana 3).
