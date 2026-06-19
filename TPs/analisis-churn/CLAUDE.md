# CLAUDE.md — Análisis de datos | Churn

Guía para trabajar en este TP. El enunciado completo y fiel está en `consigna.md`.

## Qué es

Segundo TP (Semana 2, entregado). Análisis del churn (abandono) de clientes de una empresa de telefonía iraní. El objetivo es un estudio exhaustivo de análisis de datos que después permita construir un modelo predictivo del abandono.

## Datos

En `data/`:
- `Iran_Customer_Churn.csv` — 3.150 clientes, 13 atributos (fallas de llamada, frecuencia de SMS, quejas, antigüedad, grupo etario, monto de cargo, etc.) más la variable objetivo `Churn`.
- `Information.txt` — descripción del dataset y de cada atributo.

## Entregable

Presentación (pdf) más el código (Colab o link al repo).

## Cómo encararlo

- Es un problema de clasificación binaria con clases desbalanceadas (lo normal en churn), así que conviene mirar métricas más allá de la accuracy.
- El profe pide justificar cada herramienta de visualización y análisis: para qué se usa, qué se espera y qué dio.
- Atención al desbalance al evaluar (precision/recall, F1, matriz de confusión) y al partir train/test.

## Estado

Done (Semana 2).
