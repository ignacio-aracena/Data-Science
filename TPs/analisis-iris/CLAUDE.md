# CLAUDE.md — Análisis de datos | Iris

Guía para trabajar en este TP. El enunciado completo y fiel está en `consigna.md`.

## Qué es

Primer TP de la materia (Semana 1, entregado). Análisis exploratorio del dataset clásico Iris (UCI): 150 flores, cuatro medidas (largo y ancho de sépalo y de pétalo) y tres especies (Setosa, Versicolor, Virginica). El foco es un ejercicio de clasificación: estudiar las cuatro variables y determinar si las especies son separables con aprendizaje supervisado.

## Datos

En `data/`:
- `iris.data` — las 150 observaciones (CSV sin encabezado: 4 features + especie).
- `iris.names` — descripción de las variables y notas del dataset.
- `The_iris_data_set.pdf` — material de referencia del profe.

El profe sugirió además un Colab de arranque para el análisis exploratorio (link en `consigna.md`).

## Entregable

Presentación (ppt) más el código (Colab o link al repo), antes del comienzo de la tutorial.

## Cómo encararlo

- EDA primero: distribuciones de las cuatro variables por especie, correlaciones y separabilidad. El patrón conocido es que Setosa se separa linealmente y Versicolor/Virginica se solapan.
- El profe pide justificar cada técnica: para qué se usa, qué se espera obtener y qué dio. Si no se puede responder eso, mejor no usarla.
- Hay una extensión de feature engineering y modelado (ratios y áreas de pétalo/sépalo, comparar Random Forest contra modelos de boosting, evaluar con precision/recall/F1 y no solo accuracy, y armar una tabla comparativa). Está en `consigna.md`.

## Estado

Done (Semana 1).
