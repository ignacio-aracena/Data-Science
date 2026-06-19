# CLAUDE.md — Redes neuronales secuenciales | Iris

Guía para trabajar en este TP. El enunciado completo y fiel está en `consigna.md`.

## Qué es

Sexto TP (Semana 6, entregado). Vuelve sobre el dataset Iris, pero ahora con redes neuronales secuenciales en lugar de modelos clásicos. Suma selección de características: evaluar si reducir la cantidad de variables de entrada mejora (o no) el desempeño.

## Datos

En `data/` (los mismos archivos que el TP de análisis de Iris):
- `iris.data` — 150 observaciones (4 features + especie).
- `iris.names` — descripción de las variables.
- `The_iris_data_set.pdf` — material de referencia.

## Entregable

Presentación por equipo más el link al notebook de Colab completo (o `.py`). Bonus: armar en Excel la arquitectura de dos de los modelos elegidos, como en la clase magistral.

## Cómo encararlo

- EDA breve (ya conocido del TP de análisis) y decisiones de preprocesamiento: normalización/estandarización y codificación de etiquetas.
- Selección de características: decidir cuántas y cuáles usar para uno de los modelos y justificarlo.
- Entrenar varios modelos (mismas o distintas arquitecturas) y elegir el mejor, justificando capas, neuronas, activaciones y optimizador. Split 80/20.
- Evaluar con accuracy/precision/recall/F1, matriz de confusión y curvas de entrenamiento, comparando con y sin selección de features.
- Frontera de decisión: proyectar con PCA a 2D y graficar la superficie de decisión de cada modelo con los puntos reales encima.

## Estado

Done (Semana 6).
