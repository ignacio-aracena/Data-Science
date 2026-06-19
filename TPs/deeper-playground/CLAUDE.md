# CLAUDE.md — Deeper Playground | Introducción geométrica a redes neuronales

Guía para trabajar en este TP. El enunciado completo y fiel está en `consigna.md`.

## Qué es

Quinto TP (Semana 5, entregado). Primer acercamiento a redes neuronales secuenciales sobre datasets sintéticos con forma geométrica (espirales, círculos, lóbulos, corazones), en la línea del TensorFlow Playground. Se diseña, entrena y analiza una red por cada dataset, adaptando la arquitectura a la dificultad de separar cada forma.

## Datos

En `data/`:
- `Set_de_datos_inspirados_en_tensorflow.xlsx` — los datasets geométricos (Media Luna, Infinito A/B, Círculo doble, Cuadrantes, Blobs, Espiral, Corazón dividido, Flor, etc.). Cada grupo trabaja con dos asignados por el docente, pero se recomienda probar con todos.

Referencia interactiva: deeperplayground.org (link en `consigna.md`).

## Entregable

Presentación (PDF o Google Slides) más el link al notebook de Colab. Además hay que armar la arquitectura del modelo en Excel, como en la clase magistral y en el parcial.

## Cómo encararlo

- Por cada dataset: visualizar los puntos (x vs y, color por clase), ver cantidad de clases y distribución, evaluar preprocesamiento.
- Diseñar una red por dataset y justificar las decisiones (capas, neuronas, activaciones); formas más difíciles suelen pedir modelos más complejos.
- Evaluar con accuracy/precision/recall/F1, matriz de confusión y curvas de pérdida y accuracy por época.
- La pieza distintiva: graficar la **superficie de decisión** de cada modelo con un meshgrid y superponer los puntos reales.

## Estado

Done (Semana 5).
