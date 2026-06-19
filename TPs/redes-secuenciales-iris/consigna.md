# Redes neuronales secuenciales | Iris

## Metadata
- Categoría: Redes Secuenciales
- Entrega: Semana 6
- Estado en el curso: Done
- Fuente (Notion A422): https://playful-cantaloupe-94c.notion.site/Redes-neuronales-secuenciales-Iris-1d01d85031168197b0f9e97be6427056

## Datos
Archivos en `data/`:
- `The_iris_data_set.pdf`
- `iris.data`
- `iris.names`

## Consigna

🧠 Consigna: Modelado y análisis de redes neuronales sobre el dataset Iris
En este trabajo van a diseñar, entrenar y analizar modelos de redes neuronales para resolver el problema de clasificación de especies de flores usando el dataset clásico de Iris.
Además, deberán incorporar técnicas de selección de características para evaluar si es posible mejorar el desempeño del modelo reduciendo la cantidad de variables de entrada.
📄 Descripción del dataset
El dataset contiene 150 observaciones con medidas de tres especies de flores:
Variables numéricas:
sepal length (cm)
sepal width (cm)
petal length (cm)
petal width (cm)
Variable de clase:
species (Setosa, Versicolor, Virginica)
📌 Instrucciones
1. Exploración inicial
Analizar el dataset:
Realizar un análisis exploratorio.
Determinar la cantidad de clases y su distribución.
Evaluar la necesidad de:
Normalización o estandarización de los datos.
Codificación de las etiquetas de clase.
2. Selección de características
Decidir cuántas y cuáles características utilizar para uno de los modelos.
Justificar la elección.
3. Diseño y entrenamiento de modelos
Diseñar y entrenar modelos diferentes. Seleccionar y justificar cuál consideran que es el mejor modelo logrado (presentar los mejores modelos).
⚡ Tip: Pueden optar por arquitecturas distintas o mantener la misma arquitectura para comparar solo el impacto de la selección de características.
Para cada modelo:
Justificar las decisiones de diseño (capas, neuronas, funciones de activación, optimizador, etc.).
Dividir en conjunto de entrenamiento y test (80%-20% recomendado).
Entrenar y validar.
4. Evaluación de desempeño
Especificar:
Accuracy, Precision, Recall, F1 Score.
Matriz de confusión.
Curvas de entrenamiento (pérdida y accuracy vs. epochs).
Comparación de los resultados obtenidos con y sin selección de características.
5. Visualización de la frontera de decisión
Utilizar PCA para proyectar los datos en 2 dimensiones.
Generar un gráfico de superficie de decisión:
Mostrar cómo el modelo separa las clases en el espacio reducido.
Superponer los puntos reales coloreados por clase.
Repetir este gráfico para cada modelo.
6. Reflexión y comparación
Comparar el desempeño de los modelos.
Reflexionar:
¿Mejoró la performance al reducir las características?
¿Fue más simple o más difícil entrenar el modelo con menos features?
¿Qué aprendieron sobre el dataset y el modelado?
🎯 Entregable
Una presentación por equipo:
Link al notebook de Google Colab completo o archivo.py.
🎁 BONUS
Construir en Excel la arquitectura de dos modelos seleccionados (similar a lo visto en clase magistral).
🌸
Iris - modelo de clasificación
