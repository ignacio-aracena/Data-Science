# Análisis de datos | Iris

## Metadata
- Categoría: Análisis de Datos
- Entrega: Semana 1
- Estado en el curso: Done
- Fuente (Notion A422): https://playful-cantaloupe-94c.notion.site/An-lisis-de-datos-Iris-1d01d85031168124a95ae1067c5fa7c8

## Datos
Archivos en `data/`:
- `The_iris_data_set.pdf`
- `iris.data`
- `iris.names`
Recursos / datasets externos:
- [Google Colab](https://colab.research.google.com/drive/1TA-AMVqfe0XLbKRUYjUIs8asfJrkxbjM?usp=sharing)

## Consigna

La consigna para esta semana es realizar un análisis exhaustivo en parejas del dataset iris. El interés está focalizado en llevar adelante un ejercicio de clasificación de las flores, a partir de las 4 variables disponibles (sepal length, sepal width, petal length y petal width ).
Utilice los conceptos aprendidos en analítica de datos, y las herramientas que considere necesarias para estudiar las 4 variables y determinar si el conjunto de datos es separable o no mediante un procedimiento de aprendizaje supervisado.
Se recomienda realizar el análisis del dataset con Python (porque es el lenguaje que se utilizará para la construcción de modelos), pero podrían utilizar otras herramientas como Tableau, Power BI, Excel u otras.
Cada equipo debe entregar un ppt con su analisis y el código/tablero/documento previo al comienzo de la tutorial (en el campus virtual).
Una ayuda adicional (para orientarte a cómo iniciar el análisis exploratorio)
Google Colab​
**Ejercicio de Feature Engineering y Modelado:**

Entrenamos un modelo Random Forest utilizando únicamente las 4 características originales del dataset. Como pudiste observar en los resultados, el rendimiento ya es bastante bueno.

Ahora, te invito a experimentar con diferentes combinaciones de características como *input* para el modelo Random Forest (o incluso otros modelos de clasificación). Podes crear nuevas celdas de código para:

1.  Seleccionar un subconjunto diferente de características del DataFrame `iris_df`, incluyendo los ratios (`ratio_petalo`, `ratio_sepalo`) y las características de ingeniería que creamos (`es_petalo_pequeno`, `es_ancho_petalo_pequeno`, `area_petalo`).
    *   **Ejemplos:**
        *   Solo características del pétalo (`longitud del pétalo (cm)`, `ancho del pétalo (cm)`, `ratio_petalo`, `area_petalo`).
        *   Características originales + ratios.
        *   Solo las características binarias del pétalo (`es_petalo_pequeno`, `es_ancho_petalo_pequeno`).
        *   Todas las características disponibles.
2.  Probar otras alternativas de Feature Engineering: Considerá crear nuevas características usando técnicas que discutimos anteriormente pero que aún no  implementamos, como:
    *   Transformaciones Polinómicas: Crear términos como `longitud_petalo^2` o `ancho_sepalo^3` si observaste relaciones no lineales en los gráficos.
    *   Otras Combinaciones Lineales: Explorar sumas o diferencias de características (`longitud_petalo - ancho_petalo`).
    *   Interacciones Adicionales: Si hay otras combinaciones de características que crees que podrían ser relevantes (por ejemplo, `longitud_sepalo * ancho_sepalo` para el área del sépalo).
3.  Dividir los datos (`X` seleccionado/modificado y `y`) en conjuntos de entrenamiento y prueba nuevamente (usando `train_test_split` con `stratify=y` y el mismo `random_state` para comparaciones justas).
4.  Entrenar **otros modelos de clasificación** (sin incluir Redes Neuronales). 
    *   **Modelos de Boosting** (como Gradient Boosting o AdaBoost)
    * Combinación de técnicas: Usar la importancia de las características de un modelo Random Forest entrenado con todas las características disponibles (las originales + las nuevas características creadas a partir del feature engineering) para seleccionar un subconjunto de las características más relevantes. Luego, tomar ese subconjunto de características y utilizarlo para entrenar y ajustar los hiperparámetros de un modelo de Boosting (como Gradient Boosting o XGBoost) para intentar maximizar el rendimiento.
5.  Evaluar el rendimiento de cada modelo. Además de la Accuracy, Matriz de Confusión y Reporte de Clasificación, te invito a considerar otras métricas importantes, como **Precision, Recall, F1-score, Curva ROC (si aplica por clase), Error Tipo I (Falsos Positivos) y Error Tipo II (Falsos Negativos)**.
    *   **Pensá qué métrica priorizar** para este problema específico. ¿Es más importante no clasificar erróneamente una especie particular (evitar Falsos Positivos o Falsos Negativos para una clase)? ¿O la precisión general es suficiente?
6.  Además de comparar el rendimiento, te invito a **analizar la complejidad de cada modelo**. Considerá aspectos como:
    *   **Tiempo de entrenamiento y predicción (Run Time):** ¿Cuánto tardan en entrenarse y hacer predicciones?
    *   **Cantidad de Inputs (Inputs):** ¿Cuántas características utiliza el modelo? (Esto depende de la selección de características que hagas en el paso 1 o 2).
    *   **Interpretabilidad:** ¿Qué tan fácil es entender cómo el modelo llega a sus predicciones? (Por ejemplo, un Árbol de Decisión suele ser más interpretable que un Random Forest o un SVM complejo).

Este ejercicio te ayudará a entender empíricamente cómo las diferentes características, su combinación o transformación, la elección del modelo y la selección de métricas de evaluación pueden influir en el rendimiento y la complejidad de un modelo de clasificación.

Al final, armá una tabla que incluya: Modelo, Features usadas, Accuracy, F1-macro, Tiempo de entrenamiento/predicción, Nº de features y notas de interpretabilidad. Esto facilitará la comparación entre experimentos.

​
