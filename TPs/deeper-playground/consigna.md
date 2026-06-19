# Deeper Playground | Introducción geométrica a redes neuronales

## Metadata
- Categoría: Redes Secuenciales
- Entrega: Semana 5
- Estado en el curso: Done
- Fuente (Notion A422): https://playful-cantaloupe-94c.notion.site/Deeper-Playground-Introducci-n-geom-trica-a-redes-neuronales-1d01d850311681f594d3c8e04f783553

## Datos
Archivos en `data/`:
- `Set_de_datos_inspirados_en_tensorflow.xlsx`
Recursos / datasets externos:
- [deeperplayground.org/#ac…ntum=0](https://deeperplayground.org/#activation=tanh&regularization=L2&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.99407&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&animationSpeed=100&layerwiseGradientNormalization=0&learningRateAutotuning=-1&preventLossIncreases=false&dropout=0&momentum=0)

## Consigna

🧠 Consigna: Modelado y análisis de redes neuronales sobre dos datasets
Cada grupo trabajará con dos datasets diferentes (asignados por el docente), basados en distribuciones geométricas y espaciales (por ejemplo: espirales, círculos, lóbulos, formas de corazón). El objetivo es diseñar, entrenar y analizar un modelo de red neuronal por cada dataset, adaptando el modelo a las características particulares de los datos.
Se recomienda realizar el ejercicio con todos los datasets, pero se entregan los dos datasets asignados.
Los datasets incluidos en el archivo son los siguientes:
Media Luna abstracta
Infinito A
Círculo doble abstracto
Cuadrantes
Blobs
Espiral detallada
Media luna clara
Corazón dividido
Flor detallada
Infinito B
📌 Instrucciones
Exploración inicial de los datasets
Visualizar los puntos del dataset (gráfico x vs y con color por clase).
Identificar la cantidad de clases y su distribución.
Evaluar si es necesario preprocesamiento (normalización, codificación de etiquetas, etc.).
Diseño y entrenamiento de modelos
Diseñar un modelo de red neuronal por cada dataset.
Elegí arquitecturas distintas si lo considerás necesario (por ejemplo, modelos más complejos para formas más difíciles de separar).
Justificá tus decisiones de diseño (capas, neuronas, activaciones, etc.).
Entrená y validá los modelos sobre una división train/test adecuada.
Evaluación de desempeño
Calculá métricas de clasificación: accuracy, precision, recall, F1 score.
Graficá la matriz de confusión.
Mostrá las curvas de pérdida y accuracy vs. épocas.
Visualización del resultado del modelo
Generá un gráfico de superficie de decisión para cada modelo, donde se visualice cómo el modelo clasifica el espacio en función de las coordenadas x e y.
Usá una malla de puntos (meshgrid) para estimar predicciones del modelo sobre el plano.
Mostrá los puntos reales superpuestos.
Comparación y reflexión
Compará el desempeño de ambos modelos.
Explicá en qué se diferenciaron las decisiones de diseño entre los dos casos y por qué.
Proponé posibles mejoras para lograr una mejor clasificación.
🎯 Entregable
Una presentación en PDF o Google Slides con:
Nombres de los integrantes
Breve descripción de cada dataset
Visualizaciones claras y explicadas:
Gráfico de los datos originales
Arquitectura del modelo
Métricas y gráficos de entrenamiento
Gráficos de superficie de decisión
Reflexión final con aprendizajes
Link al notebook de Google Colab con el desarrollo completo (código, visualizaciones y análisis)
Armá en Excel la arquitectura del modelo, como vieron en la clase magistral (y como el parcial).
