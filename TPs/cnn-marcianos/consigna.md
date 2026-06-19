# Clasificación de terrenos marcianos con una CNN

## Metadata
- Categoría: Empty
- Entrega: Empty
- Estado en el curso: In progress
- Fuente (Notion A422): https://playful-cantaloupe-94c.notion.site/Clasificaci-n-de-terrenos-marcianos-con-una-CNN-3741d850311680399dc1dbf4ebab195f

## Datos
Recursos / datasets externos:
- [https://www.kaggle.com/datasets/aumthaker/mars-terrain-images](https://www.kaggle.com/datasets/aumthaker/mars-terrain-images)

## Consigna

Trabajo Práctico: Clasificación de terrenos marcianos con una CNN
Contexto
Las misiones de exploración a Marte (rovers como Curiosity, Perseverance, Spirit y Opportunity) generan miles de imágenes de la superficie. Clasificar automáticamente el tipo de terreno es útil para la navegación autónoma y para priorizar zonas de interés científico. En este trabajo van a entrenar una red neuronal convolucional (CNN) que aprenda a distinguir tipos de terreno a partir de imágenes reales, y además van a investigar qué está mirando la red para tomar sus decisiones.
Dataset
Mars Terrain image dataset (Kaggle): https://www.kaggle.com/datasets/aumthaker/mars-terrain-images
Pueden descargarlo localmente o trabajar directamente en un notebook de Kaggle o Google Colab.
Objetivos de aprendizaje
Construir, entrenar y evaluar una CNN de clasificación de imágenes comprendiendo el flujo completo —exploración, preprocesamiento, diseño de la arquitectura, entrenamiento, regularización y análisis crítico— y desarrollar una mirada de interpretabilidad sobre el modelo: entender en qué se apoya para predecir y por qué falla donde falla.
Tareas
1. Exploración de datos (EDA). Inspeccionen la organización del dataset: ¿cuántas clases hay y cómo se llaman? ¿Cuántas imágenes por clase? ¿Qué resolución y formato tienen? Identifiquen en particular la categoría "Other". Muestren una grilla de ejemplos por categoría y comenten si el dataset está balanceado.
2. Preparación de los datos. Separen desde el inicio la categoría "Other" del resto: el modelo se entrena y valida únicamente con las clases de terreno bien definidas, y las imágenes de "Other" quedan reservadas aparte (sin usarse en el entrenamiento). Sobre las clases definidas, dividan en entrenamiento, validación y test (por ejemplo 70/15/15). Apliquen redimensionado a un tamaño fijo, normalización de píxeles y data augmentation (rotaciones, flips, zoom, brillo), justificando qué transformaciones tienen sentido para imágenes de terreno.
3. Diseño de la CNN. Definan una arquitectura propia desde cero (bloques de Conv2D + activación + pooling, seguidos de capas densas). Justifiquen las decisiones: cantidad de filtros, tamaño de kernel, función de activación, uso de dropout / batch normalization y la capa de salida acorde al número de clases.
4. Entrenamiento. Elijan función de pérdida, optimizador y métrica adecuados al problema. Entrenen registrando las curvas de loss y accuracy de entrenamiento y validación. Usen al menos una técnica para evitar el sobreajuste (early stopping, regularización, etc.).
5. Evaluación. Reporten el desempeño sobre el conjunto de test con métricas más allá de la accuracy: matriz de confusión, precision, recall y F1 por clase. Identifiquen qué clases se confunden entre sí y muestren ejemplos de errores del modelo.
6. Clasificación de la categoría "Other". Con el modelo ya entrenado, predigan a qué clase asignaría cada imagen de "Other". Como estas imágenes no tienen etiqueta verdadera, el análisis es distinto al del test:
Obtengan, para cada imagen, la clase predicha y su probabilidad/confianza (salida softmax).
Muestren la distribución de predicciones: ¿hacia qué clases tiende a asignarlas?, ¿se reparten parejo o se concentran en pocas?
Definan un umbral de confianza (p. ej. 0,7) y separen las predicciones seguras de las dudosas, con ejemplos visuales de ambos casos.
Discutan qué son esas imágenes de baja confianza: ¿terrenos genuinamente distintos a las clases conocidas (out-of-distribution), mezclas de varias clases, o ejemplos que en realidad encajarían en una categoría existente y podrían reetiquetarse?
7. Ideas convolucionales (qué aprende y mira la red). El objetivo es entender qué está mirando la CNN —formas, color, texturas— y no tratarla como una caja negra. Desarrollen al menos algunas de estas líneas:
¿Qué features aprende? Visualicen los mapas de activación de las primeras y últimas capas convolucionales. ¿Las iniciales detectan bordes/texturas y las profundas patrones complejos? Apóyense en visualización de feature maps, Grad-CAM (mapas de calor de dónde mira la red) u occlusion sensitivity (tapar zonas y ver cómo cae la predicción).
¿El kernel aprende formas? Apliquen rotaciones (90°, 180°, ángulos arbitrarios) y flips a imágenes del test. ¿El modelo sigue prediciendo bien o se degrada? Relacionen el resultado con haber usado (o no) rotaciones en el data augmentation.
¿El kernel aprende color? Pasen las imágenes a escala de grises, intercambien canales (RGB → BGR) o alteren tono/brillo, y midan el impacto. ¿La red se apoya en el espectro de colores del terreno o le alcanza con la estructura/textura? En terreno marciano (predominantemente rojizo-ocre) esto es especialmente interesante.
¿Por qué falla donde falla? Crucen los errores de la matriz de confusión con estos experimentos: ¿aparecen en imágenes rotadas, colores atípicos, zonas de baja textura? Busquen una explicación de los modos de falla.
Enfoque funcional / en cascada (opcional). En lugar de un único clasificador multiclase, prueben una arquitectura jerárquica: primero un clasificador binario "¿es la clase X o no?" y, si no lo es, un segundo modelo que decide entre las clases restantes (análogo a: primero decidir "¿es un ocho?" y recién después clasificar qué dígito es). Comparen contra el clasificador único: ¿mejora alguna clase difícil?, ¿a costa de qué complejidad?
Documenten cada experimento con ejemplos visuales y una breve interpretación de lo que revela.
8. Análisis y conclusiones. Discutan qué funcionó y qué no, el impacto del data augmentation y del balanceo de clases, y qué reveló la sección de interpretabilidad sobre el funcionamiento interno de la red. Reflexionen sobre la utilidad de usar el modelo para pre-etiquetar automáticamente datos sin categoría ("Other"): ¿reduce el trabajo manual de anotación?, ¿qué riesgos tiene confiar en esas etiquetas generadas por el modelo? Propongan al menos dos mejoras concretas.
Transfer learning
Comparar la CNN propia contra un modelo con transfer learning (MobileNetV2, ResNet50 o VGG16 preentrenados en ImageNet) y discutir las diferencias de desempeño y costo de entrenamiento.
Entregables
Presentación
Criterios de evaluación (sugeridos)
Correctitud del flujo y reproducibilidad (20%), calidad del EDA y del preprocesamiento (15%), diseño y justificación de la arquitectura (15%), entrenamiento y manejo del sobreajuste (10%), evaluación con métricas adecuadas (15%), clasificación y análisis de "Other" (10%), interpretabilidad / ideas convolucionales (10%) y análisis crítico y conclusiones (5%).
presentacion_mars_cnn_v4.html
66.4 KiB
Opcional collab inicial
from google.colab import drive
drive.mount('/content/drive')

​
import os, glob

# ZIP en tu Drive
ZIP_PATH = "/content/drive/MyDrive/FinalBigData-LamasyFodrini/auburn_data.zip"

# Carpeta destino en Colab donde se descomprime
EXTRACT_PATH = "/content/auburn_data"

# Carpeta donde quedan realmente las imágenes
DATA_DIR = os.path.join(EXTRACT_PATH, "Auburn_1")

print("ZIP_PATH   :", ZIP_PATH)
print("EXTRACT_PATH:", EXTRACT_PATH)
print("DATA_DIR   :", DATA_DIR)

​
import os, glob

# ¿Ya existe el dataset descomprimido con imágenes?
already_extracted = os.path.isdir(DATA_DIR) and \
                    len(glob.glob(os.path.join(DATA_DIR, "*", "*.jpg"))) > 0


if not os.path.exists(ZIP_PATH):
    print(f"❌ Error: El archivo ZIP '{ZIP_PATH}' no se encontró.")
    print("Por favor, verifica que el archivo exista en tu Google Drive y que la ruta sea correcta.")
    print(f"Puedes ejecutar `!ls -F \"{os.path.dirname(ZIP_PATH)}\"` en una celda nueva para verificar el contenido de la carpeta.")
elif already_extracted:
    print("✅ Dataset ya descomprimido, NO se vuelve a unzip.")
else:
    print("🔁 Descomprimiendo ZIP desde cero...")
    # Borramos cualquier resto previo
    !rm -rf "{EXTRACT_PATH}"
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    # -q: quiet   -o: overwrite sin preguntar
    !unzip -qo "{ZIP_PATH}" -d "{EXTRACT_PATH}"
    print("✅ Listo, ZIP descomprimido.")

# Ver contenido principal
print("\nContenido de EXTRACT_PATH:")
!ls "{EXTRACT_PATH}"

print("\nSubcarpetas dentro de DATA_DIR:")
!ls "{DATA_DIR}"

​
!ls "/content/auburn_data/Auburn_1"

​
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

​
root_dir = "/content/auburn_data/Auburn_1"

classes = sorted(os.listdir(root_dir))
print("Clases encontradas:", classes)

​
image_counts = {}

for cls in classes:
    class_path = os.path.join(root_dir, cls)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        image_counts[cls] = count

image_counts

​
