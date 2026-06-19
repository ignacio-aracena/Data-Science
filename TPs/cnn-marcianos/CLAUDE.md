# CLAUDE.md — Clasificación de terrenos marcianos con una CNN

Guía para trabajar en este TP. El enunciado completo y fiel está en `consigna.md`.

## Qué es

Es el TP **en curso** de la materia (estado *In progress*), el único de redes convolutivas que está activo. Se entrena una CNN que clasifica tipos de terreno marciano a partir de imágenes reales de los rovers, y además se investiga qué está mirando la red para decidir (interpretabilidad).

## Datos

El dataset es externo, de Kaggle (no se versiona; link en `consigna.md`):
- Mars Terrain image dataset: `kaggle.com/datasets/aumthaker/mars-terrain-images`.

Se puede bajar local o trabajar en un notebook de Kaggle o Colab. Hay una categoría `Other` que cumple un rol especial (ver más abajo).

## Entregable

Notebook que corre completo más la presentación. La parte de interpretabilidad es central, no opcional.

## Cómo encararlo

El enunciado pide ocho secciones; el hilo conductor es no tratar a la CNN como una caja negra. Lo distintivo de este TP:

- **Categoría `Other`**: se separa desde el inicio. El modelo se entrena y valida solo con las clases de terreno bien definidas; las imágenes de `Other` quedan reservadas y, ya con el modelo entrenado, se predice a qué clase las asignaría (análisis de confianza con umbral, out-of-distribution, posible re-etiquetado).
- **Arquitectura propia desde cero** (Conv2D + activación + pooling + densas), justificando cada decisión, más data augmentation con sentido para imágenes de terreno.
- **Interpretabilidad**: feature maps, Grad-CAM, occlusion sensitivity; experimentos de invarianza (rotaciones, flips) y de color (escala de grises, swap de canales) para ver si la red se apoya en forma/textura o en el color rojizo del terreno; y cruzar los errores de la matriz de confusión con esos experimentos.
- Enfoque jerárquico (binario y después multiclase) como opcional.

## Estado

In progress. Es el trabajo activo de la materia.
