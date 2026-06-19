# CLAUDE.md

Esta guía es para Claude Code (claude.ai/code) cuando trabaja dentro de esta carpeta.

## Qué es esta carpeta

Es el workspace de **Ciencia de Datos (código A422)**, materia que Ignacio cursa en la Licenciatura en Negocios Digitales de la UdeSA (primer semestre 2026). Acá viven los trabajos prácticos y parciales de machine learning de la materia. Este es el contenido que se entrega como repositorio de GitHub, así que conviene mantenerlo ordenado y reproducible.

El equipo docente de la materia es Gabriel Basaluzzo, Clara Kearney, Pedro Busato Ader, Nicolás Araujo y Tomás Morales.

## Estructura

```
Data-Science/
├── Material Teorico/        # PDFs y recursos teóricos del curso (hoy vacía)
└── TPs/                     # un proyecto por trabajo práctico
    └── <tp>/
        ├── consigna.md      # enunciado del profe, fiel al Notion del curso
        ├── CLAUDE.md        # guía de trabajo del TP (cómo encararlo)
        └── data/            # datasets del TP, cuando vienen adjuntos
```

Cada TP es independiente y tiene su propio `CLAUDE.md`; leerlo antes de trabajar adentro. La `consigna.md` es la fuente de verdad del enunciado (texto del profe sin editar); el `CLAUDE.md` es la lectura nuestra de cómo resolverlo.

## Los trabajos prácticos

Son los TPs que están en el tablero del curso en estado *Done* (entregados) más el que está *In progress*. Los que el curso tiene en *Not started* no se hicieron y no forman parte de este repo.

| Carpeta | Qué es | Tipo | Estado |
|---|---|---|---|
| `analisis-iris` | EDA y clasificación del dataset Iris | Análisis de datos | Done (Semana 1) |
| `analisis-churn` | Predicción de abandono en una telco iraní | Análisis de datos | Done (Semana 2) |
| `analisis-riesgo-credito` | Aceptación/rechazo de créditos, banco alemán | Análisis de datos | Done (Semana 3) |
| `analisis-sommelier` | Calidad y color de vino (Wine Quality, UCI) | Análisis de datos | Done (Semana 4) |
| `deeper-playground` | Redes sobre datasets geométricos (estilo TF Playground) | Redes secuenciales | Done (Semana 5) |
| `redes-secuenciales-iris` | Red secuencial + selección de features sobre Iris | Redes secuenciales | Done (Semana 6) |
| `f1` | Pregunta de negocio y modelo sobre F1 (1950-2024) | Análisis + modelo | Done |
| `cnn-marcianos` | CNN para clasificar terrenos marcianos | Redes convolutivas | In progress |

El arco de la materia va de análisis de datos puro (EDA y modelos clásicos) a redes neuronales secuenciales y funcionales, y termina en convolutivas (visión).

## Convenciones

- **Español rioplatense, sin emojis** en la documentación propia. Las `consigna.md` son la excepción: conservan el texto del profe tal cual, emojis incluidos.
- **Reproducibilidad**: cada notebook corre de punta a punta desde un kernel limpio (`Kernel → Restart and Run All`).
- **Seeds fijos** en todo lo estocástico (`RANDOM_SEED = 42`: `np.random.seed`, `tf.random.set_seed`, `random_state`).
- **Sin leakage**: scalers, encoders y embeddings se ajustan (`fit`) solo en train; `transform` en validación y test.
- **Validación exhaustiva, no por muestreo**: chequear sobre el dataset completo, no sobre `df.head()`.
- **Entregable de cada TP**: un notebook que corre completo (Colab o `.py`) más una presentación. Algunos TPs piden además armar la arquitectura del modelo en Excel.

## Sobre los datos

Los datasets livianos que el curso adjunta viven en `data/` de cada TP. Los pesados o los que el curso deja como fuente externa (Kaggle, UCI) se referencian con su link en la `consigna.md` y no se versionan, para no inflar el repo. Antes de subir a GitHub, revisar que no entre nada con copyright si el repo va a ser público.

## De dónde salió esto

El contenido de cada TP se extrajo del Notion público de la materia (A422 Ciencia de Datos). El link a la página de origen está en el encabezado de cada `consigna.md`.
