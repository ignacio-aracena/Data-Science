# Ciencia de Datos (A422) — Trabajos Prácticos

Repositorio de entregables de la materia **Ciencia de Datos** (código A422), Licenciatura en Negocios Digitales, Universidad de San Andrés (primer semestre 2026).

Cada trabajo práctico es un proyecto independiente: su enunciado, el notebook resuelto que corre de punta a punta, y los datos para reproducirlo.

## Trabajos prácticos

| Carpeta | Tema | Tipo |
|---|---|---|
| [`TPs/analisis-iris`](TPs/analisis-iris) | EDA y clasificación de especies (Iris), feature engineering y boosting | Análisis de datos |
| [`TPs/analisis-churn`](TPs/analisis-churn) | Predicción de abandono de clientes en una telco | Análisis de datos |
| [`TPs/analisis-riesgo-credito`](TPs/analisis-riesgo-credito) | Aceptación/rechazo de créditos (German Credit) | Análisis de datos |
| [`TPs/analisis-sommelier`](TPs/analisis-sommelier) | Color y calidad de vino (Wine Quality) + redes | Análisis de datos |
| [`TPs/deeper-playground`](TPs/deeper-playground) | Redes sobre datasets geométricos (estilo TF Playground) | Redes secuenciales |
| [`TPs/redes-secuenciales-iris`](TPs/redes-secuenciales-iris) | Red secuencial + selección de características (Iris) | Redes secuenciales |
| [`TPs/f1`](TPs/f1) | Pregunta de negocio y modelo sobre Fórmula 1 (1950-2024) | Análisis + modelo |
| [`TPs/cnn-marcianos`](TPs/cnn-marcianos) | CNN para clasificar terrenos marcianos + transfer learning | Redes convolutivas |

El arco va de análisis de datos puro (EDA y modelos clásicos) a redes secuenciales y funcionales, y termina en convolutivas (visión).

## Estructura

```
TPs/<tp>/
├── consigna.md      # enunciado del profe (fiel al Notion del curso)
├── CLAUDE.md        # guía de cómo se encaró el TP
├── <tp>.ipynb       # notebook resuelto, corre de punta a punta
└── data/            # datos para reproducirlo
```

Los dos TPs de redes secuenciales traen además `arquitectura_*.xlsx` con la arquitectura del modelo (capas, pesos y forward pass).

## Cómo correrlos

Los notebooks corren con Python 3 + scikit-learn; los de redes (`deeper-playground`, `redes-secuenciales-iris`, `analisis-sommelier`, `cnn-marcianos`) usan además TensorFlow/Keras. Cada notebook ya viene ejecutado con sus salidas; para reproducir desde cero, *Kernel → Restart and Run All*. Los datos necesarios están en la carpeta `data/` de cada TP.

Convenciones: español rioplatense, seeds fijos (`RANDOM_SEED = 42`), preprocesamiento sin fuga de datos (los scalers y encoders se ajustan solo sobre train), y validación sobre el dataset completo.

## Uso de IA (transparencia)

Estos trabajos se desarrollaron con asistencia de IA (Claude Code). La carpeta [`.claude/`](.claude) deja a la vista las herramientas que se usaron y cómo se trabajó:

- [`.claude/agents/`](.claude/agents) — agentes especializados que se invocaron según la tarea: `data-scientist`, `data-analyst`, `python-pro`, `code-reviewer`, `debugger` y `technical-writer`.
- [`.claude/skills/`](.claude/skills) — skills que encapsulan el flujo de un proyecto de ML de punta a punta: adquisición de datos (tabular e imágenes), EDA con narrativa, feature engineering justificado, diseño de arquitecturas CNN, iteración de redes con análisis de error, interpretabilidad de modelos e insights de negocio.

La IA se usó para acelerar el EDA, el código, las visualizaciones y la documentación. Las decisiones de modelado, la validación de resultados y las conclusiones se revisaron y verificaron en cada TP.
