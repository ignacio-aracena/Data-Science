# CLAUDE.md — Análisis de datos | Entrenando a un sommelier

Guía para trabajar en este TP. El enunciado completo y fiel está en `consigna.md`.

## Qué es

Cuarto TP (Semana 4, entregado). Análisis exhaustivo (gráfico y analítico) del dataset Wine Quality para determinar la calidad y el color del vino a partir de sus variables fisicoquímicas. Son los vinos portugueses "Vinho Verde" (tinto y blanco).

## Datos

El dataset es externo, del UCI Machine Learning Repository (id 186). Se baja con `ucimlrepo`:

```python
pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features      # 11 variables fisicoquímicas continuas
y = wine_quality.data.targets       # quality (0-10); color (red/white) como otra etiqueta
```

En `data/` quedan los materiales de referencia, no el dataset:
- `Wine.pdf` — material del profe.
- `Understanding_Wine_Chemistry_Waterhouse.pdf` — libro de referencia (tiene copyright; sacarlo si el repo va a ser público).

## Entregable

Presentación (pdf) más el código (Colab o link al repo).

## Cómo encararlo

- Análisis exhaustivo de las 11 variables fisicoquímicas, apuntando a dos preguntas: calidad (target ordinal y desbalanceado) y color.
- Las clases de calidad están desbalanceadas (muchos vinos "normales", pocos excelentes o malos); tenerlo en cuenta al evaluar.
- Los *next steps* del enunciado (red secuencial, autoencoder para reducir dimensión, modelo funcional) no eran parte de la entrega de esta semana, pero están documentados en `consigna.md`.

## Estado

Done (Semana 4).
