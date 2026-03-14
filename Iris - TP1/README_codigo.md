# README — Explicación bloque por bloque: `Iris_tp1.ipynb`
**Ignacio Aracena · Tomás Arizu | Ciencia de Datos**

Este documento describe qué hace cada celda/bloque del notebook `Iris_tp1.ipynb`.

---

## Bloque 1 — Imports y configuración global

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, time, warnings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
```

**¿Qué hace?**
- Importa todas las librerías del proyecto.
- Intenta importar XGBoost con `try/except` — si no está instalado, `XGBOOST_AVAILABLE = False` y el experimento de XGBoost se omite automáticamente.
- Define `CLASS_NAMES = ['setosa', 'versicolor', 'virginica']`, lista usada en todos los gráficos y reportes.
- Inicializa `resultados = []`, lista donde cada experimento acumula sus métricas para la tabla comparativa del final.

---

## Bloque 2 — Carga del dataset

```python
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo'])
iris_df['target']  = iris.target
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
print('Dimensiones del dataset:', iris_df.shape)
display(iris_df.describe().T.round(2))
display(iris_df.head())
```

**¿Qué hace?**
- Carga el dataset Iris desde scikit-learn y lo convierte a un DataFrame con nombres de columnas en español.
- Agrega columnas `target` (numérica: 0/1/2) y `species` (texto: nombre de la especie).
- Imprime dimensiones (150 × 6) y estadísticas descriptivas transpuestas (`.T`) para leer una fila por variable.

**Output relevante:**
```
Dimensiones del dataset: (150, 6)
                 count  mean   std  min  25%   50%  75%  max
longitud_sepalo  150.0  5.84  0.83  4.3  5.1  5.80  6.4  7.9
longitud_petalo  150.0  3.76  1.77  1.0  1.6  4.35  5.1  6.9  ← mayor varianza
ancho_petalo     150.0  1.20  0.76  0.1  0.3  1.30  1.8  2.5  ← mayor varianza
```
Las variables del pétalo tienen desviación estándar mucho mayor, anticipando mayor poder discriminativo.

---

## Bloque 3 — Balance de clases

```python
print(iris_df['species'].value_counts())
iris_df['species'].value_counts().plot(kind='bar', ...)
```

**¿Qué hace?**
- Cuenta muestras por especie e imprime el resultado.
- Genera un gráfico de barras de la distribución de clases.

**Output:** 50 muestras exactas por especie → dataset perfectamente balanceado. Esto valida usar *accuracy* como métrica principal.

---

## Bloque 4 — Histogramas con KDE

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(data=iris_df, x='longitud_sepalo', hue='species', kde=True, ax=axes[0, 0])
# ... idem para las otras 3 variables
```

**¿Qué hace?**
- Genera 4 histogramas (uno por variable) con curva de densidad KDE superpuesta, coloreados por especie.
- El argumento `kde=True` agrega la curva de densidad suavizada sobre el histograma.

**Lectura clave:** `longitud_petalo` y `ancho_petalo` muestran distribución bimodal: *setosa* forma un grupo separado en valores bajos, mientras que *versicolor* y *virginica* se superponen en valores altos.

---

## Bloque 5 — Scatter Plots

```python
pairs = [('longitud_sepalo', 'ancho_sepalo'), ('longitud_petalo', 'ancho_petalo'), ...]
for (x, y), ax in zip(pairs, axes.flatten()):
    sns.scatterplot(data=iris_df, x=x, y=y, hue='species', ax=ax, palette='Set1')
```

**¿Qué hace?**
- Grafica 4 pares de variables en scatter plots coloreados por especie usando un loop sobre los pares definidos.
- Muestra visualmente la separabilidad para cada combinación de variables.

**Lectura clave:** El par `longitud_petalo` vs `ancho_petalo` es la combinación más separable: *setosa* queda completamente aislada.

---

## Bloque 6 — Boxplots

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(data=iris_df, x='species', y='longitud_sepalo', ax=axes[0, 0], palette='Set2')
# ... idem para las otras 3 variables
```

**¿Qué hace?**
- Genera 4 boxplots (uno por variable) con la distribución por especie: mediana, cuartiles, rango y outliers.

**Lectura clave:** En las variables del pétalo, los rangos de *setosa* no se solapan en absoluto con los de las otras dos especies.

---

## Bloque 7 — Matriz de correlación global

```python
sns.heatmap(iris_df[features].corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
```

**¿Qué hace?**
- Calcula la matriz de correlación de Pearson entre las 4 variables y la visualiza como heatmap.
- `annot=True` muestra los valores numéricos en cada celda.
- `cmap='coolwarm'`: azul = correlación negativa, rojo = positiva.

**Lectura clave:** `longitud_petalo` ↔ `ancho_petalo` tienen correlación ~0.96, indicando alta redundancia entre estas dos variables.

---

## Bloque 8 — Matrices de correlación por especie

```python
for ax, sp in zip(axes, ['setosa', 'versicolor', 'virginica']):
    datos_especie = iris_df[iris_df['species'] == sp][features]
    sns.heatmap(datos_especie.corr(), annot=True, ...)
```

**¿Qué hace?**
- Filtra el DataFrame por especie y calcula/visualiza la correlación de forma independiente para cada una.
- Genera 3 heatmaps apilados verticalmente, uno por especie.

**Lectura clave:** Las correlaciones varían entre especies, lo que justifica crear features que capturen estas diferencias.

---

## Bloque 9 — PCA: varianza explicada

```python
scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(iris_df[features])
pca_full = PCA()
pca_full.fit(X_scaled)
varianza = pca_full.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza)
```

**¿Qué hace?**
1. Escala los datos con `StandardScaler` — el PCA es sensible a la escala, sin este paso las variables con mayor rango dominarían las componentes.
2. Ajusta PCA con todas las componentes (`PCA()` sin `n_components`).
3. Calcula varianza individual y acumulada con `np.cumsum`.
4. Grafica barras (varianza individual) + línea con umbral del 95% (varianza acumulada).

**Output:** Con 2 componentes se explica ~95.8% de la varianza total.

---

## Bloque 10 — PCA: visualización 2D

```python
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['species'] = iris_df['species'].values
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='species', ...)
```

**¿Qué hace?**
- Aplica PCA reduciendo a 2 dimensiones y grafica el scatter plot en el espacio reducido.
- El título del gráfico incluye el porcentaje de varianza explicada dinámicamente con `f'{pca_2d.explained_variance_ratio_.sum():.1%}'`.

**Lectura clave:** *Setosa* queda perfectamente aislada; *versicolor* y *virginica* se superponen parcialmente.

---

## Bloque 11 — Feature Engineering

```python
iris_df['ratio_petalo'] = iris_df['longitud_petalo'] / iris_df['ancho_petalo']
iris_df['ratio_sepalo'] = iris_df['longitud_sepalo'] / iris_df['ancho_sepalo']
iris_df['es_petalo_pequeno']       = (iris_df['longitud_petalo'] < 2.0).astype(int)
iris_df['es_ancho_petalo_pequeno'] = (iris_df['ancho_petalo']    < 0.6).astype(int)
iris_df['area_petalo'] = iris_df['longitud_petalo'] * iris_df['ancho_petalo']
iris_df['area_sepalo'] = iris_df['longitud_sepalo'] * iris_df['ancho_sepalo']
iris_df['longitud_petalo_2'] = iris_df['longitud_petalo'] ** 2
iris_df['ancho_petalo_2']    = iris_df['ancho_petalo']    ** 2
iris_df['diff_petalo'] = iris_df['longitud_petalo'] - iris_df['ancho_petalo']
iris_df['diff_sepalo'] = iris_df['longitud_sepalo'] - iris_df['ancho_sepalo']
iris_df['suma_petalo'] = iris_df['longitud_petalo'] + iris_df['ancho_petalo']
```

**¿Qué hace?**
- Agrega 11 nuevas columnas directamente al DataFrame existente, llegando a 15 features en total.
- Las columnas binarias usan `(condicion).astype(int)` para convertir `True/False` a `1/0`.

**Output:** `Total de features: 15` + tabla con las primeras filas del DataFrame ampliado.

---

## Bloque 12 — Preparación para el modelado

```python
y = iris_df['target']
features_originales = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo']
features_petalo     = ['longitud_petalo', 'ancho_petalo', 'ratio_petalo', 'area_petalo']
# ... más conjuntos
X_tr_orig, X_te_orig, y_train, y_test = train_test_split(
    iris_df[features_originales], y, test_size=0.2, random_state=42, stratify=y
)
# ... idem para los otros 4 conjuntos de features
```

**¿Qué hace?**
- Define 5 conjuntos de features distintos para los experimentos.
- Realiza **5 splits simultáneos** con los mismos `random_state=42` y `stratify=y`, garantizando que todos los modelos se evalúan sobre el mismo conjunto de test.
- `stratify=y` asegura proporciones iguales de clases en train y test (40/10 por clase).

**Output:** `Train: 120 muestras | Test: 30 muestras` con distribución 40/40/40 en train y 10/10/10 en test.

---

## Bloque 13 — RF Baseline (experimento 6.1)

```python
inicio = time.time()
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_tr_orig, y_train)
t_train = time.time() - inicio

inicio = time.time()
y_pred_rf_base = rf_base.predict(X_te_orig)
t_pred = time.time() - inicio

acc = accuracy_score(y_test, y_pred_rf_base)
f1  = f1_score(y_test, y_pred_rf_base, average='macro')
cm  = confusion_matrix(y_test, y_pred_rf_base)
# ... impresión de métricas, heatmap de confusión, cálculo de FP/FN por clase
resultados.append({...})
```

**¿Qué hace?**
- Mide el tiempo de entrenamiento y predicción por separado con `time.time()`.
- Imprime accuracy, F1-macro, reporte completo por clase y la matriz de confusión como heatmap.
- Calcula FP (Error Tipo I) y FN (Error Tipo II) por clase: `FP = cm.sum(axis=0)[i] - cm[i][i]`, `FN = cm.sum(axis=1)[i] - cm[i][i]`.
- Agrega un diccionario con todos los resultados a la lista global `resultados`.

Este **patrón se repite en todos los experimentos** (6.1 al 6.11), cambiando solo el modelo y el conjunto de features.

---

## Bloque 14 — RF Todas las features + importancia (experimento 6.2)

```python
rf_todas = RandomForestClassifier(n_estimators=100, random_state=42)
rf_todas.fit(X_tr_todas, y_train)
# ... métricas, confusión, FP/FN ...

importancias = pd.Series(rf_todas.feature_importances_, index=features_todas)
importancias = importancias.sort_values(ascending=False)
importancias.plot(kind='bar', ...)
print('Top 5 features:', importancias.head())
```

**¿Qué hace?** Igual que el baseline pero con 15 features. Además, extrae el atributo `.feature_importances_` del RF (importancia media de cada feature en todos los árboles), lo ordena y lo grafica. Esta importancia se reutiliza en el experimento 6.10 para seleccionar features.

---

## Bloque 15 — RF Solo pétalo (experimento 6.3)

Mismo patrón que el baseline, entrenado con `X_tr_pet` (4 features del pétalo). Evalúa si el pétalo solo es suficiente.

---

## Bloque 16 — RF Solo binarias (experimento 6.4)

Mismo patrón con `X_tr_bin` (2 features binarias). La celda de descripción previa explica explícitamente la limitación: estas variables identifican *setosa* pero no distinguen *versicolor* de *virginica*.

---

## Bloque 17 — RF Originales + Ratios (experimento 6.5)

Mismo patrón con `X_tr_ratio` (6 features: 4 originales + 2 ratios).

---

## Bloque 18 — KNN: selección de k (experimento 6.6, parte 1)

```python
k_values  = [1, 3, 5, 7, 9, 11]
cv_scores = []
for k in k_values:
    pipe_k = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])
    scores = cross_val_score(pipe_k, X_tr_orig, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
best_k = k_values[np.argmax(cv_scores)]
```

**¿Qué hace?**
- Itera sobre 6 valores de k.
- Para cada k crea un Pipeline `StandardScaler → KNN` y lo evalúa con validación cruzada de 5 folds **solo sobre el conjunto de entrenamiento**.
- Selecciona el k con mayor accuracy promedio usando `np.argmax`.
- Grafica la curva accuracy CV vs k.

**¿Por qué Pipeline aquí?** Sin Pipeline, el StandardScaler se ajustaría sobre el fold completo (incluyendo el fold de validación), filtrando información del conjunto de validación hacia el entrenamiento (*data leakage*).

---

## Bloque 19 — KNN: modelo final (experimento 6.6, parte 2)

```python
pipe_knn = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=best_k))])
pipe_knn.fit(X_tr_orig, y_train)
y_pred_knn = pipe_knn.predict(X_te_orig)
```

**¿Qué hace?** Entrena el Pipeline con el `best_k` encontrado y registra resultados con el mismo patrón que los demás experimentos.

---

## Bloque 20 — Gradient Boosting (experimento 6.7)

```python
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_tr_todas, y_train)
```

Mismo patrón. Parámetros: 100 árboles, tasa de aprendizaje 0.1, profundidad máxima 3. Usa todas las 15 features.

---

## Bloque 21 — AdaBoost (experimento 6.8)

```python
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42, algorithm='SAMME')
ada.fit(X_tr_todas, y_train)
```

Mismo patrón. El argumento `algorithm='SAMME'` es la versión discreta de AdaBoost compatible con clasificación multiclase.

---

## Bloque 22 — XGBoost (experimento 6.9)

```python
if XGBOOST_AVAILABLE:
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42,
                        eval_metric='mlogloss', verbosity=0)
    xgb.fit(X_tr_todas, y_train)
```

**¿Qué hace?** El bloque completo está dentro de un `if XGBOOST_AVAILABLE`, por lo que si la librería no está instalada simplemente imprime el mensaje de instalación y no ejecuta nada más.

---

## Bloque 23 — Pipeline RF → GB tuneado (experimento 6.10)

Este experimento tiene 3 sub-bloques:

**Sub-bloque 1 — Selección de features:**
```python
top_features = importancias[importancias > 0.05].index.tolist()
X_tr_top, X_te_top, _, _ = train_test_split(iris_df[top_features], y, test_size=0.2, random_state=42, stratify=y)
```
Filtra las features del RF (bloque 14) con importancia > 5% → resultan 9 features.

**Sub-bloque 2 — Búsqueda de hiperparámetros:**
```python
param_grid = [
    {'n_estimators': 50,  'learning_rate': 0.1,  'max_depth': 2},
    {'n_estimators': 100, 'learning_rate': 0.1,  'max_depth': 3},
    {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 3},
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 4},
]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for params in param_grid:
    scores = cross_val_score(GradientBoostingClassifier(**params), X_tr_top, y_train, cv=cv, scoring='f1_macro')
```
Itera sobre una grilla manual de 4 combinaciones, seleccionando la de mayor F1-macro CV.

**Sub-bloque 3 — Modelo final:**
```python
gb_tuned = GradientBoostingClassifier(**best_params, random_state=42)
gb_tuned.fit(X_tr_top, y_train)
```
Entrena con los mejores parámetros sobre las 9 features seleccionadas.

---

## Bloque 24 — Pipeline PCA → GB (experimento 6.11)

```python
pipe_pca_gb = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=2)),
    ('gb',     GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])
pipe_pca_gb.fit(X_tr_orig, y_train)
```

**¿Qué hace?** Pipeline de 3 etapas: escala → reduce a 2 componentes principales → clasifica con Gradient Boosting. Como siempre, las transformaciones se ajustan solo sobre train y se aplican al test.

---

## Bloque 25 — Curvas ROC

```python
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
modelos_roc = [
    ('RF Baseline', rf_base, X_te_orig),
    ('RF Todas features', rf_todas, X_te_todas),
    # ...
]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, class_idx, class_name in zip(axes, [0, 1, 2], CLASS_NAMES):
    for (nombre, modelo, X_te), color in zip(modelos_roc, colors):
        proba = modelo.predict_proba(X_te)[:, class_idx]
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{nombre} (AUC={roc_auc:.2f})')
```

**¿Qué hace?**
1. `label_binarize`: convierte `y_test` (vector de 0/1/2) en una matriz (30 × 3) de indicadores binarios, necesaria para el enfoque One-vs-Rest.
2. Define la lista de modelos a comparar con su respectivo conjunto de test (cada modelo puede haber sido entrenado con un conjunto distinto).
3. Genera 3 paneles (uno por clase): para cada modelo extrae las probabilidades de la clase actual con `.predict_proba(X_te)[:, class_idx]` y calcula la curva ROC.
4. Grafica todas las curvas juntas con la línea del clasificador aleatorio (AUC = 0.50) como referencia.

---

## Bloque 26 — Tabla comparativa final

```python
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('F1-macro', ascending=False).reset_index(drop=True)
df_resultados.index += 1
display(df_resultados)
```

**¿Qué hace?**
- Convierte la lista `resultados` (acumulada a lo largo de todos los experimentos) en un DataFrame.
- Lo ordena de mayor a menor F1-macro.
- Resetea el índice y suma 1 para que empiece en 1 en vez de 0.
- Muestra la tabla completa con todas las métricas comparadas.

---

## Flujo general del notebook

```
Carga → EDA → PCA → Feature Engineering → Splits
   ↓
Experimentos (6.1 a 6.11): cada uno sigue el patrón:
   fit → predict → accuracy/F1 → confusion matrix → FP/FN → resultados.append()
   ↓
Curvas ROC (comparación visual de todos los modelos)
   ↓
Tabla comparativa final (pd.DataFrame de resultados)
```
