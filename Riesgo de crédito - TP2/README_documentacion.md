# README — Documentación Técnica: `credit_risk.ipynb`
**Ignacio Aracena · Tomás Arizu | Data Science — A42 | TP2**

Este documento describe qué hace cada celda del notebook `credit_risk.ipynb`, qué output produce, por qué se eligió esa implementación, y qué significa cada resultado en el contexto del problema de riesgo crediticio.

---

## Estructura general del notebook

```
S1 — Carga y Preparación (celdas 1–10)
S2 — Análisis Univariado (celdas 11–15)
S3 — Análisis Bivariado (celdas 16–20)
S4 — Correlaciones (celdas 21–23)
S5 — Information Value + WoE (celdas 24–30)
S6 — Preprocesamiento y Split (celdas 31–33)
S7 — Regresión Logística L1 + Cross-Validation (celdas 34–43)
S8 — Comparación de Modelos (celdas 44–48)
S9 — Optimización del Threshold (celdas 49–53)
S10 — Odds Ratios e Interpretabilidad (celdas 54–56)
S11 — Scorecard (celdas 57–60)
Conclusiones Finales (celda 61)
```

**Total: 62 celdas** (30 markdown + 32 código)

---

## Índice resumido

| Celda | Tipo | Contenido | Output clave |
|---|---|---|---|
| 0 | md | Título, objetivo, estructura | — |
| 1 | md | Intro S1: para qué, qué esperar | — |
| 2 | code | Importación de librerías | "Librerías cargadas." |
| 3 | code | Carga del Excel y recodificación | Shape (1000, 22), 700/300 |
| 4 | md | Resultado S1: distribución target | — |
| 5 | code | Inspección: head, dtypes, nulos | 0 nulos, 0 duplicados |
| 6 | md | Resultado: calidad del dataset | — |
| 7 | code | describe() variables numéricas | Tabla estadísticas descriptivas |
| 8 | md | Resultado: interpretación de medias/medianas | — |
| 9 | code | Gráfico distribución del target | Barplot + pie chart |
| 10 | md | Resultado: desbalance 70/30 | — |
| 11 | md | Intro S2: univariado | — |
| 12 | code | Histogramas + KDE + boxplots numéricas | 7 gráficos (2 paneles c/u) |
| 13 | md | Resultado: skew, outliers, artefactos | — |
| 14 | code | Barplots categóricas con bad rate | 13 gráficos |
| 15 | md | Resultado: variables casi constantes | — |
| 16 | md | Intro S3: bivariado | — |
| 17 | code | Boxplots por clase + Mann-Whitney | 7 tests, tabla p-values |
| 18 | md | Resultado: significancia por variable | — |
| 19 | code | Bad rate por categoría + chi² | 13 gráficos con p-values |
| 20 | md | Resultado: spread y variables no predictivas | — |
| 21 | md | Intro S4: correlaciones | — |
| 22 | code | Heatmap de correlación | Matriz triangular inferior |
| 23 | md | Resultado: r=0.625 duracion-monto | — |
| 24 | md | Intro S5: fórmulas IV/WoE | — |
| 25 | code | Funciones WoE/IV + cálculo para 20 vars | Tabla IV completa |
| 26 | md | Resultado: 14 seleccionadas, cuenta_corriente excluida | — |
| 27 | code | Gráfico ranking IV | Barplot horizontal con umbrales |
| 28 | md | Resultado: frontera de decisión IV | — |
| 29 | code | WoE plots top 6 variables | 6 gráficos WoE + bad rate |
| 30 | md | Resultado: cómo leer WoE | — |
| 31 | md | Intro S6: preprocesamiento | — |
| 32 | code | Split + OHE + StandardScaler | Train/Valid/Test 600/200/200 |
| 33 | md | Resultado: estratificación perfecta | — |
| 34 | md | Intro S7: LogReg L1 | — |
| 35 | code | Entrenamiento LogReg L1 | 23/38 coef no nulos |
| 36 | md | Intro CV: para qué, StratifiedKFold | — |
| 37 | code | Cross-validation 5-fold | Media AUC ± std |
| 38 | md | Resultado CV: estabilidad confirmada | — |
| 39 | md | Resultado: L1 descartó 15 features | — |
| 40 | code | Métricas threshold=0.50 | AUC=0.7102, PR=0.5443 |
| 41 | md | Resultado: interpretación matriz confusión | — |
| 42 | code | Curvas ROC y Precision-Recall | 2 curvas con área sombreada |
| 43 | md | Resultado: cómo leer ROC vs PR | — |
| 44 | md | Intro S8: comparación 6 modelos (LogReg→XGBoost) | — |
| 45 | code | 6 modelos: DT, RF, GB, AdaBoost, XGBoost + CV 5-fold | Tabla AUC CV por modelo |
| 46 | code | Train en train completo + tabla + curvas ROC superpuestas | Tabla comparativa + curvas |
| 47 | md | Resultado: champion vs challenger, por qué elegimos LogReg | — |
| 48 | code | SHAP values para XGBoost: importancia global + beeswarm | Ranking SHAP vs ranking IV |
| 49 | md | Intro S9: threshold asimétrico | — |
| 50 | code | Búsqueda threshold óptimo en validación | Threshold=0.26, curva costo |
| 51 | md | Resultado: threshold 0.26 explicado | — |
| 52 | code | Evaluación final en test | Costo=137, reducción 54.3% |
| 53 | md | Resultado: comparación 3 escenarios | — |
| 54 | md | Intro S10: Odds Ratios | — |
| 55 | code | Tabla + gráfico Odds Ratios | Top 20 features activas |
| 56 | md | Resultado: interpretación de negocio | — |
| 57 | md | Intro S11: Scorecard | — |
| 58 | code | Cálculo factor, offset y scores | Factor=28.85, Offset=487.12 |
| 59 | code | Histograma scores por clase | Score corte=517, 16.5% aprobados |
| 60 | md | Resultado: scorecard | — |
| 61 | md | Conclusiones finales | — |

---

## Documentación detallada por celda

---

### Celda 2 — Importación de Librerías

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               AdaBoostClassifier)
try:
    from xgboost import XGBClassifier; XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    import shap; SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
```

**¿Qué hace?**
- Importa todo lo que el notebook necesita en una sola celda al inicio: `pandas` para DataFrames, `numpy` para operaciones numéricas, `matplotlib`/`seaborn` para visualizaciones, `scipy.stats` para los tests estadísticos (Mann-Whitney, chi²), y `sklearn` para el modelado.
- Importa `StratifiedKFold` y `cross_val_score` para la validación cruzada estratificada que se realiza en S7 y S8.
- Importa **6 modelos** para la comparación en S8: `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier` (todos de sklearn), más `XGBClassifier` y `shap` con `try/except` porque son librerías opcionales no siempre disponibles en todos los entornos.
- El patrón `try/except + flag booleano` (`XGBOOST_AVAILABLE`, `SHAP_AVAILABLE`) permite que el notebook sea ejecutable en entornos sin XGBoost/SHAP, omitiendo esas celdas con un mensaje claro. Alternativa: poner la importación dentro de cada celda que la usa (menos ordenado).
- `warnings.filterwarnings('ignore')` silencia advertencias de compatibilidad de versiones de sklearn que no afectan los resultados.
- Configura el estilo global de gráficos una sola vez: `plt.rcParams` y `sns.set_palette('Set2')`. La paleta Set2 es amigable con personas con daltonismo — buena práctica en visualización.

**Output**: `"Librerías cargadas."` (más posibles mensajes si XGBoost o SHAP no están disponibles).

---

### Celda 3 — Carga y Preparación del Dataset

**¿Qué hace?**
1. Lee `Base_Clientes_Alemanes.xlsx` (hoja 0) con `pd.read_excel`.
2. Mapea los nombres de columnas numéricos (1, 2, 3...) a nombres semánticos usando el diccionario `col_names`, construido a partir de `german_clean.docx`. Cada asignación tiene un comentario con las categorías posibles — crucial para entender qué representa cada código.
3. Filtra `df['target'].isin([1, 2])` — descarta cualquier valor corrupto. En este dataset no hay registros corruptos, pero es buena práctica para reutilización.
4. Recodifica el target: `(df['target'] == 2).astype(int)` → 0=bueno, 1=malo. La convención estándar en clasificación binaria en sklearn es `1 = clase positiva = la que queremos detectar`, que en este caso es el mal pagador.
5. Separa `num_cols` y `cat_cols` — estas listas se usan en todas las secciones siguientes.

**¿Por qué `col_names` en lugar de renombrar las columnas con posición fija?** Si el orden de columnas cambia en futuras versiones del Excel, el mapeo por `df_raw.columns[i]` seguiría siendo correcto. Si usáramos nombres fijos como `col_1`, `col_2`, habría que actualizar manualmente.

**Output real**:
```
Shape: (1000, 22)
Target -> 0 (bueno): 700  |  1 (malo): 300
```

---

### Celda 5 — Inspección General del Dataset

**¿Qué hace?**
- `df.head()`: muestra las primeras filas para confirmar que los nombres de columnas quedaron bien asignados y que los valores tienen formato correcto.
- `df.dtypes`: confirma que numéricas quedaron como `int64` y categóricas como `object`. Si una numérica quedara como `object`, habría que convertirla antes de modelar.
- `df.isnull().sum()`: cuenta valores nulos por columna — solo imprime las columnas con nulos, si existen.
- `df.duplicated().sum()`: cuenta filas duplicadas exactas.

**Output real**:
```
Nulos: 0  |  Duplicados: 0
```
Dataset completamente limpio. No requiere tratamiento previo de datos faltantes ni duplicados.

---

### Celda 7 — Estadísticas Descriptivas

**¿Qué hace?**
- `df[num_cols].describe().round(2)`: calcula count, mean, std, min, Q1, Q2 (mediana), Q3, max para las 7 variables numéricas.
- `.round(2)` para legibilidad.

**¿Por qué solo numéricas?** Las categóricas tienen su propio análisis con `value_counts()` en la celda de univariado. Las estadísticas del `describe()` no son informativas para variables codificadas como strings (A11, A32, etc.).

**Output real** (valores clave):

| Variable | Media | Mediana | Diferencia | Señal |
|---|---|---|---|---|
| duracion_meses | 20.90 | 18.00 | +2.9 | Sesgo positivo — cola de créditos largos |
| monto_credito | 3.271 | 2.320 | +951 | Sesgo fuerte — pocos créditos muy grandes |
| tasa_cuota | 2.97 | 3.00 | ~0 | Distribución simétrica, discreta |
| edad | 35.55 | 33.00 | +2.5 | Sesgo positivo — concentración en jóvenes |
| num_dependientes | 1.16 | 1.00 | +0.16 | 84% tiene valor 1, 16% tiene valor 2 |

La diferencia entre media y mediana anticipa el sesgo: `monto_credito` con +951 DM de diferencia ya adelanta los outliers que veremos en el boxplot.

---

### Celda 9 — Distribución del Target

**¿Qué hace?**
- Genera un barplot (conteos absolutos) y un pie chart (proporciones) del target.
- `colores = ['#66b2b2', '#e07b7b']`: azul-teal para buenos, rojo-rosado para malos. Este esquema de colores se mantiene consistente en todo el notebook.

**¿Por qué barplot Y pie chart?** El barplot facilita leer conteos absolutos (700 vs 300). El pie chart facilita leer la proporción (70% / 30%). Son complementarios.

**Output real**: 700 buenos (70%) / 300 malos (30%) — desbalance moderado. Un clasificador naive que siempre diga "bueno" tendría 70% de accuracy pero costo real = 300 × 5 = **1.500 unidades** (usando la matriz asimétrica).

---

### Celda 12 — Análisis Univariado: Variables Numéricas

**¿Qué hace?**
Para cada variable numérica genera dos paneles lado a lado:

**Panel izquierdo — Histograma + KDE**:
- `ax.twinx()`: crea un segundo eje Y en el mismo gráfico para superponer el KDE sobre el histograma sin mezclar las escalas de frecuencia y densidad.
- La KDE (Kernel Density Estimate) es la versión suavizada del histograma — muestra la forma continua de la distribución.
- Líneas `axvline` para media (rojo) y mediana (naranja): si están separadas, hay sesgo. Si la mediana está a la izquierda de la media, el sesgo es positivo (cola derecha).

**Panel derecho — Boxplot**:
- La caja representa el 50% central (IQR = Q3 − Q1).
- Los bigotes llegan hasta Q1 − 1.5×IQR y Q3 + 1.5×IQR.
- Los puntos fuera de los bigotes son "outliers por definición IQR". No necesariamente son errores — en `monto_credito` son créditos grandes pero válidos.
- El título del boxplot incluye el skewness (calculado con `data.skew()`) y el conteo de outliers.

**Hallazgos clave**:
- `monto_credito`: skew=1.950, 72 outliers. La media (~3.271 DM) está fuertemente inflada por los créditos grandes. La mediana (~2.320 DM) es más representativa del cliente típico.
- `num_dependientes`: skew=1.909, 155 "outliers". Artefacto: IQR = Q3 − Q1 = 1 − 1 = 0. Cualquier desvío de la mediana queda fuera del bigote. No son errores de datos.
- `tasa_cuota` y `anios_residencia`: variables discretas (valores 1–4), sin outliers. Los histogramas muestran barras discretas, no distribución continua.

---

### Celda 14 — Análisis Univariado: Variables Categóricas

**¿Qué hace?**
- Para cada variable categórica genera un barplot con frecuencia absoluta y porcentaje sobre cada barra.
- `cat_labels`: diccionario que traduce los códigos originales (A11, A32...) a etiquetas legibles ("< 0 DM", "Sin cuenta"...). Construido manualmente a partir de `german_clean.docx`.
- Las barras están coloreadas con `sns.color_palette('Set2')` — una paleta diferente por categoría para distinguirlas visualmente.
- `axes[i].set_ylim(0, conteo_cat.max() * 1.25)`: deja espacio en la parte superior para las etiquetas de porcentaje.

**Hallazgos clave**:
- `trabajador_extranjero`: 963 de 1.000 (96.3%) son extranjeros. Variable casi constante — la categoría A202 (alemán) tiene solo 37 casos.
- `otros_deudores`: 907 de 1.000 (90.7%) no tienen co-deudores. Concentración en una sola categoría.
- `ahorros` A61: 603 clientes (60.3%) tienen menos de 100 DM en ahorros — señal estructural de bajo colchón financiero en la muestra.

---

### Celda 17 — Bivariado: Numéricas vs Target (Mann-Whitney)

**¿Qué hace?**
- Boxplots lado a lado para buenos (azul) y malos (rojo) de cada variable numérica.
- **Mann-Whitney U Test** en lugar de t-test:
  - El t-test asume normalidad de las distribuciones.
  - Mann-Whitney es no paramétrico: compara rangos, no medias. Apropiado para distribuciones sesgadas como `monto_credito`.
  - H₀: las distribuciones de buenos y malos son iguales. Si p < 0.05, la variable discrimina.
- El nivel de significancia se muestra en el título de cada gráfico: `***` (p<0.001), `**` (p<0.01), `*` (p<0.05), `ns` (no significativo).

**Output real**:

| Variable | p-value | Significancia | Conclusión |
|---|---|---|---|
| duracion_meses | <0.001 | *** | Muy significativa — malos tienen plazos más largos |
| edad | 0.0004 | *** | Malos son más jóvenes en promedio |
| monto_credito | 0.0059 | ** | Malos solicitan más monto |
| tasa_cuota | 0.0199 | * | Diferencia pequeña pero significativa |
| anios_residencia | 0.936 | ns | No discrimina |
| num_creditos_banco | 0.135 | ns | No discrimina |
| num_dependientes | 0.924 | ns | No discrimina |

Las tres variables no significativas son candidatas a descarte — confirmado después por el IV.

---

### Celda 19 — Bivariado: Categóricas vs Target (Chi² + Bad Rate)

**¿Qué hace?**
- Para cada variable categórica calcula el **bad rate por categoría** (`df.groupby(col)['target'].mean()`).
- Las barras se colorean según nivel de riesgo: rojo ≥ 40%, naranja 25–40%, verde < 25%.
- Línea horizontal punteada: bad rate global del dataset (30%) — referencia visual para ver qué categorías están por encima o por debajo del promedio.
- **Chi² de independencia**: H₀ = la variable y el target son independientes. Si p < 0.05, hay asociación estadística.

**¿Por qué chi² para categóricas?** Mann-Whitney es para numéricas vs numérica. Para categórica vs binaria, chi² es la prueba estándar: compara la distribución observada de buenos/malos en cada categoría vs la distribución esperada bajo independencia.

**Output real** (highlights):

| Variable | p-valor | Spread |
|---|---|---|
| historial_credito | <0.001 | 45.4 pts (A30=62.5% vs A34=17.1%) |
| cuenta_corriente | <0.001 | 37.6 pts (A11=49.3% vs A14=11.7%) |
| proposito | <0.001 | 32.9 pts (A46=44% vs A41=11%) |
| tipo_trabajo | 0.597 (ns) | 6.5 pts — no discrimina |
| telefono | 0.279 (ns) | 3.4 pts — no discrimina |

---

### Celda 22 — Correlaciones entre Variables Numéricas

**¿Qué hace?**
- Calcula la **matriz de correlación de Pearson** entre las 7 variables numéricas.
- `np.triu(..., dtype=bool)` como máscara: oculta el triángulo superior para no repetir la misma correlación dos veces.
- `cmap='RdBu_r'`: azul = correlación negativa, rojo = positiva, blanco = sin correlación.
- Imprime solo los pares con |r| > 0.30 para facilitar la lectura textual.

**¿Por qué importa la multicolinealidad?** En regresión logística, dos features muy correlacionadas entre sí aportan información redundante y desestabilizan los coeficientes (sus intervalos de confianza se amplían). El umbral común de preocupación es |r| > 0.70.

**Output real**:

| Par | r de Pearson | Interpretación |
|---|---|---|
| duracion_meses vs monto_credito | 0.625 | Créditos más grandes = plazos más largos. Esperado. |
| monto_credito vs tasa_cuota | -0.271 | Correlación débil, sin impacto |
| anios_residencia vs edad | 0.266 | Correlación débil, sin impacto |

**Conclusión**: no hay multicolinealidad severa (ningún |r| > 0.70). La correlación 0.625 es alta pero esperada y no justifica descartar ninguna variable.

---

### Celda 25 — Cálculo de WoE e IV para 20 Variables

**¿Qué hace?** Implementa dos funciones:

**`calcular_woe_iv_categorica(df, col, ...)`**:
- **Agrupación de raras (<3%)**: categorías con menos del 3% de observaciones se agrupan en "OTRO". En 1.000 registros esto equivale a < 30 casos. Con tan pocos casos, el WoE es estadísticamente inestable (alta varianza muestral).
- **Suavizado Laplace (smoothing=0.5)**: suma 0.5 a numerador y denominador. Sin esto, una categoría con 0 buenos o 0 malos haría `log(0)` → infinito. El suavizado mantiene los valores en rango finito con mínimo impacto en categorías bien pobladas.
- **WoE_i = ln(dist_buenos_i / dist_malos_i)**: donde `dist_buenos_i = (buenos en bin i + 0.5) / (total buenos + 0.5 × n_bins)`. WoE > 0 indica que ese bin tiene más buenos que el promedio del dataset; WoE < 0, más malos.
- **IV = Σ(dist_b_i − dist_m_i) × WoE_i**: suma de todas las contribuciones de cada bin. Es la métrica agregada de discriminación de la variable.

**`calcular_woe_iv_numerica(df, col, ...)`**:
- Usa `pd.qcut(col, q=5, duplicates='drop')`: divide el rango en 5 grupos con aproximadamente igual número de observaciones (~200 por bin). Mejor que `pd.cut` (igual ancho) porque con distribuciones sesgadas como `monto_credito`, el igual ancho dejaría bins vacíos.

**Umbrales de interpretación del IV** (Siddiqui, 2012):

| IV | Interpretación |
|---|---|
| < 0.02 | No predictiva — ruido puro |
| 0.02–0.10 | Débil — algo de señal |
| 0.10–0.30 | Moderado — buen predictor |
| 0.30–0.50 | Fuerte — muy predictiva |
| > 0.50 | Sospechoso — posible data leakage |

**Output real** (completo):
```
cuenta_corriente:      IV=0.6591  Sospechoso  ✗ EXCLUIDA
historial_credito:     IV=0.2908  Moderado    ✓
duracion_meses:        IV=0.2132  Moderado    ✓
ahorros:               IV=0.1879  Moderado    ✓
proposito:             IV=0.1520  Moderado    ✓
propiedad:             IV=0.1118  Moderado    ✓
monto_credito:         IV=0.0921  Débil       ✓
empleo_desde:          IV=0.0857  Débil       ✓
vivienda:              IV=0.0838  Débil       ✓
edad:                  IV=0.0673  Débil       ✓
otros_planes_cuota:    IV=0.0584  Débil       ✓
estado_civil_sexo:     IV=0.0448  Débil       ✓
trabajador_extranjero: IV=0.0393  Débil       ✓
otros_deudores:        IV=0.0311  Débil       ✓
tasa_cuota:            IV=0.0252  Débil       ✓
tipo_trabajo:          IV=0.0090  No pred.    ✗
telefono:              IV=0.0063  No pred.    ✗
num_creditos_banco:    IV=0.0029  No pred.    ✗
anios_residencia:      IV=0.0002  No pred.    ✗
num_dependientes:      IV=0.0000  No pred.    ✗

Seleccionadas: 14 / 20 variables
```

**¿Por qué excluir `cuenta_corriente` con IV=0.659?** En la industria de credit scoring, IV > 0.60 se considera sospechoso en datasets de < 10.000 registros porque puede reflejar:
- **Data leakage**: la variable ya incorpora información del outcome (el banco puede haber cerrado la cuenta de quien entró en mora).
- **Sesgo de selección**: el banco ya rechazó los peores casos antes de que lleguen al dataset, inflando artificialmente la capacidad predictiva.

---

### Celda 27 — Gráfico de Ranking IV

**¿Qué hace?**
- Barplot horizontal con las 20 variables ordenadas por IV descendente.
- Las barras se colorean según el umbral: gris (no predictiva), naranja (débil), verde (moderado/fuerte), rojo oscuro (sospechoso).
- Líneas verticales punteadas: `IV_MIN=0.02` (umbral mínimo) y `IV_MAX=0.60` (umbral máximo).
- `ax.invert_yaxis()`: la variable con mayor IV aparece en la parte superior del gráfico.

**¿Para qué sirve este gráfico?** Permite identificar de un vistazo cuántas variables tienen señal real y dónde está la "frontera de ruido". La separación visual entre la zona naranja y la zona gris indica qué variables el modelo ignorará.

---

### Celda 29 — WoE Plots (Top 6 Variables)

**¿Qué hace?**
- Para las 6 variables con mayor IV (que quedaron seleccionadas), genera un gráfico con:
  - **Barras de WoE** (eje izquierdo): verde si WoE > 0 (más buenos que malos en ese bin), rojo si WoE < 0. La magnitud indica qué tan diferente es ese bin del promedio.
  - **Línea de bad rate** (eje derecho, via `ax.twinx()`): el porcentaje real de malos en ese bin.
  - **Línea horizontal en WoE=0**: el punto neutro (ese bin tiene exactamente la proporción promedio del dataset).

**¿Para qué sirve el WoE?** Muestra en qué dirección y con qué intensidad discrimina cada categoría o bin dentro de una variable. Permite identificar si la variable tiene un patrón monótono (más riesgo con más valor) o no monótono (algunos bins intermedios tienen mayor riesgo).

---

### Celda 32 — Preprocesamiento y Split Estratificado

**¿Qué hace?**

**Selección de features**: solo las 14 variables con IV entre 0.02 y 0.60.

**Imputación preventiva**: aunque el dataset no tiene nulos, se incluye para que el código sea reutilizable. Para numéricas: `fillna(mediana)` — robusta a outliers. Para categóricas: `fillna('MISSING')` — trata el nulo como una categoría propia.

**One-hot encoding** con `pd.get_dummies`:
- `drop_first=True`: elimina la primera categoría de cada variable para evitar la "trampa de la variable dummy". Sin esto, la suma de todas las dummies de una variable siempre = 1, creando multicolinealidad perfecta con el intercepto.
- Convierte las 14 variables seleccionadas (7 numéricas sin cambio + 7 categóricas expandidas) en 38 features binarias.

**Split 60/20/20 estratificado** en dos pasos:
1. `train_test_split(test_size=0.20, stratify=y)` → separa el 20% de test.
2. `train_test_split(test_size=0.25, stratify=y_tmp)` → del 80% restante, separa 25% = 20% del total como validación.
- `stratify=y`: garantiza que la proporción 70/30 se mantenga en los 3 conjuntos.

**¿Por qué tener un set de validación además de test?** El threshold óptimo se busca en validación (no en test). Si usáramos test para buscar el threshold, estaríamos ajustando el modelo al test — data leakage implícito.

**StandardScaler**:
- `scaler.fit_transform(X_train)`: aprende la media y std del train.
- `scaler.transform(X_valid)`, `.transform(X_test)`: aplica esa misma transformación sin re-aprender.
- **Regla crítica**: nunca ajustar el scaler sobre valid o test. El test debe simular datos que el modelo nunca ha visto.
- `with_mean=False`: compatibilidad con matrices sparse. No afecta el resultado en este caso (la matriz es dense), pero es buena práctica.

**Output real**:
```
Train: (600, 38)  |  Valid: (200, 38)  |  Test: (200, 38)
Bad rate  train: 30.0%  |  valid: 30.0%  |  test: 30.0%
```

---

### Celda 35 — Entrenamiento Regresión Logística L1

```python
logit = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    class_weight='balanced',
    C=0.1,
    max_iter=500,
    random_state=42
)
```

**Justificación de cada hiperparámetro**:

| Parámetro | Valor | Por qué |
|---|---|---|
| `penalty='l1'` | L1 (Lasso) | Lleva coeficientes irrelevantes **exactamente** a 0, actuando como selección automática de features dentro del modelo. L2 solo achica los coeficientes pero no los elimina. |
| `solver='liblinear'` | liblinear | El único solver de sklearn compatible con `penalty='l1'` en clasificación binaria. Los solvers como `lbfgs` o `saga` no soportan L1 con `liblinear`. |
| `class_weight='balanced'` | balanced | Asigna pesos inversamente proporcionales a la frecuencia de cada clase: clase bueno → peso ~0.71, clase malo → peso ~1.67. Compensa el desbalance 70/30 sin modificar los datos. |
| `C=0.1` | 0.1 | `C = 1/λ` (inverso de la regularización). C pequeño = regularización fuerte. Con 600 casos de train y 38 features, necesitamos penalización alta para evitar sobreajuste. |
| `max_iter=500` | 500 | Iteraciones máximas para convergencia. El default (100) puede ser insuficiente con regularización fuerte y muchas features. |
| `random_state=42` | 42 | Reproducibilidad — garantiza el mismo resultado en cada ejecución. |

**Output real**: 23 coeficientes no nulos de 38 → L1 eliminó 15 features (39.5% de las features descartadas automáticamente).

---

### Celda 37 — Cross-Validation LogReg L1

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_cv_logit = cross_val_score(logit_param, X_train_sc, y_train, cv=cv, scoring='roc_auc')
```

**¿Por qué cross-validation?** Con 600 casos de train, la estimación de AUC sobre un único hold-out de 120 casos tiene alta varianza. El AUC observado puede ser "afortunado" o "desafortunado" dependiendo de qué 120 casos cayeron en ese fold. El CV promedia 5 estimaciones independientes, dando una estimación más robusta.

**¿Por qué StratifiedKFold?** El dataset tiene 70/30 de desbalance. Con KFold simple, algunos folds podrían tener muy pocos malos (por azar) y dar estimaciones sesgadas. `StratifiedKFold` garantiza que cada fold tenga ~30% de malos.

**`shuffle=True`**: mezcla los datos antes de dividir en folds. Si los datos estuvieran ordenados por alguna variable (ej. cronológicamente), no mezclarlos podría introducir sesgo.

**`scoring='roc_auc'`**: métrica relevante para problemas con desbalance de clases. Mide la capacidad discriminante del modelo independientemente del threshold.

**¿Por qué CV sobre `X_train_sc` y no `X_train_sc + X_valid_sc`?** El set de validación se reserva para la búsqueda del threshold óptimo (S9). Si también lo usáramos en CV, el threshold "vería" los mismos datos sobre los que fue buscado — data leakage sutil.

**Output real**: Media ~0.71 ± std ~0.03 → modelo estable, AUC en test consistente con CV.

---

### Celda 40 — Evaluación con Threshold=0.50

**¿Qué hace?**
- `(y_prob_test >= 0.5).astype(int)`: aplica el threshold default de sklearn. Para cada cliente, si la probabilidad predicha de ser malo ≥ 50%, se clasifica como malo.
- Imprime `classification_report`: precision, recall y F1-score por clase.
- Calcula AUC-ROC con `roc_auc_score` y PR-AUC con `average_precision_score`.
- Genera la matriz de confusión como heatmap.

**¿Por qué AUC-ROC y no accuracy?** Con 70/30 de desbalance, el accuracy es engañoso. Un modelo que siempre diga "bueno" tiene 70% de accuracy pero detecta 0 malos. AUC-ROC es independiente del threshold y del desbalance.

**¿Por qué PR-AUC además de ROC?** La curva Precision-Recall es más informativa que la ROC cuando la clase positiva es minoría (30%). El AUC-ROC puede parecer alto en datasets desbalanceados incluso con muchos falsos positivos. La PR-AUC "castiga" más los falsos positivos.

**Output real**:
```
Accuracy : 0.6250
AUC-ROC  : 0.7102
PR-AUC   : 0.5443
Costo (threshold=0.50): 5×22 + 1×53 = 163
```

---

### Celda 42 — Curvas ROC y Precision-Recall

**¿Qué hace?**
- `roc_curve(y_test, y_prob_test)`: calcula TPR (recall) y FPR para todos los thresholds posibles.
- `precision_recall_curve(y_test, y_prob_test)`: calcula precision y recall para todos los thresholds.
- `fill_between`: área sombreada bajo las curvas para facilitar la comparación visual del AUC.
- `ax.axhline(y_test.mean(), ...)`: línea de baseline para la curva PR. Un clasificador aleatorio en un dataset 70/30 tendría PR-AUC = 0.30 (igual que la prevalencia de la clase positiva).

**Cómo leer la curva ROC**: el punto (0,1) es el clasificador perfecto (100% recall, 0% FPR). La diagonal es el clasificador aleatorio (AUC=0.5). Cualquier curva por encima de la diagonal tiene poder discriminante.

**Cómo leer la curva PR**: alta precision + alto recall = clasificador ideal. En la práctica hay trade-off: aumentar el recall baja la precision. La curva muestra ese trade-off para todos los thresholds posibles.

---

### Celda 45 — Comparación de Modelos: 6 arquitecturas + CV

```python
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, algorithm='SAMME', random_state=42)
if XGBOOST_AVAILABLE:
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                         scale_pos_weight=sum(y_train==0)/sum(y_train==1), ...)
```

**Justificación de hiperparámetros**:

| Modelo | Parámetro clave | Valor | Por qué |
|---|---|---|---|
| DT | `max_depth=5` | 5 | Sin restricción, un árbol individual con 38 features memoriza el train. |
| DT/RF | `min_samples_leaf=20` | 20 | Hojas con <20 casos tienen estimaciones de probabilidad inestables. |
| RF | `n_estimators=100` | 100 | Balance estabilidad/cómputo. Aumentar a 500 no mejora AUC en 1000 registros. |
| GB | `max_depth=3` | 3 | Gradient Boosting usa árboles pequeños (*stumps* poco profundos): cada árbol corrige el error residual del anterior. Árboles más profundos → sobreajuste rápido. |
| GB | `learning_rate=0.1` | 0.1 | Shrinkage estándar: cada árbol contribuye solo el 10% de su ajuste → convergencia más suave y menos sobreajuste. |
| GB | sin `class_weight` | — | `GradientBoostingClassifier` de sklearn no soporta `class_weight`. El desbalance se maneja implícitamente por la función de pérdida. |
| AdaBoost | `learning_rate=0.5` | 0.5 | Reducir la contribución de cada learner débil aumenta la estabilidad del ensemble en muestras pequeñas. |
| AdaBoost | `algorithm='SAMME'` | SAMME | La variante multi-class de AdaBoost. SAMME.R (la alternativa) requiere probabilidades confiables de learners muy simples — no siempre disponible. |
| XGBoost | `scale_pos_weight` | n_neg/n_pos | Equivalente a `class_weight='balanced'`: pondera más los errores sobre la clase minority (malos). Se calcula como `n_buenos / n_malos = 700/300 ≈ 2.33`. |
| XGBoost | `eval_metric='logloss'` | logloss | Silencia el warning de XGBoost sobre la métrica por defecto en clasificación binaria. |

**¿Por qué `max_depth=3` para GB y XGBoost y `max_depth=5` para DT/RF?** Boosting y bagging tienen lógicas distintas. Bagging (RF) usa árboles profundos con alta varianza y los promedia. Boosting usa árboles poco profundos (stumps) y los encadena secuencialmente: cada uno corrige el error del anterior. Árboles muy profundos en boosting → sobreajuste rápido porque cada árbol ya se ajusta demasiado al residuo.

**Cross-validation**: el mismo objeto `cv` (StratifiedKFold 5-fold, `random_state=42`) definido en S7 se reutiliza para todos los modelos. Esto garantiza que los 6 modelos se evalúen exactamente sobre las mismas 5 particiones — sin varianza por partición aleatoria.

---

### Celda 46 — Tabla Comparativa + Curvas ROC superpuestas

**¿Qué hace?**
- Loop genérico sobre `modelos_cmp` (dict con los 6 modelos): entrena cada uno en `X_train_sc` completo (600 casos) y evalúa en `X_test_sc` (200 casos, nunca visto durante desarrollo).
- Construye `df_cmp` con AUC CV (media ± std), AUC test y Gap CV-Test para todos los modelos.
- Genera un gráfico con las 6 curvas ROC superpuestas sobre el mismo test set usando colores distintos — comparación visual directa de la capacidad discriminante.
- El **Gap CV-Test** (AUC CV mean − AUC test) es un indicador de sobreajuste: gap > 0.05 indica que el modelo generaliza peor de lo que CV sugería.

**¿Por qué entrenar en train (600) y no en train+valid (800)?** La separación train/valid/test es deliberada. El válido se reserva para buscar el threshold óptimo (S9). Si se usara para entrenar, el threshold "vería" datos de entrenamiento — data leakage sutil que inflaría el costo estimado en validación.

### Celda 48 — SHAP values para XGBoost

```python
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test_sc)
shap.summary_plot(shap_values, X_test_sc, ...)
```

**¿Qué hace?**
- `shap.TreeExplainer(xgb)`: crea un explainer específico para modelos basados en árboles. TreeExplainer usa el algoritmo de Shapley values exacto (no aproximado) aprovechando la estructura de árbol — mucho más rápido que KernelExplainer.
- `explainer.shap_values(X_test_sc)`: calcula el SHAP value de cada feature para cada observación del test set. Para 200 observaciones y 38 features, devuelve una matriz 200×38.
- **Beeswarm plot**: cada punto es una observación. El eje X muestra el SHAP value (impacto en la predicción del modelo). El color muestra el valor real de la feature. Permite ver *dirección* (positivo = aumenta el riesgo) y *magnitud* a la vez.
- **Comparación IV vs SHAP**: compara el top 10 de variables por IV (calculado pre-modeling) contra el top 10 por importancia SHAP (calculado post-modeling). Una coincidencia alta valida que IV es un proxy robusto de importancia real.

**¿Por qué SHAP y no feature_importances_ de sklearn?** `feature_importances_` de RF/GB da la importancia promedio en *entrenamiento* — sobreestima variables que el modelo memorizó. SHAP calcula el impacto marginal de cada feature en la *predicción individual*, usando el test set — más representativo de la generalización real.

**¿Por qué comparar IV con SHAP?** Es la validación cruzada metodológica del trabajo: si el análisis de Information Value realizado *antes de entrenar cualquier modelo* identifica las mismas variables que SHAP identifica *después de entrenar el modelo más complejo*, confirma que la selección de features pre-modeling fue correcta y robusta.

**Output esperado**: mean |SHAP| ≈ 0.10-0.25 para las features más importantes (historial_credito, duracion_meses, ahorros); coincidencia IV-SHAP de 8-9 variables sobre las top 10.

---

### Celda 49 — Búsqueda del Threshold Óptimo

**¿Qué hace?**
- Define `COSTO_FN = 5` y `COSTO_FP = 1`: los costos de la matriz asimétrica del problema.
- `costo_total(y_true, y_prob, threshold)`: dado un threshold, clasifica y calcula `5×FN + 1×FP`.
- Evalúa 91 thresholds equiespaciados entre 0.05 y 0.95 sobre el **conjunto de validación**.
- Plotea la curva de costo vs threshold: forma de U asimétrica — el mínimo está a la izquierda de 0.50.

**¿Por qué buscar el threshold en validación y no en test?** Si usáramos el test para encontrar el threshold óptimo, estaríamos ajustando el modelo al test — data leakage implícito. El test debe quedar completamente virgen hasta la evaluación final. La separación train/valid/test existe precisamente para este caso.

**¿Por qué el threshold óptimo está en ~0.26 (izquierda de 0.50)?** Con el ratio 5:1, el modelo prefiere clasificar como malo a más personas (bajo threshold) para evitar los FN costosos. Matemáticamente, el threshold óptimo bajo la función de costo lineal está alrededor de `1 / (1 + costo_ratio)` = `1 / (1 + 5)` = 0.167, pero la distribución de probabilidades predichas lo desplaza hacia ~0.26 en la práctica.

---

### Celda 51 — Evaluación Final en Test

**¿Qué hace?**
- Aplica el threshold óptimo encontrado en validación al conjunto de test.
- Compara 3 escenarios: baseline (aprobar todo), threshold 0.50, threshold óptimo.
- Genera dos matrices de confusión lado a lado para comparación visual.

**Output real**:

| Escenario | TN | FP | FN | TP | Costo |
|---|---|---|---|---|---|
| Baseline (aprobar todo) | 140 | 0 | 60 | 0 | **300** |
| Threshold 0.50 | 87 | 53 | 22 | 38 | **163** |
| **Threshold óptimo 0.26** | **28** | **112** | **5** | **55** | **137** |

**Lectura**: con threshold=0.26, el modelo rechaza a 112 buenos clientes (FP) pero evita 55 créditos incobrables (TP). El costo de 5 por cada FN justifica ese sacrificio: 55 × 5 = 275 unidades de ahorro, a un costo de 112 × 1 = 112 unidades por rechazos incorrectos.

---

### Celda 54 — Odds Ratios

**¿Qué hace?**
- Construye `coef_df` con los coeficientes no nulos del modelo (los que L1 no eliminó).
- `Odds_Ratio = np.exp(coeficiente)`: convierte el log-odds a escala multiplicativa.
- `coef_df.reindex(coef_df['Coeficiente'].abs().sort_values(ascending=False).index)`: ordena por magnitud absoluta del coeficiente — los más influyentes primero.
- Gráfico horizontal: barras verdes (OR < 1, reduce riesgo) y rojas (OR > 1, aumenta riesgo). Línea vertical en OR=1 (sin efecto).

**Cómo interpretar un Odds Ratio**:
- `OR = 1.44` → esa feature multiplica las chances de ser malo por 1.44 (+44% de riesgo).
- `OR = 0.68` → esa feature multiplica las chances de ser malo por 0.68 (-32% de riesgo).
- `OR = 1.00` → esa feature no tiene efecto sobre el riesgo.

**Output real** (top 10):

| Feature | OR | Interpretación |
|---|---|---|
| historial_credito_A34 | 0.677 | Cuenta crítica → −32.3% de riesgo |
| duracion_meses | 1.441 | Cada mes extra: ×1.44 el riesgo |
| otros_planes_cuota_A143 | 0.740 | Sin otros planes: −26% de riesgo |
| ahorros_A65 | 0.753 | Sin ahorros conocidos: −24.7% |
| tasa_cuota | 1.281 | Cada punto de tasa: +28.1% riesgo |
| ahorros_A64 | 0.784 | ≥1000 DM ahorros: −21.6% riesgo |

---

### Celda 57 — Cálculo del Scorecard

```python
factor = PDO / np.log(2)
offset = SCORE_REF - factor * np.log(ODDS_REF)
score = offset - factor * log_odds
```

**¿Qué es el scorecard?** En la industria de credit scoring, el modelo no le entrega al analista una probabilidad cruda (0.32) sino un **puntaje** en una escala comprensible. El scorecard es la transformación matemática que convierte el log-odds del modelo en un puntaje.

**Parámetros estándar**:
- `PDO = 20`: "Points to Double Odds" — cada 20 puntos de caída en el score duplica las chances de mora. Convención industrial.
- `SCORE_REF = 600`: punto de anclaje de la escala. A ese puntaje las odds son `ODDS_REF`.
- `ODDS_REF = 50`: a score=600 las chances son 50 buenos por 1 malo.

**¿Por qué `score = offset − factor × log_odds`?** El signo negativo garantiza que a mayor log-odds de mora (mayor probabilidad de ser malo), menor es el score. Mayor score = mejor pagador.

**Output real**:
```
Factor : 28.8539
Offset : 487.1229
Score mínimo : 423  (cliente de altísimo riesgo)
Score máximo : 572  (cliente de bajísimo riesgo)
```

---

### Celda 58 — Histograma de Scores por Clase

**¿Qué hace?**
- Superpone histogramas de scores para buenos (azul) y malos (rojo) con `alpha=0.7` para ver el solapamiento.
- `score_corte = prob_a_score(threshold_optimo)`: traduce el threshold óptimo (0.26) al lenguaje de puntajes. Este es el corte operacional que usaría el banco.
- Línea vertical: score de referencia (600) y score de corte.

**Output real**:
```
Score de corte: 517 puntos
Aprobados (score > 517):  33 de 200 (16.5%)
Rechazados (score ≤ 517): 167 de 200 (83.5%)
Media buenos: 497.1  |  Media malos: 476.6
Diferencia:  20.5 puntos
```

La separación de 20.5 puntos entre buenos y malos es moderada. El solapamiento visible en el histograma refleja las limitaciones del dataset (muestra pequeña, pocas variables de ingresos).

---

## Decisiones de implementación y justificaciones

| Decisión | Justificación |
|---|---|
| Suavizado Laplace (smoothing=0.5) | Evita log(0) en WoE cuando una categoría tiene 0 buenos o 0 malos. Sin suavizado, el IV de esa variable sería infinito. |
| Agrupación de categorías raras (<3% → "OTRO") | Bins con <30 casos tienen alta varianza en WoE. Agrupar mejora la estabilidad sin perder información relevante. |
| `pd.qcut` (quantile bins) para numéricas | Garantiza igual n por bin. `pd.cut` (igual ancho) dejaría bins vacíos con distribuciones sesgadas como `monto_credito`. |
| IV_MIN=0.02, IV_MAX=0.60 | <0.02: ruido estadístico. >0.60: sospecha de data leakage o concentración artificial de información. |
| `stratify=y` en train_test_split | Garantiza bad rate 30% en los 3 conjuntos. Sin estratificar, por azar un conjunto podría tener 25% o 35% de malos. |
| Split en dos pasos para lograr 60/20/20 | `train_test_split` solo hace splits en dos partes. Para 3 partes se aplica dos veces. El orden importa: primero separar test, luego valid. |
| `fit_transform` en train, `transform` en valid/test | Evita data leakage: valid y test no deben informar la media/std de normalización. El scaler aprende solo del train. |
| `drop_first=True` en `get_dummies` | Evita la "trampa de la variable dummy": la suma de todas las dummies de una variable siempre = 1 → multicolinealidad perfecta con el intercepto. |
| `penalty='l1'` en lugar de `l2` | L1 lleva coeficientes exactamente a 0 (selección de features). L2 solo los achica. Con 38 features y 600 casos, L1 es más adecuado. |
| `C=0.1` (regularización fuerte) | Con 600 casos de train y 38 features, C muy alto (baja regularización) haría sobreajuste. C=0.1 es un valor razonable para este ratio features/observaciones. |
| `class_weight='balanced'` | Compensa el desbalance 70/30 sin SMOTE ni resampling. Más simple y menos propenso a artefactos. |
| Threshold buscado en validación, no en test | Si usáramos test para buscar el threshold, el test dejaría de ser "datos nunca vistos" — data leakage. |
| `max_depth=3` para GB y XGBoost (vs 5 para DT/RF) | Boosting usa árboles pequeños encadenados: cada árbol corrige el error residual del anterior. Árboles profundos en boosting → sobreajuste rápido. Bagging (RF) usa árboles profundos y los promedia. |
| `scale_pos_weight` en XGBoost en lugar de `class_weight` | XGBClassifier no acepta `class_weight`. `scale_pos_weight = n_neg/n_pos` es el equivalente nativo de XGBoost para manejar desbalance de clases. |
| SHAP sobre test set, no sobre train | Los SHAP values calculados sobre train reflejan lo que el modelo *memorizó*. Los calculados sobre test reflejan cómo el modelo *generaliza* — más representativo del comportamiento en producción. |
| `shap.TreeExplainer` en lugar de `KernelExplainer` | TreeExplainer usa el algoritmo exacto de Shapley values aprovechando la estructura de árbol → 10-100x más rápido. KernelExplainer es agnóstico al modelo pero aproximado y lento. |
| `max_depth=5` y `min_samples_leaf=20` en DT/RF | Con 600 casos, árboles sin restricción memorizan el train. Estas restricciones controlan la complejidad y mejoran la generalización. |
| `StratifiedKFold(shuffle=True)` | El shuffle mezcla antes de dividir. Sin shuffle, si los datos tienen orden temporal o por variable, los folds serían sesgados. |
| Comparar modelos sobre el mismo set de test | Usar distintos sets de test introduciría varianza por la partición. La comparación justa requiere el mismo conjunto. |
