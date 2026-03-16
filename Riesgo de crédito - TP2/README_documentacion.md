# Documentación Técnica — credit_risk.ipynb

**Notebook**: `credit_risk.ipynb` (versión 2 — mejorada para comprensión y reutilización)
**Dataset**: `Base_Clientes_Alemanes.xlsx`
**Kernel**: Python 3 (Anaconda) | **Dependencias**: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
**Ejecutar con**: `/opt/anaconda3/bin/python3` (tiene todas las dependencias instaladas)

---

## Resumen de outputs reales por sección

> Todos los valores provienen de la ejecución real del notebook sobre los 1.000 registros del dataset.

| Sección | Resultado clave |
|---|---|
| S1 — Carga | 1.000 filas, 0 nulos, 0 duplicados. Buenos: 700 (70%), Malos: 300 (30%) |
| S2 — Univariado | monto_credito skew=1.950; num_dependientes 155 outliers (artefacto de discretización, no errores) |
| S3 — Bivariado | duracion p<0.001 (***); tipo_trabajo p=0.597 (ns); historial_credito spread=45.4% |
| S4 — Correlaciones | duracion vs monto: r=0.625 (mayor correlación del dataset; sin multicolinealidad severa) |
| S5 — IV | cuenta_corriente IV=0.659 EXCLUIDA (>IV_MAX=0.60); 14/20 variables seleccionadas |
| S6 — Split | Train=600, Valid=200, Test=200; bad rate=30.0% en los 3 conjuntos (estratificación perfecta) |
| S7 — Modelo | AUC-ROC=0.7102; PR-AUC=0.5443; L1 descartó 15 features → 23/38 coef no nulos |
| S8 — Costo | Threshold óptimo=0.26 (buscado en valid, evaluado en test); costo=137 vs baseline=300 (−54.3%) |
| S9 — Odds Ratios | historial_A34 OR=0.677 (−32.3% riesgo); duracion_meses OR=1.441 (+44% por mes) |
| S10 — Scorecard | factor=28.8539; offset=487.1229; rango 423-572; corte=517 pts; aprobados=33/200 (16.5%) |

---

## Índice de Celdas

> 53 celdas en total (26 markdown + 27 código). Cada celda de código es seguida por una celda markdown
> con el resultado e interpretación inmediatamente debajo.

| # | ID | Tipo | Contenido |
|---|---|---|---|
| 0 | md-001 | Markdown | Título, objetivo, desafíos y estructura del notebook |
| 1 | md-002 | Markdown | Intro S1: qué hacer, qué esperar, nota para otros proyectos |
| 2 | cd-003 | Código | Importación de librerías y configuración global de gráficos |
| 3 | cd-004 | Código | Carga del Excel, renombre de columnas (con comentarios por código), recodificación del target |
| 4 | md-005 | Markdown | Resultado: 1.000 filas, 700/300, explicación de la recodificación 1/2 → 0/1 |
| 5 | cd-006 | Código | Inspección: head(), dtypes, nulos, duplicados |
| 6 | md-007 | Markdown | Resultado: calidad del dataset y significado de dtype object |
| 7 | cd-008 | Código | Estadísticas descriptivas (describe) de numéricas |
| 8 | md-009 | Markdown | Resultado: tabla comparativa media vs mediana, señal de sesgo por variable |
| 9 | cd-010 | Código | Gráfico distribución del target (barplot + pie chart) |
| 10 | md-011 | Markdown | Resultado: desbalance 70/30, baseline ingenuo = costo 1.500, motivación del costo asimétrico |
| 11 | md-012 | Markdown | Intro S2: herramientas (histograma, KDE, boxplot, skewness) y qué esperamos |
| 12 | cd-013 | Código | Histogramas + KDE + boxplots para las 7 variables numéricas |
| 13 | md-014 | Markdown | Resultado: skew por variable, outliers reales vs artefactos, impacto en modelos |
| 14 | cd-015 | Código | Barplots de frecuencia con etiquetas legibles — variables categóricas |
| 15 | md-016 | Markdown | Resultado: variables casi constantes (trabajador_extranjero 96.3%), categorías sin casos |
| 16 | md-017 | Markdown | Intro S3: Mann-Whitney vs t-test, chi-cuadrado, qué esperamos |
| 17 | cd-018 | Código | Boxplots por clase + Mann-Whitney — numéricas vs target |
| 18 | md-019 | Markdown | Resultado: tabla de significancia con interpretación de negocio por variable |
| 19 | cd-020 | Código | Bad rate por categoría + chi-cuadrado — categóricas vs target |
| 20 | md-021 | Markdown | Resultado: spread por variable, tipo_trabajo p=0.597 (ns), cuenta_corriente 38pts spread |
| 21 | md-022 | Markdown | Intro S4: qué es multicolinealidad, cuándo preocupa, umbral |r|>0.70 |
| 22 | cd-023 | Código | Heatmap de correlación entre numéricas |
| 23 | md-024 | Markdown | Resultado: r=0.625 duracion-monto, sin multicolinealidad severa |
| 24 | md-025 | Markdown | Intro S5: fórmulas WoE/IV, tabla de umbrales, nota para otros proyectos |
| 25 | cd-026 | Código | Funciones WoE/IV con comentarios extensos (Laplace, qcut, dist_b/dist_m) |
| 26 | md-027 | Markdown | Resultado: tabla IV completa, paradoja de cuenta_corriente, 14/20 seleccionadas |
| 27 | cd-028 | Código | Gráfico ranking IV horizontal con umbrales IV_MIN/IV_MAX coloreados |
| 28 | md-029 | Markdown | Resultado: lista de 14 variables seleccionadas con sus IVs |
| 29 | cd-030 | Código | WoE plots para las top 6 variables (barras WoE + línea bad rate) |
| 30 | md-031 | Markdown | Resultado: cómo leer WoE, monotonismo en numéricas, paradoja historial A34 |
| 31 | md-032 | Markdown | Intro S6: roles de train/valid/test, por qué 60/20/20, one-hot vs label encoding |
| 32 | cd-033 | Código | Split estratificado + one-hot encoding + StandardScaler (con comentarios sobre data leakage) |
| 33 | md-034 | Markdown | Resultado: Train=600/Valid=200/Test=200, bad rate=30% en los 3, regla del scaler |
| 34 | md-035 | Markdown | Intro S7: L1 vs L2, explicación de cada hiperparámetro (C, solver, class_weight) |
| 35 | cd-036 | Código | Entrenamiento LogisticRegression L1 con comentarios por hiperparámetro |
| 36 | md-037 | Markdown | Resultado: 23/38 no nulos, qué significa ajustar C, cómo mejorar la selección |
| 37 | cd-038 | Código | Métricas en test (threshold=0.50) + matriz de confusión |
| 38 | md-039 | Markdown | Resultado: AUC=0.7102, lectura de la matriz TN/FP/FN/TP con números reales, costo=163 |
| 39 | cd-040 | Código | Curvas ROC y Precision-Recall con área sombreada |
| 40 | md-041 | Markdown | Resultado: cómo leer ROC vs PR, diferencia entre ambas, cuándo usar cada una |
| 41 | md-042 | Markdown | Intro S8: intuición del threshold, función de costo, por qué el óptimo baja de 0.50 |
| 42 | cd-043 | Código | Búsqueda del threshold óptimo en validación (91 valores, 5×FN + 1×FP) |
| 43 | md-044 | Markdown | Resultado: threshold=0.26, curva en U asimétrica, explicación de la asimetría |
| 44 | cd-045 | Código | Evaluación final en test con threshold óptimo + comparación de matrices |
| 45 | md-046 | Markdown | Resultado: tabla comparativa 3 escenarios con números reales, reducción 54.3% |
| 46 | md-047 | Markdown | Intro S9: fórmula OR, interpretación OR>1/OR<1/OR=1, para qué sirve la auditoría |
| 47 | cd-048 | Código | Tabla y gráfico de odds ratios (solo coeficientes no nulos) |
| 48 | md-049 | Markdown | Resultado: OR concretos con traducción a lenguaje de negocio |
| 49 | md-050 | Markdown | Intro S10: qué es el scorecard, PDO, fórmulas con significado de cada término |
| 50 | cd-051 | Código | Cálculo de factor, offset y scores del test con función prob_a_score |
| 51 | cd-052 | Código | Histograma de scores por clase con score de corte equivalente al threshold |
| 52 | md-053 | Markdown | Resultado: rango 423-572, corte=517, 33/200 aprobados, separación visual |

---

## Documentación Detallada por Bloque

---

### Celda 2 — Importación de Librerías

**Librerías**: pandas, numpy, matplotlib, seaborn, scipy.stats, sklearn (train_test_split, StandardScaler, LogisticRegression, métricas).
**Config global**: figure.figsize=(10,5), sin bordes superiores/derechos, paleta Set2.
**Output**: "Librerias cargadas."

---

### Celda 3 — Carga y Preparación

**Input**: `Base_Clientes_Alemanes.xlsx`, sheet 0.

**Proceso**:
1. Lee 1.000 filas × 22 columnas (incluyendo cliente_id y target).
2. Mapea columnas 0-21 a nombres semánticos via `col_names` (basado en german_clean.docx).
3. Filtra target in {1, 2} — en este dataset no hay registros inválidos.
4. Recodifica: target=1 → 0 (bueno), target=2 → 1 (malo).

**Output real**:
```
Shape: (1000, 22)
Buenos: 700 (70.0%)  Malos: 300 (30.0%)
```

**Variables globales**: `df` (1000×22), `num_cols` (7 vars), `cat_cols` (13 vars).

---

### Celda 5 — Inspección General

**Output real**:
```
Nulos: 0  |  Duplicados: 0
```
Dataset completamente limpio. Tipos: numéricas como int64, categóricas como object (códigos tipo A11, A32).

---

### Celda 7 — Estadísticas Descriptivas

**Output real** (variables numéricas):

```
                   mean    median    std     min      max
duracion_meses    20.90     18.00  12.06    4.00    72.00
monto_credito   3271.26   2319.50 2822.74  250.00 18424.00
tasa_cuota         2.97      3.00   1.12    1.00     4.00
anios_residencia   2.85      3.00   1.10    1.00     4.00
edad              35.55     33.00  11.38   19.00    75.00
num_creditos_banco  1.41      1.00   0.58    1.00     4.00
num_dependientes   1.16      1.00   0.36    1.00     2.00
```

---

### Celda 9 — Distribución del Target

**Output real**: 700 buenos (70.0%) / 300 malos (30.0%). Barplot + pie chart.

---

### Celda 12 — Univariado Numéricas (Histograma + Boxplot)

**Output real por variable**:

| Variable | Skew | Outliers | Observación |
|---|---|---|---|
| duracion_meses | 1.094 | 70 | Sesgo positivo moderado |
| monto_credito | 1.950 | 72 | Sesgo positivo fuerte |
| tasa_cuota | -0.531 | 0 | Discreta (1-4), sin outliers |
| anios_residencia | -0.273 | 0 | Discreta (1-4), sin outliers |
| edad | 1.021 | 23 | Sesgo positivo moderado |
| num_creditos_banco | 1.273 | 6 | 83% con valor=1 |
| num_dependientes | 1.909 | 155 | Artefacto: 84.4% valor=1, 15.6% valor=2 |

---

### Celda 14 — Univariado Categóricas (Barplots)

**Output real destacado**:
- `ahorros` A61: 603 clientes (60.3%) con menos de 100 DM
- `trabajador_extranjero` A201: 963 clientes (96.3%) son extranjeros
- `otros_deudores` A101: 907 clientes (90.7%) sin otros deudores
- `estado_civil_sexo` A93: 550 clientes (55%) hombres solteros
- `tipo_trabajo` A173: 630 clientes (63%) empleados calificados

---

### Celda 17 — Bivariado Numéricas (Mann-Whitney)

**Output real**:

| Variable | Media buenos | Media malos | p-value | Sig |
|---|---|---|---|---|
| duracion_meses | 19.21 | 24.86 | 0.0000 | *** |
| edad | 36.22 | 33.96 | 0.0004 | *** |
| monto_credito | 2985.46 | 3938.13 | 0.0059 | ** |
| tasa_cuota | 2.92 | 3.10 | 0.0199 | * |
| anios_residencia | 2.84 | 2.85 | 0.9358 | ns |
| num_creditos_banco | 1.42 | 1.37 | 0.1348 | ns |
| num_dependientes | 1.16 | 1.15 | 0.9242 | ns |

---

### Celda 19 — Bivariado Categóricas (Chi2 + Bad Rate)

**Output real** (bad rates por categoría, casos más relevantes):

```
historial_credito:  A30=62.5%(n=40)  A31=57.1%(n=49)  A32=14.7%(n=530)  spread=45.4% ***
cuenta_corriente:   A11=49.3%(n=274) A12=39.0%(n=269) A14=11.7%(n=394)  spread=37.6% ***
proposito:          A46=44.0%(n=50)  A40=36.1%(n=234) A41=11.1%(n=103)  spread=32.9% ***
tipo_trabajo:       spread=6.5%  p=0.597  ns
telefono:           spread=3.4%  p=0.279  ns
```

---

### Celda 22 — Correlaciones

**Output real** (pares con |r| > 0.20):

```
duracion_meses vs monto_credito:   r = 0.625
monto_credito  vs tasa_cuota:      r = -0.271
anios_residencia vs edad:          r = 0.266
```

No se detecta multicolinealidad severa. La correlación 0.625 entre duración y monto es esperada (créditos grandes = plazos largos) y no justifica descartar ninguna.

---

### Celda 25 — Cálculo WoE/IV

**Funciones implementadas**:

`calcular_woe_iv_categorica(df, col, target='target', smoothing=0.5)`:
- Agrupa categorías con frecuencia <3% en "OTRO"
- WoE_i = ln(dist_buenos_i / dist_malos_i), con suavizado Laplace
- IV = sum((dist_b_i - dist_m_i) × WoE_i)

`calcular_woe_iv_numerica(df, col, target='target', n_bins=5, smoothing=0.5)`:
- pd.qcut con hasta 5 bins por quantiles, duplicates='drop'
- Mismo cálculo de WoE/IV

**Output real** (IV completo):

```
cuenta_corriente:    IV=0.6591  Sospechoso  ✗ EXCLUIDA
historial_credito:   IV=0.2908  Moderado    ✓
duracion_meses:      IV=0.2132  Moderado    ✓
ahorros:             IV=0.1879  Moderado    ✓
proposito:           IV=0.1520  Moderado    ✓
propiedad:           IV=0.1118  Moderado    ✓
monto_credito:       IV=0.0921  Débil       ✓
empleo_desde:        IV=0.0857  Débil       ✓
vivienda:            IV=0.0838  Débil       ✓
edad:                IV=0.0673  Débil       ✓
otros_planes_cuota:  IV=0.0584  Débil       ✓
estado_civil_sexo:   IV=0.0448  Débil       ✓
trabajador_extranjero:IV=0.0393 Débil       ✓
otros_deudores:      IV=0.0311  Débil       ✓
tasa_cuota:          IV=0.0252  Débil       ✓
tipo_trabajo:        IV=0.0090  No pred.    ✗
telefono:            IV=0.0063  No pred.    ✗
num_creditos_banco:  IV=0.0029  No pred.    ✗
anios_residencia:    IV=0.0002  No pred.    ✗
num_dependientes:    IV=0.0000  No pred.    ✗

Seleccionadas: 14 / 20 variables
```

---

### Celda 32 — Preprocesamiento y Split

**Output real**:
```
Train: (600, 38)  Valid: (200, 38)  Test: (200, 38)
Bad rates: train=30.0%  valid=30.0%  test=30.0%
Features tras one-hot encoding: 38
```

El one-hot encoding convirtió 14 variables (7 num + 7 cat seleccionadas) en 38 columnas binarias.

---

### Celda 35 — Entrenamiento Modelo

**Configuración**: `LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.1, max_iter=500, random_state=42)`

**Output real**: 23 coeficientes no nulos de 38 (el L1 eliminó el 39.5% de features automáticamente).

---

### Celda 37 — Métricas con Threshold=0.50

**Output real**:
```
Accuracy:  0.6250
AUC-ROC:   0.7102
PR-AUC:    0.5443

Matriz de confusión:
           Pred Bueno   Pred Malo
Real Bueno    TN=87       FP=53
Real Malo     FN=22       TP=38

Classification report:
              precision  recall  f1-score  support
Bueno              0.80    0.62      0.70      140
Malo               0.42    0.63      0.50       60
accuracy                             0.62      200
```

Costo total: 5×22 + 1×53 = **163**

---

### Celda 42 — Búsqueda Threshold Óptimo

**Función**: `costo_total(y_true, y_prob, threshold) = 5×FN + 1×FP`
**Grid**: np.linspace(0.05, 0.95, 91) — 91 valores de threshold evaluados en validación.

**Output real**:
```
Threshold óptimo (validación): 0.26
```

---

### Celda 44 — Evaluación Final con Threshold Óptimo

**Output real**:
```
Threshold óptimo: 0.26

Matriz de confusión (threshold=0.26):
           Pred Bueno   Pred Malo
Real Bueno    TN=28      FP=112
Real Malo     FN=5        TP=55

Classification report:
              precision  recall  f1-score  support
Bueno              0.85    0.20      0.32      140
Malo               0.33    0.92      0.48       60
accuracy                             0.41      200

Costos:
  Baseline (todo aprobado): 300
  Threshold 0.50:           163   (5×22 + 1×53)
  Threshold óptimo 0.26:    137   (5×5  + 1×112)
  Reducción vs baseline:     54.3%
  Reducción vs 0.50:         16.0%
```

---

### Celda 47 — Odds Ratios

**Output real** (top 15 por |coeficiente|, total features no nulas: 23):

```
Feature                     Coef     OR
historial_credito_A34      -0.3907  0.6766  (cuenta critica -> -32.3% riesgo)
duracion_meses              0.3654  1.4411  (cada mes: x1.44 el riesgo)
otros_planes_cuota_A143    -0.3017  0.7395  (sin otros planes: -26% riesgo)
ahorros_A65                -0.2843  0.7526
tasa_cuota                  0.2479  1.2813  (cada punto de tasa: +28.1% riesgo)
ahorros_A64                -0.2431  0.7842  (>=1000 DM: -21.6% riesgo)
estado_civil_sexo_A93      -0.2013  0.8177
proposito_A41              -0.1976  0.8207  (auto usado: -17.9% riesgo)
proposito_A43              -0.1749  0.8395  (TV/Radio: -16.0% riesgo)
empleo_desde_A72            0.1285  1.1372  (<1 año: +13.7% riesgo)
monto_credito               0.1284  1.1370
vivienda_A152              -0.1146  0.8917
ahorros_A63                -0.0933  0.9109
otros_planes_cuota_A142    -0.0762  0.9267
otros_deudores_A102         0.0744  1.0773
```

---

### Celda 50 — Scorecard

**Output real**:
```
PDO=20  Score_ref=600  Odds_ref=50:1
Factor = 28.8539
Offset = 487.1229

Score en test:
  Mínimo:             423.0
  Máximo:             571.7
  Media total:        490.9
  Media buenos (0):   497.1
  Media malos (1):    476.6
  Diferencia media:    20.5 puntos
```

---

### Celda 51 — Histograma de Scores

**Output real**:
```
Score de corte (threshold=0.26): 517.3 puntos
Aprobados (score > 517):          33 de 200 (16.5%)
Rechazados (score <= 517):        167 de 200 (83.5%)
```

La separación de 20.5 puntos entre buenos y malos es moderada pero consistente. El score de corte de 517.3 es conservador (recomienda rechazar al 83.5%), lo que refleja la preferencia del banco por minimizar los FN dado el costo 5:1.

---

## Notas de Implementación

| Aspecto | Decisión | Justificación |
|---|---|---|
| Suavizado Laplace (smoothing=0.5) | Evitar WoE = ±∞ | Categorías con 0 buenos o 0 malos → log(0) sin suavizado |
| Agrupación categorías <3% en OTRO | Estabilidad del WoE | Bins con <30 casos tienen alta varianza en WoE/IV |
| pd.qcut (quantile bins) para numéricas | Igual n por bin | pd.cut (igual ancho) deja bins vacíos con distribuciones sesgadas |
| IV_MIN=0.02, IV_MAX=0.60 | Filtro de variables | <0.02 = ruido; >0.60 = sospecha de data leakage (excluye cuenta_corriente) |
| Estratificación en split (stratify=y) | Bad rate uniforme | Sin estratificar, por azar puede haber conjuntos con 25% vs 35% de malos |
| Split en dos pasos (test_size=0.25 en 2° split) | Lograr 60/20/20 | 0.25 × 0.80 = 0.20 → 20% de validación del total |
| with_mean=False en StandardScaler | Compatibilidad sparse | get_dummies puede producir matrices sparse; buena práctica aunque aquí sea dense |
| Fit scaler solo en train | Evitar data leakage | Valid/test no deben informar la media/std de normalización |
| drop_first=True en get_dummies | Evitar multicolinealidad perfecta | Sin drop_first, la suma de todas las dummies de una variable = 1 siempre |
| penalty='l1', solver='liblinear' | Selección automática de features | L1 lleva coeficientes irrelevantes exactamente a 0; liblinear = único solver compatible |
| C=0.1 | Regularización fuerte | Muestra pequeña (600 train) con 38 features → necesita más penalización |
| class_weight='balanced' | Compensar desbalance 70/30 | Asigna peso 1/freq a cada clase sin modificar los datos |
| Threshold optimizado en validación | Separación train/valid/test | Buscar threshold en test = data leakage implícito (ajuste a datos de evaluación) |
| random_state=42 | Reproducibilidad | Resultados idénticos en cada ejecución |

---

## Decisiones de diseño pedagógico (v2)

El notebook v2 incorpora las siguientes mejoras respecto a la versión original:

| Mejora | Dónde |
|---|---|
| Cada celda de código tiene comentarios que explican el POR QUÉ, no solo el QUÉ | Todas las celdas de código |
| Las celdas de Resultado incluyen los números reales de la ejecución | md-005, md-009, md-011, md-014, md-016, md-019, md-021, md-024, md-027, md-034, md-037, md-039, md-044, md-046, md-049, md-053 |
| Notas explícitas sobre cuándo aplicar cada técnica en otro proyecto | md-002, md-010, md-019, md-024, md-029, md-031, md-034, md-037, md-041, md-046, md-049, md-050 |
| La intro de S7 explica cada hiperparámetro de la LogisticRegression | md-035 |
| La función de costo explica por qué el threshold se busca en validación y no en test | cd-043 |
| El scorecard incluye la interpretación económica del PDO con ejemplo concreto | md-050 |
