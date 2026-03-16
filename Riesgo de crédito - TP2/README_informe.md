# Informe: Análisis de Riesgo de Crédito — Banco Alemán

**Materia**: Data Science — A42
**Trabajo Práctico**: TP2 — Riesgo de Crédito
**Dataset**: German Credit Dataset (Prof. Dr. Hans Hofmann, Universität Hamburg)

---

## 1. Introducción y Problema

El presente trabajo analiza 1.000 solicitudes de préstamos de un banco alemán con el objetivo de construir un modelo que reproduzca el criterio utilizado para aceptar o rechazar solicitudes. Cada cliente está descripto por 20 variables socioeconómicas y crediticias, y clasificado como buen o mal pagador.

El **desafío central** planteado por la consigna es que la muestra es relativamente pequeña para la cantidad de variables disponibles, lo que implica que muchas variables serán poco representativas o poco relevantes. Esto hace que la selección de variables sea crítica: de las 20 variables disponibles, el análisis de Information Value descartó 6 por no aportar poder predictivo real.

**Costo asimétrico**: el banco penaliza de forma desigual los dos tipos de error:
- Clasificar a un mal pagador como bueno → costo = **5**
- Clasificar a un buen pagador como malo → costo = **1**

Esta asimetría se incorpora explícitamente para optimizar el threshold de clasificación.

---

## 2. Descripción del Dataset

| Característica | Valor |
|---|---|
| Observaciones | 1.000 |
| Variables predictoras | 20 (7 numéricas + 13 categóricas) |
| Variable objetivo | `target`: 0=bueno, 1=malo |
| Buenos pagadores (0) | 700 (70.0%) |
| Malos pagadores (1) | 300 (30.0%) |
| Valores faltantes | 0 |
| Duplicados | 0 |

### Variables numéricas — Estadísticas descriptivas reales

| Variable | Media | Mediana | Std | Min | Max | Skew | Outliers |
|---|---|---|---|---|---|---|---|
| duracion_meses | 20.90 | 18.00 | 12.06 | 4 | 72 | 1.094 | 70 |
| monto_credito | 3.271 | 2.320 | 2.823 | 250 | 18.424 | 1.950 | 72 |
| tasa_cuota | 2.97 | 3.00 | 1.12 | 1 | 4 | -0.531 | 0 |
| anios_residencia | 2.85 | 3.00 | 1.10 | 1 | 4 | -0.273 | 0 |
| edad | 35.55 | 33.00 | 11.38 | 19 | 75 | 1.021 | 23 |
| num_creditos_banco | 1.41 | 1.00 | 0.58 | 1 | 4 | 1.273 | 6 |
| num_dependientes | 1.16 | 1.00 | 0.36 | 1 | 2 | 1.909 | 155 |

*Montos en Deutsche Marks (DM). Outliers por criterio IQR×1.5.*

---

## 3. Análisis Exploratorio

### 3.1 Distribución del target

El dataset presenta un desbalance moderado: 700 buenos (70%) vs 300 malos (30%). No es extremo, pero es suficiente para que un clasificador naive que apruebe todo alcance 70% de accuracy sin detectar ningún caso malo. Se aborda con `class_weight='balanced'` en el modelo y optimización del threshold por costo asimétrico.

### 3.2 Análisis univariado — Variables numéricas

- **monto_credito**: skew=1.950, el más sesgado. Media (3.271 DM) muy superior a mediana (2.320 DM), con 72 outliers hacia valores altos. Máximo de 18.424 DM.
- **duracion_meses**: skew=1.094, 70 outliers. Media 20.9 meses vs mediana 18 meses. Rango 4-72 meses.
- **edad**: skew=1.021, distribución concentrada entre 19 y 75 años. Media 35.55, mediana 33.00.
- **tasa_cuota** y **anios_residencia**: valores discretos del 1 al 4, sin outliers. Skew negativo leve. Poca variabilidad.
- **num_dependientes**: skew=1.909 pero 155 outliers (el 15.5% del dataset), todos con valor=2. Es un artefacto de la discretización: 84.4% tiene valor 1 y 15.6% tiene valor 2, lo que el criterio IQR detecta como outlier.
- **num_creditos_banco**: 83% de los clientes tienen exactamente 1 crédito.

### 3.3 Análisis univariado — Variables categóricas

| Variable | N cats | Categoría dominante | % | Cats con <5% |
|---|---|---|---|---|
| cuenta_corriente | 4 | A14 (sin cuenta) | 39% | 0 |
| historial_credito | 5 | A32 (pago OK actual) | 53% | 2 |
| proposito | 10 | A43 (TV/Radio) | 28% | 4 |
| ahorros | 5 | A61 (<100 DM) | 60% | 1 |
| empleo_desde | 5 | A73 (1-4 años) | 34% | 0 |
| estado_civil_sexo | 4 | A93 (H soltero) | 55% | 0 |
| otros_deudores | 3 | A101 (ninguno) | 91% | 1 |
| vivienda | 3 | A152 (propia) | 71% | 0 |
| tipo_trabajo | 4 | A173 (calificado) | 63% | 1 |
| trabajador_extranjero | 2 | A201 (sí) | 96% | 1 |

`otros_deudores` (91% en una categoría) y `trabajador_extranjero` (96%) anticipan bajo poder discriminante.

### 3.4 Análisis bivariado — Numéricas vs target (Mann-Whitney)

| Variable | Media buenos | Media malos | p-value | Significancia |
|---|---|---|---|---|
| duracion_meses | 19.21 | 24.86 | <0.001 | *** |
| edad | 36.22 | 33.96 | 0.0004 | *** |
| monto_credito | 2.985 | 3.938 | 0.0059 | ** |
| tasa_cuota | 2.92 | 3.10 | 0.0199 | * |
| anios_residencia | 2.84 | 2.85 | 0.9358 | ns |
| num_creditos_banco | 1.42 | 1.37 | 0.1348 | ns |
| num_dependientes | 1.16 | 1.15 | 0.9242 | ns |

Las variables sin diferencia significativa (`anios_residencia`, `num_creditos_banco`, `num_dependientes`) son candidatas a descarte, confirmado luego por el IV.

### 3.5 Análisis bivariado — Categóricas vs target (chi-cuadrado y bad rate)

| Variable | p-value | Bad rate min | Bad rate max | Spread |
|---|---|---|---|---|
| historial_credito | <0.001 *** | 17.1% | 62.5% | 45.4% |
| cuenta_corriente | <0.001 *** | 11.7% | 49.3% | 37.6% |
| proposito | <0.001 *** | 11.1% | 44.0% | 32.9% |
| propiedad | <0.001 *** | 21.3% | 43.5% | 22.2% |
| ahorros | <0.001 *** | 12.5% | 36.0% | 23.5% |
| vivienda | <0.001 *** | 26.1% | 40.7% | 14.7% |
| empleo_desde | 0.001 ** | 22.4% | 40.7% | 18.3% |
| otros_planes_cuota | 0.002 ** | 27.5% | 41.0% | 13.5% |
| trabajador_extranjero | 0.016 * | 10.8% | 30.7% | 19.9% |
| estado_civil_sexo | 0.022 * | 26.6% | 40.0% | 13.4% |
| otros_deudores | 0.036 * | 19.2% | 43.9% | 24.7% |
| tipo_trabajo | 0.597 ns | 28.0% | 34.5% | 6.5% |
| telefono | 0.279 ns | 28.0% | 31.4% | 3.4% |

Casos destacados: `historial_credito` A30 (sin créditos previos) tiene bad rate del 62.5% (n=40), y A14 en `cuenta_corriente` (sin cuenta) tiene solo 11.7% de malos. `tipo_trabajo` y `telefono` no muestran asociación significativa.

### 3.6 Correlaciones entre variables numéricas

| Par | r de Pearson |
|---|---|
| duracion_meses vs monto_credito | 0.625 |
| monto_credito vs tasa_cuota | -0.271 |
| anios_residencia vs edad | 0.266 |

La correlación más relevante es entre duración y monto (r=0.625): créditos más grandes tienen plazos más largos. No hay multicolinealidad severa que justifique descartar variables por este criterio.

---

## 4. Selección de Variables: Information Value (IV)

### IV real obtenido por variable

| Variable | IV | Interpretación | Seleccionada |
|---|---|---|---|
| cuenta_corriente | 0.6591 | **Sospechoso** (>0.60) | ✗ excluida |
| historial_credito | 0.2908 | Moderado | ✓ |
| duracion_meses | 0.2132 | Moderado | ✓ |
| ahorros | 0.1879 | Moderado | ✓ |
| proposito | 0.1520 | Moderado | ✓ |
| propiedad | 0.1118 | Moderado | ✓ |
| monto_credito | 0.0921 | Débil | ✓ |
| empleo_desde | 0.0857 | Débil | ✓ |
| vivienda | 0.0838 | Débil | ✓ |
| edad | 0.0673 | Débil | ✓ |
| otros_planes_cuota | 0.0584 | Débil | ✓ |
| estado_civil_sexo | 0.0448 | Débil | ✓ |
| trabajador_extranjero | 0.0393 | Débil | ✓ |
| otros_deudores | 0.0311 | Débil | ✓ |
| tasa_cuota | 0.0252 | Débil | ✓ |
| tipo_trabajo | 0.0090 | No predictiva | ✗ |
| telefono | 0.0063 | No predictiva | ✗ |
| num_creditos_banco | 0.0029 | No predictiva | ✗ |
| anios_residencia | 0.0002 | No predictiva | ✗ |
| num_dependientes | 0.0000 | No predictiva | ✗ |

**Resultado**: 14 variables seleccionadas (IV entre 0.02 y 0.60). 6 descartadas.

**Hallazgo importante**: `cuenta_corriente`, que era la mejor discriminante en el análisis bivariado (spread del 37.6%), obtiene IV=0.6591 y queda **excluida por superar el umbral máximo de 0.60**. Esto refleja que su poder discriminante podría ser excesivamente alto para un dataset de 1.000 registros, sugiriendo posible concentración de información o sesgo de selección en la muestra.

---

## 5. Preprocesamiento

- **Features usadas**: 14 variables seleccionadas por IV
- **Split estratificado 60/20/20**: train=600, valid=200, test=200
- **Bad rate por conjunto**: train=30.0%, valid=30.0%, test=30.0% (estratificación perfecta)
- **Tras one-hot encoding**: 38 features binarias
- **StandardScaler** aplicado solo sobre train, transformado a valid y test

---

## 6. Modelo: Regresión Logística L1

### Parámetros

| Parámetro | Valor |
|---|---|
| penalty | l1 (Lasso) |
| solver | liblinear |
| class_weight | balanced |
| C | 0.1 (regularización fuerte) |
| max_iter | 500 |

### Resultados con threshold=0.50

| Métrica | Valor |
|---|---|
| Accuracy | 0.6250 |
| AUC-ROC | 0.7102 |
| PR-AUC | 0.5443 |
| Coeficientes no nulos (L1) | 23 / 38 |

**Matriz de confusión (threshold=0.50)**:

|  | Pred Bueno | Pred Malo |
|---|---|---|
| **Real Bueno** | TN=87 | FP=53 |
| **Real Malo** | FN=22 | TP=38 |

Costo total con threshold=0.50: **163** (5×22 + 1×53 = 110 + 53)

---

## 7. Optimización del Threshold con Costo Asimétrico

**Función objetivo**: Costo = 5×FN + 1×FP

| Escenario | Threshold | TN | FP | FN | TP | Costo |
|---|---|---|---|---|---|---|
| Baseline (todo aprobado) | — | 140 | 0 | 60 | 0 | 300 |
| Threshold default | 0.50 | 87 | 53 | 22 | 38 | 163 |
| **Threshold óptimo** | **0.26** | **28** | **112** | **5** | **55** | **137** |

- **Reducción vs baseline**: 54.3%
- **Reducción vs threshold 0.50**: 16.0%
- Con threshold=0.26, el recall de malos sube de 63.3% a 91.7% (de 60 malos en test, detecta 55)
- El costo de 5 por cada FN justifica aceptar más FP (rechazos incorrectos de buenos clientes)

---

## 8. Interpretabilidad: Odds Ratios (top 10 por |coeficiente|)

| Feature | Coeficiente | Odds Ratio | Interpretación |
|---|---|---|---|
| historial_credito_A34 | -0.3907 | 0.677 | Cuenta crítica → 32.3% menos chances de mora |
| duracion_meses | +0.3654 | 1.441 | Cada mes extra multiplica por 1.44 el riesgo |
| otros_planes_cuota_A143 | -0.3017 | 0.740 | Sin otros planes → 26% menos chances de mora |
| ahorros_A65 | -0.2843 | 0.753 | Sin ahorros (categoría base) → reduce OR |
| tasa_cuota | +0.2479 | 1.281 | Cada punto de tasa sube riesgo en 28.1% |
| ahorros_A64 | -0.2431 | 0.784 | Ahorros >=1000 DM → 21.6% menos chances de mora |
| estado_civil_sexo_A93 | -0.2013 | 0.818 | H soltero → 18.2% menos chances de mora |
| proposito_A41 | -0.1976 | 0.821 | Auto usado → 17.9% menos chances de mora |
| proposito_A43 | -0.1749 | 0.840 | TV/Radio → 16.0% menos chances de mora |
| empleo_desde_A72 | +0.1285 | 1.137 | Empleo <1 año → 13.7% más chances de mora |

El L1 eliminó 15 de los 38 coeficientes (39.5% de features descartadas automáticamente).

---

## 9. Scorecard

| Parámetro | Valor |
|---|---|
| PDO | 20 |
| Score de referencia | 600 puntos |
| Odds de referencia | 50:1 buenos:malos |
| Factor | 28.8539 |
| Offset | 487.1229 |

**Distribución de scores en test**:

| Grupo | Score medio | Score min | Score max |
|---|---|---|---|
| Buenos pagadores | 497.1 | — | — |
| Malos pagadores | 476.6 | — | — |
| Total | 490.9 | 423.0 | 571.7 |

- **Score de corte** (equivalente al threshold óptimo 0.26): **517.3 puntos**
- Clientes aprobados en test (score > 517): **33 de 200 (16.5%)**
- Clientes rechazados (score ≤ 517): **167 de 200 (83.5%)**

La diferencia de 20.5 puntos entre el score medio de buenos (497.1) y malos (476.6) refleja la separación lograda por el modelo.

---

## 10. Conclusiones

### Respuesta al objetivo del trabajo

El trabajo logró construir un modelo predictivo que reproduce el criterio del banco para clasificar solicitudes de crédito. Usando regresión logística con regularización L1, calibrada con un threshold optimizado por costo asimétrico (5:1), el modelo **reduce el costo económico de clasificación en un 54.3% respecto al baseline** (de 300 a 137 unidades en el set de test de 200 clientes).

El resultado concreto: con el threshold óptimo de 0.26, el modelo detecta **55 de los 60 malos pagadores** presentes en el test (recall = 91.7%), dejando escapar solo 5. Esos 55 créditos habrían sido aprobados por el banco y probablemente entrado en mora.

### ¿Qué variables explican el riesgo?

De las 20 variables disponibles, **14 muestran poder predictivo real** (IV ≥ 0.02). Las cuatro más influyentes dentro del umbral aceptable son:

| Variable | IV | Hallazgo clave |
|---|---|---|
| historial_credito | 0.291 | El predictor más confiable. Historial sin créditos previos (A30) → bad rate 62.5%. |
| duracion_meses | 0.213 | Cada mes adicional multiplica el riesgo ×1.44 (OR=1.441). |
| ahorros | 0.188 | Indica colchón financiero. Sin ahorros → bad rate ~35%; con ≥1000 DM → ~14%. |
| proposito | 0.152 | El destino del crédito diferencia significativamente el perfil de riesgo. |

La variable con mayor capacidad discriminante bivariada — `cuenta_corriente` (spread=37.6%) — quedó **excluida por IV=0.659 > 0.60**. Esto confirma la advertencia de la consigna: con 1.000 registros, algunas variables aparentan ser extremadamente predictivas por artefactos del muestreo. Las variables `num_dependientes`, `anios_residencia`, `num_creditos_banco`, `telefono` y `tipo_trabajo` (IV < 0.01) no aportan información útil y fueron descartadas antes de modelar.

### Perfil del cliente de alto riesgo

Crédito de larga duración (mediana de 24 meses vs 19 en buenos pagadores), sin historial crediticio previo, con escasos ahorros y empleo reciente de menos de un año. Estos factores combinados representan la señal más fuerte que el modelo aprendió a detectar.

### Resultado cuantitativo final

| Escenario | FN (malos no detectados) | Costo total | Reducción |
|---|---|---|---|
| Baseline — aprobar todo | 60 / 60 | 300 | — |
| Threshold 0.50 (default) | 22 / 60 | 163 | −45.7% |
| **Threshold óptimo 0.26** | **5 / 60** | **137** | **−54.3%** |

El costo asimétrico fue incorporado correctamente: el threshold se desplazó de 0.50 a 0.26 porque clasificar un mal pagador como bueno pesa 5 veces más que el error inverso. El modelo acepta más rechazos incorrectos (112 FP) a cambio de reducir drásticamente los créditos malos aprobados (5 FN).

### Limitaciones

- **AUC-ROC = 0.71**: aceptable en credit scoring, pero modelos no lineales (Random Forest, XGBoost) podrían mejorar la discriminación. La contrapartida es perder la interpretabilidad de los Odds Ratios y la posibilidad de construir el scorecard.
- **Score de corte conservador (517 pts)**: rechaza al 83.5% del test. En producción, el banco calibraría el corte según su apetito de riesgo y capacidad operativa, posiblemente implementando zonas de revisión manual.
- **Muestra pequeña**: 1.000 registros de los años '90 de una sola institución alemana. Algunas categorías tienen bajo n (historial A30: n=40), con alta varianza en sus estimaciones de WoE. Su transferibilidad a otro contexto requiere reentrenamiento con datos locales.

---

## 11. Validación contra la Consigna

| Requisito | Implementación | Estado |
|---|---|---|
| Analizar base de datos del banco alemán | Secciones 1–4: EDA completo con 20 variables | ✅ |
| Usar decodificador german_clean.docx | Renombre semántico en celda 3, cat_labels en celdas 14/19 | ✅ |
| Construir modelo predictivo | Regresión Logística L1, AUC-ROC=0.71 | ✅ |
| Considerar desafío de muestra pequeña | IV filtra 6 variables; L1 elimina 15 de 38 coeficientes | ✅ |
| Explicar para qué se usa cada técnica | Intro de cada sección + comentarios en código | ✅ |
| Explicar qué se espera obtener | Párrafo "Qué esperamos" en cada intro | ✅ |
| Reportar qué resultados se obtuvieron | Celda "Resultado" tras cada bloque de código | ✅ |
| Derivar conclusiones | Incorporadas en cada "Resultado" | ✅ |
| Costo asimétrico 5:1 | Sección 8: threshold óptimo 0.26, reducción de costo del 54.3% vs baseline | ✅ |
| Herramientas de visualización | Histogramas, boxplots, bad rates, heatmap, ROC, PR, WoE, scorecard | ✅ |
