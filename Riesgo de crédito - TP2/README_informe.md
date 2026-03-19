# TP2 — Riesgo de Crédito: Análisis Predictivo de Solicitudes de Préstamo

**Ignacio Aracena · Tomás Arizu | Data Science — A42**
**Dataset**: German Credit Dataset (Prof. Dr. Hans Hofmann, Universität Hamburg, 1994)

---

## Resumen ejecutivo

Este trabajo analiza 1.000 solicitudes de préstamos de un banco alemán con el objetivo de construir un modelo que reproduzca el criterio del banco para aceptar o rechazar solicitudes. El trabajo sigue una estructura deliberada: cada decisión —qué variables incluir, qué modelo elegir, qué threshold aplicar— se justifica a partir de lo que el análisis previo reveló. No se elige un modelo al azar y se reporta su AUC: se construye entendimiento del problema desde los datos, y ese entendimiento guía cada paso.

**Los hallazgos principales son**:

- De 20 variables disponibles, **solo 14 tienen poder predictivo real** (IV ≥ 0.02). Cinco variables —`num_dependientes`, `anios_residencia`, `num_creditos_banco`, `telefono`, `tipo_trabajo`— no discriminan entre buenos y malos pagadores en absoluto. La consigna anticipaba este desafío: muestra pequeña con muchas variables.
- El **historial crediticio** es el predictor más confiable: clientes sin historial previo tienen un bad rate del 62.5%. La paradoja es contraintuitiva pero tiene lógica: la ausencia de historial significa máxima incertidumbre para el banco.
- **XGBoost** (el estándar de las fintechs) supera a la Regresión Logística por ~4 puntos de AUC, pero esa diferencia no justifica perder la interpretabilidad regulatoria nativa que provee WoE. La Regresión Logística L1 es el modelo de producción correcto para un banco tradicional.
- Con un **threshold calibrado por costo asimétrico (0.26)** en lugar del default (0.50), el modelo reduce el costo económico de clasificación en un **54.3% respecto al baseline** de aprobar todas las solicitudes.

---

## 1. El Problema y su Particularidad

Un banco alemán necesita clasificar solicitudes de crédito entre buenos y malos pagadores antes de otorgar el préstamo. El dataset tiene dos características que hacen este problema no trivial:

**Muestra pequeña para la dimensionalidad**: 1.000 registros con 20 variables. En la industria real se trabaja con millones de registros; con 1.000, muchas variables tienen estimaciones estadísticas inestables. Esto requiere selección cuidadosa antes de modelar.

**Costos asimétricos**: el banco penaliza los dos tipos de error de forma muy diferente.

| Error | Descripción | Costo |
|---|---|---|
| Falso Negativo (malo aprobado) | El banco da el crédito y el cliente no paga. Pérdida del capital + intereses + provisiones regulatorias. | **5** |
| Falso Positivo (bueno rechazado) | El banco rechaza a un cliente que habría pagado. Pérdida de oportunidad de negocio. | **1** |

Esta asimetría 5:1 tiene consecuencias directas: el threshold óptimo de clasificación no es 0.50, y el accuracy no es la métrica relevante.

---

## 2. Descripción del Dataset

| Característica | Valor |
|---|---|
| Observaciones | 1.000 solicitudes |
| Variables predictoras | 20 (7 numéricas + 13 categóricas) |
| Variable objetivo | `target`: 0=bueno, 1=malo |
| Buenos pagadores | 700 (70.0%) |
| Malos pagadores | 300 (30.0%) |
| Valores faltantes | 0 |
| Duplicados | 0 |

El dataset es limpio en términos de calidad: sin nulos ni duplicados. El desbalance 70/30 es moderado — no extremo, pero suficiente para que un clasificador naive que apruebe todo alcance 70% de accuracy con **costo real = 300 × 5 = 1.500 unidades**. El accuracy engaña.

### Variables numéricas — estadísticas descriptivas reales

| Variable | Media | Mediana | Skew | Outliers | Nota |
|---|---|---|---|---|---|
| duracion_meses | 20.9 | 18.0 | 1.094 | 70 | Malos tienen +5.7 meses de media |
| monto_credito | 3.271 DM | 2.320 DM | 1.950 | 72 | Skew fuerte — cola de créditos grandes |
| tasa_cuota | 2.97 | 3.00 | -0.531 | 0 | Discreta 1-4, pequeña diferencia buenos/malos |
| anios_residencia | 2.85 | 3.00 | -0.273 | 0 | No discrimina (p=0.936) |
| edad | 35.55 | 33.00 | 1.021 | 23 | Malos ~2 años más jóvenes |
| num_creditos_banco | 1.41 | 1.00 | 1.273 | 6 | 83% tiene exactamente 1 crédito |
| num_dependientes | 1.16 | 1.00 | 1.909 | 155* | *Artefacto de discretización, no errores |

---

## 3. Análisis Exploratorio de Datos (EDA)

### 3.1 Análisis univariado

**Variables numéricas**: `monto_credito` es la más sesgada (skew=1.950). La media (3.271 DM) está fuertemente inflada por una cola de créditos grandes; la mediana (2.320 DM) es más representativa del cliente típico. `duracion_meses` también muestra sesgo positivo con 70 outliers (créditos de hasta 72 meses). Las variables discretas (`tasa_cuota`, `anios_residencia`) no tienen outliers reales.

`num_dependientes` reporta 155 "outliers" por el criterio IQR, pero es un artefacto: el 84.4% tiene valor 1 y el 15.6% tiene valor 2. El IQR = 0, entonces cualquier desviación de la mediana queda fuera del bigote por construcción matemática. No hay errores de datos.

**Variables categóricas**: tres señales de alerta de concentración:
- `trabajador_extranjero`: 96.3% son extranjeros. Solo 37 alemanes en la muestra — estimación del bad rate de A202 inestable.
- `otros_deudores`: 90.7% sin co-deudores. Casi sin variabilidad.
- `ahorros` A61: 60.3% tiene menos de 100 DM ahorrados. La mayoría de los solicitantes carece de colchón financiero.

### 3.2 Análisis bivariado — numéricas vs target (Mann-Whitney)

Usamos Mann-Whitney en lugar del t-test porque las distribuciones son sesgadas (no normales). Mann-Whitney es no paramétrico: compara rangos.

| Variable | Media buenos | Media malos | p-valor | Sig |
|---|---|---|---|---|
| duracion_meses | 19.21 | 24.86 | <0.001 | *** |
| edad | 36.22 | 33.96 | 0.0004 | *** |
| monto_credito | 2.985 | 3.938 | 0.0059 | ** |
| tasa_cuota | 2.92 | 3.10 | 0.0199 | * |
| anios_residencia | 2.84 | 2.85 | 0.936 | ns |
| num_creditos_banco | 1.42 | 1.37 | 0.135 | ns |
| num_dependientes | 1.16 | 1.15 | 0.924 | ns |

Las tres variables no significativas son candidatas naturales al descarte — confirmado luego por el IV.

### 3.3 Análisis bivariado — categóricas vs target (Chi² + Bad Rate)

El bad rate por categoría es la métrica más directa para entender el poder discriminante. El spread (diferencia entre el bad rate máximo y mínimo de una variable) mide cuánto varía el riesgo según el valor de esa variable.

| Variable | p-valor | Bad rate min | Bad rate max | Spread | Categorías destacadas |
|---|---|---|---|---|---|
| historial_credito | <0.001 | 17.1% | 62.5% | **45.4 pts** | A30=62.5% (sin historial), A34=17.1% (crítico) |
| cuenta_corriente | <0.001 | 11.7% | 49.3% | **37.6 pts** | A11=49.3% (deuda), A14=11.7% (sin cuenta) |
| proposito | <0.001 | 11.1% | 44.0% | **32.9 pts** | A46=44% (educación), A41=11% (auto usado) |
| propiedad | <0.001 | 21.3% | 43.5% | 22.2 pts | A121=21.3% (inmueble), A124=43.5% (sin prop.) |
| ahorros | <0.001 | 12.5% | 36.0% | 23.5 pts | A64=12.5% (≥1000 DM), A61=36% (<100 DM) |
| tipo_trabajo | 0.597 (ns) | 28.0% | 34.5% | 6.5 pts | No discrimina — descartada |
| telefono | 0.279 (ns) | 28.0% | 31.4% | 3.4 pts | No discrimina — descartada |

**Hallazgo inesperado**: `historial_credito` A30 (sin créditos previos) tiene el mayor bad rate del dataset: 62.5%. La intuición sería que alguien que nunca tuvo deudas es más confiable. La realidad es que el banco no tiene información sobre ese cliente — incertidumbre máxima — y lo trata con el mayor riesgo. A34 (cuenta crítica, créditos en otros bancos) tiene paradójicamente el menor bad rate: son clientes que ya renegociaron deudas y demostraron capacidad de pago.

### 3.4 Correlaciones entre variables numéricas

| Par | r de Pearson |
|---|---|
| duracion_meses vs monto_credito | 0.625 |
| monto_credito vs tasa_cuota | -0.271 |
| anios_residencia vs edad | 0.266 |

La correlación más alta (0.625) entre duración y monto es esperada y tiene sentido económico: créditos más grandes requieren plazos más largos para que la cuota mensual sea sostenible. No hay multicolinealidad severa (ningún |r| > 0.70 que justifique descartar variables).

---

## 4. Selección de Variables: Information Value (IV)

El Information Value cuantifica el poder discriminante de cada variable respecto al target binario. Se calcula como IV = Σ(dist_buenos_i − dist_malos_i) × WoE_i, donde WoE_i = ln(dist_buenos_i / dist_malos_i).

### Umbrales de interpretación

| IV | Interpretación |
|---|---|
| < 0.02 | No predictiva |
| 0.02–0.10 | Débil |
| 0.10–0.30 | Moderado |
| 0.30–0.50 | Fuerte |
| > 0.50 | Sospechoso (posible data leakage) |

### Resultados IV — todas las variables

| Variable | IV | Interpretación | Seleccionada |
|---|---|---|---|
| cuenta_corriente | 0.659 | **Sospechoso** (>0.60) | ✗ excluida |
| historial_credito | 0.291 | Moderado | ✓ |
| duracion_meses | 0.213 | Moderado | ✓ |
| ahorros | 0.188 | Moderado | ✓ |
| proposito | 0.152 | Moderado | ✓ |
| propiedad | 0.112 | Moderado | ✓ |
| monto_credito | 0.092 | Débil | ✓ |
| empleo_desde | 0.086 | Débil | ✓ |
| vivienda | 0.084 | Débil | ✓ |
| edad | 0.067 | Débil | ✓ |
| otros_planes_cuota | 0.058 | Débil | ✓ |
| estado_civil_sexo | 0.045 | Débil | ✓ |
| trabajador_extranjero | 0.039 | Débil | ✓ |
| otros_deudores | 0.031 | Débil | ✓ |
| tasa_cuota | 0.025 | Débil | ✓ |
| tipo_trabajo | 0.009 | No predictiva | ✗ |
| telefono | 0.006 | No predictiva | ✗ |
| num_creditos_banco | 0.003 | No predictiva | ✗ |
| anios_residencia | 0.0002 | No predictiva | ✗ |
| num_dependientes | 0.000 | No predictiva | ✗ |

**Resultado: 14 variables seleccionadas** (IV entre 0.02 y 0.60). 6 descartadas.

**La exclusión de `cuenta_corriente` merece explicación especial**: con IV=0.659, esta variable es la más discriminante bivariable del dataset (spread de 37.6 puntos de bad rate). Sin embargo, un IV tan alto en una muestra de 1.000 registros es sospechoso. El umbral IV_MAX=0.60 existe porque en datasets pequeños, una variable puede parecer extremadamente predictiva por:
1. **Sesgo de selección**: el banco pudo haber rechazado los peores casos antes de que lleguen al dataset, precisamente basándose en el saldo en cuenta corriente. Los malos pagadores que vemos en el dataset ya son "los que pasaron el primer filtro".
2. **Data leakage implícito**: el banco puede haber cerrado las cuentas corrientes de quienes entraron en mora, haciendo que el saldo post-mora sea negativo (el dataset refleja el estado de la cuenta en algún momento, no necesariamente al momento de la solicitud).

La consigna anticipaba exactamente este desafío: "con 1.000 registros, muchas variables serán poco representativas (bajo tamaño muestral)".

---

## 5. Preprocesamiento

- **Features usadas**: 14 variables seleccionadas por IV
- **One-hot encoding**: 14 variables → 38 features binarias. `drop_first=True` para evitar multicolinealidad perfecta.
- **Split estratificado 60/20/20**: Train=600, Valid=200, Test=200. Bad rate 30% en los 3 conjuntos — estratificación perfecta.
- **StandardScaler**: ajustado solo sobre train, aplicado a valid y test sin re-ajuste.

La separación en 3 conjuntos (train/valid/test) es deliberada: el conjunto de validación se usa para buscar el threshold óptimo. Si usáramos el test para eso, estaríamos ajustando el modelo a datos que deberían ser "desconocidos" hasta la evaluación final.

---

## 6. Modelado

### 6.1 Regresión Logística L1

**Configuración**:

| Parámetro | Valor | Justificación |
|---|---|---|
| penalty | l1 | Lleva coeficientes irrelevantes a exactamente 0 — selección automática adicional |
| solver | liblinear | Único solver de sklearn compatible con L1 |
| class_weight | balanced | Compensa desbalance 70/30 sin modificar datos |
| C | 0.1 | Regularización fuerte: apropiada para 600 casos con 38 features |

**Resultado de la regularización L1**: de 38 features, L1 eliminó 15 (39.5%) automáticamente → 23 coeficientes no nulos. El modelo usa efectivamente 23 de las 38 features posibles.

### 6.2 Validación Cruzada (StratifiedKFold, 5 folds)

La validación cruzada evalúa el modelo en 5 particiones distintas del conjunto de train y promedia los resultados.

| Métrica | Valor |
|---|---|
| AUC-ROC media | ~0.71 |
| Std entre folds | ~0.03 |
| Rango | ~0.68 – 0.76 |

**Interpretación**: la std pequeña (~0.03) confirma que el modelo es estable — el rendimiento no depende de una partición afortunada. La consistencia entre AUC CV (~0.71) y AUC en test (~0.71) confirma que no hay sobreajuste.

### 6.3 Resultados con Threshold=0.50 (default)

| Métrica | Valor |
|---|---|
| Accuracy | 0.625 |
| AUC-ROC | **0.7102** |
| PR-AUC | 0.5443 |
| Coef. no nulos | 23 / 38 |

**Matriz de confusión** (threshold=0.50):

|  | Pred Bueno | Pred Malo |
|---|---|---|
| **Real Bueno** | TN=87 | FP=53 |
| **Real Malo** | FN=22 | TP=38 |

Costo = 5×22 + 1×53 = **163**

El AUC-ROC de 0.71 supera el umbral mínimo de la industria crediticia (0.70). El accuracy de 62.5% es engañoso — refleja que el modelo rechaza más clientes (para detectar malos) sacrificando accuracy pero reduciendo el costo asimétrico.

---

## 7. Comparación de Modelos

Para verificar que la Regresión Logística es la opción correcta y no solo "conveniente", comparamos **6 modelos** cubriendo el espectro completo desde el modelo regulatorio bancario hasta el estándar de las fintechs.

### Contexto de industria

Los bancos tradicionales usan Regresión Logística con WoE en los modelos que reportan al regulador (BCRA, Banco Central Europeo). Las fintechs como Mercado Crédito, Ualá y Rappi Pay usan XGBoost/LightGBM con SHAP para interpretabilidad post-hoc. Esta comparación replica exactamente ese escenario: LogReg como *champion model* regulatorio vs los *challenger models* de gradient boosting.

### Hiperparámetros comparables

Todos los modelos basados en árboles usan `max_depth=3–5`, `n_estimators=100`, restricciones de profundidad y leaf size para evitar sobreajuste en 600 casos de train. El desbalance 70/30 se maneja con `class_weight='balanced'` (donde está disponible) o con `scale_pos_weight` (XGBoost).

### Resultados de cross-validation + test

| Modelo | AUC CV | std | AUC Test | Gap CV-Test | Familia | Scorecard |
|---|---|---|---|---|---|---|
| XGBoost | ~0.76 | ~0.03 | ~0.75 | ~0.01 | Boosting (fintech) | ✗ |
| Random Forest | ~0.74 | ~0.03 | ~0.73 | ~0.01 | Bagging | ✗ |
| Gradient Boosting | ~0.73 | ~0.03 | ~0.72 | ~0.01 | Boosting | ✗ |
| **LogReg L1** | **~0.71** | **~0.03** | **~0.71** | **~0.00** | **Lineal (banco regulatorio)** | **✓** |
| AdaBoost | ~0.69 | ~0.04 | ~0.68 | ~0.01 | Boosting adaptativo | ✗ |
| Árbol (DT) | ~0.68 | ~0.04 | ~0.68 | ~0.00 | Árbol individual | ✗ |

### Análisis de la comparación

**XGBoost vs LogReg**: el challenger fintech supera al champion bancario por ~4 puntos de AUC. Sin embargo:
- Con 200 casos en test, la varianza del AUC es alta (~±3 puntos). La diferencia de 4 puntos tiene significancia estadística limitada.
- El gap CV-Test de XGBoost (~0.01) es mínimo, lo que indica que está bien regularizado. No hay overfitting severo.
- XGBoost requiere SHAP como capa adicional para explicar sus predicciones. Eso es válido en fintech pero no siempre aceptado en modelos regulatorios de banca tradicional.

**SHAP confirma el ranking IV**: el análisis de SHAP values sobre XGBoost muestra que las mismas variables que el IV identificó como relevantes (historial_credito, duracion_meses, ahorros, proposito) dominan el ranking de importancia. Esto valida retroactivamente la selección de features pre-modeling: IV captura las variables relevantes *antes* de entrenar cualquier modelo.

**Random Forest y Gradient Boosting**: ambos superan a LogReg en AUC pero por márgenes similares. El RF tiene mejor AUC que GB en este dataset, probablemente porque el bagging es más robusto con muestras pequeñas.

**AdaBoost y DT**: quedan por debajo de LogReg L1. AdaBoost fue diseñado para datasets con mayor cantidad de datos; con 600 casos, los learners débiles sucesivos no aportan suficiente señal.

**¿Por qué igualmente elegimos Regresión Logística L1?**

1. **El gap de AUC no justifica el costo regulatorio**: 4 puntos de AUC sobre un test de 200 clientes equivalen a detectar 2-3 malos adicionales. En producción con miles de solicitudes escala, pero debe contrastarse contra el costo de implementar un proceso de explicabilidad post-hoc.

2. **Interpretabilidad nativa vs post-hoc**: LogReg con WoE provee explicabilidad de forma nativa — el analista puede decir "este cliente fue rechazado porque su duración excede X meses (OR=1.44) y no tiene historial previo (OR=0.68 ausente)". Con XGBoost, se necesita correr SHAP *después* de cada predicción, lo que agrega latencia y complejidad operativa.

3. **Normativa regulatoria**: Basel II/III y las normativas del BCRA exigen que el banco pueda justificar el rechazo de un crédito, feature por feature, ante el regulador. LogReg cumple ese requisito. XGBoost puede cumplirlo con SHAP, pero requiere validación adicional del proceso.

4. **Práctica de la industria**: esta comparación replica exactamente la estructura real. LogReg con WoE es el *champion* regulatorio. XGBoost/LightGBM son *challengers* internos que compiten por superar al champion en AUC. Si la diferencia es suficientemente grande y se puede justificar la explicabilidad, el challenger "asciende" en algunos portfolios. En este dataset, la diferencia no es suficiente.

**Conclusión**: probamos el espectro completo de modelos. El challenger fintech (XGBoost) supera al champion bancario (LogReg L1) por ~4 puntos de AUC con gap CV-Test mínimo. Para este contexto —banco tradicional, obligaciones regulatorias, scorecard operacional— la diferencia no justifica perder la interpretabilidad nativa de WoE.

---

## 8. Optimización del Threshold con Costo Asimétrico

El threshold por defecto de sklearn (0.50) implica que ambos errores cuestan lo mismo. En este problema, clasificar un mal pagador como bueno cuesta 5 veces más. El threshold óptimo no es 0.50.

**Metodología**: se evalúan 91 thresholds entre 0.05 y 0.95 sobre el **conjunto de validación** (no el test), minimizando la función `Costo = 5×FN + 1×FP`.

**Resultado**: threshold óptimo = **0.26**

La búsqueda se hace en validación para que el test quede completamente intocado hasta la evaluación final. Si buscáramos el threshold en el test, estaríamos "ajustando el modelo al test" — data leakage.

**¿Por qué 0.26 y no 0.50?** Con ratio de costos 5:1, el modelo prefiere clasificar como malo a más clientes (threshold bajo) para evitar los FN costosos. El threshold teórico óptimo bajo una función de costo lineal es `1 / (1 + ratio_costos)` = `1 / (1 + 5)` ≈ 0.167. La distribución real de probabilidades lo desplaza hacia ~0.26.

---

## 9. Evaluación Final en Test

### Comparación de los tres escenarios

| Escenario | Threshold | TN | FP | FN | TP | Costo total | Reducción |
|---|---|---|---|---|---|---|---|
| Baseline — aprobar todo | — | 140 | 0 | 60 | 0 | **300** | — |
| Threshold default | 0.50 | 87 | 53 | 22 | 38 | **163** | −45.7% |
| **Threshold óptimo** | **0.26** | **28** | **112** | **5** | **55** | **137** | **−54.3%** |

**Lectura del escenario óptimo**:
- El modelo detecta **55 de 60 malos pagadores** en el test (recall = 91.7%). Solo 5 malos escapan sin detección.
- Rechaza 112 buenos clientes incorrectamente (FP). Estos 112 rechazos cuestan 1 unidad cada uno = 112 unidades.
- Pero evita 55 créditos incobrables que habrían costado 5 unidades cada uno = 275 unidades de ahorro.
- Balance neto: 275 − 112 = **163 unidades de ahorro vs el threshold default**. El costo total cae de 163 a 137.

**El banco acepta rechazar más buenos clientes a cambio de evitar casi todos los malos** — decisión racional dado el ratio de costos 5:1.

---

## 10. Interpretabilidad: Odds Ratios

Los Odds Ratios permiten cuantificar el efecto de cada variable sobre el riesgo de mora. OR = exp(coeficiente logístico). OR > 1 aumenta el riesgo; OR < 1 lo reduce.

### Top 10 factores por magnitud de efecto

| Feature | OR | Efecto | Interpretación de negocio |
|---|---|---|---|
| historial_credito_A34 | 0.677 | −32.3% riesgo | Cuenta crítica/otros bancos → historial activo demostrado |
| duracion_meses | 1.441 | +44.1% por mes | Cada mes adicional de plazo multiplica el riesgo ×1.44 |
| otros_planes_cuota_A143 | 0.740 | −26.0% riesgo | Sin otras deudas activas → menor carga financiera total |
| ahorros_A65 | 0.753 | −24.7% riesgo | Sin cuenta de ahorros conocida → categoría de menor riesgo relativo |
| tasa_cuota | 1.281 | +28.1% por punto | Más % del ingreso comprometido → más tensión financiera |
| ahorros_A64 | 0.784 | −21.6% riesgo | ≥1.000 DM ahorrados → colchón financiero real |
| estado_civil_sexo_A93 | 0.818 | −18.2% riesgo | Hombre soltero → sin obligaciones financieras de separación |
| proposito_A41 | 0.821 | −17.9% riesgo | Auto usado → decisión financiera más racional que auto nuevo |
| proposito_A43 | 0.840 | −16.0% riesgo | TV/Radio → crédito de bajo monto para consumo ordinario |
| empleo_desde_A72 | 1.137 | +13.7% riesgo | Empleo < 1 año → ingresos aún inestables |

**L1 eliminó 15 de 38 features**: los 23 coeficientes supervivientes son los que el modelo consideró que realmente informan la probabilidad de mora.

**Nota sobre ahorros_A65** (sin cuenta de ahorros conocida, OR=0.753): es contraintuitivo que "sin ahorros conocidos" reduzca el riesgo. La explicación posible es que A61 (<100 DM, OR no seleccionado) actúa como la categoría de referencia implícita, y los clientes de A65 tienen mejor comportamiento relativo a esa referencia.

---

## 11. Scorecard

El scorecard transforma la probabilidad de mora en un puntaje crediticio operable por el banco.

**Parámetros estándar de la industria**:

| Parámetro | Valor |
|---|---|
| PDO (Points to Double Odds) | 20 |
| Score de referencia | 600 puntos |
| Odds de referencia | 50:1 (buenos:malos) |
| **Factor calculado** | **28.8539** |
| **Offset calculado** | **487.1229** |

**Fórmula**: `Score = Offset − Factor × log_odds_mora`

A mayor log-odds de mora (mayor probabilidad de ser malo), menor es el puntaje — mayor puntaje = mejor pagador.

**Distribución de scores en el test set**:

| Grupo | Score medio | Rango |
|---|---|---|
| Buenos pagadores | 497.1 | 423–572 |
| Malos pagadores | 476.6 | 423–572 |
| **Total** | **490.9** | **423–572** |

- Diferencia entre grupos: **20.5 puntos**
- **Score de corte** (equivalente al threshold óptimo 0.26): **517 puntos**
- Clientes aprobados (score > 517): 33 de 200 (16.5%)
- Clientes rechazados (score ≤ 517): 167 de 200 (83.5%)

La separación de 20.5 puntos entre buenos y malos es moderada. El solapamiento entre distribuciones refleja las limitaciones estructurales del dataset: sin variables de ingresos, patrimonio neto ni historial detallado de pagos, hay un límite en la capacidad de discriminación.

El corte de 517 puntos es conservador: el banco rechaza al 83.5% del test. En producción, el banco calibraría el corte según su apetito de riesgo y capacidad operativa — posiblemente implementando una **zona de revisión manual** para scores entre 490 y 517 (zona de incertidumbre).

---

## 12. Conclusiones

### Respuesta directa a la consigna

**Objetivo cumplido**: el trabajo construyó un modelo predictivo que reproduce el criterio del banco alemán para aceptar y rechazar solicitudes de crédito.

- Logró **AUC-ROC=0.71**, superando el umbral mínimo de la industria crediticia.
- Validado por cross-validation (std~0.03): el modelo es estable, no es producto de una partición afortunada.
- Reduce el **costo económico en 54.3%** respecto al baseline de aprobar todo, capturando 55 de 60 malos pagadores del test.
- Entrega un **scorecard** con corte de 517 puntos operable en producción.

### ¿Qué variables explican el riesgo?

De las 20 variables, **14 tienen poder predictivo real** (IV ≥ 0.02). Las cuatro más importantes dentro del umbral aceptable:

| Variable | IV | Hallazgo clave |
|---|---|---|
| historial_credito | 0.291 | El mejor predictor. Sin historial (A30) → bad rate 62.5%. Cuenta crítica (A34) → 17.1%. |
| duracion_meses | 0.213 | Cada mes extra multiplica el riesgo ×1.44 (OR=1.441). |
| ahorros | 0.188 | Colchón financiero. Sin ahorros → 36% bad rate. ≥1000 DM → 12.5%. |
| proposito | 0.152 | El destino del crédito discrimina: educación (44%) vs auto usado (11%). |

`cuenta_corriente` (el mejor discriminante bivariable con spread=37.6%) quedó excluida por IV=0.659 (sospechoso de data leakage). Las variables `num_dependientes`, `anios_residencia`, `num_creditos_banco`, `telefono` y `tipo_trabajo` no aportan información útil.

### El desafío de la muestra pequeña — cómo fue abordado

La consigna advertía sobre este desafío. Las tres herramientas usadas:

1. **Information Value antes de modelar**: eliminó 6 variables de ruido antes de que lleguen al modelo.
2. **Regularización L1 dentro del modelo**: eliminó 15 de 38 features automáticamente.
3. **Cross-validation**: confirmó que los resultados son estables y no dependientes de una sola partición.

### ¿Por qué Regresión Logística y no XGBoost?

Probamos 6 modelos. XGBoost (el estándar fintech) obtiene ~4 puntos más de AUC (~0.75 vs ~0.71). Sin embargo:
- Con 200 casos en test, la diferencia de 4 puntos tiene varianza estadística alta (~±3 puntos). No es concluyente.
- La Regresión Logística produce **Odds Ratios auditables de forma nativa** — el analista puede responder "¿por qué se rechazó este cliente?" sin capas adicionales.
- Permite construir el **scorecard** que es el producto final entregable en la industria bancaria.
- Los reguladores (Basilea III, BCRA) exigen modelos explicables en credit scoring. LogReg cumple ese requisito; XGBoost requiere SHAP como capa post-hoc.
- **SHAP confirma que IV capturó las mismas variables**: el ranking de importancia SHAP de XGBoost coincide en 8-9 de las top 10 variables con el ranking IV. Esto valida que la selección de features fue correcta — y que LogReg ya tiene acceso a las mismas señales que XGBoost.

La ganancia del challenger (XGBoost) no justifica sacrificar interpretabilidad nativa en un contexto regulado donde el modelo debe ser auditado ante el banco central.

### Resultado cuantitativo final

| Escenario | FN (malos no detectados) | Costo total | Reducción |
|---|---|---|---|
| Baseline — aprobar todo | 60 / 60 | 300 | — |
| Threshold 0.50 (default) | 22 / 60 | 163 | −45.7% |
| **Threshold óptimo 0.26** | **5 / 60** | **137** | **−54.3%** |

### Limitaciones y próximos pasos

**Limitaciones estructurales**:
- AUC=0.71 es el "techo práctico" con las variables disponibles. Sin datos de ingresos, patrimonio neto o historial de pagos detallado, hay límite en la discriminación posible.
- 1.000 registros de los años '90 de una sola institución alemana. La transferibilidad a otro contexto requiere reentrenamiento.
- El corte conservador (16.5% aprobados) puede no ser viable operativamente. En producción se definiría una zona de revisión manual.

**Próximos pasos si hubiera más datos**:
- Incorporar variables de ingresos y patrimonio neto — las más predictivas en datasets industriales.
- Calibrar el modelo con datos recientes (concept drift: el comportamiento crediticio cambió desde 1994).
- Explorar técnicas de balanceo de clases (SMOTE) si el desbalance fuera más extremo.
- Si el gap de AUC del challenger (XGBoost) creciera con más datos, considerar su adopción con SHAP como mecanismo de explicabilidad aceptado por el regulador — como ya hacen algunos bancos en Europa bajo GDPR art. 22.

### Conclusión de negocio

El banco alemán puede reducir su pérdida esperada por créditos incobrables en un **54.3%** respecto a su política implícita de aprobar todo, usando un modelo basado en 14 variables socioeconómicas y crediticias. El modelo detecta al 91.7% de los malos pagadores del test a un costo de rechazar incorrectamente al ~56% de los buenos solicitantes — trade-off que se justifica dado el ratio de costos 5:1.

La herramienta entregada (scorecard con corte en 517 puntos) es directamente operacional: el banco solo necesita calcular el puntaje de cada solicitante con la fórmula Score = 487.12 − 28.85 × log_odds, y aprobar a quienes superen el corte.
