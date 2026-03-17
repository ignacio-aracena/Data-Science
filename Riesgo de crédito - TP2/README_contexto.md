# Contexto del Problema — Riesgo de Crédito Banco Alemán

**Fuente original**: German Credit Dataset — Prof. Dr. Hans Hofmann, Institut für Statistik und Ökonometrie, Universität Hamburg (1994)
**Archivo de datos**: `Base_Clientes_Alemanes.xlsx`
**Decodificador oficial**: `german_clean.docx`
**Consigna del trabajo**: `Caso_credit_risk.pdf`

---

## 1. El problema que está resolviendo el banco

Un banco alemán necesita decidir si aprobar o rechazar una solicitud de préstamo. Para tomar esa decisión, el analista de crédito revisa una serie de variables socioeconómicas del solicitante y construye una opinión sobre si esa persona es un **buen pagador** (devuelve el crédito según lo acordado) o un **mal pagador** (entra en mora o no cumple las condiciones).

El objetivo del trabajo es **reproducir ese criterio de decisión** usando machine learning: dado un nuevo cliente con sus 20 variables, el modelo debe predecir si es bueno o malo antes de otorgar el crédito.

### ¿Por qué esto es difícil?

1. **Muestra pequeña para la cantidad de variables**: 1.000 registros y 20 variables predictoras. En problemas de credit scoring industriales se trabaja con millones de registros. Con 1.000, muchas variables tienen estimaciones estadísticas inestables, lo que hace que la selección de variables sea crítica antes de modelar.

2. **Los errores no cuestan lo mismo**: el banco no penaliza igual los dos tipos de error posibles.

---

## 2. La matriz de costos — el corazón del problema

El documento oficial del dataset establece explícitamente una **matriz de costos asimétrica**:

```
                 Predicción
                 Bueno (1)   Malo (2)
Real Bueno (1)      0           1
Real Malo  (2)      5           0
```

**Traducción directa**: clasificar a un mal pagador como bueno cuesta **5 veces más** que clasificar a un buen pagador como malo.

### ¿Por qué esta asimetría?

| Tipo de error | Consecuencia real para el banco | Costo |
|---|---|---|
| Malo aprobado (Falso Negativo) | El banco otorga el crédito. El cliente no paga. El banco pierde el capital prestado más intereses caídos. Puede generar provisiones regulatorias. | **5** |
| Bueno rechazado (Falso Positivo) | El banco pierde la oportunidad de negocio. El cliente bueno se va a otro banco. Pérdida de margen financiero potencial. | **1** |

Esta asimetría tiene implicaciones directas en cómo se construye y evalúa el modelo:
- El accuracy del 70% de un clasificador naive (que siempre aprueba) es una métrica **engañosa** — su costo real es 300 × 5 = **1.500 unidades**.
- El threshold de clasificación no debe ser 0.50 sino el valor que **minimiza** `5 × FN + 1 × FP`.
- Las métricas de evaluación relevantes son AUC-ROC, PR-AUC y el **costo total**, no el accuracy.

---

## 3. El dataset: qué representa cada registro

Cada fila de `Base_Clientes_Alemanes.xlsx` (hoja "Raw data") representa **una solicitud de crédito individual** procesada por el banco. El dataset tiene:

| Característica | Valor |
|---|---|
| Registros totales | 1.000 solicitudes |
| Variables predictoras | 20 (7 numéricas + 13 categóricas) |
| Variable objetivo (`Rechazo`) | 1 = Aprobado (buen pagador), 2 = Rechazado (mal pagador) |
| Buenos pagadores (1→0) | 700 (70.0%) |
| Malos pagadores (2→1) | 300 (30.0%) |
| Valores faltantes | 0 |
| Duplicados | 0 |
| Período | Banco alemán, datos de los años '90 |
| Moneda | Deutsche Marks (DM) |

El dataset tiene **dos hojas**:
- **"Raw data"**: 22 columnas con códigos originales (A11, A32, etc.) — la que usamos en el modelo.
- **"Variables_Separadas"**: 68 columnas ya en formato one-hot — pre-procesada, pero sin los nombres semánticos.

---

## 4. Diccionario completo de variables

A continuación se detalla cada una de las 20 variables, su significado, sus categorías posibles, la distribución real en el dataset y el **bad rate** (proporción de malos pagadores) por categoría. El bad rate es la métrica más directa para entender el poder discriminante de cada variable.

> **Bad rate global del dataset: 30.0%** — cualquier categoría con bad rate notoriamente por encima o debajo de ese promedio es una señal de riesgo útil.

---

### Variable 1 — `cuenta_corriente` (Cualitativa)
**Descripción**: Estado del saldo en la cuenta corriente del solicitante al momento de la solicitud.
Esta es la variable con mayor poder discriminante del dataset (spread de bad rate = 37.6 puntos porcentuales).

| Código | Descripción | N | Bad Rate | Interpretación |
|---|---|---|---|---|
| A11 | Saldo negativo (< 0 DM) | 274 | **49.3%** | Alto riesgo — debe dinero en su cuenta |
| A12 | Saldo entre 0 y 200 DM | 269 | 39.0% | Riesgo moderado-alto |
| A13 | Saldo ≥ 200 DM o asignación salarial ≥ 1 año | 63 | 22.2% | Riesgo moderado |
| A14 | Sin cuenta corriente | 394 | **11.7%** | Menor riesgo — no tiene deudas activas |

**Nota del decodificador**: A11 y A12 son variables numéricas entre A11 y A13 (representan un rango de saldo). A14 es una dummy (sin cuenta). La paradoja es que "sin cuenta" tiene el menor bad rate: quien no tiene cuenta corriente probablemente es más cauteloso con el crédito.

**IV = 0.659** — excluida del modelo por superar el umbral de sospecha (IV_MAX = 0.60).

---

### Variable 2 — `duracion_meses` (Numérica)
**Descripción**: Duración del crédito solicitado, en meses.

| Estadística | Buenos pagadores | Malos pagadores |
|---|---|---|
| Media | 19.2 meses | 24.9 meses |
| Mediana | 18.0 meses | 24.0 meses |
| Rango global | 4 a 72 meses | — |

**Distribución global**: media=20.9, mediana=18.0, std=12.1. Sesgo positivo (skew=1.094): la mayoría de los créditos son cortos, pero hay una cola de créditos largos de hasta 6 años.

**Interpretación**: los malos pagadores tienen plazos ~30% más largos en promedio. Créditos de larga duración son más difíciles de sostener ante cambios en la situación económica del deudor.

**IV = 0.213** (moderado). OR en el modelo = 1.441: cada mes adicional de plazo multiplica las chances de mora por 1.44.

**Nota del decodificador**: se sugiere categorizar en estratos ≤10, ≤20, ≤30, ≤40, >40 meses para el análisis WoE.

---

### Variable 3 — `historial_credito` (Cualitativa)
**Descripción**: Comportamiento crediticio pasado del solicitante. Es la segunda variable más predictiva dentro del umbral aceptable.

| Código | Descripción | N | Bad Rate | Interpretación |
|---|---|---|---|---|
| A30 | Sin créditos previos / todos pagados correctamente | 40 | **62.5%** | Paradoja: sin historial = alto riesgo |
| A31 | Todos los créditos en este banco pagados correctamente | 49 | **57.1%** | Alto riesgo — historial limitado al propio banco |
| A32 | Créditos existentes pagados al día hasta ahora | 530 | 31.9% | Riesgo promedio — historial activo y corriente |
| A33 | Demora en pagos en el pasado | 88 | 31.8% | Similar a A32 (nota del decodificador: "raro que dé igual") |
| A34 | Cuenta crítica / otros créditos en otros bancos | 293 | **17.1%** | Menor riesgo — historial mixto pero activo |

**Spread de bad rate**: 62.5% − 17.1% = **45.4 puntos** — el mayor spread del dataset entre las variables seleccionadas.

**Interpretación**: la paradoja del historial crediticio. A30 (sin créditos previos) tiene el mayor bad rate porque el banco no tiene información previa sobre ese cliente — la incertidumbre es máxima. A34 (cuenta "crítica" con créditos en otros bancos) tiene el menor bad rate, posiblemente porque son clientes que ya renegociaron deudas exitosamente y demostraron capacidad de pago.

**IV = 0.291** (moderado). OR de A34 en el modelo = 0.677 (−32.3% de riesgo respecto a la categoría base).

---

### Variable 4 — `proposito` (Cualitativa)
**Descripción**: Destino declarado del crédito solicitado.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A40 | Auto nuevo | 234 | 38.0% |
| A41 | Auto usado | 103 | **16.5%** |
| A42 | Muebles / equipamiento | 181 | 32.0% |
| A43 | Radio / televisión | 280 | 22.1% |
| A44 | Electrodomésticos | 12 | 33.3% |
| A45 | Reparaciones | 22 | 36.4% |
| A46 | Educación | 50 | **44.0%** |
| A48 | Recapacitación / reentrenamiento | 9 | 11.1% |
| A49 | Negocio | 97 | 35.1% |
| A410 | Otros | 12 | 41.7% |

**Interpretación**: el destino del crédito refleja la urgencia y la racionalidad financiera detrás de la solicitud. Autos usados (A41) tienen el menor riesgo — el cliente ya evaluó que no necesita un auto nuevo. Educación (A46) tiene alto riesgo: los créditos educativos tienen plazos largos y retorno incierto. Auto nuevo tiene bad rate del 38%, significativamente por encima del promedio (30%).

**IV = 0.152** (moderado). Nota: A47 (vacaciones) aparece en el decodificador pero con "does not exist?" — efectivamente no hay ningún caso en el dataset.

---

### Variable 5 — `monto_credito` (Numérica)
**Descripción**: Monto solicitado en Deutsche Marks (DM).

| Estadística | Buenos pagadores | Malos pagadores |
|---|---|---|
| Media | 2.986 DM | 3.938 DM |
| Mediana | 2.244 DM | 2.575 DM |
| Mínimo global | 250 DM | — |
| Máximo global | 18.424 DM | — |

**Distribución global**: media=3.271 DM, mediana=2.320 DM, std=2.823 DM. El sesgo es fuerte (skew=1.950): la mayoría de los créditos son de bajo monto pero hay una cola de créditos muy grandes que elevan la media.

**Interpretación**: los malos pagadores solicitan montos en promedio un 32% más altos. Montos más grandes implican cuotas más altas y mayor tensión financiera sostenida.

**IV = 0.092** (débil). OR en el modelo = 1.137: cada unidad adicional de monto (escalado) aumenta el riesgo un 13.7%.

---

### Variable 6 — `ahorros` (Cualitativa)
**Descripción**: Saldo en la cuenta de ahorros o bonos del solicitante.

| Código | Descripción | N | Bad Rate | Interpretación |
|---|---|---|---|---|
| A61 | Menos de 100 DM | 603 | 36.0% | Mayor riesgo — sin colchón financiero |
| A62 | Entre 100 y 500 DM | 103 | 33.0% | Riesgo moderado-alto |
| A63 | Entre 500 y 1.000 DM | 63 | 17.5% | Riesgo moderado |
| A64 | Más de 1.000 DM | 48 | **12.5%** | Menor riesgo — colchón financiero significativo |
| A65 | Desconocido / sin cuenta de ahorros | 183 | 17.5% | Riesgo moderado (similar a quien tiene 500-1000 DM) |

**Interpretación**: la capacidad de ahorro es un indicador de disciplina financiera y resiliencia ante imprevistos. El 60.3% de los solicitantes tiene menos de 100 DM ahorrados, lo que los expone a cualquier imprevisto que interrumpa el pago de la cuota. Quienes tienen ≥1.000 DM ahorrados tienen un bad rate de solo el 12.5%.

**IV = 0.188** (moderado). OR de A64 en el modelo = 0.784 (−21.6% de riesgo).

---

### Variable 7 — `empleo_desde` (Cualitativa)
**Descripción**: Antigüedad en el empleo actual del solicitante.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A71 | Desempleado | 62 | 37.1% |
| A72 | Menos de 1 año | 172 | **40.7%** |
| A73 | Entre 1 y 4 años | 339 | 30.7% |
| A74 | Entre 4 y 7 años | 174 | 22.4% |
| A75 | 7 años o más | 253 | 25.3% |

**Interpretación**: los solicitantes con empleo reciente (<1 año) tienen el mayor bad rate, incluso superior al de los desempleados. Esto puede explicarse porque los recién empleados tienen ingresos aún inestables y menor margen para absorber shocks. La estabilidad laboral de 4+ años reduce el riesgo notoriamente.

**IV = 0.086** (débil pero seleccionado). OR de A72 en el modelo = 1.137 (+13.7% de riesgo).

---

### Variable 8 — `tasa_cuota` (Numérica)
**Descripción**: Tasa de cuota como porcentaje del ingreso disponible del solicitante. Escala 1 a 4 (1 = menor proporción del ingreso, 4 = mayor proporción).

**Distribución**: variable discreta, rango 1-4, media=2.97, mediana=3. Distribución casi uniforme.

| Estadística | Buenos | Malos |
|---|---|---|
| Media | 2.92 | 3.10 |

**Interpretación**: a mayor proporción del ingreso comprometida en la cuota, mayor tensión financiera y mayor riesgo de impago. La diferencia entre buenos y malos es pequeña pero estadísticamente significativa (Mann-Whitney p=0.020).

**IV = 0.025** (débil, apenas sobre el umbral mínimo). OR en el modelo = 1.281 (+28.1% de riesgo por cada nivel de tasa adicional).

---

### Variable 9 — `estado_civil_sexo` (Cualitativa)
**Descripción**: Combinación de sexo y estado civil del solicitante. En el dataset alemán de los '90 estas dos dimensiones estaban codificadas juntas.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A91 | Hombre divorciado / separado | 50 | 40.0% |
| A92 | Mujer divorciada / separada / casada | 310 | 35.2% |
| A93 | Hombre soltero | 548 | 26.6% |
| A94 | Hombre casado / viudo | 92 | 27.2% |
| A95 | Mujer soltera | **0 casos** | — |

**Nota importante**: A95 (mujer soltera) tiene **cero observaciones** en el dataset. Esta categoría desaparece automáticamente al aplicar one-hot encoding. El decodificador lo indica explícitamente: "(0 cases)".

**Interpretación**: los hombres solteros y casados tienen menor bad rate que los divorciados. Los hombres divorciados tienen el mayor riesgo posiblemente por la carga financiera de obligaciones post-separación. Las mujeres casadas/divorciadas tienen riesgo intermedio.

**IV = 0.045** (débil). OR de A93 en el modelo = 0.818 (−18.2% de riesgo).

---

### Variable 10 — `otros_deudores` (Cualitativa)
**Descripción**: Si existe algún co-solicitante o garante en la operación.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A101 | Sin co-deudores ni garantes | 907 | 30.7% |
| A102 | Co-solicitante (co-applicant) | 41 | 43.9% |
| A103 | Garante | 52 | 19.2% |

**Interpretación**: la presencia de un garante reduce significativamente el riesgo (19.2% vs 30.7% promedio): el banco tiene una segunda fuente de repago si el titular falla. Los co-solicitantes en cambio tienen mayor riesgo — posiblemente porque son solicitudes conjuntas de parejas o socios en situaciones de mayor necesidad financiera.

**IV = 0.031** (débil, seleccionado).

---

### Variable 11 — `anios_residencia` (Numérica)
**Descripción**: Tiempo en la residencia actual del solicitante. Escala 1 a 4.

**Distribución**: variable discreta, rango 1-4, media=2.85, mediana=3. Sin outliers.

**Nota del decodificador**: "Poca relevancia (solamente valor 1 ligeramente superior al promedio)".

**Interpretación**: la antigüedad en la residencia no discrimina entre buenos y malos pagadores de forma significativa (Mann-Whitney p=0.936). Refleja estabilidad habitacional, pero el banco alemán de los '90 ya capturaba esa información a través de otras variables.

**IV = 0.0002** — **descartada por no predictiva**.

---

### Variable 12 — `propiedad` (Cualitativa)
**Descripción**: Tipo de propiedad principal que posee el solicitante, en orden de valor decreciente.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A121 | Inmueble propio (bien raíz) | 282 | **21.3%** |
| A122 | Seguro de vida / ahorro constructivo (si no tiene A121) | 232 | 30.6% |
| A123 | Auto u otra propiedad (si no tiene A121/A122) | 332 | 30.7% |
| A124 | Sin propiedad conocida | 154 | **43.5%** |

**Interpretación**: la propiedad actúa como proxy de patrimonio y como garantía implícita. Quien tiene inmueble propio tiene el menor bad rate (21.3%). Sin propiedad el riesgo sube al 43.5%. La jerarquía de categorías (A121 → A122 → A123 → A124) refleja niveles decrecientes de patrimonio y garantías disponibles.

**IV = 0.112** (moderado).

---

### Variable 13 — `edad` (Numérica)
**Descripción**: Edad del solicitante en años.

| Estadística | Buenos | Malos |
|---|---|---|
| Media | 36.2 años | 34.0 años |
| Mediana | 34.0 años | 31.0 años |
| Rango global | 19 a 75 años | — |

**Distribución global**: media=35.6, mediana=33.0, sesgo positivo (skew=1.021). Concentración entre 25-45 años.

**Interpretación**: los malos pagadores son en promedio ~2 años más jóvenes. Los clientes jóvenes tienen historiales crediticios más cortos, ingresos aún en desarrollo y mayor volatilidad laboral. La diferencia es estadísticamente significativa (Mann-Whitney p<0.001).

**Nota del decodificador**: "segmentar en menor de 30 (1) o mayor de 30 (0), categorizado por décadas se ve que de 40 en adelante da igual".

**IV = 0.067** (débil, seleccionado).

---

### Variable 14 — `otros_planes_cuota` (Cualitativa)
**Descripción**: Otros planes de pago en cuotas activos del solicitante.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A141 | Planes de cuota en banco | 139 | 41.0% |
| A142 | Planes de cuota en comercios | 47 | 27.5% |
| A143 | Sin otros planes de cuota | 814 | 28.6% |

**Interpretación**: tener deudas activas en otros bancos aumenta el riesgo (41.0%), ya que el solicitante está comprometiendo ingresos con múltiples acreedores simultáneamente. Planes en comercios tienen riesgo similar al promedio.

**Nota del decodificador**: "No dan distintos PD entre sí bank y stores, da igual que usar A143". Sin embargo el banco (A141) sí muestra bad rate elevado.

**IV = 0.058** (débil, seleccionado). OR de A143 en el modelo = 0.740 (−26.0% de riesgo vs tener deudas bancarias).

---

### Variable 15 — `vivienda` (Cualitativa)
**Descripción**: Tipo de vivienda donde reside el solicitante.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A151 | Alquiler | 179 | 39.1% |
| A152 | Vivienda propia | 713 | 26.1% |
| A153 | Vivienda sin costo (familiar, etc.) | 108 | 40.7% |

**Interpretación**: quien tiene vivienda propia tiene el menor bad rate. Quienes alquilan o viven sin costo tienen mayor riesgo — posiblemente porque no tienen patrimonio inmobiliario propio y su situación habitacional es menos estable.

**IV = 0.084** (débil, seleccionado).

---

### Variable 16 — `num_creditos_banco` (Numérica)
**Descripción**: Número de créditos activos que el solicitante tiene en este mismo banco.

**Distribución**: rango 1-4, media=1.41, mediana=1.00. El 83% tiene exactamente 1 crédito activo.

**Interpretación**: prácticamente todos los clientes tienen 1 crédito. La variable no discrimina entre buenos y malos (Mann-Whitney p=0.135). El hecho de que haya muy pocos clientes con 2+ créditos hace que las estimaciones de bad rate sean inestables.

**IV = 0.003** — **descartada por no predictiva**.

---

### Variable 17 — `tipo_trabajo` (Cualitativa)
**Descripción**: Categoría laboral del solicitante.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A171 | Desempleado / no calificado no residente | 22 | 31.8% |
| A172 | No calificado residente | 200 | 28.0% |
| A173 | Empleado calificado / funcionario | 630 | 29.5% |
| A174 | Gerencia / autónomo / altamente calificado | 148 | 34.5% |

**Interpretación**: sorprendentemente, el tipo de trabajo no discrimina entre buenos y malos (chi² p=0.597). Los cuatro tipos tienen bad rates muy similares, entre 28% y 34.5%. Esto puede deberse a que el nivel de ingresos (no captado directamente en el dataset) sería más relevante que la categoría laboral.

**IV = 0.009** — **descartada por no predictiva**.

---

### Variable 18 — `num_dependientes` (Numérica)
**Descripción**: Número de personas a cargo del solicitante.

**Distribución**: solo toma valores 1 y 2. El 84.4% tiene 1 dependiente y el 15.6% tiene 2. Media=1.16, mediana=1.

**Interpretación**: la variable no aporta información discriminante (Mann-Whitney p=0.924). La razón es simple: la escala es tan estrecha (solo dos valores posibles) que no captura variación real en la carga de dependientes.

**Nota técnica**: el criterio IQR×1.5 detecta 155 "outliers" (todos los casos con valor=2), pero es un artefacto estadístico — no hay errores de datos. El IQR es 0, entonces cualquier desvío de la mediana queda fuera del bigote.

**IV = 0.000** — **descartada, la variable menos predictiva del dataset**.

---

### Variable 19 — `telefono` (Cualitativa)
**Descripción**: Si el solicitante tiene teléfono registrado a su nombre.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A191 | Sin teléfono | 596 | 31.4% |
| A192 | Sí, registrado a su nombre | 404 | 28.0% |

**Interpretación**: la diferencia es mínima (3.4 puntos) y no es estadísticamente significativa (chi² p=0.279). En los años '90 tener teléfono fijo era una señal de estabilidad, pero en este dataset no discrimina.

**IV = 0.006** — **descartada por no predictiva**.

---

### Variable 20 — `trabajador_extranjero` (Cualitativa)
**Descripción**: Si el solicitante es un trabajador extranjero en Alemania.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A201 | Sí, es extranjero | 963 | 30.7% |
| A202 | No, es alemán | 37 | 10.8% |

**Interpretación**: el 96.3% de los solicitantes son trabajadores extranjeros (Gastarbeiter), lo que era muy común en Alemania Occidental de los años '90. La variable casi no tiene variación — solo 37 casos son alemanes. Aunque hay una diferencia en bad rate (30.7% vs 10.8%), la escasa cantidad de alemanes en la muestra hace que esa estimación sea poco fiable estadísticamente.

**IV = 0.039** (débil, seleccionado por estar sobre el umbral de 0.02, aunque marginalmente).

---

## 5. Resumen del poder predictivo de las variables

La siguiente tabla consolida todos los indicadores de poder predictivo, ordenada por IV:

| Variable | Tipo | IV | Seleccionada | Bad rate min | Bad rate max | Spread |
|---|---|---|---|---|---|---|
| cuenta_corriente | Cat | 0.659 | ✗ sospechoso | 11.7% | 49.3% | 37.6 pts |
| historial_credito | Cat | 0.291 | ✓ | 17.1% | 62.5% | 45.4 pts |
| duracion_meses | Num | 0.213 | ✓ | — | — | monotónico ↑ |
| ahorros | Cat | 0.188 | ✓ | 12.5% | 36.0% | 23.5 pts |
| proposito | Cat | 0.152 | ✓ | 11.1% | 44.0% | 32.9 pts |
| propiedad | Cat | 0.112 | ✓ | 21.3% | 43.5% | 22.2 pts |
| monto_credito | Num | 0.092 | ✓ | — | — | monotónico ↑ |
| empleo_desde | Cat | 0.086 | ✓ | 22.4% | 40.7% | 18.3 pts |
| vivienda | Cat | 0.084 | ✓ | 26.1% | 40.7% | 14.7 pts |
| edad | Num | 0.067 | ✓ | — | — | leve ↓ riesgo con edad |
| otros_planes_cuota | Cat | 0.058 | ✓ | 27.5% | 41.0% | 13.5 pts |
| estado_civil_sexo | Cat | 0.045 | ✓ | 26.6% | 40.0% | 13.4 pts |
| trabajador_extranjero | Cat | 0.039 | ✓ | 10.8% | 30.7% | 19.9 pts* |
| otros_deudores | Cat | 0.031 | ✓ | 19.2% | 43.9% | 24.7 pts |
| tasa_cuota | Num | 0.025 | ✓ | — | — | leve ↑ riesgo con tasa |
| tipo_trabajo | Cat | 0.009 | ✗ | 28.0% | 34.5% | 6.5 pts |
| telefono | Cat | 0.006 | ✗ | 28.0% | 31.4% | 3.4 pts |
| num_creditos_banco | Num | 0.003 | ✗ | — | — | sin diferencia |
| anios_residencia | Num | 0.0002 | ✗ | — | — | sin diferencia |
| num_dependientes | Num | 0.000 | ✗ | — | — | sin diferencia |

*`trabajador_extranjero`: el spread es alto pero la categoría A202 tiene solo 37 casos — estimación inestable.

---

## 6. Perfil del cliente de alto riesgo

Basándose en los bad rates reales del dataset, el perfil del solicitante con mayor probabilidad de mora combina:

| Factor | Señal de riesgo | Bad rate observado |
|---|---|---|
| Cuenta corriente | Saldo negativo (< 0 DM) | 49.3% |
| Historial crediticio | Sin créditos previos (A30) | 62.5% |
| Ahorros | Menos de 100 DM | 36.0% |
| Empleo | Menos de 1 año en el empleo actual | 40.7% |
| Propósito | Educación | 44.0% |
| Propiedad | Sin propiedad | 43.5% |
| Vivienda | Alquiler o sin costo | ~40% |
| Duración | Crédito de larga duración | Bad rate crece con el plazo |

**Perfil del cliente de menor riesgo**:

| Factor | Señal de bajo riesgo | Bad rate observado |
|---|---|---|
| Cuenta corriente | Sin cuenta corriente (A14) | 11.7% |
| Historial crediticio | Cuenta crítica / otros bancos (A34) | 17.1% |
| Ahorros | ≥ 1.000 DM | 12.5% |
| Empleo | 4 a 7 años en el empleo actual | 22.4% |
| Propósito | Auto usado | 16.5% |
| Propiedad | Inmueble propio | 21.3% |
| Duración | Crédito de corto plazo | Bad rate decrece con el plazo |

---

## 7. El scorecard: cómo se comunica el resultado al banco

El modelo no le entrega al analista de crédito una "probabilidad de mora". Le entrega un **puntaje** entre 0 y 1.000 (en este caso, entre 423 y 572 dado el poder del modelo). El puntaje se construye con la convención estándar de la industria:

- **Mayor puntaje = menor riesgo** (mejor pagador).
- **PDO = 20**: cada 20 puntos de caída en el score duplica las chances de mora.
- **Score 600 = odds 50:1**: en ese punto, por cada 51 clientes con ese puntaje hay 50 buenos y 1 malo.

La política de crédito se define como un **corte**:
- Score > 517 → **Aprobado** (16.5% de los solicitantes en el set de test)
- Score ≤ 517 → **Rechazado o revisión manual** (83.5%)

Este corte refleja el threshold óptimo de 0.26 encontrado al minimizar la función de costo asimétrica (5×FN + 1×FP). Es un corte conservador — el banco prefiere rechazar buenos clientes a aprobar malos.

---

## 8. Fuente y limitaciones del dataset

### Origen
El dataset fue donado en 1994 por el Prof. Dr. Hans Hofmann de la Universidad de Hamburgo como benchmark de machine learning para clasificación binaria con costos asimétricos. Es uno de los datasets más citados en la literatura de credit scoring académico.

### Limitaciones que afectan la interpretación

1. **Datos históricos (años '90)**: el comportamiento crediticio y las variables relevantes cambian con el tiempo. Variables que predecían mora en 1994 pueden no ser igualmente predictivas hoy.

2. **Una sola institución, un solo país**: los patrones son específicos del sistema bancario alemán de esa época. La proporción de trabajadores extranjeros (96.3%) es un reflejo histórico de la política de Gastarbeiter, no representativa de otros contextos.

3. **Muestra pequeña**: 1.000 registros con 20 variables. En la industria bancaria se trabaja con millones de registros y modelos que se recalibran trimestralmente.

4. **Variables no disponibles**: no están incluidos ingresos, patrimonio neto, deudas totales, score previo ni historial de pagos detallado — información que un banco real sí tendría.

5. **Categorías con bajo n**: historial A30 (n=40), A31 (n=49), propósito A44 (n=12), A48 (n=9) — las estimaciones de bad rate y WoE para estas categorías tienen alta varianza estadística.

6. **Variable `cuenta_corriente` potencialmente problemática**: su IV de 0.659 es sospechosamente alto para un dataset de 1.000 registros. Podría reflejar que el banco ya tomó decisiones de aprobación basadas en ese saldo, generando un sesgo de selección en los datos observados.
