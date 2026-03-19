# Contexto del Problema — Riesgo de Crédito Banco Alemán

**Ignacio Aracena · Tomás Arizu | Data Science — A42 | TP2**

**Fuente original**: German Credit Dataset — Prof. Dr. Hans Hofmann, Institut für Statistik und Ökonometrie, Universität Hamburg (1994)
**Archivo de datos**: `Base_Clientes_Alemanes.xlsx`
**Decodificador oficial**: `german_clean.docx`

---

## 1. El problema que está resolviendo el banco

Un banco alemán necesita decidir si aprobar o rechazar una solicitud de préstamo. Para tomar esa decisión, el analista de crédito revisa variables socioeconómicas del solicitante y construye una opinión sobre si esa persona es un **buen pagador** (devuelve el crédito según lo acordado) o un **mal pagador** (entra en mora o no cumple las condiciones).

El objetivo del trabajo es **reproducir ese criterio de decisión** usando machine learning: dado un nuevo cliente con sus 20 variables, el modelo debe predecir si es bueno o malo antes de otorgar el crédito.

### ¿Por qué esto es difícil?

1. **Muestra pequeña para la cantidad de variables**: 1.000 registros y 20 variables predictoras. En problemas de credit scoring industriales se trabaja con millones de registros. Con 1.000, muchas variables tienen estimaciones estadísticas inestables, lo que hace que la selección de variables sea crítica antes de modelar.

2. **Los errores no cuestan lo mismo**: el banco no penaliza igual ambos tipos de error posibles. Un crédito otorgado a alguien que no va a pagar tiene consecuencias económicas muy distintas a rechazar a alguien que sí habría pagado.

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
| Variable objetivo (`Rechazo`) | 1 = Buen pagador, 2 = Mal pagador (recodificado a 0/1) |
| Buenos pagadores (0) | 700 (70.0%) |
| Malos pagadores (1) | 300 (30.0%) |
| Valores faltantes | 0 |
| Duplicados | 0 |
| Período | Banco alemán, datos de los años '90 |
| Moneda | Deutsche Marks (DM) |

El dataset tiene **dos hojas**:
- **"Raw data"**: 22 columnas con códigos originales (A11, A32, etc.) — la que usamos en el modelo.
- **"Variables_Separadas"**: 68 columnas ya en formato one-hot — pre-procesada, pero sin los nombres semánticos.

---

## 4. Diccionario completo de variables

A continuación se detalla cada una de las 20 variables, su significado según `german_clean.docx`, sus categorías posibles, la distribución real en el dataset y el **bad rate** (proporción de malos pagadores) por categoría. El bad rate es la métrica más directa para entender el poder discriminante de cada variable antes de modelar.

> **Bad rate global del dataset: 30.0%** — cualquier categoría con bad rate notoriamente por encima o debajo de ese promedio es una señal de riesgo útil.

---

### Variable 1 — `cuenta_corriente` (Cualitativa)
**Descripción**: Estado del saldo en la cuenta corriente del solicitante al momento de la solicitud.
Esta es la variable con mayor spread de bad rate del dataset (37.6 puntos porcentuales).

| Código | Descripción | N | Bad Rate | Interpretación |
|---|---|---|---|---|
| A11 | Saldo negativo (< 0 DM) | 274 | **49.3%** | Alto riesgo — debe dinero en su cuenta |
| A12 | Saldo entre 0 y 200 DM | 269 | 39.0% | Riesgo moderado-alto |
| A13 | Saldo ≥ 200 DM o asignación salarial ≥ 1 año | 63 | 22.2% | Riesgo moderado |
| A14 | Sin cuenta corriente | 394 | **11.7%** | Menor riesgo — no tiene deudas activas |

**Nota del decodificador**: la paradoja es que "sin cuenta" tiene el menor bad rate: quien no tiene cuenta corriente probablemente es más cauteloso con el crédito.

**IV calculado = 0.659** — excluida del modelo por superar el umbral de sospecha (IV_MAX = 0.60). Un IV tan alto en una muestra de 1.000 registros sugiere posible sesgo de selección.

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

**IV = 0.213** (moderado). Seleccionada para el modelo.

---

### Variable 3 — `historial_credito` (Cualitativa)
**Descripción**: Comportamiento crediticio pasado del solicitante. Es la variable más predictiva dentro del umbral aceptable.

| Código | Descripción | N | Bad Rate | Interpretación |
|---|---|---|---|---|
| A30 | Sin créditos previos / todos pagados | 40 | **62.5%** | Paradoja: sin historial = máxima incertidumbre |
| A31 | Todos los créditos en este banco pagados | 49 | **57.1%** | Historial limitado al propio banco |
| A32 | Créditos existentes pagados al día | 530 | 31.9% | Riesgo promedio |
| A33 | Demora en pagos en el pasado | 88 | 31.8% | Similar a A32 (raro que dé igual) |
| A34 | Cuenta crítica / otros créditos en otros bancos | 293 | **17.1%** | Menor riesgo — historial mixto pero activo |

**Spread de bad rate**: 62.5% − 17.1% = **45.4 puntos** — el mayor spread del dataset entre las variables seleccionadas.

**Interpretación**: A30 (sin créditos previos) tiene el mayor bad rate porque el banco no tiene información previa — la incertidumbre es máxima. A34 (cuenta "crítica") tiene el menor bad rate: posiblemente son clientes que renegociaron deudas exitosamente y demostraron capacidad de pago.

**IV = 0.291** (moderado). Seleccionada.

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

**Interpretación**: el destino del crédito refleja la racionalidad financiera detrás de la solicitud. Autos usados (A41) tienen el menor riesgo — el cliente ya evaluó que no necesita un auto nuevo. Educación (A46) tiene alto riesgo: plazos largos y retorno incierto. Auto nuevo supera el promedio del 30% con un 38%.

**Nota**: A47 (vacaciones) aparece en el decodificador pero no hay ningún caso en el dataset.

**IV = 0.152** (moderado). Seleccionada.

---

### Variable 5 — `monto_credito` (Numérica)
**Descripción**: Monto solicitado en Deutsche Marks (DM).

| Estadística | Buenos pagadores | Malos pagadores |
|---|---|---|
| Media | 2.986 DM | 3.938 DM |
| Mediana | 2.244 DM | 2.575 DM |
| Rango global | 250 a 18.424 DM | — |

**Distribución global**: media=3.271 DM, mediana=2.320 DM, skew=1.950. Fuerte sesgo positivo: la mayoría de los créditos son de bajo monto pero hay una cola de créditos muy grandes que elevan la media.

**Interpretación**: los malos pagadores solicitan montos un 32% más altos en promedio. Montos más grandes implican cuotas más altas y mayor tensión financiera sostenida.

**IV = 0.092** (débil). Seleccionada.

---

### Variable 6 — `ahorros` (Cualitativa)
**Descripción**: Saldo en la cuenta de ahorros o bonos del solicitante.

| Código | Descripción | N | Bad Rate | Interpretación |
|---|---|---|---|---|
| A61 | Menos de 100 DM | 603 | 36.0% | Mayor riesgo — sin colchón financiero |
| A62 | Entre 100 y 500 DM | 103 | 33.0% | Riesgo moderado-alto |
| A63 | Entre 500 y 1.000 DM | 63 | 17.5% | Riesgo moderado |
| A64 | Más de 1.000 DM | 48 | **12.5%** | Menor riesgo — colchón financiero significativo |
| A65 | Desconocido / sin cuenta de ahorros | 183 | 17.5% | Riesgo moderado |

**Interpretación**: la capacidad de ahorro indica disciplina financiera y resiliencia ante imprevistos. El 60.3% de los solicitantes tiene menos de 100 DM ahorrados. Quienes tienen ≥1.000 DM ahorrados tienen bad rate del 12.5%.

**IV = 0.188** (moderado). Seleccionada.

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

**Interpretación**: los solicitantes con empleo reciente (<1 año) tienen el mayor bad rate, incluso superior al de los desempleados. Ingresos aún inestables y menor margen para absorber shocks. La estabilidad laboral de 4+ años reduce el riesgo.

**IV = 0.086** (débil). Seleccionada.

---

### Variable 8 — `tasa_cuota` (Numérica)
**Descripción**: Tasa de cuota como porcentaje del ingreso disponible. Escala discreta 1 a 4 (1 = menor proporción, 4 = mayor proporción).

| Estadística | Buenos | Malos |
|---|---|---|
| Media | 2.92 | 3.10 |

**Interpretación**: a mayor proporción del ingreso comprometida en la cuota, mayor tensión financiera y mayor riesgo. La diferencia entre buenos y malos es pequeña pero significativa (Mann-Whitney p=0.020).

**IV = 0.025** (débil, sobre el umbral mínimo). Seleccionada.

---

### Variable 9 — `estado_civil_sexo` (Cualitativa)
**Descripción**: Combinación de sexo y estado civil. En el dataset alemán de los '90 estas dos dimensiones estaban codificadas juntas.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A91 | Hombre divorciado / separado | 50 | 40.0% |
| A92 | Mujer divorciada / separada / casada | 310 | 35.2% |
| A93 | Hombre soltero | 548 | 26.6% |
| A94 | Hombre casado / viudo | 92 | 27.2% |
| A95 | Mujer soltera | **0 casos** | — |

**Nota**: A95 tiene cero observaciones en el dataset — desaparece automáticamente en el one-hot encoding.

**Interpretación**: los hombres divorciados tienen el mayor riesgo posiblemente por la carga financiera de obligaciones post-separación.

**IV = 0.045** (débil). Seleccionada.

---

### Variable 10 — `otros_deudores` (Cualitativa)
**Descripción**: Si existe algún co-solicitante o garante en la operación.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A101 | Sin co-deudores ni garantes | 907 | 30.7% |
| A102 | Co-solicitante (co-applicant) | 41 | 43.9% |
| A103 | Garante | 52 | 19.2% |

**Interpretación**: la presencia de un garante reduce significativamente el riesgo (19.2%): el banco tiene una segunda fuente de repago. Los co-solicitantes en cambio tienen mayor riesgo — posiblemente solicitudes conjuntas en situaciones de mayor necesidad financiera.

**IV = 0.031** (débil). Seleccionada.

---

### Variable 11 — `anios_residencia` (Numérica)
**Descripción**: Tiempo en la residencia actual. Escala 1 a 4.

**Distribución**: discreta, rango 1-4, media=2.85. Sin outliers reales.

**Interpretación**: no discrimina entre buenos y malos pagadores (Mann-Whitney p=0.936). Refleja estabilidad habitacional, pero el banco alemán ya capturaba esa información a través de otras variables.

**IV = 0.0002** — **descartada**. No predictiva.

---

### Variable 12 — `propiedad` (Cualitativa)
**Descripción**: Tipo de propiedad principal del solicitante, en orden de valor decreciente.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A121 | Inmueble propio (bien raíz) | 282 | **21.3%** |
| A122 | Seguro de vida / ahorro constructivo | 232 | 30.6% |
| A123 | Auto u otra propiedad | 332 | 30.7% |
| A124 | Sin propiedad conocida | 154 | **43.5%** |

**Interpretación**: la propiedad actúa como proxy de patrimonio y garantía implícita. La jerarquía A121→A122→A123→A124 refleja niveles decrecientes de patrimonio.

**IV = 0.112** (moderado). Seleccionada.

---

### Variable 13 — `edad` (Numérica)
**Descripción**: Edad del solicitante en años.

| Estadística | Buenos | Malos |
|---|---|---|
| Media | 36.2 años | 34.0 años |
| Mediana | 34.0 años | 31.0 años |

**Distribución global**: media=35.6, mediana=33.0, skew=1.021. Concentración 25-45 años.

**Interpretación**: los malos pagadores son ~2 años más jóvenes en promedio. Los clientes jóvenes tienen historiales crediticios más cortos e ingresos aún en desarrollo.

**Nota del decodificador**: "segmentar en menor de 30 (1) o mayor de 30 (0); de 40 en adelante da igual".

**IV = 0.067** (débil). Seleccionada.

---

### Variable 14 — `otros_planes_cuota` (Cualitativa)
**Descripción**: Otros planes de pago en cuotas activos del solicitante.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A141 | Planes de cuota en banco | 139 | 41.0% |
| A142 | Planes de cuota en comercios | 47 | 27.5% |
| A143 | Sin otros planes de cuota | 814 | 28.6% |

**Interpretación**: tener deudas activas en otros bancos aumenta el riesgo (41.0%), ya que el solicitante compromete ingresos con múltiples acreedores simultáneamente.

**IV = 0.058** (débil). Seleccionada.

---

### Variable 15 — `vivienda` (Cualitativa)
**Descripción**: Tipo de vivienda donde reside el solicitante.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A151 | Alquiler | 179 | 39.1% |
| A152 | Vivienda propia | 713 | 26.1% |
| A153 | Vivienda sin costo (familiar, etc.) | 108 | 40.7% |

**Interpretación**: quien tiene vivienda propia tiene menor bad rate. Quienes alquilan o viven sin costo tienen mayor riesgo — probablemente no tienen patrimonio inmobiliario propio.

**IV = 0.084** (débil). Seleccionada.

---

### Variable 16 — `num_creditos_banco` (Numérica)
**Descripción**: Número de créditos activos en este mismo banco.

**Distribución**: rango 1-4, media=1.41. El 83% tiene exactamente 1 crédito activo.

**Interpretación**: prácticamente todos los clientes tienen 1 crédito. La variable no discrimina (Mann-Whitney p=0.135).

**IV = 0.003** — **descartada**. No predictiva.

---

### Variable 17 — `tipo_trabajo` (Cualitativa)
**Descripción**: Categoría laboral del solicitante.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A171 | Desempleado / no calificado no residente | 22 | 31.8% |
| A172 | No calificado residente | 200 | 28.0% |
| A173 | Empleado calificado / funcionario | 630 | 29.5% |
| A174 | Gerencia / autónomo / altamente calificado | 148 | 34.5% |

**Interpretación**: sorprendentemente, el tipo de trabajo no discrimina (chi² p=0.597). Los cuatro tipos tienen bad rates muy similares. El nivel de ingresos (no captado directamente) sería más relevante que la categoría laboral.

**IV = 0.009** — **descartada**. No predictiva.

---

### Variable 18 — `num_dependientes` (Numérica)
**Descripción**: Número de personas a cargo del solicitante.

**Distribución**: solo toma valores 1 y 2. El 84.4% tiene 1 dependiente y el 15.6% tiene 2.

**Interpretación**: la variable no aporta información discriminante (Mann-Whitney p=0.924). La escala tan estrecha no captura variación real.

**Nota técnica**: el criterio IQR×1.5 detecta 155 "outliers" (todos los casos con valor=2), pero es un artefacto estadístico — el IQR es 0, entonces cualquier desvío de la mediana queda fuera del bigote.

**IV = 0.000** — **descartada**. La variable menos predictiva del dataset.

---

### Variable 19 — `telefono` (Cualitativa)
**Descripción**: Si el solicitante tiene teléfono registrado a su nombre.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A191 | Sin teléfono | 596 | 31.4% |
| A192 | Sí, registrado a su nombre | 404 | 28.0% |

**Interpretación**: la diferencia es mínima (3.4 puntos) y no significativa (chi² p=0.279). En los años '90 tener teléfono fijo era una señal de estabilidad, pero en este dataset no discrimina.

**IV = 0.006** — **descartada**. No predictiva.

---

### Variable 20 — `trabajador_extranjero` (Cualitativa)
**Descripción**: Si el solicitante es un trabajador extranjero en Alemania.

| Código | Descripción | N | Bad Rate |
|---|---|---|---|
| A201 | Sí, es extranjero | 963 | 30.7% |
| A202 | No, es alemán | 37 | 10.8% |

**Interpretación**: el 96.3% de los solicitantes son trabajadores extranjeros (Gastarbeiter), muy común en Alemania Occidental de los años '90. Aunque hay diferencia en bad rate (30.7% vs 10.8%), la escasa cantidad de alemanes (37 casos) hace que esa estimación sea poco fiable.

**IV = 0.039** (débil, sobre el umbral de 0.02). Seleccionada con reservas.

---

## 5. Resumen del poder predictivo — tabla consolidada

| Variable | Tipo | IV | Seleccionada | Spread bad rate |
|---|---|---|---|---|
| cuenta_corriente | Cat | 0.659 | ✗ sospechoso | 37.6 pts |
| historial_credito | Cat | 0.291 | ✓ | 45.4 pts |
| duracion_meses | Num | 0.213 | ✓ | monotónico ↑ |
| ahorros | Cat | 0.188 | ✓ | 23.5 pts |
| proposito | Cat | 0.152 | ✓ | 32.9 pts |
| propiedad | Cat | 0.112 | ✓ | 22.2 pts |
| monto_credito | Num | 0.092 | ✓ | monotónico ↑ |
| empleo_desde | Cat | 0.086 | ✓ | 18.3 pts |
| vivienda | Cat | 0.084 | ✓ | 14.7 pts |
| edad | Num | 0.067 | ✓ | leve ↓ con edad |
| otros_planes_cuota | Cat | 0.058 | ✓ | 13.5 pts |
| estado_civil_sexo | Cat | 0.045 | ✓ | 13.4 pts |
| trabajador_extranjero | Cat | 0.039 | ✓ | 19.9 pts* |
| otros_deudores | Cat | 0.031 | ✓ | 24.7 pts |
| tasa_cuota | Num | 0.025 | ✓ | leve ↑ con tasa |
| tipo_trabajo | Cat | 0.009 | ✗ | 6.5 pts |
| telefono | Cat | 0.006 | ✗ | 3.4 pts |
| num_creditos_banco | Num | 0.003 | ✗ | sin diferencia |
| anios_residencia | Num | 0.0002 | ✗ | sin diferencia |
| num_dependientes | Num | 0.000 | ✗ | sin diferencia |

*`trabajador_extranjero`: el spread es alto pero la categoría A202 tiene solo 37 casos — estimación inestable.

**Conclusión pre-modelado**: de las 20 variables disponibles, anticipamos que 14 tienen poder predictivo real. Las 6 restantes serán descartadas antes de modelar para reducir ruido y sobreajuste — especialmente crítico con solo 1.000 registros.

---

## 6. Perfil del cliente de alto y bajo riesgo (basado en bad rates)

### Alto riesgo
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

### Bajo riesgo
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

El modelo no le entrega al analista de crédito una "probabilidad de mora". Le entrega un **puntaje** entre 0 y 1.000. El puntaje se construye con la convención estándar de la industria:

- **Mayor puntaje = menor riesgo** (mejor pagador).
- **PDO = 20**: cada 20 puntos de caída en el score duplica las chances de mora.
- **Score 600 = odds 50:1**: en ese punto, por cada 51 clientes hay 50 buenos y 1 malo.

La política de crédito se define como un **corte**:
- Score > corte → **Aprobado**
- Score ≤ corte → **Rechazado o revisión manual**

El corte específico se determina en el notebook optimizando la función de costo asimétrica.

---

## 8. Fuente y limitaciones del dataset

### Origen
Dataset donado en 1994 por el Prof. Dr. Hans Hofmann de la Universidad de Hamburgo como benchmark de ML para clasificación binaria con costos asimétricos. Es uno de los más citados en la literatura académica de credit scoring.

### Limitaciones que afectan la interpretación

1. **Datos históricos (años '90)**: el comportamiento crediticio cambia con el tiempo. Variables predictivas en 1994 pueden no serlo hoy.

2. **Una sola institución, un solo país**: los patrones son específicos del sistema bancario alemán de esa época. La proporción de trabajadores extranjeros (96.3%) refleja la política histórica de Gastarbeiter.

3. **Muestra pequeña**: 1.000 registros con 20 variables. En la industria se trabaja con millones y modelos que se recalibran trimestralmente.

4. **Variables no disponibles**: no están incluidos ingresos, patrimonio neto, deudas totales, score previo ni historial de pagos detallado — información que un banco real sí tendría.

5. **Categorías con bajo n**: historial A30 (n=40), A31 (n=49), propósito A44 (n=12), A48 (n=9) — estimaciones de bad rate y WoE con alta varianza estadística.

6. **Variable `cuenta_corriente` potencialmente problemática**: IV=0.659 es sospechosamente alto. Podría reflejar que el banco ya tomó decisiones basadas en ese saldo, generando sesgo de selección en los datos observados.
