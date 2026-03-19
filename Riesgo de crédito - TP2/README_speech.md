# Speech — TP2: Análisis de Riesgo de Crédito
**Ignacio Aracena · Tomás Arizu | Data Science A42**

> Guía de presentación slide por slide. Duración estimada total: **12–15 minutos**.
> Tono: claro, seguro, orientado al negocio. Evitar leer los números literalmente — contarlos como historia.

---

## Slide 1 — Portada + Índice
*~45 segundos*

> "Buenas, somos Ignacio y Tomás. El trabajo que vamos a presentar hoy toma un dataset de 1.000 solicitudes de préstamo de un banco alemán y busca responder una sola pregunta: ¿cómo decide un banco a quién le presta y a quién no?
>
> Lo que vamos a mostrar no es solo un modelo que predice. Es un sistema completo: desde entender el problema, limpiar las variables, elegir las que realmente importan, calibrar el modelo para que tome las decisiones correctas dadas las consecuencias económicas reales, y traducir todo eso en una herramienta operacional que el banco puede usar mañana mismo.
>
> La estructura la tienen en pantalla — los vamos a llevar por ese camino."

---

## Slide 2 — Problema & Dataset
*~90 segundos*

> "Arranquemos con el problema. Tenemos 1.000 solicitudes de crédito: 700 buenos pagadores y 300 que entraron en mora. El banco nos pide reproducir su criterio de aprobación.
>
> Ahora bien, hay algo que hace este problema distinto a una clasificación binaria común: **los errores no cuestan lo mismo**. Y eso cambia todo.
>
> Si el modelo aprueba a alguien que no va a pagar, el banco pierde el capital prestado, los intereses, y encima tiene que hacer provisiones regulatorias. Eso vale **5**. Si en cambio rechaza a alguien que sí habría pagado, solo pierde la oportunidad de negocio — ese cliente se va a otro banco. Eso vale **1**.
>
> Esta asimetría 5 a 1 tiene una consecuencia directa que van a ver repetida en toda la presentación: el threshold de aprobación no puede ser 0.50, y el accuracy no es la métrica que importa. El objetivo real es minimizar ese costo asimétrico."

---

## Slide 3 — Análisis Exploratorio: Variables Numéricas
*~90 segundos*

> "Antes de modelar, miramos los datos. De las 7 variables numéricas, 4 discriminan entre buenos y malos pagadores y 3 no aportan nada útil.
>
> Los hallazgos más concretos: los malos pagadores piden créditos que duran en promedio casi **5 meses más** que los buenos. Eso tiene mucho sentido — un plazo más largo implica sostener el esfuerzo financiero durante más tiempo, y cualquier imprevisto en el medio puede romper la cadena de pagos.
>
> También son en promedio casi **1.000 Deutsche Marks más caros** los créditos que piden los malos pagadores. Más monto, más tensión.
>
> Y en el otro extremo, tres variables que en principio parecerían relevantes — cuántos años llevan en la misma casa, cuántos créditos tienen en el banco, cuántos dependientes — resultan que no discriminan nada. El banco captura esa información por otras vías.
>
> Usamos Mann-Whitney para los tests, porque las distribuciones son sesgadas y el t-test clásico no aplica."

---

## Slide 4 — Análisis Exploratorio: Variables Categóricas
*~2 minutos*

> "Acá es donde aparecen las cosas más interesantes — y las más contraintuitivas.
>
> La variable con mayor spread de riesgo dentro de las que vamos a usar en el modelo es el **historial crediticio**: hay 45 puntos de diferencia entre la categoría de mayor y menor riesgo.
>
> El dato que más llama la atención: el cliente que **nunca tuvo un crédito previo** tiene el bad rate más alto del dataset — 62.5%. La intuición diría lo contrario, que alguien sin deudas es más confiable. La realidad es otra: el banco no tiene información sobre esa persona. Es máxima incertidumbre. Y ante la incertidumbre, el riesgo histórico es alto.
>
> El segundo dato paradójico: los clientes con **cuenta crítica o créditos en otros bancos** tienen el menor bad rate, 17%. Son personas que ya negociaron deudas, ya demostraron capacidad de pago. El sistema bancario los conoce.
>
> Y una tercera paradoja: tener **empleo hace menos de un año** es más riesgoso que estar desempleado. El desempleado ya sabe que no puede pagar mucho — el recién empleado todavía tiene ingresos inestables y a veces sobreestima su capacidad futura.
>
> Estos patrones no son ruido. Son señales reales que el modelo va a capturar."

---

## Slide 5 — Análisis Exploratorio: Correlaciones
*~45 segundos*

> "Un paso que no se puede saltear antes de modelar: revisar si hay variables que digan lo mismo, lo que en la jerga se llama multicolinealidad.
>
> La correlación más alta que encontramos es entre la duración del crédito y el monto: 0.63. Tiene todo el sentido — si pedís más plata, necesitás más tiempo para devolverla. Pero 0.63 no es lo suficientemente alto como para descartar ninguna de las dos. El umbral de preocupación real está en 0.70.
>
> Conclusión: no hay multicolinealidad severa. Las variables pueden convivir en el modelo sin problemas."

---

## Slide 6 — Selección de Variables: Information Value
*~90 segundos*

> "Llegamos a uno de los pasos más críticos del trabajo: decidir con qué variables modelar. Tenemos 20, pero más no es siempre mejor — especialmente con solo 1.000 registros.
>
> Usamos una métrica llamada **Information Value** que cuantifica cuánta información aporta cada variable para separar buenos de malos. El resultado: 14 de las 20 variables tienen señal real. Las otras 6 son prácticamente ruido.
>
> Pero el hallazgo más importante no es ese. Es que la variable con **mayor** poder predictivo de todo el dataset — la cuenta corriente — la excluimos deliberadamente.
>
> ¿Por qué? Su IV es 0.66, muy por encima del umbral de sospecha. Con 1.000 registros, ese número tan alto es una señal de alerta. El banco probablemente ya usaba el saldo en cuenta corriente para decidir quién llegaba al dataset en primer lugar. Si la incluimos, el modelo estaría capturando ese sesgo histórico, no el riesgo real del cliente. Preferimos un modelo más honesto.
>
> Las 14 variables que sí seleccionamos tienen los IV entre 0.02 y 0.30 — señal real, sin exagerar."

---

## Slide 7 — Modelo & Pipeline
*~90 segundos*

> "Con las 14 variables seleccionadas, construimos el pipeline. Cuatro pasos bien definidos: selección de variables, división del dataset en tres partes, encoding de variables categóricas, y normalización.
>
> El modelo elegido es **Regresión Logística con regularización L1**. Y la decisión no es por comodidad — la justificamos contra cinco modelos alternativos, incluyendo XGBoost.
>
> XGBoost nos da 4 puntos más de AUC. Pero en este contexto, esa diferencia no justifica lo que perdemos: interpretabilidad nativa. Con regresión logística, podemos decirle al banco exactamente por qué rechazamos cada cliente — feature por feature. Eso es lo que el regulador exige. XGBoost necesita una capa adicional de explicabilidad que suma complejidad operativa.
>
> El resultado del modelo: AUC de 0.71. Por encima del umbral mínimo de la industria para credit scoring. Y de las 38 variables que entran al modelo luego del encoding, la regularización L1 elimina automáticamente 15 — el modelo decide que no las necesita."

---

## Slide 8 — Resultados con Threshold 0.50
*~60 segundos*

> "Evaluemos primero el modelo con el threshold default del 50%. El resultado: detecta 38 de los 60 malos pagadores en el set de test, y comete 53 rechazos incorrectos.
>
> Costo total: 163 unidades. Comparado con el baseline de aprobar todo — que costaría 300 — ya es una reducción importante del 45%.
>
> Pero hay un problema. Con threshold 0.50, el modelo trata los dos tipos de error como si costaran lo mismo. Y ya sabemos que no es así. Podemos hacerlo mejor."

---

## Slide 9 — Costo Asimétrico: Optimización del Threshold
*~2 minutos*

> "Acá está el salto más importante del trabajo. Llevamos el threshold de 0.50 a **0.26**.
>
> ¿Qué significa eso? Que el modelo empieza a clasificar como malo a cualquier cliente que tenga más del 26% de probabilidad de no pagar, en lugar del 50%.
>
> El resultado: pasa de detectar 38 malos a detectar **55 de los 60**. Solo 5 malos se escapan sin detección. El costo total cae a 137 — una reducción del **54.3% respecto al baseline**.
>
> Sí, ahora rechaza incorrectamente a 112 buenos clientes en lugar de 53. Y el accuracy cae al 41%. Pero ese número no importa. Lo que importa es el costo económico real. Y ese costo bajó.
>
> El razonamiento es simple: cada mal pagador que se escapa cuesta 5 unidades. Cada buen cliente rechazado de más cuesta 1. Rechazar a 59 buenos adicionales cuesta 59 unidades. A cambio, evitamos 33 malos que habrían costado 165 unidades. El balance es ampliamente positivo.
>
> Y un detalle metodológico importante: buscamos ese threshold óptimo en el set de validación, no en el de test. El test tiene que quedar completamente intocado hasta la evaluación final. Si no, estaríamos ajustando el modelo a datos que se supone no conoce."

---

## Slide 10 — Interpretabilidad: Odds Ratios
*~90 segundos*

> "Una de las ventajas más grandes de haber elegido regresión logística es que podemos explicar exactamente qué mueve el riesgo — y cuánto.
>
> Los números se llaman Odds Ratios. Un OR mayor a 1 aumenta el riesgo, uno menor a 1 lo reduce.
>
> Los tres drivers más fuertes que encontramos:
>
> Primero, **la duración del crédito**: cada mes adicional de plazo multiplica el riesgo por 1.44. No es aditivo, es multiplicativo — los créditos largos escalan el riesgo de forma exponencial.
>
> Segundo, **el historial crediticio**: los clientes con cuenta crítica o créditos en otros bancos tienen un 32% menos de riesgo que la categoría base. Demostraron que pueden manejar deuda.
>
> Tercero, **los ahorros**: tener más de 1.000 Deutsche Marks ahorrados reduce el riesgo un 22%. El colchón financiero es una señal de resiliencia.
>
> Este tipo de explicación es exactamente lo que el banco necesita para justificar una decisión ante el cliente o ante el regulador."

---

## Slide 11 — Scorecard
*~90 segundos*

> "El último paso es traducir todo esto al lenguaje que usa el banco. En la industria bancaria, el modelo no le entrega a un analista una probabilidad de 0.32. Le entrega un **puntaje**: el scorecard.
>
> La lógica es simple: a mayor puntaje, mejor pagador. Usamos la convención estándar de la industria, donde cada 20 puntos de caída en el score duplica el riesgo de mora.
>
> En nuestro test, los scores van de 423 a 572 puntos. Los buenos pagadores sacan en promedio 497, los malos 477. Una diferencia de 20 puntos.
>
> El corte óptimo está en **517 puntos**: quien supere ese score, aprobado. Quien no, rechazado o revisión manual.
>
> Con ese corte, solo el 16.5% de los clientes del test aprueba. Es una política conservadora — coherente con el costo 5 a 1. En producción, el banco podría ajustar ese corte según cuánto riesgo está dispuesto a tolerar, o crear una zona intermedia para revisión humana."

---

## Slide 12 — Perfiles de Riesgo
*~60 segundos*

> "Para hacer más tangible lo que aprendimos, construimos dos perfiles concretos: el cliente de alto riesgo y el de bajo riesgo.
>
> El cliente de alto riesgo pide un crédito largo, no tiene historial previo, tiene casi nada ahorrado, lleva menos de un año en su trabajo, y alquila. Cada uno de esos factores, por separado, ya eleva el riesgo. Combinados, lo disparan.
>
> El cliente de bajo riesgo tiene historial demostrado con otros bancos, plazo corto, más de 1.000 DM ahorrados, lleva varios años en el mismo empleo, y tiene su propia vivienda. El perfil de alguien con estabilidad financiera real.
>
> Esta tabla es útil no solo para el modelo — es útil para que el analista de crédito entienda la lógica detrás de cada decisión."

---

## Slide 13 — Conclusiones
*~2 minutos*

> "Seis conclusiones que resumen el trabajo.
>
> Una: **el objetivo se cumplió**. El modelo detecta 55 de 60 malos pagadores en el test, con un costo total de 137 — el 54% menos que aprobar todo.
>
> Dos: **14 de 20 variables aportan señal real**. Las otras 6 son ruido estadístico. Con 1.000 registros, incluirlas solo habría dañado el modelo.
>
> Tres: **el threshold correcto no es 0.50**. La función de costo asimétrica lo desplaza a 0.26. El accuracy puede caer al 41% y eso está bien — no es la métrica que importa.
>
> Cuatro: **la variable más poderosa del dataset la dejamos afuera**. La cuenta corriente tiene un IV sospechosamente alto que sugiere endogeneidad o sesgo histórico. Incluirla habría sido hacer trampa.
>
> Cinco: **AUC de 0.71 es el techo práctico con estas variables**. Sin datos de ingresos, patrimonio o historial de pagos detallado, hay un límite a lo que cualquier modelo puede hacer. XGBoost sube a 0.75 — pero no justifica perder la interpretabilidad nativa en un contexto bancario regulado.
>
> Seis: **el scorecard es directamente operable**. El banco solo necesita calcular un puntaje con una fórmula, compararlo con el corte de 517, y tomar la decisión. No hace falta entender el modelo — solo aplicarlo."

---

## Slide 14 — Cierre / Gracias
*~30 segundos*

> "En síntesis: tomamos 1.000 solicitudes de préstamo, entendimos la lógica económica detrás del problema, seleccionamos las variables que realmente importan, calibramos el modelo para minimizar el costo correcto — no el accuracy — y entregamos un scorecard listo para producción.
>
> El costo cayó de 300 a 137. Detectamos 55 de 60 malos pagadores. Y el banco puede explicar cada decisión.
>
> Gracias, y quedamos para preguntas."

---

## Notas generales para la presentación

- **Timing sugerido**: 12–15 minutos totales. Slides 2, 4, 9 y 13 tienen más contenido — no apresurarse en ellas.
- **Números clave a mencionar de memoria**: 300→137 (reducción de costo), 55/60 (malos detectados), AUC 0.71, threshold 0.26, score de corte 517.
- **Preguntas probables del jurado**:
  - *¿Por qué no usaron XGBoost si da mejor AUC?* → Respuesta lista en slides 7 y 13.
  - *¿Por qué excluyeron cuenta_corriente?* → IV sospechoso, posible data leakage / sesgo de selección.
  - *¿El accuracy del 41% no es demasiado bajo?* → Accuracy engañoso. El objetivo es minimizar 5×FN + 1×FP, no maximizar aciertos.
  - *¿Qué harían con más datos?* → Incorporar variables de ingresos y patrimonio, recalibrar por concept drift, evaluar si el gap de XGBoost justifica su adopción con SHAP como mecanismo de explicabilidad.
