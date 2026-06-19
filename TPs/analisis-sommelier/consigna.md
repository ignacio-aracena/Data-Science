# Análisis de datos | Entrenando a un sommelier

## Metadata
- Categoría: Análisis de Datos
- Entrega: Semana 4
- Estado en el curso: Done
- Fuente (Notion A422): https://playful-cantaloupe-94c.notion.site/An-lisis-de-datos-Entrenando-a-un-sommelier-1d01d850311681988f62d2724360a49c

## Datos
Archivos en `data/`:
- `Understanding_Wine_Chemistry_Waterhouse.pdf`
- `Wine.pdf`
Recursos / datasets externos:
- [archive.ics.uci.edu/dat…uality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)
- [CVRVVVinho Verde](https://www.vinhoverde.pt/en/)

## Consigna

Entrenando a un sommelier
Vaya al repositorio UCI Machine Learning Repository y descargue el archivo sobre clasificación de vinos. Si lo quiere descargar directamente en Python, puede hacerlo también usando el siguiente script
#Install the ucimlrepo

pip install ucimlrepo

#Import the dataset into your code

from ucimlrepo import fetch_ucirepo

# Fetch dataset
wine_quality = fetch_ucirepo(id=186)

# Data (as pandas dataframes)
X = wine_quality.data.features
y = wine_quality.data.targets

# Metadata
print(wine_quality.metadata)

# Variable information
print(wine_quality.variables)

​
Les comparto alguna información de los que produjeron el dataset:
“The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
For more details, consult: CVRVVVinho Verde or the reference [Cortez et al., 2009].
Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones).
Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.”
Información sobre las variables
Asimismo, les comparto información sobre las variables:
Variable Name
	
Role
	
Type
	
Description


fixed_acidity
	
Feature
	
Continuous
	


volatile_acidity
	
Feature
	
Continuous
	


citric_acid
	
Feature
	
Continuous
	


residual_sugar
	
Feature
	
Continuous
	


chlorides
	
Feature
	
Continuous
	


free_sulfur_dioxide
	
Feature
	
Continuous
	


total_sulfur_dioxide
	
Feature
	
Continuous
	


density
	
Feature
	
Continuous
	


pH
	
Feature
	
Continuous
	


sulphates
	
Feature
	
Continuous
	


alcohol
	
Feature
	
Continuous
	


quality
	
Target
	
Integer
	
score between 0 and 10


color
	
Other
	
Categorical
	
red or white
Objetivo:
Realizar un análisis exhaustivo de variables (gráfico y analítico) para determinar la calidad y el color del vino. Entregable: presentación (pdf) + código (collab o link al repositorio)
Next steps (no forma parte de la entrega de esta semana)
Construir un modelo de redes secuencial que permita determinar la calidad y el color del vino en base a sus variables fisicoquímicas. Estudie la posible presencia de overfitting, y en caso de observarla, plantee posibles soluciones para el problema. Analice la bondad de ajuste del modelo.
Utilice un autoencoder para reducir la dimensionalidad del problema. Explique en base a lo encontrado en el punto 1 porque podría o no tener sentido hacer esto, y comente sobre cuan efectivo resultó en la reducción de la dimensionalidad.
Reestime el modelo, pero ahora usando un modelo funcional. Compare con los resultados encontrados en el punto 2.
