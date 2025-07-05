import pandas as pd

df = pd.read_csv('./housing.csv')

#print(df.head())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Cargar datos
df = pd.read_csv('./housing.csv')

df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']
df['rooms_per_household'] = df['total_rooms'] / df['households']

#Estos ratios y densidades aumentan la información para el modelo.
#casas con más dormitorios por cuarto pueden ser más grandes o mejor distribuidas.
#Puede detectar relaciones no lineales y factores sociales o económicos implícitos 

df = pd.get_dummies(df, columns=['ocean_proximity'])

# Separar X y y
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Evaluar modelo
print("Score de entrenamiento:", modelo.score(X_train, y_train))
print("Score de prueba:", modelo.score(X_test, y_test))

#¿El resultado fue mejor o peor?
#Fue mejor que sin procesar los datos.

#¿Por qué crees que es así?
#Porque al imputar nulos, crear nuevas características y codificar variables categóricas,
#el modelo tiene más información para aprender relaciones y predecir un poco mejor
