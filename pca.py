import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

# Ubicar 'covid_DB.xlsx'
file_path = 'covid_DB.xlsx'

# Leer los datos del archivo Excel
data = pd.read_excel(file_path, usecols='A:DG')


# Excluir la columna 'Patient ID' 
data = data.drop('Patient ID', axis=1)
# Convertir valores categóricos a numéricos
data = data.replace({'negative': 0, 'positive': 1})
data = data.replace({'detected': 1, 'not_detected': 0})
data = data.replace({'absent': 0,'normal': 1,'present': 1, 'not_done': 0})
data = data.replace({'clear': 0.2, 'lightly_cloudy': 0.4, 'cloudy': 0.6, 'altered_coloring': 0.8})
data = data.replace({'<1000': 500})
data = data.replace({'Não Realizado': '0.5'})
data = data.replace({'Ausentes': 0, 'Urato Amorfo --+':1, 'Oxalato de Cálcio -++':1, 'Oxalato de Cálcio +++':1, 'Urato Amorfo +++':1})
data = data.replace({'light_yellow': 0.3, 'yellow':0.6, 'citrus_yellow':0.75, 'orange':0.9})


# Excluir columnas vacias
data = data.dropna(axis=1, how='all')
# Imputar los valores NaN restantes con la media de cada columna
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data_imputed = imputer.fit_transform(data)


#print(data_imputed)

# Escalar los datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_imputed)

# Aplicar PCA con todos los componentes
pca = PCA()
pca.fit(scaled_data)

# Calcular la varianza explicada acumulada
varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)

# Graficar la varianza explicada acumulada
plt.figure()
plt.plot(varianza_acumulada)
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.show()

# Encontrar el número de componentes para una varianza explicada >= 95%
n_componentNumber = np.where(varianza_acumulada >= 0.95)[0][0] + 1
print("Número mínimo de componentes para explicar al menos el 95% de la varianza:", n_componentNumber)


# Aplica PCA
pca = PCA(n_components=n_componentNumber)  # número de componentes a reducir
principal_components = pca.fit_transform(scaled_data)

print(principal_components)
