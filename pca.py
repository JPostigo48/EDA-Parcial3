from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ubicar 'covid_DB.xlsx'
file_path = 'covid_DB.xlsx'

# Leer los datos del archivo Excel
data = pd.read_excel(file_path, usecols='A:DG')
# Excluir la columna 'Patient ID' y excluir columnas vacias
data = data.drop('Patient ID', axis=1)
data = data.dropna(axis=1, how='all')

# Convertir valores categóricos a numéricos
data = data.replace({'negative': 0, 'positive': 1})
data = data.replace({'detected': 1, 'not_detected': 0})
data = data.replace({'absent': 0,'normal': 1,'present': 1, 'not_done': 0})
data = data.replace({'clear': 0.2, 'lightly_cloudy': 0.4, 'cloudy': 0.6, 'altered_coloring': 0.8})
data = data.replace({'<1000': 500})
data = data.replace({'Não Realizado': '0.5'})
data = data.replace({'Ausentes': 0, 'Urato Amorfo --+':1, 'Oxalato de Cálcio -++':1, 'Oxalato de Cálcio +++':1, 'Urato Amorfo +++':1})
data = data.replace({'light_yellow': 0.3, 'yellow':0.6, 'citrus_yellow':0.75, 'orange':0.9})


# Imputar los valores NaN restantes con la media de cada columna
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data_imputed = imputer.fit_transform(data)

print(data_imputed)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_imputed)

# Aplica PCA
pca = PCA(n_components=65)  # número de componentes a reducir
principal_components = pca.fit_transform(scaled_data)

print(principal_components)
