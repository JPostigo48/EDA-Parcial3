import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import vtk

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


# Aplicar PCA
pca = PCA(n_components=n_componentNumber)  # número de componentes a reducir
principal_components = pca.fit_transform(scaled_data)

# Cargar los datos originales incluyendo la columna "Patient ID"
data_with_id = pd.read_excel('covid_DB.xlsx')

# Crear un DataFrame con los componentes y el ID del paciente
df_componentes = pd.DataFrame(principal_components, columns=[f'Componente_{i}' for i in range(1, n_componentNumber + 1)])
df_componentes['Patient ID'] = data_with_id['Patient ID']  # Agregar la columna ID del Paciente

# Guardar el DataFrame en un archivo CSV
df_componentes.to_csv('componentes_principales.csv', index=False)

print(principal_components)

# Primero, escalar los datos. PCA es sensible a las escalas de las variables
scaler = StandardScaler()
data_scaled = scaler.fit_transform(principal_components)

# Inicializar PCA para reducir los datos a 3 dimensiones
pca = PCA(n_components=3)
data_3d = pca.fit_transform(data_scaled)

# Crear un objeto vtkPoints para almacenar los datos tridimensionales
points = vtk.vtkPoints()
for i in range(len(data_3d)):
    points.InsertNextPoint(data_3d[i])

# Crear un polydata para los puntos
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)

# Visualizar los puntos con esferas pequeñas
sphere = vtk.vtkSphereSource()
sphere.SetRadius(0.01)

glyph = vtk.vtkGlyph3D()
glyph.SetInputData(polydata)
glyph.SetSourceConnection(sphere.GetOutputPort())
glyph.SetScaleModeToScaleByScalar()
glyph.SetColorModeToColorByScalar()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(glyph.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Crear la ventana y el renderizador
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.1, 0.1)

# Establecer posición inicial para la cámara
camera = renderer.GetActiveCamera()
camera.SetPosition(0, 0, 5)
camera.SetFocalPoint(0, 0, 0)
camera.SetViewUp(0, 1, 0)

render_window.Render()
render_window_interactor.Start()
