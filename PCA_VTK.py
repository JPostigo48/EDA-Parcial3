import vtk
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Cargar los datos
data = pd.read_csv('covid_DB.csv', delimiter=';')

# Seleccionar columnas numéricas para PCA
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data_numeric = data[numeric_cols]

# Imputar valores faltantes
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data_numeric)

# Escalar los datos al rango (0, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Aplicar PCA
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