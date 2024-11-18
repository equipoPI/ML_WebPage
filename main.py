#librerias
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import os
import gdown
import requests
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate

#componentes
from componentes.componente_prediccion import C_prediccion
from componentes.componente_clasificacion import C_clasificacion
from componentes.componente_eda import C_visualizacion

# Configuración de la página
st.set_page_config(page_title="Análisis y Predicción de Empleados", layout="wide")

# Usar HTML para personalizar el tamaño del texto
# Usar HTML para mostrar el logo con borde circular y el texto al lado, con fuente Georgia en 80px
# Usar HTML para mostrar solo el texto "equipoPI" con fuente Georgia en 80px
st.markdown("""
    <style>
        .custom-font {
            font-family: 'Georgia', serif;
            font-size: 100px;  /* Tamaño de fuente 80px */
            color: #FF6347;  /* Cambia el color a tu gusto */
        }
    </style>
    <h1 class="custom-font">equipoPI</h1>
""", unsafe_allow_html=True)


st.markdown("<h1 style='font-size: 50px;'>Análisis del Empleado</h1>", unsafe_allow_html=True)

# Creación de pestañas
tab1, tab2, tab3 = st.tabs(["Visualización EDA", "Modelo de Predicción", "Modelo de Clasificación"])

'''
# URLs de los modelos en Google Drive
url_modelprediccion = "https://drive.google.com/file/d/1HcdA69bo2Px8VB6divzKa_SVhqS-2mzn"
url_modelclasificacion = "https://drive.google.com/file/d/1BJAa4C4L_DLKorh3xOxGUL31c6J210Dd"
'''
# URLs de los modelos en Google Drive (usando el formato adecuado para gdown)
url_modelprediccion = "https://drive.google.com/uc?id=1HcdA69bo2Px8VB6divzKa_SVhqS-2mzn"
url_modelclasificacion = "https://drive.google.com/uc?id=1BJAa4C4L_DLKorh3xOxGUL31c6J210Dd"


# Directorio donde se guardarán los modelos
modelos_dir = "modelos"
os.makedirs(modelos_dir, exist_ok=True)

# Función para descargar un archivo si no existe
def descargar_modelo(url, path):
    if not os.path.exists(path):
        print(f"Descargando {os.path.basename(path)}...")
        try:
            # Utilizamos gdown para descargar el archivo
            gdown.download(url, path, quiet=False)
            print(f"Archivo descargado y guardado en {path}.")
        except Exception as e:
            print(f"Error en la descarga: {e}")
    else:
        print(f"El archivo {os.path.basename(path)} ya existe.")

# Descargar los modelos y asegurar que se complete antes de continuar

descargar_modelo(url_modelprediccion, os.path.join(modelos_dir, "modelprediccion.pkl"))
descargar_modelo(url_modelclasificacion, os.path.join(modelos_dir, "modelclasificacion.pkl"))

# Inicializar los modelos como None
model2 = None
model1 = None

'''
# Cargar los modelos con pickle solo si se descargaron correctamente
try:
    with open(os.path.join(modelos_dir, "modelprediccion.pkl"), "rb") as file1:
        model2 = pickle.load(file1)
    print("Modelo de predicción cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo de predicción: {e}")

try:
    with open(os.path.join(modelos_dir, "modelclasificacion.pkl"), "rb") as file:
        model1 = pickle.load(file)
    print("Modelo de clasificación cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo de clasificación: {e}")

# Verificar que los modelos están cargados antes de continuar
if model2 is not None and model1 is not None:
    print("Modelos cargados exitosamente. Continuando con el procesamiento...")
else:
    print("Error: No se pudieron cargar los modelos. El código no continuará.")
'''

with open('modelos/modelprediccion.pkl', 'rb') as file1:
    model2 = pickle.load(file1)

with open('modelos/modelclasificacion.pkl', 'rb') as file:
    model1 = pickle.load(file)
# Verifica que los modelos se hayan cargado correctamente antes de pasarlos a las funciones

df = pd.read_csv("modelos/archivo_combinado.csv")

# Contenido de la pestaña 1: Visualización EDA
with tab1:
    C_visualizacion(df)

# Contenido de la pestaña 2: Modelo de Predicción
with tab2:
    C_prediccion(model2)

# Contenido de la pestaña 3: Modelo de Clasificación
with tab3:
    C_clasificacion(model1)
    