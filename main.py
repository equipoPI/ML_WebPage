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

# URLs de los modelos en Google Drive (usar enlaces directos con el formato "uc?id=FILE_ID")
url_modelprediccion = "https://drive.google.com/file/d/1HcdA69bo2Px8VB6divzKa_SVhqS-2mzn"
url_modelclasificacion = "https://drive.google.com/file/d/1BJAa4C4L_DLKorh3xOxGUL31c6J210Dd"

# Directorio donde se guardarán los modelos
modelos_dir = "modelos"
os.makedirs(modelos_dir, exist_ok=True)  # Crear la carpeta si no existe

# Rutas locales de los modelos
path_modelprediccion = os.path.join(modelos_dir, "modelprediccion.pkl")
path_modelclasificacion = os.path.join(modelos_dir, "modelclasificacion.pkl")

# Función para descargar un archivo si no existe
def descargar_modelo(url, path):
    if not os.path.exists(path):
        print(f"Descargando {os.path.basename(path)}...")
        response = requests.get(url)
        response.raise_for_status()  # Verificar si hubo errores
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"Guardado en {path}.")
    else:
        print(f"El archivo {os.path.basename(path)} ya existe en {path}.")

# Descargar los modelos
descargar_modelo(url_modelprediccion, path_modelprediccion)
descargar_modelo(url_modelclasificacion, path_modelclasificacion)

# Cargar los modelos con pickle
with open(path_modelprediccion, "rb") as file1:
    model2 = pickle.load(file1)

with open(path_modelclasificacion, "rb") as file:
    model1 = pickle.load(file)

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
    