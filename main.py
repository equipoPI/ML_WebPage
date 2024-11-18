# Librerías
import streamlit as st
import pandas as pd
import pickle
import os
import gdown

# Componentes
from componentes.componente_prediccion import C_prediccion
from componentes.componente_clasificacion import C_clasificacion
from componentes.componente_eda import C_visualizacion

# Configuración de la página
st.set_page_config(page_title="Análisis y Predicción de Empleados", layout="wide")

# Usar HTML para personalizar el tamaño del texto
st.markdown("""
    <style>
        .custom-font {
            font-family: 'Georgia', serif;
            font-size: 100px;  /* Tamaño de fuente 100px */
            color: #FF6347;  /* Cambia el color a tu gusto */
        }
    </style>
    <h1 class="custom-font">equipoPI</h1>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 50px;'>Análisis del Empleado</h1>", unsafe_allow_html=True)

# Creación de pestañas
tab1, tab2, tab3 = st.tabs(["Visualización EDA", "Modelo de Predicción", "Modelo de Clasificación"])

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

# Descargar los modelos solo si no existen
descargar_modelo(url_modelprediccion, os.path.join(modelos_dir, "modelprediccion.pkl"))
descargar_modelo(url_modelclasificacion, os.path.join(modelos_dir, "modelclasificacion.pkl"))

# Función para cargar los modelos en caché usando st.cache_resource
@st.cache_resource
def cargar_modelo(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

# Cargar los modelos (se cargan solo una vez)
model2 = cargar_modelo(os.path.join(modelos_dir, "modelprediccion.pkl"))
model1 = cargar_modelo(os.path.join(modelos_dir, "modelclasificacion.pkl"))

# Cargar el archivo combinado (usado para las visualizaciones)
df = pd.read_csv("modelos/archivo_combinado.csv")

# Función para limpiar el estado cuando se cambia de tab
def limpiar_estado_tab_actual(tab_seleccionado):
    if tab_seleccionado != "Visualización EDA" and 'input_data_eda' in st.session_state:
        del st.session_state['input_data_eda']
    if tab_seleccionado != "Modelo de Predicción" and 'input_data_prediccion' in st.session_state:
        del st.session_state['input_data_prediccion']
    if tab_seleccionado != "Modelo de Clasificación" and 'input_data_clasificacion' in st.session_state:
        del st.session_state['input_data_clasificacion']

# Contenido de la pestaña 1: Visualización EDA
with tab1:
    limpiar_estado_tab_actual("Visualización EDA")  # Limpiar las otras pestañas al entrar a esta
    C_visualizacion(df)
    '''
    # URLs de los modelos en Google Drive
    url_modelprediccion = "https://drive.google.com/file/d/1HcdA69bo2Px8VB6divzKa_SVhqS-2mzn"
    url_modelclasificacion = "https://drive.google.com/file/d/1BJAa4C4L_DLKorh3xOxGUL31c6J210Dd"
    '''

# Contenido de la pestaña 2: Modelo de Predicción
with tab2:
    limpiar_estado_tab_actual("Modelo de Predicción")  # Limpiar las otras pestañas al entrar a esta
    C_prediccion(model2)

# Contenido de la pestaña 3: Modelo de Clasificación
with tab3:
    limpiar_estado_tab_actual("Modelo de Clasificación")  # Limpiar las otras pestañas al entrar a esta
    C_clasificacion(model1)
