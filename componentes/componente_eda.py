def C_visualizacion(df):
    import streamlit as st
    import pandas as pd
    import pickle
    import numpy as np
    import plotly.express as px
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler
    from scipy import interpolate
    from scipy.ndimage import uniform_filter1d


    st.header("Visualización del Análisis Exploratorio de Datos (EDA)")
    # Agrega tu código para visualizaciones EDA aquí
    st.markdown(
    '''
    <div style="text-align: justify;">
        Nuestro grupo de trabajo decidió centrarse en el análisis de la plantilla laboral de las empresas con el fin de implementar algoritmos de Machine Learning que permitan a los directivos o sectores de la empresa tomar decisiones basadas en los resultados de los algoritmos. Para cumplir con el objetivo, utilizamos los siguientes datos, que se obtuvieron al combinar dos datasets con información relacionada a empleados del sector privado. Con base en esos datos, se analizaron diferentes variables y situaciones para determinar un patrón o similitudes.
    </div>
    ''',
    unsafe_allow_html=True
    )

    st.dataframe(df)

    # Agrupar por 'Deserción'
    df_count = df.groupby('Deserción').count().reset_index()
    fig = px.pie(df_count, values="ID Empleado", names="Deserción", title="Deserción")
    fig.update_layout(title_x=0.5)  # Centrar el título

    #Agrupar Satisfaccion del Trabajo
    df_count5 = df.groupby('Satisfacción del Trabajo').count().reset_index()
    # Crear gráfico de pastel para 'Satisfacción del Trabajo'
    fig5 = px.pie(df_count5, values="ID Empleado", names="Satisfacción del Trabajo", title="Satisfacción del Trabajo")
    # Centrar el título
    fig5.update_layout(title_x=0.5)


    st.header("Graficos Ilustrativos:")

    # Crear dos columnas para mostrar los gráficos
    col1, col2 = st.columns(2)

    # Mostrar el primer gráfico en la primera columna
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    # Mostrar el segundo gráfico en la segunda columna
    with col2:
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
    '''
    <div style="text-align: justify;">
        Para analizar la deserción y la satisfacción laboral, utilizamos gráficos de torta. En el gráfico de deserción, se observa que el 64.1% de los empleados permanecen en la empresa (Stayed), mientras que el 35.9% ha dejado la organización (Left). Por otro lado, el gráfico de satisfacción laboral muestra que el 42.6% de los empleados se encuentran en un nivel de satisfacción 'High', el 24.4% en 'Very High', el 13.5% en 'Low', y el 19.4% en 'Medium'. Estos datos proporcionan una visión clara de la retención y el bienestar de los empleados dentro de la empresa.
    </div>
    ''',
    unsafe_allow_html=True
    )


    #definimos la creacion del grafico de Pareto
    df_pareto = df['Rol de Trabajo'].value_counts()
    pareto_cumulative = df_pareto.cumsum() / df_pareto.sum() * 100

    # Crear la figura
    fig1 = go.Figure()

    # Barras de frecuencia
    fig1.add_trace(go.Bar(
        x=df_pareto.index,
        y=df_pareto.values,
        name='Frecuencia',
        marker_color='blue',
        yaxis='y1'
    ))

    # Línea de porcentaje acumulado
    fig1.add_trace(go.Scatter(
        x=df_pareto.index,
        y=pareto_cumulative,
        name='Porcentaje Acumulado',
        mode='lines+markers',
        line=dict(color='orange', width=2),
        marker=dict(symbol='circle'),
        yaxis='y2'
    ))

    # Agregar línea del 80%
    fig1.add_shape(
        type="line",
        x0=-0.5,
        x1=len(df_pareto) - 0.5,
        y0=80,
        y1=80,
        line=dict(color='red', dash='dash'),
        xref="x",
        yref="y2"
    )

    # Configuración de los ejes y el diseño
    fig1.update_layout(
        title=dict(
            text='Gráfico de Pareto',
            x=0.5,  # Centrar el título
            xanchor='center'
        ),
        xaxis=dict(title='Categoría'),
        yaxis=dict(
            title='Frecuencia',
            showgrid=False,
            side='left'
        ),
        yaxis2=dict(
            title='Porcentaje Acumulado (%)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(x=0.5, y=-0.2, orientation='h'),
        template='plotly_white',
        height=600,
        width=900
    )

    # Mostrar el gráfico centrado en la pantalla de Streamlit
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(
    '''
    <div style="text-align: justify;">
        Utilizamos este gráfico para ver qué sectores contaban con un mayor número de trabajadores. Las primeras seis categorías (Technology, Education, Healthcare, Media, Finance, Sales Executive) parecen cubrir aproximadamente el 80% de la frecuencia total. Estas se podrían considerar las categorías más relevantes en el análisis y serían el 'punto de enfoque' si se busca priorizar esfuerzos o recursos para obtener resultados generales. Las categorías restantes tienen menos impacto en términos de frecuencia, pero se podrían utilizar para hacer análisis más reducidos y realizar enfoques especializados en grupos pequeños de las empresas.
    </div>
    ''',
    unsafe_allow_html=True
    )

    # Gráfico de violín para 'Distancia a Casa' vs 'Trabajo Remoto'
    fig_violin = px.violin(
        df,
        x='Trabajo Remoto',
        y='Distancia a Casa',
        box=True,
        points=False,
        color='Trabajo Remoto',
        title="Distancia a Casa vs Trabajo Remoto",
        color_discrete_sequence=['orange', 'purple']
    )

    # Centrar el título
    fig_violin.update_layout(title_x=0.4)

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig_violin, use_container_width=True, key="violin_plot")

    st.markdown(
    '''
    <div style="text-align: justify;">
        A su vez, nos interesaba el impacto del trabajo remoto en la vida de los trabajadores, ya que, a partir de la pandemia, muchas compañías que no contaban con esta modalidad la introdujeron. Usamos un gráfico de violín en el que se puede apreciar que los empleados que trabajan de forma remota parecen vivir más lejos de su lugar de trabajo, o que la distancia a su casa no tiene mucha importancia. En cambio, los empleados que no trabajan de forma remota tienden a vivir más cerca, con la mayoría concentrada en distancias de hasta 20 km, ya que vivir a más distancia puede ocasionarles mayores gastos de dinero y tiempo en el transporte, en comparación con vivir más cerca de su lugar de trabajo.
    </div>
    ''',
    unsafe_allow_html=True
    )

    # Variables a graficar
    hist_columns = {
        'Edad': 42,
        'Antigüedad en la Empresa': 43,
        'Ingreso Mensual': 100
    }

    # Crear tres columnas para mostrar los gráficos
    hist_col1, hist_col2, hist_col3 = st.columns(3)
    hist_columns_layout = [hist_col1, hist_col2, hist_col3]  # Lista de columnas para asignar gráficos

    for i, (col, bins) in enumerate(hist_columns.items()):
        # Crear histograma con el número de bins especificado para cada variable
        hist_values, bin_edges = np.histogram(df[col], bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Calcular el promedio móvil (suavizado) de las barras
        smoothed_values = uniform_filter1d(hist_values, size=3)  # Tamaño de la ventana de suavizado
        
        # Crear figura
        fig_hist = go.Figure()
        
        # Barras del histograma
        fig_hist.add_trace(go.Bar(
            x=bin_centers,
            y=hist_values,
            name="Frecuencia",
            marker_color='blue',
            width=(bin_edges[1] - bin_edges[0]) * 0.8  # Controla el ancho de las barras
        ))

        # Curva de promedio móvil
        fig_hist.add_trace(go.Scatter(
            x=bin_centers,
            y=smoothed_values,
            mode='lines',
            name='Promedio Móvil',
            line=dict(color='red', width=2)
        ))
        
        # Configurar diseño
        fig_hist.update_layout(
        title=dict(
            text=f"{col}",
            x=0.5,  # Centrar el título
            xanchor='center'
        ),
        xaxis=dict(title=col),
        yaxis=dict(title="Frecuencia"),
        bargap=0.2,  # Controla la separación entre barras
        template="plotly_white"
    )
        
        # Mostrar gráfico en la columna correspondiente
        with hist_columns_layout[i]:  # Asignar gráfico a cada columna
            st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{col}")

    st.markdown("""
        ### Distribución de Variables Clave

        1. **Edad**: 
        - La mayoría de los empleados se encuentran en un rango de edad joven a mediana.
        - Hay una disminución en la cantidad de empleados mayores de 45 años, lo que podría indicar una tendencia de deserción o jubilación en edades más avanzadas.

        2. **Antigüedad en la Empresa**: 
        - La mayoría de los empleados son relativamente nuevos en la empresa, con pocos alcanzando más de 10 años de antigüedad.
        - Esto podría indicar una alta rotación de personal o una estructura joven de la empresa.

        3. **Ingreso Mensual**: 
        - La mayoría de los empleados tienen ingresos en un rango de **2000 a 12500**.
        - Una minoría recibe sueldos más altos, posiblemente debido a diferencias en roles o niveles de experiencia.
        """)
    