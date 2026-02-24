import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="VIMS Predictor - Cat 797F", layout="wide")

# --- CUSTOMIZACIÓN DE CABECERA ---
# Puedes ajustar estos valores
ALTO_CABECERA = "120px"
ANCHO_LOGO = 482 # px
ALTO_LOGO = 81  # px

st.markdown(f"""
    <style>
    .header-container {{
        display: flex;
        align-items: center;
        height: {ALTO_CABECERA};
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
    }}
    .logo-section {{
        flex: 1;
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .title-section {{
        flex: 2;
        padding-left: 20px;
    }}
    </style>
    """, unsafe_allow_html=True)

# HTML para la cabecera
st.markdown(f"""
    <div class="header-container">
        <div class="logo-section">
            <img src="fondo1.jpg" width="{ANCHO_LOGO}" height="{ALTO_LOGO}">
        </div>
        <div class="title-section">
            <h1>Sistema de Predicción de Fallas Críticas - Flota Cat 797F</h1>
            <p>Monitoreo Predictivo basado en Telemetría VIMS - Andes Peruanos</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR / MENÚ ---
st.sidebar.title("Navegación")
opcion = st.sidebar.radio("Ir a:", [
    "2.0 Inicio", 
    "2.1 Cargar Data", 
    "2.2 Análisis de Data", 
    "2.3 Entrenamiento ML", 
    "2.4 Predicción de Guardia"
])

# --- LÓGICA DE SESIÓN (Para persistir el modelo) ---
if 'modelo' not in st.session_state:
    st.session_state.modelo = None
    st.session_state.scaler = None
    st.session_state.data = None

# --- SECCIONES ---

# 2.0 INICIO
if opcion == "2.0 Inicio":
    st.subheader("Contexto del Proyecto")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        **Caso:** Estamos analizando la flota de transporte de una unidad minera en los Andes peruanos. 
        El objetivo es predecir si un **Caterpillar 797F** terminará su guardia de 12 horas de forma normal 
        o si sufrirá una falla mecánica que obligue a remolcarlo, basándonos en la telemetría 
        recolectada por el sistema VIMS.
        """)
    with col2:
        # Aquí cargarías tu imagen local: st.image("camion.jpg")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/CAT_797F_Mining_Truck.jpg/640px-CAT_797F_Mining_Truck.jpg", caption="Cat 797F en operación")

# 2.1 CARGAR DATA
elif opcion == "2.1 Cargar Data":
    st.subheader("Carga de Historial de Telemetría (CSV)")
    uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success("¡Datos cargados correctamente!")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)

# 2.2 ANÁLISIS DE DATA
elif opcion == "2.2 Análisis de Data":
    if st.session_state.data is not None:
        df = st.session_state.data
        st.subheader("Visualización Avanzada de Telemetría")
        
        c1, c2 = st.columns(2)
        
        # 1. Distribución del Estado (Pie Chart)
        fig1 = px.pie(df, names='estado_maquina', title="Proporción Normal vs Fallo", hole=0.4, color_discrete_sequence=['#2ecc71', '#e74c3c'])
        c1.plotly_chart(fig1)

        # 2. Matriz de Correlación (Heatmap)
        fig2 = px.imshow(df.corr(), text_auto=True, title="Correlación de Sensores", color_continuous_scale='RdBu_r')
        c2.plotly_chart(fig2)

        # 3. Vibración vs Temperatura (Scatter)
        fig3 = px.scatter(df, x='temperatura_motor', y='vibracion', color='estado_maquina', 
                         title="Relación Temperatura / Vibración", trendline="ols")
        st.plotly_chart(fig3, use_container_width=True)

        c3, c4 = st.columns(2)
        
        # 4. Boxplot Horas vs Fallo
        fig4 = px.box(df, x='estado_maquina', y='horas_operacion', title="Impacto de Horas de Operación", color='estado_maquina')
        c3.plotly_chart(fig4)

        # 5. Histograma de Carga
        fig5 = px.histogram(df, x='carga_promedio', nbins=30, title="Distribución de Carga (Toneladas)", color_discrete_sequence=['#f1c40f'])
        c4.plotly_chart(fig5)

        # 6. Parallel Coordinates (Innovador para ver patrones)
        fig6 = px.parallel_coordinates(df, color="estado_maquina", title="Patrones Multivariables de Fallo")
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.warning("Primero carga la data en la sección 2.1")

# --- 2.3 GENERACIÓN Y APLICACIÓN DE ML ---
elif opcion == "2.3 Entrenamiento ML":
    if st.session_state.data is not None:
        st.subheader("Pipeline de Entrenamiento")
        df = st.session_state.data.copy()

        # 1. Manejo de Nulos
        df = df.dropna()
        st.write("✔️ Valores nulos procesados.")

        # 2. Manejo de Outliers (¡OJO AQUÍ!)
        # Solo aplicamos limpieza a variables que no definen el fallo directamente
        # o usamos un multiplicador más alto (3.0 en lugar de 1.5) para no borrar los fallos.
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        # Filtramos pero manteniendo un margen amplio
        df = df[~((df < (Q1 - 3 * IQR)) | (df > (Q3 + 3 * IQR))).any(axis=1)]
        st.write("✔️ Outliers filtrados (Margen amplio para conservar fallos).")

        # 3. Verificación de Clases (Seguro de vida para tu código)
        if df['estado_maquina'].nunique() < 2:
            st.error("⚠️ La limpieza de datos eliminó todos los casos de fallo. Se procedió a restaurar la data original para el entrenamiento.")
            df = st.session_state.data.copy()

        # 4. Preparación y Escalamiento
        X = df.drop('estado_maquina', axis=1)
        y = df['estado_maquina']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        st.write("✔️ Escalamiento de variables completado.")

        # 5. Entrenamiento
        model = LogisticRegression(class_weight='balanced') # Ayuda si hay pocos fallos
        model.fit(X_train_scaled, y_train)
        
        # Guardar en la sesión
        st.session_state.modelo = model
        st.session_state.scaler = scaler
        
        st.success("✅ Modelo de ML entrenado y generado")
        
        # Evaluación visual para la clase
        acc = accuracy_score(y_test, model.predict(X_test_scaled))
        st.metric("Precisión del Modelo", f"{acc:.2%}")
    else:
        st.warning("Por favor, carga la data en la sección 2.1")
# 2.4 PREDICCIÓN
elif opcion == "2.4 Predicción de Guardia":
    if st.session_state.modelo is not None:
        st.subheader("Inspección de Camión para Próxima Guardia")
        
        # Simulamos la data de un camión específico
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            t = st.number_input("Temperatura Motor", value=95.0)
            v = st.number_input("Vibración", value=5.2)
            p = st.number_input("Presión Aceite", value=45.0)
        with col_p2:
            h = st.number_input("Horas Operación", value=5000)
            c = st.number_input("Carga Promedio", value=180.0)
            a = st.number_input("Antigüedad Componente", value=12)

        datos_camion = np.array([[t, v, p, h, c, a]])
        
        if st.button("Aplicar modelo ML"):
            # Escalar los datos de entrada
            datos_scaled = st.session_state.scaler.transform(datos_camion)
            prediccion = st.session_state.modelo.predict(datos_scaled)
            probabilidad = st.session_state.modelo.predict_proba(datos_scaled)[0][1]

            if prediccion[0] == 1:
                st.error(f"ALERTA: El camión tiene un {probabilidad:.2%} de probabilidad de falla. SE RECOMIENDA MANTENIMIENTO.")
            else:
                st.success(f"OPERACIÓN NORMAL: El camión puede iniciar guardia. Riesgo de falla: {probabilidad:.2%}")
    else:

        st.error("Debes entrenar el modelo en la sección 2.3 antes de predecir.")

