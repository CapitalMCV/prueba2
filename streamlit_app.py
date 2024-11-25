import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Clasificación de Horarios Pico en Área de Alimentos")
st.markdown("Esta aplicación predice si un horario es **pico** en el área de alimentos basado en datos históricos.")

# Subir el archivo CSV
st.sidebar.header("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    # Cargar el archivo
    df = pd.read_csv(uploaded_file)
    st.subheader("Vista Previa de los Datos")
    st.write(df.head())

    # Separar características (X) y variable objetivo (y)
    X = df[['hora', 'dia_semana', 'ventas_alimentos', 'tiempo_espera', 'personal_asignado']]
    y = df['horario_pico_alimentos']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    # Evaluación del modelo
    predicciones = modelo.predict(X_test)
    precision = accuracy_score(y_test, predicciones)
    f1 = f1_score(y_test, predicciones)
    matriz_confusion = confusion_matrix(y_test, predicciones)

    # Mostrar métricas
    st.subheader("Métricas del Modelo")
    st.write(f"**Precisión:** {precision * 100:.2f}%")
    st.write(f"**F1-Score:** {f1:.2f}")
    st.text("Informe de Clasificación:")
    st.text(classification_report(y_test, predicciones))

    # Visualizar matriz de confusión
    st.subheader("Matriz de Confusión")
    fig, ax = plt.subplots()
    sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=['No Pico', 'Pico'], yticklabels=['No Pico', 'Pico'])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    st.pyplot(fig)

    # Predicción manual
    st.sidebar.header("Predicción Manual")
    st.sidebar.markdown("Ingresa los valores para predecir si es horario pico o no:")

    hora = st.sidebar.slider("Hora (8-23)", 8, 23, 12)
    dia_semana = st.sidebar.slider("Día de la Semana (1=Lunes, 7=Domingo)", 1, 7, 1)
    ventas_alimentos = st.sidebar.number_input("Ventas de Alimentos", 20, 1000, 100)
    tiempo_espera = st.sidebar.number_input("Tiempo de Espera (minutos)", 1, 60, 10)
    personal_asignado = st.sidebar.slider("Personal Asignado", 1, 10, 3)

    if st.sidebar.button("Predecir"):
        # Crear un DataFrame con los valores ingresados
        entrada = pd.DataFrame({
            'hora': [hora],
            'dia_semana': [dia_semana],
            'ventas_alimentos': [ventas_alimentos],
            'tiempo_espera': [tiempo_espera],
            'personal_asignado': [personal_asignado],
        })

        # Realizar la predicción
        resultado = modelo.predict(entrada)[0]
        if resultado == 1:
            st.sidebar.success("Predicción: **Horario Pico**")
        else:
            st.sidebar.info("Predicción: **No Pico**")

else:
    st.sidebar.info("Por favor, sube un archivo CSV para empezar.")