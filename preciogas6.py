import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------
# Título e imagen
# ----------------------
st.write("# Predicción del precio de gasolina por estado")

# Mostrar imagen solo si existe
if os.path.exists("gasolina.jpg"):
    st.image("gasolina.jpg", caption="Precio estimado de gasolina por estado")
else:
    st.warning("Imagen 'gasolina.jpg' no encontrada. Continúa sin imagen.")

# ----------------------
# Entrada de usuario
# ----------------------
def user_input_features():
    Estado = st.selectbox('Estado:', [
        'Aguascalientes','Baja California','Baja California Sur','Campeche','Coahuila',
        'Colima','Chiapas','Chihuahua','Ciudad de México','Durango','Guanajuato',
        'Guerrero','Hidalgo','Jalisco','México','Michoacán','Morelos','Nayarit',
        'Nuevo León','Oaxaca','Puebla','Querétaro','Quintana Roo','San Luis Potosí',
        'Sinaloa','Sonora','Tabasco','Tamaulipas','Tlaxcala','Veracruz','Yucatán','Zacatecas'
    ])
    Mes = st.number_input('Mes (1-12):', min_value=1, max_value=12, value=1, step=1)
    Año = st.number_input('Año:', min_value=2000, max_value=2100, value=2025, step=1)
    return pd.DataFrame({'Estado': [Estado], 'Mes': [Mes], 'Año': [Año]})

df = user_input_features()

# ----------------------
# Cargar encoder y modelo
# ----------------------
if not os.path.exists('encoder_gasolina.joblib') or not os.path.exists('modelo_gasolina.joblib'):
    st.error("Faltan los archivos 'encoder_gasolina.joblib' o 'modelo_gasolina.joblib'. "
             "Por favor, genera y sube estos archivos.")
else:
    encoder = joblib.load('encoder_gasolina.joblib')
    model = joblib.load('modelo_gasolina.joblib')

    try:
        # Convertir columna categórica a string
        df['Estado'] = df['Estado'].astype(str)

        # Transformar columna categórica
        estado_encoded = encoder.transform(df[['Estado']])

        # Reconstruir DataFrame con los mismos nombres de columnas que se usaron al entrenar
        encoded_cols = encoder.get_feature_names_out(['Estado'])
        X_encoded = pd.DataFrame(
            np.hstack([estado_encoded, df[['Mes','Año']].values]),
            columns=list(encoded_cols) + ['Mes','Año']
        )

        # Predicción
        prediccion = model.predict(X_encoded)

        # Mostrar resultado
        st.subheader('Precio estimado de gasolina')
        st.write('El precio estimado es:', round(float(prediccion), 2))

    except Exception as e:
        st.error(f"Ocurrió un error al transformar los datos o hacer la predicción: {e}")
