import streamlit as st
import pandas as pd
import joblib
import os

st.title("Predicción del precio de gasolina por estado")

# Mostrar imagen si existe
if os.path.exists("gasolina.jpg"):
    st.image("gasolina.jpg", caption="Precio estimado de gasolina por estado")
else:
    st.warning("Imagen 'gasolina.jpg' no encontrada. La app funcionará sin ella.")

# Entrada de datos del usuario
def user_input_features():
    estados = [
        'Aguascalientes','Baja California','Baja California Sur','Campeche','Coahuila',
        'Colima','Chiapas','Chihuahua','Ciudad de México','Durango','Guanajuato',
        'Guerrero','Hidalgo','Jalisco','México','Michoacán','Morelos','Nayarit',
        'Nuevo León','Oaxaca','Puebla','Querétaro','Quintana Roo','San Luis Potosí',
        'Sinaloa','Sonora','Tabasco','Tamaulipas','Tlaxcala','Veracruz','Yucatán','Zacatecas'
    ]
    Estado = st.selectbox('Estado:', estados)
    Mes = st.number_input('Mes (1-12):', min_value=1, max_value=12, value=1, step=1)
    Año = st.number_input('Año:', min_value=2000, max_value=2100, value=2025, step=1)
    return pd.DataFrame({'Estado': [Estado], 'Mes': [Mes], 'Año': [Año]})

df = user_input_features()

# Verificar si existen los archivos del modelo
if not os.path.exists('encoder_gasolina.joblib') or not os.path.exists('modelo_gasolina.joblib'):
    st.error("No se encontraron los archivos del modelo. Por favor sube 'encoder_gasolina.joblib' y 'modelo_gasolina.joblib'.")
else:
    # Cargar encoder y modelo
    encoder = joblib.load('encoder_gasolina.joblib')
    model = joblib.load('modelo_gasolina.joblib')

    # Convertir columnas a string y transformar
    df_encoded = encoder.transform(df.astype(str))

    # Predicción
    prediccion = model.predict(df_encoded)

    st.subheader('Precio estimado de gasolina')
    st.write('El precio estimado es:', round(float(prediccion), 2))
