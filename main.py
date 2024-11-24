import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="House pricing predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("House pricing prediction ML-powered 🔮")
st.write("Use this app to predict your home price")

st.markdown("""### Notas

---

Mirar qqplot para normalidad""")

st.image("https://cdn.midjourney.com/96cb8658-e9f0-46a1-890b-0614391245f3/0_0.png",
         caption="Tu próximo hogar")

# Columnas
col1, col2 = st.columns(2)

with col1:
    barrio = st.selectbox("Barrio", ["A", "B", "C"])
    st.write(barrio)
    tipo_casa = st.selectbox("Tipo de casa", ["Casa", "Piso", "Castillo"])
    st.write(tipo_casa)

with col2:
    habitaciones = st.number_input("Número de habitaciones", min_value=1, max_value=7, step=1)
    st.write(habitaciones)
    area = st.number_input("Tamaño (m2)", min_value=1, max_value=700, step=10)
    st.write(area)

diccionario_respuesta = {
    "Rooms": habitaciones,
    "HouseType": tipo_casa,
    "Neighborhood": barrio,
    "Area": area
}

df_pred = pd.DataFrame(diccionario_respuesta, index = [0])

st.table(df_pred)

df_pred_copy = df_pred.copy()

# Carga de los modelos
#target, standard, modelo = sm.load_models()

numeric_col = df_pred.select_dtypes(include = np.number)

if st.button("Predecir precio"):

    # Estandarizar numéricas
    #df_pred_copy[numeric_col] = standard.transform(df_pred_copy[numeric_col])

    # Encoding (a target se le pasa todo el df)
    #df_pred_copy[df_pred_copy.columns] = target.transform(df_pred_copy)

    #prediccion = modelo.predict(df_pred_copy)
    #st.write(prediccion[0])

    st.balloons()