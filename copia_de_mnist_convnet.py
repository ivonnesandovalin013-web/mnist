import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Configuración del título de la página
st.title("🧠 Red Neuronal Convolucional - MNIST")
st.write("Sube una imagen de un número escrito a mano (en formato JPG o PNG) para que el modelo lo reconozca.")

# 1. FUNCIÓN PARA CARGAR EL MODELO ENTRENADO
@st.cache_resource
def cargar_modelo_entrenado():
    try:
        # Busca el archivo que acabas de subir a GitHub
        return keras.models.load_model("modelo_mnist.h5")
    except Exception as e:
        return None

model = cargar_modelo_entrenado()

# 2. VERIFICAR SI EL MODELO CARGÓ CORRECTAMENTE
if model is not None:
    st.success("✅ ¡Modelo entrenado cargado con éxito desde el repositorio!")
    
    # Interfaz para que el usuario suba su imagen
    uploaded_file = st.file_uploader("Elige una imagen...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Abrir y mostrar la imagen subida
        imagen = Image.open(uploaded_file)
        st.image(imagen, caption="Imagen subida por el usuario", width=150)
        
        if st.button("Hacer Predicción"):
            # Procesar la imagen para que coincida con lo que espera MNIST (28x28 en escala de grises)
            img_gris = imagen.convert("L").resize((28, 28))
            
            # Convertir a matriz numérica y normalizar (/255) como hiciste en Colab
            img_array = np.array(img_gris).astype("float32") / 255
            
            # 🚨 AQUÍ ESTÁ EL ARREGLO: Invertir colores si la imagen tiene fondo blanco
            # Si el promedio de los píxeles es alto (más cerca de blanco), invertimos la matriz
            if np.mean(img_array) > 0.5:
                img_array = 1.0 - img_array
            
            # Ajustar dimensiones a (1, 28, 28, 1) -> (Lote, Alto, Ancho, Canal de color)
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Realizar la predicción con el modelo .h5
            predicciones = model.predict(img_array)
            clase_predicha = np.argmax(predicciones)
            certeza = np.max(predicciones) * 100
            
            # Mostrar el resultado final en pantalla
            st.metric(label="Número Detectado", value=str(clase_predicha))
            st.info(f"Certeza del modelo: {certeza:.2f}%")
else:
    st.warning("⚠️ No se pudo cargar el archivo 'modelo_mnist.h5'. Asegúrate de que se haya subido correctamente a la carpeta principal de tu repositorio de GitHub.")
