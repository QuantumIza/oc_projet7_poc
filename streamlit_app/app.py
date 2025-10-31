import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.convnext import preprocess_input as preprocess_convnext

# --- CONFIGURATION DES CHEMINS ET LIENS DRIVE
MODEL_PATH = "model/convnext_tiny_bestval_loss.keras"
ENCODER_PATH = "outputs/encoder_classes_transfer_learning.joblib"

# https://drive.google.com/file/d/1wgStOawKhJvhH3Lsngsf50BtjXUBJO52/view?usp=drive_link
MODEL_DRIVE_ID = "1wgStOawKhJvhH3Lsngsf50BtjXUBJO52"
# https://drive.google.com/file/d/1qPlTqV7RIV1AlphyMu2QJQSzbw8qtiJ1/view?usp=drive_link
ENCODER_DRIVE_ID = "1qPlTqV7RIV1AlphyMu2QJQSzbw8qtiJ1"

# --- TELECHARGEMENT DEPUIS DRIVE SI FICHIERS ABSENTS
def download_from_drive(file_path, file_id):
    if not os.path.exists(file_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False)

download_from_drive(MODEL_PATH, MODEL_DRIVE_ID)
download_from_drive(ENCODER_PATH, ENCODER_DRIVE_ID)

# --- CHARGEMENT DU MODELE ET DE L'ENCODEUR
@st.cache_resource
def load_model_and_encoder():
    model = load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

model, encoder = load_model_and_encoder()

# --- PRETRAITEMENT IMAGE
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size).convert("RGB")
    img_array = np.array(img)
    img_proc = preprocess_convnext(img_array)
    img_batch = np.expand_dims(img_proc, axis=0)
    return img_batch

# --- INTERFACE STREAMLIT
st.set_page_config(page_title="Projet 7 – Prédiction", layout="centered")
st.title("🐶 Classification de races de chiens")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Image chargée", use_column_width=True)

    img_batch = preprocess_image(img)
    y_pred = model.predict(img_batch)
    pred_class = encoder.inverse_transform([np.argmax(y_pred)])[0]
    st.success(f"Race prédite : **{pred_class}**")

    # Affichage des probabilités
    probas = pd.Series(y_pred[0], index=encoder.classes_).sort_values(ascending=False)
    st.bar_chart(probas)

