import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.models import load_model
from pymongo import MongoClient
from bson.binary import Binary

# Se connecter à la base de données MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['tumeur_database']
collection_with_tumor = db['tumeurs_malignes']
collection_without_tumor = db['tumeurs_benignes']

# Charger le meilleur modèle
best_model = load_model(filepath='models/cnn-parameters-improvement-22-0.90.model')

# Fonction de prédiction
def predict_image(img_array):
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Redimensionner l'image à la taille attendue par le modèle
    img_array = tf.image.resize(img_array, (240, 240))

    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, axis=0)

    # Faire une prédiction avec le modèle chargé
    prediction = best_model.predict(img_array)

    return prediction[0][0] > 0.5, prediction[0][0]

# Fonction principale pour l'application Streamlit
def main():
    st.title("Détection de Tumeurs")

    # Widget d'upload d'image
    uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])

    # Centrer le bouton et augmenter sa taille
    col1, col2, col3 = st.columns([1, 2, 2])  # Adjusted column widths

    with col2:
        if uploaded_file is not None:
            img = Image.open(uploaded_file).resize((224, 224)).convert("RGB")
            img_array = np.array(img)

            # Redimensionner l'image pour l'affichage
            display_img = img.resize((500, 500))  # Augmenter la taille de l'image

            # Centrer l'image en utilisant une colonne
            st.image(display_img, caption="Image téléchargée", use_column_width=True)

            
            # Utiliser st.button pour obtenir l'événement de clic
            if st.button("Prédire", key="prediction_button"):
                # Augmenter la taille d'affichage des messages avec st.markdown
                result, prediction_value = predict_image(img_array)
                if result:
                    st.markdown("<p style='font-size: 1.5em; color: red;'>Tumeur détectée</p>", unsafe_allow_html=True)
                    save_image_to_mongodb(uploaded_file, prediction_value, collection_with_tumor)
                else:
                    st.markdown("<p style='font-size: 1.5em;'>Pas de tumeur détectée</p>", unsafe_allow_html=True)
                    save_image_to_mongodb(uploaded_file, prediction_value, collection_without_tumor)

# Fonction pour sauvegarder une image dans MongoDB
def save_image_to_mongodb(uploaded_file, prediction_value, collection):
    image_data = uploaded_file.read()
    encoded_image = Binary(image_data)

    document = {
        "image": encoded_image,
        "filename": uploaded_file.name,
        "content_type": uploaded_file.type,
        "prediction": prediction_value.item(),  # Convertir en type natif Python
    }

    collection.insert_one(document)

# Exécuter l'application principale
if __name__ == "__main__":
    main()