import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# --- Load necessary assets ---

# Load the tokenizer (assuming you saved it earlier)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load the label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load the model
model = tf.keras.models.load_model("pokemon_name_model.h5")

# --- Define Input UI ---
st.title("🔍 Pokémon Name Predictor")
st.markdown("Enter Pokémon characteristics below to predict its name:")

info = st.text_area("Description (info):", "A small green Pokémon with plant features.")

col1, col2 = st.columns(2)
with col1:
    height = st.number_input("Height", min_value=0.0, value=0.5)
    weight = st.number_input("Weight", min_value=0.0, value=5.0)
    hp = st.number_input("HP", min_value=1, value=45)
    attack = st.number_input("Attack", min_value=1, value=49)

with col2:
    defense = st.number_input("Defense", min_value=1, value=49)
    s_attack = st.number_input("Special Attack", min_value=1, value=65)
    s_defense = st.number_input("Special Defense", min_value=1, value=65)
    speed = st.number_input("Speed", min_value=1, value=45)

# --- Predict Button ---
if st.button("Predict Pokémon Name"):
    # Process text
    seq = tokenizer.texts_to_sequences([info])
    padded = pad_sequences(seq, maxlen=30, padding='post')

    # Process numerical features
    num_features = np.array([[height, weight, hp, attack, defense, s_attack, s_defense, speed]])
    num_scaled = scaler.transform(num_features)

    # Make prediction
    pred_probs = model.predict({'text_input': padded, 'num_input': num_scaled})
    pred_id = np.argmax(pred_probs, axis=1)[0]
    pred_name = label_encoder.inverse_transform([pred_id])[0]

    st.success(f"🎉 Predicted Pokémon: **{pred_name}**")

    # Optional: Show top 3
    top_k = 3
    top_k_ids = np.argsort(pred_probs[0])[-top_k:][::-1]
    top_k_names = label_encoder.inverse_transform(top_k_ids)
    st.markdown("#### 🔝 Top 3 Predictions:")
    for i, name in enumerate(top_k_names):
        st.write(f"{i+1}. {name}")
