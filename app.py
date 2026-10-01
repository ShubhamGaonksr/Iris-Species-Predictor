import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Species label mapping (matches sklearn's default Iris target encoding)
SPECIES_MAP = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

# --- Page config ---
st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="centered")

# --- Header ---
st.title("🌸 Iris Species Predictor")
st.markdown("Enter the flower measurements below and click **Predict** to identify the species.")

st.divider()

# --- Input form ---
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

st.divider()

# --- Prediction ---
if st.button("🔍 Predict Species", use_container_width=True):
    # Feature order: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    species = SPECIES_MAP.get(prediction, str(prediction))

    st.success(f"### Predicted Species: **{species}**")