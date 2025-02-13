import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model('my_model.keras')

# Define class names
class_names = ['Benign', 'Early', 'Pre', 'Pro']

# Streamlit UI
st.set_page_config(page_title="Lymphoma Classification", layout="centered")
st.title("Lymphoma Image Classification")
st.markdown("Upload an image to predict the class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 224, 224, 3)

    # Make prediction
    label = model.predict(img_array)
    predicted_class_index = np.argmax(label)
    predicted_class = class_names[predicted_class_index]

    # Display image and prediction
    st.image(uploaded_file, caption=f"Predicted Class: {predicted_class}", use_column_width=True)
    st.success(f"Prediction: {predicted_class}")
    
    # Show confidence scores
    st.subheader("Confidence Scores:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {label[0][i]:.4f}")
