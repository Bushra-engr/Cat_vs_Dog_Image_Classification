import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
# Import the specific MobileNetV2 preprocessing
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- CONFIGURATION ---
MODEL_PATH = 'cat_dog_cnn_model.keras'  # Make sure this matches your saved filename
IMG_SIZE = (128, 128) # Fixed from 256 to 128 as per your model's error message

# --- LOAD MODEL ---
@st.cache_resource
def get_model():
    # It's better to load the model once and cache it
    model = load_model(MODEL_PATH)
    return model

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")
st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image to see if it's a Cat or a Dog!")

try:
    model = get_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Check if {MODEL_PATH} exists in the same folder.")
    st.stop()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # --- PREPROCESSING ---
    with st.spinner('Analyzing...'):
        # 1. Convert PIL image to RGB (handles PNGs with transparency too)
        img_rgb = img.convert('RGB')
        
        # 2. Resize to 128x128
        img_resized = img_rgb.resize(IMG_SIZE)
        
        # 3. Convert to numpy array
        img_array = np.array(img_resized)
        
        # 4. Add Batch Dimension (shape becomes 1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 5. MobileNetV2 Preprocessing (This was the missing piece!)
        # This handles the scaling correctly so you don't get the 0.7 vs 0.7 error
        prepared_img = preprocess_input(img_array.astype('float32'))
        
        # --- PREDICTION ---
        prediction = model.predict(prepared_img)
        result = float(prediction[0][0]) # Get the sigmoid value
        
        # --- LOGIC ---
        # Based on our previous test: 0 = Cat, 1 = Dog
        st.divider()
        if result > 0.5:
            st.balloons()
            st.success(f"### It's a **DOG**! 🐶")
            st.write(f"**Confidence:** {result:.2%}")
        else:
            st.snow()
            st.success(f"### It's a **CAT**! 🐱")
            st.write(f"**Confidence:** {1 - result:.2%}")

st.divider()
st.caption("Built with TensorFlow, MobileNetV2 & Streamlit")