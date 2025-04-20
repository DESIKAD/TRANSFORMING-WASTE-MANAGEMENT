import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import os
from pathlib import Path

# Load the model
model = load_model('waste_classifier_model.h5')

# Define class labels and tips
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
waste_tips = {
    'cardboard': "Flatten boxes to save space.",
    'glass': "Rinse and remove caps before recycling glass.",
    'metal': "Empty and rinse cans before disposal.",
    'paper': "Avoid recycling oily or greasy paper.",
    'plastic': "Rinse plastic containers before recycling.",
    'trash': "Dispose responsibly if it's non-recyclable waste."
}
recycling_uses = {
    'cardboard': "Cardboard can be recycled into new cardboard boxes, paperboard, or packaging material.",
    'glass': "Glass can be recycled into new bottles, jars, or tiles.",
    'metal': "Recycled metal is used in construction materials, new cans, or tools.",
    'paper': "Recycled paper becomes newspaper, tissue, or cardboard.",
    'plastic': "Plastic can be turned into containers, clothing fibers, or furniture.",
    'trash': "Non-recyclable waste is often used for energy recovery or disposed in landfills.Not all trash is recycled"
}

# Custom UI styling
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .result {
        font-size: 30px;
        font-weight: bold;
        color: green;
        text-align: center;
    }
    .tip, .recycle-info {
        font-size: 24px;
        text-align: center;
        margin-top: 10px;
        color: #444444;
    }
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .product-title { 
        font-size: 30px; font-weight: bold; text-align: center; margin-top: 30px; 
            }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Transforming Waste into Resources</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)  # updated to remove deprecation warning

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 224, 224, 3))

    # Predict class
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Display prediction results
    st.markdown(f'<div class="result">{predicted_class.capitalize()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="tip"><b>Tip:</b> {waste_tips[predicted_class]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="recycle-info">{recycling_uses[predicted_class]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="product-title">Example Recycled Products:</div>', unsafe_allow_html=True)
    # Display product images from images/<class>_product/
    product_dir = Path("images") / f"{predicted_class}_product"
    if product_dir.exists():
        product_images = list(product_dir.glob("*.png")) + list(product_dir.glob("*.jpg")) + list(product_dir.glob("*.jpeg"))
        for img_path in product_images:
            st.image(str(img_path), caption=f"Recycled {predicted_class} product", use_container_width=True)
    else:
        st.info(f"No recycled product images available for {predicted_class}.")


    