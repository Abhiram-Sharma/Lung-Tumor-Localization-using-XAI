import streamlit as st
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import requests
import matplotlib.pyplot as plt

# Set page configuration for a wide layout and professional appearance
st.set_page_config(
    page_title="Lung Cancer Classification",
    layout="wide",
    page_icon="🩺"
)
st.markdown(
    """
    <style>
 body {
        background-color: #f9f9fc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS for flip cards and styling
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #dbeafe, #f0f4ff);
    font-family: 'Helvetica', sans-serif;
}
h1, h2, h3, h4 {
    color: #0f172a;
    font-weight: 700;
    animation: fadeIn 2s ease-in;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0px); }
}
.stButton > button {
    background-color: #4f46e5;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 20px;
    cursor: pointer;
    transition: 0.4s;
}
.stButton > button:hover {
    background-color: #6366f1;
    box-shadow: 0px 0px 10px #6366f1;
}
.stDownloadButton > button {
    background-color: #0f766e;
    color: white;
    border-radius: 12px;
    transition: 0.3s;
}
.stDownloadButton > button:hover {
    background-color: #14b8a6;
    box-shadow: 0px 0px 8px #14b8a6;
}
.flip-card {
    background-color: transparent;
    width: 250px;
    height: 350px;
    perspective: 1000px;
    margin: 10px;
    transition: transform 0.6s;
}
.flip-card-inner {
    position: relative;
    width: 100%;
    height: 100%;
    text-align: center;
    transform-style: preserve-3d;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}
.flip-card:hover .flip-card-inner {
    transform: rotateY(180deg);
}
.flip-card-front, .flip-card-back {
    position: absolute;
    width: 100%;
    height: 100%;
    backface-visibility: hidden;
    border-radius: 15px;
}
.flip-card-front {
    background-color: #f0f4ff;
}
.flip-card-back {
    background: radial-gradient(circle, #6366f1, #4338ca);
    color: white;
    transform: rotateY(180deg);
    padding: 20px;
    display: flex;
    justify-content: center;
    align-items: center;
}
.metric-container {
    padding: 10px;
    border-radius: 10px;
    background-color: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)



# --- Model Loading and Setup ---
@st.cache_resource
def load_model():
    """Load the pre-trained DenseNet121 model from torchxrayvision."""
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to('cpu')  # Move model to CPU
    return model

# Initialize the model
model = load_model()

# Define the target layer for Grad-CAM (last convolutional layer in DenseNet121)
target_layers = [model.features.denseblock4.denselayer16.conv2]

# --- Global Lung Cancer Statistics Section ---
st.header("Global Lung Cancer Statistics")
st.markdown("View sample statistics on lung cancer cases worldwide (demonstration data).")
# Sample data since real-time API is not implemented
data = pd.DataFrame({
    'Year': [2018, 2019, 2020, 2021],
    'Cases': [2200000, 2250000, 2300000, 2350000],
    'Deaths': [1800000, 1820000, 1840000, 1860000]
})
col1, col2 = st.columns(2)
with col1:
    st.metric("New Cases (2021)", f"{data['Cases'].iloc[-1]:,}")
with col2:
    st.metric("Deaths (2021)", f"{data['Deaths'].iloc[-1]:,}")
st.line_chart(data.set_index('Year'))

# --- Lung Cancer Detection Section ---
st.header("Lung Cancer Detection")
st.markdown("""
Upload a chest X‑ray image to detect potential lung cancer indications using the DenseNet121 model 
and visualize the results with a Grad‑CAM heatmap. The model predicts the probability of a mass, which 
may indicate lung cancer.
""")
uploaded_file = st.file_uploader("Upload a chest X‑ray image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
  with st.spinner("Analyzing X-ray and generating heatmap..."):
    try:
        # Load and preprocess the image
        img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        img_resized = img.resize((224, 224))  # Resize to model input size
        img_array = np.array(img_resized) / 255.0  # Normalize to [0,1]
        
        # Prepare image for model input (normalize to [-1,1])
        img_input = (img_array - 0.5) / 0.5
        img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).float()  # Shape: (1, 1, 224, 224)
        img_tensor = img_tensor.to('cpu')
        
        # Run the model to get predictions
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]  # Get probabilities for all pathologies
        
        # Extract probability for "Mass" (indicative of potential cancer)
        mass_index = model.pathologies.index("Mass")
        mass_prob = probs[mass_index]
        
        # Generate Grad-CAM heatmap
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(mass_index)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        # Ensure grayscale_cam is 2D
        if len(grayscale_cam.shape) == 3:
            grayscale_cam = grayscale_cam[0]
        elif len(grayscale_cam.shape) != 2:
            raise ValueError(f"Unexpected shape for grayscale_cam: {grayscale_cam.shape}")
        
        # Create a RGB version of the grayscale image for overlay
        img_resized_rgb = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)
        visualization = show_cam_on_image(img_resized_rgb, grayscale_cam, use_rgb=True)
        
        # --- Display Results ---
        st.image(img_resized, caption="Processed Chest X‑ray (224x224)", use_column_width=True)
        st.image(visualization, caption="Grad‑CAM Heatmap Highlighting Mass Location", use_column_width=True)
        st.write(f"**Probability of Mass Detection:** {mass_prob*100:.2f}%")
        if mass_prob > 0.5:
            st.success("Mass detected – Potential indication of lung cancer.")
            st.balloons()  # Add fun feedback
        else:
            st.info("No mass detected.")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# --- Authors Section ---
st.header("Authors")
st.markdown("Meet the researchers behind this study (demonstration placeholders).")
authors = [
    {"name": "Dr. Alice Smith", "bio": "Expert in medical imaging and AI.", "photo": "https://via.placeholder.com/250"},
    {"name": "Dr. Bob Johnson", "bio": "Specialist in deep learning applications.", "photo": "https://via.placeholder.com/250"}
]
cols = st.columns(len(authors))
for col, author in zip(cols, authors):
    with col:
        flip_card_html = f"""
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <img src="{author['photo']}" alt="{author['name']}" style="width:100%;height:100%;border-radius:10px;">
                </div>
                <div class="flip-card-back">
                    <h3>{author['name']}</h3>
                    <p>{author['bio']}</p>
                </div>
            </div>
        </div>
        """
        st.markdown(flip_card_html, unsafe_allow_html=True)

# --- Research Paper Section ---
st.header("Research Paper")
st.markdown("Download a sample research paper for demonstration purposes.")
sample_pdf_url = "/assets/paper.pdf"
response = requests.get(sample_pdf_url)
st.download_button(
    label="Download Sample Research Paper",
    data=response.content,
    file_name="sample_research_paper.pdf",
    mime="application/pdf"
)

# --- Footer ---
st.markdown("""
---
**Contact Us**  
Email: research.team@example.com  

**Portfolio**  
[GitHub](https://github.com) | [LinkedIn](https://linkedin.com)
""")
