import streamlit as st
import numpy as np
import torch
import torchxrayvision as xrv
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import skimage.transform

# --- Model Loading ---
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

# --- Streamlit App Interface ---
st.title("Lung Cancer Detection with XAI Heatmap")
st.write(
    """
    Upload a chest image image to detect and localize potential lung cancer using a deep learning model and explainable AI. 
    The app processes the image, predicts the presence of a mass (which may indicate cancer), and highlights the regions 
    influencing the prediction with a heatmap.
    """
)

# File uploader for chest image image
uploaded_file = st.file_uploader(
    "Upload Chest image Image", 
    type=["jpg", "png", "jpeg"], 
    help="Upload a chest image image in JPG, PNG, or JPEG format."
)

# Process the uploaded image and display results
if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img_resized = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img_resized) / 255.0  # Normalize to [0,1]
    
    # Prepare image for model input (normalize to [-1,1])
    img_input = (img_array - 0.5) / 0.5
    img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).float()  # Shape: (1, 1, 224, 224)
    img_tensor = img_tensor.to('cpu')  # Move input tensor to CPU

    # Run the model to get predictions
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]  # Get probabilities for all pathologies
    
    # Extract probability for "Mass" (indicative of potential cancer)
    mass_index = model.pathologies.index("Mass")
    mass_prob = probs[mass_index]

    # Generate Grad-CAM heatmap
    # Generate Grad-CAM heatmap
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(mass_index)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

    # Ensure grayscale_cam is a 2D array
    if len(grayscale_cam.shape) == 3:
        grayscale_cam = grayscale_cam[0]  # Assuming shape is (1, height, width)
    elif len(grayscale_cam.shape) != 2:
        raise ValueError(f"Unexpected shape for grayscale_cam: {grayscale_cam.shape}")

    # Visualize the heatmap overlaid on the image
    img_resized_rgb = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)  # Convert grayscale to RGB
    visualization = show_cam_on_image(img_resized_rgb, grayscale_cam, use_rgb=True)

    # --- Display Results ---
    st.subheader("Analysis Results")
    
    # Display original resized image
    st.image(img_resized, caption="Processed Input Image (224x224)", use_column_width=True)
    
    # Display heatmap overlay
    st.image(visualization, caption="Grad-CAM Heatmap Highlighting Mass Location", use_column_width=True)
    
    # Display prediction
    st.write(f"**Probability of Mass Detection:** {mass_prob*100:.2f}%")
    if mass_prob > 0.5:
        st.success("Mass detected - Potential indication of lung cancer.")
    else:
        st.info("No mass detected.")

# --- Sidebar Information ---
st.sidebar.title("About the Application")
st.sidebar.write(
    """
    **Model:** This app utilizes a DenseNet121 deep learning model infused with features extracted from MobilenetV2 and ResNet50 models on multiple large-scale chest image datasets. 
    It is designed to detect various pathologies, including masses that may indicate lung cancer.

    **XAI Method:** We employ Grad-CAM (Gradient-weighted Class Activation Mapping) to generate a heatmap, 
    visually explaining the model's predictions by highlighting the regions in the image most influential 
    to the detection of a mass.

    **Purpose:** This tool demonstrates how AI can assist in medical imaging analysis by providing 
    interpretable results.
    """
)
st.sidebar.warning(
    "**Disclaimer:** This application is for demonstration purposes only. "
    "Always consult a healthcare professional for medical advice."
)

# --- Footer ---
st.write("---")
st.write("This project has been developed under guidance of Dr. Suganya R by Abhiram Sharma, KP Behara and team at VIT Chennai, 2025.")
