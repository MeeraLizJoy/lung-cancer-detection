"""
Lung Cancer Detection - Streamlit Web App
==========================================
Interactive web interface for lung cancer classification from CT scans.

Usage:
    streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']
CLASS_COLORS = ['#2ca02c', '#d62728', '#1f77b4']  # Green, Red, Blue
MODEL_PATH = 'models/efficientnet_b0_final.pth'
DROPOUT = 0.2
NUM_CLASSES = 3

# Device configuration
device = torch.device("cpu")

# Cache model loading
@st.cache_resource
def load_model():
    """Load the trained EfficientNet-B0 model."""
    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(in_features, NUM_CLASSES)
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess uploaded image for model input."""
    mean, std = [0.5], [0.5]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

def predict(model, image_tensor):
    """Make prediction on preprocessed image."""
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item()
    
    probs_dict = {
        class_name: prob.item() 
        for class_name, prob in zip(CLASS_NAMES, probabilities)
    }
    
    return predicted_class, confidence_score, probs_dict

def create_probability_chart(probs_dict):
    """Create interactive bar chart for class probabilities."""
    fig = go.Figure(data=[
        go.Bar(
            x=list(probs_dict.keys()),
            y=list(probs_dict.values()),
            marker_color=CLASS_COLORS,
            text=[f'{v:.1%}' for v in probs_dict.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig

# Main app
def main():
    # Header
    st.title("🫁 Lung Cancer Detection System")
    st.markdown("### AI-Powered Classification of Lung CT Scans")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        **Architecture:** EfficientNet-B0  
        **Classes:** Benign, Malignant, Normal  
        **Validation Accuracy:** 98.93%  
        **Training Dataset:** 1377 CT scans  
        """)
        
        st.markdown("---")
        st.header("ℹ️ Instructions")
        st.markdown("""
        1. Upload a lung CT scan image
        2. Click "Analyze Image"
        3. View prediction results
        4. Check confidence scores
        """)
        
        st.markdown("---")
        st.markdown("**⚠️ Disclaimer:** This is a research tool. Not for clinical diagnosis.")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a lung CT scan image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a CT scan image for analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded CT Scan", use_column_width=True)
            
            # Analyze button
            analyze_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
        else:
            analyze_button = False
            st.info("👆 Please upload an image to begin analysis")
    
    with col2:
        st.header("📋 Results")
        
        if uploaded_file is not None and analyze_button:
            with st.spinner("Analyzing image..."):
                # Load model
                model = load_model()
                
                # Preprocess and predict
                image_tensor = preprocess_image(image)
                predicted_class, confidence, probs = predict(model, image_tensor)
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Prediction result with color coding
                if predicted_class == "Malignant":
                    result_color = "red"
                    emoji = "🔴"
                elif predicted_class == "Benign":
                    result_color = "green"
                    emoji = "🟢"
                else:
                    result_color = "blue"
                    emoji = "🔵"
                
                st.markdown(f"### {emoji} Predicted Class: :{result_color}[{predicted_class}]")
                st.markdown(f"### Confidence: **{confidence:.2%}**")
                
                # Progress bar for confidence
                st.progress(confidence)
                
                st.markdown("---")
                
                # Probability chart
                st.plotly_chart(create_probability_chart(probs), use_container_width=True)
                
                # Detailed probabilities
                st.markdown("#### 📊 Detailed Probabilities")
                for class_name, prob in probs.items():
                    st.metric(label=class_name, value=f"{prob:.2%}")
                
        elif not uploaded_file:
            st.info("Upload an image to see results here")
        else:
            st.info("Click 'Analyze Image' to get prediction")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit and PyTorch</p>
            <p>Model: EfficientNet-B0 | Dataset: Lung Cancer CT Scans</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
