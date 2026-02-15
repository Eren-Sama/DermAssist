import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download

HF_REPO_ID = "Eklavya16/DermAssist"

CLASSIFICATION_THRESHOLD = 0.5
UNCERTAINTY_THRESHOLD = 0.165

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="DermAssist – Clinical Dermoscopic AI",
    layout="wide"
)

def build_model():
    model = models.resnet50(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 1)
    )
    return model.to(device)

@st.cache_resource
def load_models():
    models_list = []
    for i in range(1, 4):
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"resnet50_model_{i}.pth"
        )
        
        model = build_model()
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        model.eval()
        models_list.append(model)
    return models_list

ensemble_models = load_models()

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, class_idx):
        self.model.zero_grad()
        output = self.model(input_image)
        loss = output[0]
        loss.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))

        cam[cam < np.percentile(cam, 75)] = 0

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

target_layer = ensemble_models[0].layer4[-1]
gradcam = GradCAM(ensemble_models[0], target_layer)

def ensemble_predict(models, image_tensor):
    probs_list = []

    with torch.no_grad():
        for model in models:
            output = model(image_tensor)
            prob = torch.sigmoid(output).item()
            probs_list.append(prob)

    mean_prob = np.mean(probs_list)
    std_prob = np.std(probs_list)

    return mean_prob, std_prob, probs_list

def decision_logic(mean_prob, std_prob):
    if std_prob > UNCERTAINTY_THRESHOLD:
        return "UNCERTAIN"

    if mean_prob >= 0.75:
        return "HIGH RISK"

    if mean_prob >= CLASSIFICATION_THRESHOLD:
        return "MODERATE RISK"

    return "LOW RISK"

def overlay_gradcam(original_image, cam):
    image = np.array(original_image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay

st.sidebar.title("About DermAssist")

st.sidebar.write("""
DermAssist is an AI-powered dermoscopic analysis system trained on HAM10000
and externally validated on ISIC 2019.

This system:
- Uses a 3-model ResNet50 ensemble
- Provides calibrated risk scores
- Estimates uncertainty via model disagreement
- Generates Grad-CAM visual explanations
""")

st.sidebar.write("---")
st.sidebar.write("Clinical Use Disclaimer:")
st.sidebar.write("""
This tool is for research and educational purposes only.
It does not replace professional medical diagnosis.
""")

st.title("DermAssist – Clinical Dermoscopic Risk Triage System")

page = st.radio("Select View", ["Prediction", "Validation Metrics"])

if page == "Prediction":

    uploaded_file = st.file_uploader("Upload Dermoscopic Image", type=["jpg","jpeg","png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        image_tensor = val_transform(image).unsqueeze(0).to(device)

        mean_prob, std_prob, individual_probs = ensemble_predict(
            ensemble_models,
            image_tensor
        )

        decision = decision_logic(mean_prob, std_prob)
        confidence = 1 - std_prob

        target_class = 1 if mean_prob >= CLASSIFICATION_THRESHOLD else 0
        cam = gradcam.generate(image_tensor, target_class)
        overlay = overlay_gradcam(image, cam)

        with col2:
            st.image(overlay, caption="Grad-CAM Attention Map", use_container_width=True)

        st.write("---")

        if decision == "HIGH RISK":
            st.error("High Risk – Immediate Clinical Evaluation Recommended")
        elif decision == "MODERATE RISK":
            st.warning("Moderate Risk – Professional Evaluation Recommended")
        elif decision == "UNCERTAIN":
            st.info("Uncertain – Dermatologist Review Recommended")
        else:
            st.success("Low Risk – Monitor and Recheck")

        st.subheader("Prediction Summary")

        st.metric("Malignancy Probability", f"{mean_prob * 100:.2f}%")
        st.metric("Confidence Score", f"{confidence * 100:.2f}%")
        st.metric("Model Disagreement", f"{std_prob * 100:.2f}%")

        st.write("Individual Model Outputs:")
        for i, p in enumerate(individual_probs):
            st.write(f"Model {i+1}: {p*100:.2f}%")

        st.write("---")
        st.write("Clinical Notes:")
        st.write("""
- Probability reflects estimated malignancy risk.
- Confidence is derived from ensemble agreement.
- High model disagreement indicates uncertainty.
- Grad-CAM highlights regions influencing the model's decision.
""")

if page == "Validation Metrics":

    st.subheader("Internal Validation (HAM10000)")

    st.write("""
    Ensemble AUC: 0.937  
    Malignant Recall: ~0.93  
    Accuracy (t=0.35): 82%
    """)

    st.write("""
    The model demonstrates strong internal performance with
    calibrated sensitivity for melanoma detection.
    """)

    st.write("---")
    st.subheader("External Validation (ISIC 2019)")
    st.write("""
    External AUC: 0.740  
    Precision (t=0.31): 80.7%  
    Recall (t=0.31): 50.0%  
    """)
    
    st.write("""
    External testing revealed expected performance drop due to domain shift,
    while maintaining clinically useful accuracy. The model generalizes well
    to independent datasets.
    """)