# DermAssist  
## AI-Powered Dermoscopic Skin Cancer Risk Triage System

DermAssist is an AI-based dermoscopic image analysis system designed to estimate malignancy risk, quantify prediction confidence, and provide visual explainability through attention mapping.

The system supports dermatological triage workflows by combining deep learning ensemble modeling with uncertainty estimation.

---

## Overview

DermAssist provides:

- Malignancy Probability (%)
- Confidence Score
- Uncertainty Estimation
- Grad-CAM Visual Attention Map
- Risk Category Recommendation

The model was trained on HAM10000 and evaluated internally and on an external dataset (ISIC 2019) to assess generalization performance.

---

## Technology Stack

<p align="left">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black"/>
</p>

---

## Model Architecture

| Component | Specification |
|------------|---------------|
| Backbone | ResNet50 (ImageNet pretrained) |
| Fine-Tuning | Full network fine-tuning |
| Ensemble | 3 independently trained models |
| Optimizer | AdamW (lr = 1e-4, weight_decay = 1e-4) |
| Loss Function | BCEWithLogitsLoss (class-weighted) |
| Early Stopping | Validation AUC |
| Explainability | Grad-CAM |
| Uncertainty | Ensemble standard deviation |

---

## Internal Validation (HAM10000)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.937 |
| Accuracy | 0.86 |
| Malignant Recall | > 0.90 |
| Malignant Precision | ~0.59 |
| Benign Precision | ~0.96 |

Confusion Matrix (Threshold = 0.5)

|              | Predicted Benign | Predicted Malignant |
|--------------|------------------|---------------------|
| Actual Benign | 1395 | 236 |
| Actual Malignant | 54 | 339 |

---

## External Validation (ISIC 2019 – Filtered Classes)

| Metric | Value |
|--------|-------|
| AUC-ROC | ~0.72 |
| Accuracy | ~0.57 |
| Domain Shift Observed | Yes |

Performance differences across datasets highlight the importance of cross-dataset validation in medical AI systems.

---

## Clinical Decision Framework

Each prediction includes:

- Malignancy Probability (%)
- Confidence Score
- Uncertainty Level
- Risk Category

Risk Categories:

- Low Risk – Routine Monitoring  
- Moderate Risk – Professional Evaluation Recommended  
- High Risk – Urgent Dermatological Review  
- Uncertain – Manual Dermatologist Review Recommended  

Decision logic integrates both probability thresholds and calibrated uncertainty.

---

## Key Methodological Strengths

- Lesion-level data split to prevent leakage  
- Class imbalance handling  
- Ensemble-based uncertainty quantification  
- External dataset validation  
- Grad-CAM explainability  
- Deployment-ready Streamlit interface  

---

## Deployment

### Run Locally
