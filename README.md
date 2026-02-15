# DermAssist  
## Uncertainty-Aware Dermoscopic Skin Cancer Risk Triage System

DermAssist is an AI-based dermoscopic image analysis system designed to estimate malignancy risk, quantify prediction confidence, and provide visual explainability.

The system supports dermatological triage workflows by combining deep learning ensemble modeling with uncertainty estimation.

---

## Overview

DermAssist provides:

- Malignancy Probability (%)
- Confidence Score
- Uncertainty Estimation
- Grad-CAM Visual Attention Map
- Risk Category Recommendation

The model was trained on HAM10000 and externally evaluated on ISIC 2019 to assess generalization performance.

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

Final prediction = mean probability across models.  
Uncertainty = standard deviation across ensemble outputs.

---

## Internal Validation (HAM10000)

### Base Performance (Classification Threshold = 0.35)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.937 |
| Accuracy | 0.82 |
| Malignant Recall | 0.93 |
| Malignant Precision | 0.52 |
| False Negatives | 28 |

Confusion Matrix:

|              | Predicted Benign | Predicted Malignant |
|--------------|------------------|---------------------|
| Actual Benign | 1288 | 343 |
| Actual Malignant | 28 | 365 |

---

### Uncertainty-Aware Performance  
(Uncertainty Threshold = 0.165)

Total Cases Flagged as Uncertain: 477

Confident Cases Only:

| Metric | Value |
|--------|-------|
| Accuracy | 0.89 |
| Malignant Recall | 0.95 |
| Malignant Precision | 0.64 |
| False Negatives (Confident Only) | 15 |

Confusion Matrix (Confident Cases):

|              | Predicted Benign | Predicted Malignant |
|--------------|------------------|---------------------|
| Actual Benign | 1100 | 155 |
| Actual Malignant | 15 | 277 |

Uncertainty filtering reduced false negatives from 28 to 15 among confident predictions, while flagging high-risk ambiguous cases for manual review.

---

## External Validation (ISIC 2019 – Filtered Classes)

| Metric | Value |
|--------|-------|
| AUC-ROC | ~0.72 |
| Accuracy | ~0.57 |
| Domain Shift Observed | Yes |

Performance differences across datasets highlight distribution variability and emphasize the importance of cross-dataset validation in medical AI systems.

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

## Key Features

- Lesion-level data splitting to prevent leakage  
- Class imbalance handling  
- Ensemble-based uncertainty estimation  
- External dataset validation  
- Grad-CAM explainability  
- Streamlit deployment  
- HuggingFace-hosted model weights  

---

## Deployment

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://your-website-url.com)

