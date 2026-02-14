# DermAssist – AI-Assisted Dermoscopic Risk Assessment

DermAssist is a deep learning–based dermoscopic analysis system designed to estimate malignancy risk in skin lesion images.  
The system combines ensemble modeling, uncertainty estimation, and visual explainability to provide clinically interpretable predictions.

---

## Overview

DermAssist was trained on the HAM10000 dermoscopic dataset and externally validated on ISIC 2019.  
The system is designed as a research-grade clinical AI prototype for melanoma risk assessment.

Key features:

- 3× ResNet50 ensemble (fully fine-tuned)
- Binary malignant vs benign classification
- Uncertainty estimation via ensemble standard deviation
- Grad-CAM visual explanation
- External validation on independent dataset
- Streamlit-based clinical interface

---

## Model Architecture

Backbone:
- ResNet50 (ImageNet pretrained)

Modifications:
- Fully fine-tuned network
- Custom classification head:
  - Linear → ReLU → Dropout (0.5) → Linear
- Binary output (malignant probability)

Training Strategy:
- AdamW optimizer (lr=1e-4, weight_decay=1e-4)
- Early stopping on validation AUC
- GroupShuffleSplit to prevent lesion-level leakage
- Threshold calibration for clinical sensitivity

---

## Ensemble Strategy

Three independently trained ResNet50 models were combined via probability averaging.

Benefits:
- Reduced variance
- Improved calibration
- Increased robustness
- Enables uncertainty estimation

Uncertainty is computed as:
Standard deviation across model predictions.

---

## Performance

Internal Validation (HAM10000):
- AUC: ~0.93
- Malignant Recall: ~0.90

External Validation (ISIC 2019, filtered classes):
- Demonstrated domain sensitivity differences
- Highlights importance of calibration across datasets

---

## Clinical Interpretation

Output includes:

- Malignancy probability (%)
- Confidence score (ensemble agreement)
- Model disagreement (uncertainty)
- Grad-CAM attention visualization

Decision categories:
- Low Risk – Routine Monitoring Advised
- Moderate Risk – Dermatological Consultation Recommended
- High Risk – Urgent Dermatological Assessment Advised
- Uncertain – Dermatologist Review Recommended

---

## Installation

Clone repository:

```bash
git clone https://github.com/yourusername/DermAssist.git
cd DermAssist
