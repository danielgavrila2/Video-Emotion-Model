# Video Emotion Recognition Project

A complete system to perform **emotion recognition from video** using deep learning and cloud infrastructure (AWS SageMaker, S3, etc.). This project demonstrates an end‑to‑end pipeline: data ingestion → preprocessing → model training → deployment / inference → visualization / analysis.

---

## Table of Contents

1. [Overview](#overview)  
2. [Architecture & Design](#architecture--design)  
3. [Core AI / ML Concepts](#core-ai--ml-concepts)  
4. [Cloud & Infrastructure](#cloud--infrastructure)  
5. [Setup & Usage](#setup--usage)  
6. [File Structure](#file-structure)  
7. [Technologies Used](#technologies-used)  
8. [Evaluation & Metrics](#evaluation--metrics)  
9. [Limitations & Future Work](#limitations--future-work)  
10. [Acknowledgements & References](#acknowledgements--references)  

---

## 1. Overview

The goal of this project is to detect and classify emotional states (e.g. “happy”, “sad”, “angry”, “neutral”, etc.) from videos. Rather than static images only, the system handles temporal sequences (frames) and optionally leverages multimodal cues.

Key goals:

- Process video input (frame extraction, face detection, alignment)
- Extract visual features (via CNN / transfer learning)
- Model temporal dynamics (via LSTM, 3D CNN, or temporal fusion)
- Train, validate, and version models
- Deploy / serve the model in a scalable, cloud‑based environment
- Store data (raw videos, intermediate frames, logs) reliably
- Monitor and analyze model performance

---

## 2. Architecture & Design

High‑level pipeline:

```
Video Input → Frame Extraction → Face Detection / Crop →  
Preprocessing / Augmentation → Feature Extraction (CNN) →  
Temporal Module (LSTM / 3D conv / attention) → Output Emotions
```

Example research diagrams for multimodal emotion systems:

![Emotion Recognition Diagram](https://images.openai.com/thumbnails/url/cibpi3icu5meUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw4KMg0rNw_LS8zzKgp3Dw0y1nV2D_Io8AgO9jf1jqrydIqKSDKxDCsvcXYtVyu2NTQAAPrjJGQ)

Infrastructure mapping:

- **Client / UI / Ingest** → video upload  
- **Storage (S3)** → raw video, frames, logs  
- **Compute (SageMaker, EC2)** → training & inference  
- **Model repository / serving** → hosted endpoint  
- **Monitoring (CloudWatch)** → metrics, logs  

---

## 3. Core AI / ML Concepts

### CNN / Transfer Learning  
Use pretrained CNNs (e.g. ResNet, EfficientNet) to extract features and fine‑tune for emotion classification.

### Temporal Modeling  
- LSTM / GRU to model time dependencies  
- 3D CNN for spatio‑temporal features  
- Temporal attention layers for context awareness  

### Loss & Training  
- Categorical cross‑entropy  
- Class weighting for imbalance  
- Data augmentation (flip, crop, brightness)  

### Evaluation Metrics  
- Accuracy, Precision, Recall, F1‑score  
- Confusion Matrix  
- ROC / AUC curves  

---

## 4. Cloud & Infrastructure (AWS)

### Amazon S3  
Used for storing raw videos, extracted frames, processed datasets, and model artifacts.

```
s3://video-emotion-project/
   raw-videos/
   frames/
   processed-data/
   model-artifacts/
   logs/
```

### AWS SageMaker  
Handles model training and deployment.

- Training jobs (distributed, GPU‑accelerated)
- Hyperparameter tuning
- Real‑time endpoints
- Batch transform jobs
- Experiment tracking & versioning

### IAM & Security  
- Proper roles for SageMaker to access S3  
- Encrypted storage and secure endpoints  

---

## 5. Setup & Usage

### Prerequisites

- AWS account (S3, SageMaker access)  
- Python 3.8+  
- `awscli` configured  

### Clone Repository

```bash
git clone https://github.com/danielgavrila2/Video-Emotion-Project.git
cd Video-Emotion-Project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Local Training

```bash
python train_local.py --data-dir ./data --epochs 20 --batch-size 32
```

### AWS SageMaker Training

```bash
python train_sagemaker.py   --s3-bucket my-bucket   --role-arn arn:aws:iam::123456789012:role/SageMakerRole   --region us-east-1   --epochs 30   --batch-size 32
```

### Deploy Model Endpoint

```python
from sagemaker import Model

model = Model(
    model_data='s3://my-bucket/model-artifacts/model.tar.gz',
    role='arn:aws:iam::123456789012:role/SageMakerRole',
    image_uri='your-container-uri'
)

predictor = model.deploy(initial_instance_count=1, instance_type='ml.m5.large')
```

---

## 6. File Structure

```
Video-Emotion-Project/
├── training/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   └── utils.py
├── deployment/
│   ├── inference.py
│   └── serve_script.py
├── client/
│   ├── client.py
│   └── video_utils.py
├── train_sagemaker.py
├── batch_inference.py
├── requirements.txt
└── README.md
```

---

## 7. Technologies Used

| Layer | Technology | Purpose |
|-------|-------------|----------|
| DL Framework | PyTorch / TensorFlow | Model definition, training |
| Vision | OpenCV, Dlib | Frame extraction, face detection |
| AWS SDK | boto3, sagemaker | Cloud integration |
| Cloud | S3, SageMaker, IAM | Storage, compute, security |
| Deployment | Docker, AWS ECR | Containerization |
| Data | NumPy, Pandas | Processing |
| Logging | TensorBoard, CloudWatch | Metrics tracking |

---

## 8. Evaluation & Metrics

Include confusion matrix, precision/recall/F1 plots, and learning curves.  
Sample evaluation:

![Confusion Matrix](https://images.openai.com/thumbnails/url/b64tuHicu5meUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw5O9fPIKvCPKAzIqIosLQkNdS5Jzsw1DnAPi4rwq3L2cAt2jHTPdzF38wl0VCu2NTQAACGLJR0)

---

## 9. Limitations & Future Work

- Dataset imbalance  
- Mixed emotions  
- Lighting / occlusion issues  
- Generalization across demographics  
- Real-time deployment optimization  
- Add multimodal (audio/text) data  

---

## 10. Acknowledgements & References

- FER‑2013, AFEW datasets  
- AWS SageMaker documentation  
- PyTorch & TensorFlow communities  

---

**Author:** Daniel Gavrila  
**License:** MIT  
