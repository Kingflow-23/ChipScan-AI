# Interactive Segmentation Correction with SAM and Active Learning

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Motivation](#motivation)  
3. [System Architecture](#system-architecture)  
4. [Dataset](#dataset)  
5. [Active Learning Workflow](#active-learning-workflow)  
6. [Segment Anything Model (SAM) Integration](#segment-anything-model-sam-integration)  
7. [Frontend: Batch Correction Interface](#frontend-batch-correction-interface)  
8. [Correction Logic](#correction-logic)  
9. [Data Organization](#data-organization)  
10. [Retraining Pipeline](#retraining-pipeline)  
11. [Performance Considerations](#performance-considerations)  
12. [Limitations and Future Work](#limitations-and-future-work)  
13. [Installation and Usage](#installation-and-usage)  
14. [Acknowledgments](#acknowledgments)  

---

## Project Overview

This repository implements an **interactive human-in-the-loop correction system** for object detection and segmentation. The goal is to **improve model performance iteratively** using minimal manual labeling by focusing only on incorrect predictions.  

Key components:
- Pre-trained object detection/segmentation model (YOLO)
- Web-based correction interface
- **Segment Anything Model (SAM)** for precise mask generation
- Active learning loop for iterative retraining

> Principle: **Correct only what is wrong; preserve what is already correct.**

---

## Motivation

Manual annotation is costly and time-consuming, especially for high-resolution images requiring precise segmentation masks.  

This system addresses this by:
- Using an initial trained model to generate predictions on new images
- Allowing users to correct only erroneous regions
- Leveraging corrections to improve the model iteratively
- Reducing annotation workload while maintaining high-quality data  

---

## System Architecture

### High-Level Pipeline

1. **Initial Training**
   - Model trained on manually labeled dataset from Roboflow.
2. **Prediction**
   - Model predicts objects on new image batches.
   - Visual overlays are generated for user inspection.
3. **User Correction**
   - Corrections are made through a web-based interface.
4. **SAM Refinement**
   - Bounding boxes are refined into precise segmentation masks.
5. **Feedback Collection**
   - Metadata and corrections are stored in structured JSON.
6. **Retraining**
   - Original predictions and corrections are merged.
   - Model is fine-tuned incrementally.
   
This workflow forms a **closed-loop active learning system**.

---

## Dataset

### Source

The baseline dataset was sourced from **Roboflow**:
- Images manually labeled with object classes such as:
  - `Chip`
  - `Void`
- Annotations exported in detection/segmentation formats (YOLO-compatible)

### Role in Active Learning

- Baseline data used for initial model training
- Subsequent corrections expand dataset iteratively

---

## Active Learning Workflow

Instead of re-labeling entire datasets:
1. Model predicts on new images
2. Users review predictions
3. Only misclassified regions are corrected
4. Corrections are fed back into the retraining pipeline  

This reduces labeling effort while improving model accuracy over time.

---

## Segment Anything Model (SAM) Integration

SAM is integrated to convert **coarse user bounding boxes** into **high-quality masks**.

### Key Details:
- SAM loaded once at application startup
- Predictor reused across all correction requests
- Reduces the need for manual polygon/mask drawing
- Improves mask accuracy without increasing annotation burden

**Note:** SAM is applied **only during correction**, not during inference.

---

## Frontend: Batch Correction Interface

### User Interaction

- Displays one image at a time
- Shows model prediction overlay
- Users draw bounding boxes around incorrect regions
- Each bounding box is assigned a class (`Chip`, `Void`)
- Undo functionality included
- Submitting with no corrections indicates the prediction is correct

### Design Choice

- Focused on **minimal user interaction**
- No “mark as checked” buttons
- Encourages efficiency and reduces cognitive load

---

## Correction Logic

### Relabel-Only-Errors Strategy

- Original model predictions are **preserved by default**
- User corrections **override only erroneous regions**
- During retraining:
  - Corrected masks replace incorrect predictions
  - Unmodified predictions remain unchanged

This ensures **valid annotations are never accidentally removed**.

---

## Data Organization

### Results Directory

- Stores visual outputs:
  - Model overlays
  - Corrected overlays
  - SAM-generated masks (`.npy` files)

### Correction Metadata

Stored separately in JSON:
- Image ID
- Correction status (`ok`, `corrected`)
- Bounding boxes
- Class labels
- Mask references

Benefits:
- Clean retraining datasets
- Easy debugging
- Clear separation between visuals and structured data

---

## Retraining Pipeline

- Server-side script merges corrections with original predictions
- YOLO model is retrained using:
  - Original Roboflow labels
  - Model predictions
  - User corrections
- Incremental retraining reduces epochs for new data while preserving prior knowledge
- Forms a **closed-loop active learning pipeline**

---

## Performance Considerations

- SAM is loaded **once**, minimizing redundant computations
- Minimal disk writes; intermediate results handled efficiently
- Frontend polling for retraining status is adjustable
- Batch-based processing prevents server overload

---

## Limitations and Future Work

### Current Limitations
- Corrections are **box-based**, not free-form
- No **real-time SAM preview**
- Retraining is batch-based, not fully online
- Single-user assumption for correction sessions

### Future Improvements
- Confidence-based sample selection
- Incremental/partial retraining
- Multi-user correction sessions
- Dataset versioning and tracking per iteration
- Automated model performance reporting

---

## Installation and Usage

### Requirements

- Python 3.10+
- PyTorch with CUDA support
- YOLOv8 (Ultralytics)
- Flask
- SAM dependencies (see Meta AI repository)
- Roboflow dataset exported in YOLO format

### Steps

1. Clone repository:
   ```bash
   git clone 
   

2. Install dependencies:

