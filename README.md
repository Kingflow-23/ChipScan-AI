# Interactive Segmentation Correction with SAM and Active Learning

## Overview

This project implements an **interactive human-in-the-loop correction system** for object detection and segmentation models.  
It is designed to **improve model performance over time** by allowing users to correct only incorrect predictions and feed those corrections back into a retraining pipeline.

The system combines:
- A pre-trained detection/segmentation model
- A web-based batch correction interface
- Meta AI’s **Segment Anything Model (SAM)** for precise mask refinement
- An **active learning loop** for iterative retraining

The guiding principle is:

> **Correct only what is wrong — keep what is already correct.**

---

## Dataset

### Source

The initial dataset used in this project comes from **Roboflow**.

- Images were **manually labeled** using Roboflow’s annotation tools
- Labels include object classes such as:
  - `Chip`
  - `Void`
- Annotations were exported in a standard detection/segmentation format

This manually labeled dataset serves as the **baseline training data** for the first model version.

---

## Why Active Learning?

Manual annotation is expensive and time-consuming.

Instead of repeatedly labeling full datasets, this project:
- Uses an existing trained model
- Lets the model make predictions on new images
- Allows users to **correct only prediction errors**
- Uses those corrections to retrain and improve the model

This dramatically reduces labeling effort while continuously improving accuracy.

---

## System Architecture

### High-Level Pipeline

1. **Initial Training**
   - Model trained on Roboflow-labeled dataset

2. **Prediction**
   - Model generates predictions on new image batches
   - Visual overlays are saved for review

3. **User Correction**
   - User reviews predictions in a web UI
   - Only incorrect regions are corrected
   - Corrections are provided as bounding boxes

4. **SAM Refinement**
   - Bounding boxes are passed to SAM
   - SAM produces high-quality segmentation masks

5. **Feedback Collection**
   - Corrections are stored as structured metadata
   - Images with no corrections are marked as valid

6. **Retraining**
   - Original predictions + corrections are merged
   - Dataset is prepared for fine-tuning
   - Model improves iteratively

---

## Segment Anything Model (SAM)

### Purpose

SAM is used to convert **coarse user input** (bounding boxes) into **precise segmentation masks**.

This avoids:
- Manual pixel-level annotation
- Polygon drawing
- Mask painting

### Implementation Details

- SAM is loaded **once** at application startup
- The predictor is reused across all correction requests
- This avoids repeated model loading and performance issues

SAM is used **only during correction**, not during inference.

---

## Frontend: Batch Correction Interface

### User Experience

- One image at a time
- Model prediction overlay shown
- User draws bounding boxes over incorrect regions only
- Each box is assigned a class:
  - **Chip** → Red
  - **Void** → Yellow
- Undo supported
- Submit button advances to the next image

### Key Design Choice

There is **no “Mark as Checked” button**.

- Submitting with **no boxes** means:
  > “This image is correct.”
- Submitting with boxes means:
  > “These regions were incorrect and are corrected.”

This keeps the workflow simple and intuitive.

---

## Correction Logic

### Relabel-Only-Errors Strategy

The system does **not** assume that unlabeled regions are empty.

Instead:
- Original model predictions are preserved by default
- User corrections override specific regions
- During retraining:
  - Corrected masks replace incorrect predictions
  - Unmodified predictions remain unchanged

This prevents accidental deletion of valid annotations.

---

## Data Organization

### Results Directory

Used only for **visual outputs**:
- Model prediction overlays
- Corrected overlays
- SAM-generated masks (`.npy`)

### Correction Metadata

Stored separately as structured JSON:
- Image ID
- Correction status (`ok`, `corrected`)
- Bounding boxes
- Class labels
- Mask references

This separation ensures:
- Readability
- Clean retraining datasets
- Easy debugging

---

## Retraining Pipeline

- Correction data is exported server-side
- Training datasets are rebuilt automatically
- Model fine-tuning uses:
  - Original Roboflow labels
  - Model predictions
  - User corrections

This creates a **closed active learning loop**.

---

## Performance Considerations

- SAM loaded once
- No redundant disk writes
- Lightweight frontend logic
- Efficient batch handling

---

## Current Limitations

- Corrections are box-based (not free-form)
- No real-time SAM preview (by design)
- Retraining is batch-based, not online

---

## Future Improvements

- Confidence-based sample selection
- Partial or incremental retraining
- Multi-user correction sessions
- Dataset versioning
- Model performance tracking per iteration

---

## Summary

This project extends a **manually labeled Roboflow dataset** into a **scalable active learning system**.

By combining:
- Human correction
- SAM refinement
- Smart data merging
- Iterative retraining

It provides a practical and production-ready approach to continuously improving segmentation models with minimal annotation effort.

---

## Acknowledgments

- **Roboflow** for dataset management and initial annotations
- **Meta AI** for Segment Anything Model (SAM)