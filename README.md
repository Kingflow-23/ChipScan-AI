# Interactive Segmentation Correction with SAM and Active Learning

## Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation](#motivation)
3. [System Architecture](#system-architecture)
4. [Dataset](#dataset)
5. [Active Learning Workflow](#active-learning-workflow)
6. [Segment Anything Model (SAM) Integration](#segment-anything-model-sam-integration)
7. [Frontend: Web-Based Correction Interface](#frontend-web-based-correction-interface)
8. [Correction Logic](#correction-logic)
9. [Data Organization](#data-organization)
10. [Retraining Pipeline](#retraining-pipeline)
11. [Performance Considerations](#performance-considerations)
12. [Limitations and Future Work](#limitations-and-future-work)
13. [Installation and Usage](#installation-and-usage)
14. [Deployment on Azure App Service](#deployment-on-azure-app-service)
15. [Current Deployment Status and Blocking Issues](#current-deployment-status-and-blocking-issues)

---

## Project Overview

This project implements an **interactive human-in-the-loop correction system** for object detection and segmentation. The objective is to iteratively improve model performance while **minimizing manual annotation effort** by focusing only on incorrect predictions.

Key components:

* YOLO-based object detection and segmentation model
* Web-based frontend for prediction review and correction
* **Segment Anything Model (SAM)** for high-quality mask refinement
* Active learning loop with incremental retraining

> Core principle: **Correct only what is wrong; preserve what is already correct.**

---

## Motivation

High-quality segmentation annotation is expensive and time-consuming, particularly for high-resolution industrial imagery. This project addresses that problem by:

* Using a pretrained model to generate predictions on new data
* Asking the user to correct only erroneous predictions
* Leveraging SAM to automate fine-grained mask creation
* Feeding corrections back into the training loop

The result is a scalable **active learning system** that improves with minimal human intervention.

---

## System Architecture

### High-Level Pipeline

1. **Initial Training**

   * YOLO model trained on a Roboflow-labeled dataset.
2. **Prediction**

   * Model performs inference on new image batches.
   * Bounding boxes and masks are generated.
3. **Results Review**

   * Visual overlays and per-object metrics are displayed.
4. **User Correction**

   * Users correct only incorrect predictions via the frontend.
5. **SAM Refinement**

   * User-drawn bounding boxes are converted into precise masks.
6. **Feedback Collection**

   * Corrections and metadata are stored in structured JSON.
7. **Retraining**

   * Original predictions and corrections are merged.
   * Model is fine-tuned incrementally.

This architecture forms a **closed-loop active learning system**.

---

## Dataset

### Source

The baseline dataset was sourced from **Roboflow** and includes:

* Images annotated with classes such as:

  * `Chip`
  * `Void`
* YOLO-compatible detection and segmentation formats
* Data augmentation (blur, rotation, etc.) applied via Roboflow

### Role in Active Learning

* Initial dataset used for baseline training
* User corrections progressively expand and refine the dataset

---

## Active Learning Workflow

Instead of re-labeling entire datasets:

1. The model predicts on unseen images
2. The user reviews predictions
3. Only incorrect regions are corrected
4. Unsatisfactory predicted boxes can be deleted
5. Corrections are reintegrated into retraining

This approach significantly reduces annotation cost while improving model accuracy.

---

## Segment Anything Model (SAM) Integration

SAM is used to convert **coarse user corrections** into **high-quality segmentation masks**.

Key characteristics:

* SAM is used only during correction, never during inference
* Bounding boxes are sufficient input for mask generation
* Manual polygon drawing is avoided entirely

This design balances annotation speed with segmentation precision.

---

## Frontend: Web-Based Correction Interface

The frontend is a lightweight Flask-based web interface designed to support the active learning loop. It is organized into four main pages.

---

### 1. Prediction Page

**Purpose:** Trigger model inference on new images.

**Functionality:**

* Launch batch prediction
* Generate overlays with bounding boxes and class labels
* Store prediction results for review

<img width="1877" height="1003" alt="image" src="https://github.com/user-attachments/assets/ae413b2c-98da-4e4a-9662-c2e37bf53a2f" />

<img width="1861" height="997" alt="image" src="https://github.com/user-attachments/assets/bb971b96-0f92-422b-8388-a7da4c1eca99" />

---

### 2. Results Page

**Purpose:** Review model predictions before correction.

**Functionality:**

* Display predicted images sequentially
* Show bounding boxes, masks, and class labels
* Display per-chip metrics

Submitting no correction implicitly validates the prediction.

<img width="1862" height="1001" alt="image" src="https://github.com/user-attachments/assets/3f450d2e-97a1-4805-b269-73e86ca1df30" />

<img width="948" height="521" alt="image" src="https://github.com/user-attachments/assets/0963e263-3fa7-4415-bb0b-c6bf80d59112" />

---

### 3. Correction Page

**Purpose:** Correct only erroneous predictions.

**Functionality:**

* Draw bounding boxes over incorrect regions
* Assign class labels (`Chip`, `Void`)
* Delete incorrect predicted boxes
* Undo and modify corrections

Submitted boxes are refined into masks using SAM.

<img width="1865" height="1000" alt="image" src="https://github.com/user-attachments/assets/1f0fb0e2-a272-4221-a56e-d927dc2c7cbf" />

<img width="1876" height="999" alt="image" src="https://github.com/user-attachments/assets/c2838c77-d23b-4eef-ad50-bb70be60cf17" />

If no changes has been made on the batch, UI inform user and don't let him retrain.

<img width="1877" height="1000" alt="image" src="https://github.com/user-attachments/assets/955d707e-465d-47d3-8232-bebb18e153bd" />


---

### 4. Retraining Page

**Purpose:** Monitor and control retraining.

**Functionality:**

* Trigger retraining manually
* Display retraining status (idle, running, completed, failed)
* Surface backend errors if any

<img width="1878" height="1001" alt="image" src="https://github.com/user-attachments/assets/b20a8cef-8b4a-42f1-83a6-272c99d3fdf9" />

---

## Correction Logic

### Relabel-Only-Errors Strategy

* Original predictions are preserved by default
* Corrections override only erroneous regions
* During retraining:

  * Corrected masks replace incorrect predictions
  * Valid predictions remain unchanged

This guarantees annotation integrity.

---

## Data Organization

### Results Directory

Stores visual artifacts:

* Prediction overlays
* Corrected overlays
* SAM-generated masks (`.npy`)

### Correction Metadata

Stored as structured JSON:

* Image ID
* Correction status (`ok`, `corrected`)
* Bounding boxes and class labels
* Mask references

This separation simplifies retraining and debugging.

---

## Retraining Pipeline

* Corrections are merged with original labels and predictions
* YOLO is retrained incrementally
* Fewer epochs are used for new data to avoid catastrophic forgetting

This enables efficient iterative improvement.

---

## Performance Considerations

* SAM invoked only when needed
* Minimal disk I/O
* Batch-based retraining
* Asynchronous background retraining

---

## Limitations and Future Work

### Current Limitations

* Box-based corrections only
* No real-time SAM preview
* Batch retraining (not fully online)
* Single-user workflow assumption

### Future Improvements

* Confidence-based sample selection
* Incremental online learning
* Multi-user correction sessions
* Dataset versioning
* Automated evaluation dashboards
* Scaling up: Better model for correction (sam-vit-h instead of sam-vit-b), more data for traininng a better yolo model (yolo11X maybe instead of 11m ...)

---

## Installation and Usage

### Requirements

* Python 3.10+
* PyTorch (CPU or CUDA)
* YOLOv8 (Ultralytics)
* Flask
* Segment Anything Model dependencies

See the rest of the dependencies in the requirements.txt

### Local Usage

1. Clone the repository
2. Install dependencies
3. Run the Flask application
4. Access the web interface locally

---

## Deployment on Azure App Service

The application was containerized using Docker and deployed on **Azure App Service for Linux** using **Azure Container Registry (ACR)**.

### Deployment Steps

1. **Containerization**

   * Application packaged into a Docker image
   * Gunicorn used as the production WSGI server
   * Application listens on port `5000`

2. **Azure Container Registry (ACR)**

   * ACR instance created
   * Docker image pushed to ACR
   * Admin credentials enabled for initial testing

3. **Azure Web App (App Service)**

   * Linux Web App created
   * Container-based deployment selected
   * Image pulled from ACR
   * Authentication configured (ACR admin or managed identity)

4. **Port Configuration**

   * Azure automatically injects the `PORT` environment variable
   * The container exposes port `5000`
   * Azure maps traffic correctly when the app responds in time

---

## Current Deployment Status and Blocking Issues

At the time of writing:

* The container image **pulls successfully** from ACR
* The container **starts correctly**
* The application **runs locally without issues** using `docker run`

However, on Azure App Service:

* The **startup probe fails after several minutes**
* Azure stops the container even though it initially reports "Container is running"

### Root Causes Identified

1. **Free F1 App Service Plan Limitations**

   * Very limited CPU and memory
   * Not suitable for large ML containers

2. **Large Image Size**

   * Image size ~2–3 GB after optimization
   * Heavy dependencies (`torch`, `ultralytics`, SAM)

3. **Cold Start and Model Load Time**

   * SAM and YOLO model loading exceeds Azure startup probe timeout
   * App does not respond fast enough to HTTP health checks

4. **No GPU Support on App Service**

   * CUDA dependencies are unused
   * CPU-only inference increases startup time

### Current Blocking Point

The application exceeds the **startup and resource limits** of the Free App Service plan. While the container itself is valid, Azure terminates it before the app becomes responsive.

### Recommended Next Steps

* Upgrade to **Basic (B1) or higher App Service Plan**
* Or migrate to **Azure Container Apps** or **Azure VM**
* Further reduce image size and lazy-load models
* Implement a lightweight health-check endpoint

---

**Status:** Architecture and implementation complete. Deployment blocked by infrastructure constraints, not application correctness.
