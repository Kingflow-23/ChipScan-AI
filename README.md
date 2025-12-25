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
15. [Demo](#demo)

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

We can save the results in a csv file at the bottom of the page.



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
* Display retraining is happening by a spinning wheel

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
* Scaling up: Better model for correction (sam-vit-h instead of sam-vit-b), more data for traininng a better yolo model (yolo11X maybe instead of actual 11m ...)

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

## Deployment on Microsoft Azure

The application was deployed as a containerized AI web service using **Azure Container Apps (ACA)**. This solution enables scalable execution of a compute-intensive AI workload without managing virtual machines.

---

### 1. Containerization

The application was packaged into a Docker container to ensure portability and environment consistency.

- Flask application served with **Gunicorn**
- AI inference and processing logic embedded in the container
- Application listens on **port 5000**
- All dependencies included inside the image

<img width="2239" height="552" alt="image" src="https://github.com/user-attachments/assets/c0621e05-32c2-4f66-b560-3413bc6aef6e" />

---

### 2. Azure Resource Group

A dedicated **Azure Resource Group** was created to host all deployment resources.

- Logical organization of cloud components
- Simplified lifecycle management and cost tracking
- Isolation from other Azure projects

<img width="2173" height="1019" alt="image" src="https://github.com/user-attachments/assets/47a070a9-9f73-4202-beb0-c56bb78faf84" />

On the free tier, you need to be particularly attentive to the location you choose (Azure gives you your allowed set of region for your student account).

---

### 3. Azure Container Registry (ACR)

An **Azure Container Registry** was used to store and distribute the application image.

- Docker image built locally
- Image pushed to a **private ACR instance**
- Secure image pull by Azure Container Apps

**Purpose of ACR:**
- Hosts custom AI containers (models, frameworks, dependencies)
- Supports large container images
- Provides secure, high-performance access inside Azure

<img width="2180" height="1219" alt="image" src="https://github.com/user-attachments/assets/282807d1-f206-4b43-b8ae-5b8febc4eeb4" />

---

### 4. Azure Container Apps Environment

A **Container Apps Environment** was created to act as a secure runtime boundary.

- Manages networking, logging, and scaling
- Provides isolation between container apps
- Fully managed by Azure

---

### 5. Azure Container App Configuration

The container app was configured to run the AI service.

- Image source: Azure Container Registry
- Automatic authentication to ACR
- Default container entrypoint (Gunicorn)

**Resource allocation:**
- CPU and memory allocated for AI inference workload
- Resources sufficient to load deep learning models at startup

**Pricing model:**
- Consumption-based pricing
- No upfront payment required
- Costs incurred only when the app is running

---

### 6. Ingress and Networking

Public access was enabled using Azure Container Apps ingress.

- Ingress enabled
- HTTP traffic allowed
- Target port set to **5000**
- Azure-generated public URL provided

This allows direct access to the web interface and API endpoints.

<img width="2155" height="1008" alt="image" src="https://github.com/user-attachments/assets/ab9e6fde-0c44-4e71-9ac8-8522fef4e9d5" />

<img width="1506" height="412" alt="image" src="https://github.com/user-attachments/assets/6ecb78b6-06da-43da-8f56-06b4050a80d6" />

---

### 7. Deployment Validation

The deployment was validated using Azure logs and diagnostics.

- Container startup confirmed
- Gunicorn successfully listening on port 5000
- AI models loaded correctly
- Web interface and API endpoints operational

---

### Final Outcome

The application is successfully deployed as a **fully managed AI web service** on Azure Container Apps.

- No virtual machines to manage
- Scalable and resilient architecture
- Secure container image delivery
- Suitable for production-grade AI inference workloads

---

## Demo

### Prediction

https://github.com/user-attachments/assets/4adcaa2c-4195-4f92-af1c-9b79fb5931ea

### Results

https://github.com/user-attachments/assets/c3997b63-0e56-463c-bb10-5c6ade1751f9

### Correction

https://github.com/user-attachments/assets/2d6fe9ff-c9a6-4674-a55e-2312fbe001f3

### Retraining

https://github.com/user-attachments/assets/744c4534-1b8f-4aa2-8691-32ce776ca198

---
