import cv2
import json
import uuid
import threading
from pathlib import Path

from ultralytics import YOLO
from flask import Flask, request, jsonify, render_template

from services.inference import run_inference_service
from services.correction import correct_segmentation_service
from services.retrain import start_retraining
from utils.sam_model import load_sam_model, initialize_predictor

from config import UPLOAD_DIR, RESULT_DIR, T_MODEL_PATH, RESULT_JSON_DIR

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_JSON_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLO model once
model = YOLO(str(T_MODEL_PATH))

# Load SAM predictor once
_sam_predictor = initialize_predictor(load_sam_model())

# ---------------------------------------------------------------------------
# Routes – Pages
# ---------------------------------------------------------------------------


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/results/batch/<batch_id>", methods=["GET"])
def results_batch(batch_id):
    batch_path = RESULT_JSON_DIR / f"batch_{batch_id}.json"
    if not batch_path.exists():
        return "Batch results not found", 404

    with open(batch_path, "r") as f:
        results = json.load(f)

    return render_template("results.html", results=results)


@app.route("/admin", methods=["GET"])
def admin():
    return render_template("admin.html")


# ---------------------------------------------------------------------------
# Routes – API
# ---------------------------------------------------------------------------


@app.route("/predict", methods=["POST"])
def predict():
    if "images" not in request.files:
        return jsonify({"error": "No images provided"}), 400

    files = request.files.getlist("images")
    batch_id = str(uuid.uuid4())
    batch_results = []

    for image_file in files:
        image_id = str(uuid.uuid4())
        ext = Path(image_file.filename).suffix or ".jpg"
        image_path = UPLOAD_DIR / f"{image_id}{ext}"
        image_file.save(image_path)

        # -------------------------------
        # Inference
        # -------------------------------
        metrics, yolo_result = run_inference_service(str(image_path), return_raw=True)

        # -------------------------------
        # Overlay generation
        # -------------------------------
        overlay = yolo_result.plot()
        overlay_path = RESULT_DIR / f"{image_id}_overlay{ext}"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        json_payload = {
            "batch_id": batch_id,
            "filename": image_file.filename,
            "image_id": image_id,
            "image_url": f"/static/uploads/{image_path.name}",
            "overlay_url": f"/static/results/{overlay_path.name}",
            **metrics,
        }

        # Save per-image JSON (still useful for correction)
        with open(RESULT_JSON_DIR / f"{image_id}.json", "w") as f:
            json.dump(json_payload, f, indent=2)

        batch_results.append(json_payload)

    # Save batch index (THIS IS THE KEY)
    batch_path = RESULT_JSON_DIR / f"batch_{batch_id}.json"
    with open(batch_path, "w") as f:
        json.dump(batch_results, f, indent=2)

    return jsonify({"batch_id": batch_id, "count": len(batch_results)})


@app.route("/correct/batch/<batch_id>", methods=["GET"])
def correct_batch_page(batch_id):
    batch_path = RESULT_JSON_DIR / f"batch_{batch_id}.json"
    if not batch_path.exists():
        return "Batch not found", 404

    with open(batch_path, "r") as f:
        results = json.load(f)

    return render_template("correct.html", results=results, batch_id=batch_id)


@app.route("/correct", methods=["POST"])
def batch_correct():
    """
    Processes user corrections with SAM, saves overlay, mask, YOLO labels.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    batch_id = data.get("batch_id")
    updates = data.get("updates", [])

    if not batch_id or not updates:
        return jsonify({"error": "Missing batch_id or updates"}), 400

    for update in updates:
        image_id = update.get("image_id")
        status = update.get("status", "skipped")
        correction = update.get("correction", {})

        json_path = RESULT_JSON_DIR / f"{image_id}.json"
        if not json_path.exists():
            continue

        with open(json_path, "r") as f:
            info = json.load(f)

        info["status"] = status

        # Reset annotations
        info["annotations"] = []

        # Process each manual bounding box using SAM
        for c in correction.get("chips", []):
            bbox = c["bbox"]
            class_id = c["class_id"]

            try:
                result = correct_segmentation_service(
                    image_id=image_id,
                    bounding_box=bbox,
                    class_id=class_id,
                )
                info["annotations"].append(
                    {
                        "class_id": class_id,
                        "bbox": bbox,
                        "source": "manual",
                        "mask_path": result["mask_path"],
                        "overlay_path": result["overlay_path"],
                        "yolo_label_path": result["yolo_label_path"],
                    }
                )
            except Exception as e:
                print(f"[ERROR] SAM correction failed for {image_id}, bbox {bbox}: {e}")
                continue

        # Save updated JSON
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)

    return jsonify({"status": "corrections_saved"})


@app.route("/retrain", methods=["POST"])
def retrain():
    thread = threading.Thread(target=lambda: start_retraining(resume=True), daemon=True)
    thread.start()
    return jsonify({"status": "retraining_started"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
