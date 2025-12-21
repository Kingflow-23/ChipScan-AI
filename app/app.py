import cv2
import json
import uuid
import logging
import threading
from pathlib import Path

from flask import Flask, request, jsonify, render_template

from services.inference import run_inference_service
from services.correction import correct_segmentation_service
from services.retrain import start_retraining
from utils.sam_model import load_sam_model, initialize_predictor

from config import *

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_JSON_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Directories ensured: UPLOAD_DIR, RESULT_DIR, RESULT_JSON_DIR")

# Load SAM predictor once
logger.info("Loading SAM model and initializing predictor")
_sam_predictor = initialize_predictor(load_sam_model())

# ---------------------------------------------------------------------------
# Routes – Pages
# ---------------------------------------------------------------------------


@app.route("/", methods=["GET"])
def index():
    logger.info("Serving index page")
    return render_template("index.html")


@app.route("/results/batch/<batch_id>", methods=["GET"])
def results_batch(batch_id):
    batch_path = RESULT_JSON_DIR / f"batch_{batch_id}.json"
    if not batch_path.exists():
        logger.warning(f"Batch {batch_id} not found")
        return "Batch results not found", 404

    with open(batch_path, "r") as f:
        results = json.load(f)

    logger.info(f"Serving results page for batch {batch_id}, {len(results)} images")
    return render_template("results.html", results=results)


@app.route("/admin", methods=["GET"])
def admin():
    logger.info("Serving admin page")
    return render_template("admin.html")


# ---------------------------------------------------------------------------
# Routes – API
# ---------------------------------------------------------------------------


@app.route("/predict", methods=["POST"])
def predict():
    if "images" not in request.files:
        logger.warning("No images provided for prediction")
        return jsonify({"error": "No images provided"}), 400

    files = request.files.getlist("images")
    batch_id = str(uuid.uuid4())
    batch_results = []

    logger.info(f"Processing {len(files)} images for new batch {batch_id}")

    for image_file in files:
        image_id = str(uuid.uuid4())
        ext = Path(image_file.filename).suffix or ".jpg"
        image_path = UPLOAD_DIR / f"{image_id}{ext}"
        image_file.save(image_path)
        logger.info(f"Saved uploaded image: {image_path}")

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
        logger.info(f"Saved overlay image: {overlay_path}")

        json_payload = {
            "batch_id": batch_id,
            "filename": image_file.filename,
            "image_id": image_id,
            "image_url": f"/static/uploads/{image_path.name}",
            "overlay_url": f"/static/results/{overlay_path.name}",
            **metrics,
        }

        # Save per-image JSON (still useful for correction)
        json_path = RESULT_JSON_DIR / f"{image_id}.json"
        with open(json_path, "w") as f:
            json.dump(json_payload, f, indent=2)
        logger.info(f"Saved JSON result for image {image_id}: {json_path}")

        batch_results.append(json_payload)

    # Save batch index (THIS IS THE KEY)
    batch_path = RESULT_JSON_DIR / f"batch_{batch_id}.json"
    with open(batch_path, "w") as f:
        json.dump(batch_results, f, indent=2)
    logger.info(f"Saved batch JSON for batch {batch_id}: {batch_path}")

    return jsonify({"batch_id": batch_id, "count": len(batch_results)})


@app.route("/correct/batch/<batch_id>", methods=["GET", "POST"])
def correct_batch(batch_id):
    batch_path = RESULT_JSON_DIR / f"batch_{batch_id}.json"
    if not batch_path.exists():
        logger.warning(f"Batch {batch_id} not found for correction")
        return "Batch not found", 404

    if request.method == "GET":
        with open(batch_path, "r") as f:
            results = json.load(f)
        logger.info(f"Serving correction page for batch {batch_id}")
        return render_template("correct.html", results=results, batch_id=batch_id)

    # POST: process corrections
    data = request.get_json()
    updates = data.get("updates", [])
    if not updates:
        return jsonify({"error": "No corrections submitted"}), 400

    logger.info(f"Processing {len(updates)} corrections for batch {batch_id}")

    with open(batch_path, "r") as f:
        batch_results = json.load(f)

    for update in updates:
        image_id = update["image_id"]
        chips = update.get("correction", {}).get("chips", [])
        voids = update.get("correction", {}).get("voids", [])

        all_annotations = chips + voids

        if not all_annotations:
            logger.warning(f"No annotations for image {image_id}, skipping")
            continue

        # Collect all bounding boxes and class_ids
        bounding_boxes = [ann["bbox"] for ann in all_annotations]
        class_ids = [ann["class_id"] for ann in all_annotations]

        # Find the image JSON
        json_path = RESULT_JSON_DIR / f"{image_id}.json"
        if not json_path.exists():
            logger.warning(f"JSON for image {image_id} not found, skipping")
            continue

        with open(json_path, "r") as f:
            info = json.load(f)

        info["annotations"] = []

        try:
            # Call SAM correction once per image with all boxes
            result = correct_segmentation_service(
                image_id=image_id,
                bounding_boxes=bounding_boxes,
                class_ids=class_ids,
                predictor=_sam_predictor,
            )

            # Generate YOLO labels for all contours at once
            h, w = result["mask_shape"]
            yolo_lines = []

            for obj in result["objects"]:
                class_id = obj["class_id"]

                for cnt in obj["contours"]:
                    if len(cnt) < 3:
                        continue  # need at least 3 points for a polygon

                    polygon = []
                    for pt in cnt.squeeze():
                        x = pt[0] / w
                        y = pt[1] / h
                        polygon.append(f"{x:.6f}")
                        polygon.append(f"{y:.6f}")

                    yolo_lines.append(f"{class_id} " + " ".join(polygon))

            # Write YOLO label file once
            with open(result["yolo_label_path"], "w") as f:
                f.write("\n".join(yolo_lines))

            # Update annotations in JSON
            for ann, bbox in zip(all_annotations, bounding_boxes):
                info["annotations"].append(
                    {
                        "class_id": ann["class_id"],
                        "bbox": bbox,
                        "source": "final",
                        "mask_path": result["mask_path"],
                        "overlay_path": result["overlay_path"],
                        "yolo_label_path": result["yolo_label_path"],
                    }
                )

        except Exception as e:
            logger.error(f"SAM correction failed for {image_id}: {e}")

        # Save updated JSON
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)

        # Update batch_results entry
        for i, img_entry in enumerate(batch_results):
            if img_entry["image_id"] == image_id:
                batch_results[i] = info
                break

    # Save updated batch JSON
    with open(batch_path, "w") as f:
        json.dump(batch_results, f, indent=2)
    logger.info(f"Batch {batch_id} JSON updated with corrections")

    return jsonify({"status": "corrections_saved"})


retraining_status = {
    "running": False,
    "error": None,
    "progress": 0,        # 0-100 %
    "current_epoch": 0,   # current epoch number
    "total_epochs": 0
}


@app.route("/retrain", methods=["POST"])
def retrain():
    global retraining_status

    if retraining_status["running"]:
        return jsonify({"status": "already_running"})

    def run():
        try:
            start_retraining(retrain=True, retraining_status=retraining_status)
        except Exception as e:
            retraining_status["error"] = str(e)
            logger.error(f"Retraining failed: {e}")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    logger.info("Retraining started in background")
    return jsonify({"status": "started"})


@app.route("/retrain/status", methods=["GET"])
def retrain_status():
    return jsonify(retraining_status)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Flask app")
    app.run(debug=True, use_reloader=False)
