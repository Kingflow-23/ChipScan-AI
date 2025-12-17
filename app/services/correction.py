import numpy as np
import cv2
from pathlib import Path
import glob

from utils.sam_model import load_sam_model
from utils.sam_segmentation import segment_with_sam
from config import UPLOAD_DIR, RESULT_DIR, DATA_DIR

# Load SAM once
_predictor = load_sam_model()


def find_image_path(image_id: str) -> Path:
    """
    Finds an image in UPLOAD_DIR with any extension based on image_id.
    """
    patterns = [f"{image_id}.*"]
    files = []
    for p in patterns:
        files.extend(glob.glob(str(UPLOAD_DIR / p)))
    if not files:
        raise FileNotFoundError(f"No image found for ID {image_id} in {UPLOAD_DIR}")
    return Path(files[0])


def correct_segmentation_service(
    image_id: str,
    bounding_box: list,
    class_id: int
) -> dict:
    """
    Refine YOLO prediction using SAM, save overlay, mask, and YOLO label.

    Args:
        image_id: unique ID of the image (used to locate file)
        bounding_box: [x1, y1, x2, y2]
        class_id: 0=chip, 1=void

    Returns:
        dict with:
            - mask_path: path to saved mask
            - overlay_path: path to overlay image
            - yolo_label_path: path to YOLO segmentation label
            - class_id
    """

    # --- Locate image (any extension) ---
    image_path = find_image_path(image_id)
    overlay_path = RESULT_DIR / f"{image_path.stem}_overlay{image_path.suffix}"
    mask_path = RESULT_DIR / f"{image_path.stem}_corrected_mask.npy"
    yolo_label_path = DATA_DIR / "labels" / f"{image_path.stem}.txt"

    # --- Load image ---
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- SAM segmentation ---
    masks = segment_with_sam(
        image=image_rgb,
        bounding_boxes=[bounding_box],
        predictor=_predictor,
    )
    mask = masks[0].astype(np.uint8)

    # --- Overlay for UI ---
    overlay = image.copy()
    color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
    overlay[mask > 0] = cv2.addWeighted(overlay[mask > 0], 0.5, np.array(color, dtype=np.uint8), 0.5, 0)

    # Save overlay
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), overlay)

    # Save mask
    np.save(str(mask_path), mask)

    # --- Convert mask to YOLO segmentation label ---
    # YOLO segmentation format: class_id x_center y_center width height (normalized)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    label_lines = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        width = bw / w
        height = bh / h
        label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save YOLO label file
    yolo_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yolo_label_path, "w") as f:
        f.write("\n".join(label_lines))

    return {
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "yolo_label_path": str(yolo_label_path),
        "class_id": class_id
    }
