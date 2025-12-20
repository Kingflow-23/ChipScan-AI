import cv2
import glob
import numpy as np
from pathlib import Path

from utils.sam_segmentation import segment_with_sam
from utils.mask_utils import visualize_component_voids

from config import UPLOAD_DIR, RESULT_DIR, LABELS_DIR

# Internal singleton for SAM predictor
_sam_predictor = None


def get_sam_predictor():
    global _sam_predictor
    return _sam_predictor


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
    image_id: str, bounding_box: list, class_id: int, predictor
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

    # --- Locate image ---
    image_path = find_image_path(image_id)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    overlay_path = RESULT_DIR / f"{image_path.stem}_overlay{image_path.suffix}"
    mask_path = RESULT_DIR / f"{image_path.stem}_corrected_mask.npy"
    yolo_label_path = LABELS_DIR / f"{image_path.stem}.txt"

    # --- Load image ---
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- SAM segmentation ---
    masks = segment_with_sam(
        image=image_rgb,
        bounding_boxes=[bounding_box],
        predictor=predictor,
    )
    mask = masks[0].astype(np.uint8)

    # --- Overlay using standardized function ---
    if class_id == 0:
        component_masks = [mask]
        void_masks = []
    else:
        component_masks = []
        void_masks = [mask]

    overlay = visualize_component_voids(image, component_masks, void_masks)

    # Save overlay and mask
    cv2.imwrite(str(overlay_path), overlay)
    np.save(str(mask_path), mask)

    # --- Convert mask to YOLO segmentation label ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    label_lines = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        width = bw / w
        height = bh / h
        label_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    # Save YOLO label file in LABELS_DIR
    with open(yolo_label_path, "a") as f:
        f.write("\n".join(label_lines))

    return {
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "yolo_label_path": str(yolo_label_path),
        "class_id": class_id,
    }
