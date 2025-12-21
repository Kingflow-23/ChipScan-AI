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
    image_id: str, bounding_boxes: list, class_ids: list, predictor
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
            - contours: list of polygon contours
    """

    # --- Locate image ---
    image_path = find_image_path(image_id)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    overlay_path = (
        RESULT_DIR / f"{image_path.stem}_overlay_corrected{image_path.suffix}"
    )
    mask_path = RESULT_DIR / f"{image_path.stem}_corrected_mask.npy"
    yolo_label_path = LABELS_DIR / f"{image_path.stem}.txt"

    # --- Load image ---
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize empty lists
    component_masks = []
    void_masks = []

    object_contours = []

    # Loop through all boxes and generate SAM masks
    for bbox, class_id in zip(bounding_boxes, class_ids):
        masks = segment_with_sam(
            image=image_rgb, bounding_boxes=[bbox], predictor=predictor
        )

        if not masks:
            continue
        mask = masks[0].astype(np.uint8)

        if class_id == 0:
            component_masks.append(mask)
        else:
            void_masks.append(mask)

        # --- Extract contours for YOLO labels ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        object_contours.append({"class_id": class_id, "contours": contours})

    # Merge all masks into one overlay
    overlay = visualize_component_voids(image, component_masks, void_masks)

    # Save overlay and individual masks
    cv2.imwrite(str(overlay_path), overlay)

    combined_mask = None

    # Combine all masks into one
    if component_masks or void_masks:
        combined_mask = np.zeros_like(
            component_masks[0] if component_masks else void_masks[0]
        )
        for m in component_masks + void_masks:
            combined_mask = np.maximum(combined_mask, m)
        np.save(str(mask_path), combined_mask)

    return {
        "mask_path": str(mask_path) if combined_mask is not None else None,
        "overlay_path": str(overlay_path),
        "yolo_label_path": str(yolo_label_path),
        "objects": object_contours,
        "mask_shape": (
            combined_mask.shape if combined_mask is not None else image.shape[:2]
        ),
    }
