import cv2
import numpy as np
from utils.yolo_inference import run_inference
from utils.mask_utils import compute_mask_area, mask_to_polygon

from config import CHIP_CLASS, VOID_CLASS


def run_inference_service(
    image_path: str, conf: float = 0.25, return_raw=False
) -> dict:
    results = run_inference(image_path, conf=conf)
    result = results[0]

    if result.masks is None or result.boxes is None:
        return {
            "image": image_path,
            "num_chips": 0,
            "num_voids": 0,
            "chip_area": 0,
            "void_area": 0,
            "void_rate": 0.0,
            "chips": [],
            "voids": [],
        }

    masks = (result.masks.data > 0.5).cpu().numpy().astype(np.uint8)
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    chip_entries = []
    void_entries = []
    void_masks = []

    for mask, box, cls in zip(masks, boxes, classes):
        if cls == CHIP_CLASS:
            chip_entries.append({"mask": mask.astype(bool), "bbox": box.tolist()})
        elif cls == VOID_CLASS:
            void_masks.append(mask.astype(bool))
            void_entries.append({"bbox": box.tolist()})

    chips_metrics = []
    total_chip_area = 0
    total_void_area = 0

    for idx, chip in enumerate(chip_entries, start=1):
        chip_mask = chip["mask"]
        bbox = chip["bbox"]

        chip_area = compute_mask_area(chip_mask)
        total_chip_area += chip_area

        chip_void_area = 0
        max_void_area = 0

        for void_mask in void_masks:
            overlap = np.logical_and(chip_mask, void_mask)
            overlap_area = compute_mask_area(overlap)
            chip_void_area += overlap_area
            max_void_area = max(max_void_area, overlap_area)

        total_void_area += chip_void_area

        chips_metrics.append(
            {
                "chip_id": idx,
                "chip_area": int(chip_area),
                "void_rate_pct": (
                    round((chip_void_area / chip_area) * 100, 2)
                    if chip_area > 0
                    else 0.0
                ),
                "max_void_pct": (
                    round((max_void_area / chip_area) * 100, 2)
                    if chip_area > 0
                    else 0.0
                ),
                "bbox": bbox,
            }
        )

    global_void_rate = (
        round((total_void_area / total_chip_area) * 100, 2)
        if total_chip_area > 0
        else 0.0
    )

    metrics_dict = {
        "image": image_path,
        "num_chips": len(chip_entries),
        "num_voids": len(void_masks),
        "chip_area": int(total_chip_area),
        "void_area": int(total_void_area),
        "void_rate": global_void_rate,
        "chips": chips_metrics,
        "voids": void_entries,
    }

    if return_raw:
        return metrics_dict, result  # return both metrics and YOLO result

    return metrics_dict
