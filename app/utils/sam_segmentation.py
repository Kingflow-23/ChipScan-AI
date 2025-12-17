import cv2
import torch
import numpy as np

from typing import List, Tuple
from segment_anything.segment_anything.predictor import SamPredictor


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Loads and preprocesses an image for SAM.

    Args:
        image_path (str): Path to the input image.

    Returns:
        image: Loaded image as a NumPy array (RGB).
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def segment_with_sam(
    image: np.ndarray,
    bounding_boxes: List[Tuple[int, int, int, int]],
    predictor: SamPredictor,
) -> List[np.ndarray]:
    """
    Uses SAM to segment regions defined by bounding boxes.

    Args:
        image (np.ndarray): RGB image.
        bounding_boxes (List[Tuple[int, int, int, int]]): List of bounding boxes (x1, y1, x2, y2).
        predictor (SamPredictor): Initialized SAM predictor.

    Returns:
        List[np.ndarray]: List of binary masks for each bounding box.
    """
    predictor.set_image(image)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        torch.tensor(bounding_boxes, dtype=torch.float), image.shape[:2]
    ).to(predictor.device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return [mask.squeeze(0).cpu().numpy() for mask in masks]


def visualize_masks(image: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    """
    Overlays masks on the image for visualization.

    Args:
        image (np.ndarray): Original RGB image.
        masks (List[np.ndarray]): List of binary masks.

    Returns:
        np.ndarray: Image with masks overlayed.
    """
    overlay = image.copy()
    for mask in masks:
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 255, 0]  # Green overlay
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)
    return overlay
