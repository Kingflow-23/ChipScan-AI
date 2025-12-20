import numpy as np
import cv2
from typing import List, Dict, Tuple


def mask_to_polygon(mask: np.ndarray, use_hierarchy: bool = False) -> List[np.ndarray]:
    """
    Converts a binary mask to polygon contours.

    Args:
        mask (np.ndarray): Binary mask (0s and 1s).
        use_hierarchy (bool): If True, retrieves all nested contours (useful for voids inside components)

    Returns:
        List[np.ndarray]: List of polygon contours
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    mode = cv2.RETR_TREE if use_hierarchy else cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(mask_uint8, mode, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def compute_mask_area(mask: np.ndarray) -> int:
    """Computes the area (number of pixels) of a binary mask."""
    return int(np.sum(mask > 0))


def compute_polygon_area(contour: np.ndarray) -> float:
    """Computes the area of a single polygon contour."""
    return cv2.contourArea(contour)


def draw_polygon_on_image(
    image: np.ndarray,
    contours: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draws polygon contours on an image."""
    return cv2.drawContours(image.copy(), contours, -1, color, thickness)


def compute_void_rate(
    component_mask: np.ndarray, void_masks: List[np.ndarray]
) -> float:
    """
    Computes the void rate for a single component mask given all void masks.

    Args:
        component_mask (np.ndarray): Binary mask of the component
        void_masks (List[np.ndarray]): List of binary masks for voids

    Returns:
        float: void rate = total_void_area / component_area
    """
    component_area = compute_mask_area(component_mask)
    total_void_area = 0

    for void_mask in void_masks:
        # Only count pixels that are inside the component
        overlap = np.logical_and(component_mask, void_mask)
        total_void_area += compute_mask_area(overlap)

    return total_void_area / component_area if component_area > 0 else 0.0


def compute_void_rates(
    component_masks: List[np.ndarray], void_masks: List[np.ndarray]
) -> List[float]:
    """
    Computes void rates for multiple components.

    Returns:
        List[float]: List of void rates corresponding to component_masks
    """
    return [compute_void_rate(comp_mask, void_masks) for comp_mask in component_masks]


def visualize_component_voids(
    image: np.ndarray, component_masks: List[np.ndarray], void_masks: List[np.ndarray]
) -> np.ndarray:
    """
    Draw components in red and voids in yellow on an image.

    Returns:
        np.ndarray: annotated image
    """
    annotated = image.copy()

    # Chips = red
    for comp_mask in component_masks:
        contours = mask_to_polygon(comp_mask)
        annotated = draw_polygon_on_image(
            annotated, contours, color=(0, 0, 255), thickness=2
        )

    # Voids = yellow
    for void_mask in void_masks:
        contours = mask_to_polygon(void_mask)
        annotated = draw_polygon_on_image(
            annotated, contours, color=(0, 255, 255), thickness=2
        )
    return annotated
