"""
Postprocessing utilities for handling detection masks and annotations.
This module provides functions for text extraction, mask manipulation (filtering, refinement),
and non-maximum suppression, along with tools to convert masks to Supervision detections objects
for further processing and visualization.
"""

# https://github.com/roboflow/multimodal-maestro/tree/develop/maestro/postprocessing
import re
import cv2
import numpy as np
import supervision as sv
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple, Union


class MarkMode(Enum):
    """
    An enumeration for different marking modes.
    """
    NUMERIC = "NUMERIC"
    ALPHABETIC = "ALPHABETIC"

    @classmethod
    def list(cls) -> List[str]:
        """
        Returns a list of all enumeration values.
        
        Returns:
            List[str]: List of enumeration values as strings.
        """
        return list(map(lambda c: c.value, cls))


def extract_marks_in_brackets(text: str, mode: MarkMode) -> List[str]:
    """
    Extracts all unique marks enclosed in square brackets from a given string, based
        on the specified mode. Duplicates are removed and the results are sorted in
        descending order.

    Args:
        text (str): The string to be searched.
        mode (MarkMode): The mode to determine the type of marks to extract (NUMERIC or
            ALPHABETIC).

    Returns:
        List[str]: A list of unique marks found within square brackets, sorted in
            descending order.
            
    Raises:
        ValueError: If the provided mode is not recognized.
    """
    if mode == MarkMode.NUMERIC:
        pattern = r'\[(\d+)\]'
    elif mode == MarkMode.ALPHABETIC:
        pattern = r'\[([A-Za-z]+)\]'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    found_marks = re.findall(pattern, text)
    unique_marks: Set[str] = set(found_marks)

    if mode == MarkMode.NUMERIC:
        return sorted(unique_marks, key=int, reverse=False)
    else:
        return sorted(unique_marks, reverse=False)


def extract_relevant_masks(
    text: str,
    detections: sv.Detections
) -> Dict[str, np.ndarray]:
    """
    Extracts relevant masks from the detections based on marks found in the given text.

    Args:
        text (str): The string containing marks in square brackets to be searched for.
        detections (sv.Detections): An object containing detection information,
            including masks indexed by numeric identifiers.

    Returns:
        Dict[str, np.ndarray]: A dictionary where each key is a mark found in the text,
            and each value is the corresponding mask from detections.
    """
    marks = extract_marks_in_brackets(text=text, mode=MarkMode.NUMERIC)
    return {
        mark: detections.mask[int(mark)]
        for mark
        in marks
    }


class FeatureType(Enum):
    """
    An enumeration to represent the types of features for mask adjustment in image
    segmentation.
    """
    ISLAND = 'ISLAND'
    HOLE = 'HOLE'

    @classmethod
    def list(cls) -> List[str]:
        """
        Returns a list of all enumeration values.
        
        Returns:
            List[str]: List of enumeration values as strings.
        """
        return list(map(lambda c: c.value, cls))


def compute_mask_iou_vectorized(masks: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of the Intersection over Union (IoU) for all pairs of masks.

    Args:
        masks (np.ndarray): A 3D numpy array with shape `(N, H, W)`, where `N` is the
            number of masks, `H` is the height, and `W` is the width.

    Returns:
        np.ndarray: A 2D numpy array of shape `(N, N)` where each element `[i, j]` is
            the IoU between masks `i` and `j`.

    Raises:
        ValueError: If any of the masks is found to be empty.
    """
    if np.any(masks.sum(axis=(1, 2)) == 0):
        raise ValueError(
            "One or more masks are empty. Please filter out empty masks before using "
            "`compute_iou_vectorized` function."
        )

    masks_bool = masks.astype(bool)
    masks_flat = masks_bool.reshape(masks.shape[0], -1)
    intersection = np.logical_and(masks_flat[:, None], masks_flat[None, :]).sum(axis=2)
    union = np.logical_or(masks_flat[:, None], masks_flat[None, :]).sum(axis=2)
    iou_matrix = intersection / union
    return iou_matrix


def mask_non_max_suppression(
    masks: np.ndarray,
    iou_threshold: float = 0.6
) -> np.ndarray:
    """
    Performs Non-Max Suppression on a set of masks by prioritizing larger masks and
        removing smaller masks that overlap significantly.

    When the IoU between two masks exceeds the specified threshold, the smaller mask
    (in terms of area) is discarded. This process is repeated for each pair of masks,
    effectively filtering out masks that are significantly overlapped by larger ones.

    Args:
        masks (np.ndarray): A 3D numpy array with shape `(N, H, W)`, where `N` is the
            number of masks, `H` is the height, and `W` is the width.
        iou_threshold (float): The IoU threshold for determining significant overlap.

    Returns:
        np.ndarray: A 3D numpy array of filtered masks.
    """
    num_masks = masks.shape[0]
    areas = masks.sum(axis=(1, 2))
    sorted_idx = np.argsort(-areas)
    keep_mask = np.ones(num_masks, dtype=bool)
    iou_matrix = compute_mask_iou_vectorized(masks)
    for i in range(num_masks):
        if not keep_mask[sorted_idx[i]]:
            continue

        overlapping_masks = iou_matrix[sorted_idx[i]] > iou_threshold
        overlapping_masks[sorted_idx[i]] = False
        overlapping_indices = np.where(overlapping_masks)[0]
        keep_mask[sorted_idx[overlapping_indices]] = False

    return masks[keep_mask]


def filter_masks_by_relative_area(
    masks: np.ndarray,
    minimum_area: float = 0.01,
    maximum_area: float = 1.0
) -> np.ndarray:
    """
    Filters masks based on their relative area within the total area of each mask.

    Args:
        masks (np.ndarray): A 3D numpy array with shape `(N, H, W)`, where `N` is the
            number of masks, `H` is the height, and `W` is the width.
        minimum_area (float): The minimum relative area threshold. Must be between `0`
            and `1`.
        maximum_area (float): The maximum relative area threshold. Must be between `0`
            and `1`.

    Returns:
        np.ndarray: A 3D numpy array containing masks that fall within the specified
            relative area range.

    Raises:
        ValueError: If `minimum_area` or `maximum_area` are outside the `0` to `1`
            range, or if `minimum_area` is greater than `maximum_area`.
    """

    if not (isinstance(masks, np.ndarray) and masks.ndim == 3):
        raise ValueError("Input must be a 3D numpy array.")

    if not (0 <= minimum_area <= 1) or not (0 <= maximum_area <= 1):
        raise ValueError("`minimum_area` and `maximum_area` must be between 0 and 1.")

    if minimum_area > maximum_area:
        raise ValueError("`minimum_area` must be less than or equal to `maximum_area`.")

    total_area = masks.shape[1] * masks.shape[2]
    relative_areas = masks.sum(axis=(1, 2)) / total_area
    return masks[(relative_areas >= minimum_area) & (relative_areas <= maximum_area)]


def adjust_mask_features_by_relative_area(
    mask: np.ndarray,
    area_threshold: float,
    feature_type: FeatureType = FeatureType.ISLAND
) -> np.ndarray:
    """
    Adjusts a mask by removing small islands or filling small holes based on a relative
    area threshold.

    Args:
        mask (np.ndarray): A 2D numpy array with shape `(H, W)`, where `H` is the
            height, and `W` is the width.
        area_threshold (float): Threshold for relative area to remove or fill features.
        feature_type (FeatureType): Type of feature to adjust (`ISLAND` for removing
            islands, `HOLE` for filling holes).

    Returns:
        np.ndarray: A 2D numpy array containing the adjusted mask.
        
    Note:
        Running this function on a mask with small islands may result in empty masks.
    """
    height, width = mask.shape
    total_area = width * height

    mask = np.uint8(mask * 255)
    operation = (
        cv2.RETR_EXTERNAL
        if feature_type == FeatureType.ISLAND
        else cv2.RETR_CCOMP
    )
    contours, _ = cv2.findContours(mask, operation, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        relative_area = area / total_area
        if relative_area < area_threshold:
            cv2.drawContours(
                image=mask,
                contours=[contour],
                contourIdx=-1,
                color=(0 if feature_type == FeatureType.ISLAND else 255),
                thickness=-1
            )
    return np.where(mask > 0, 1, 0).astype(bool)


def masks_to_marks(masks: np.ndarray, labels: Optional[List[int]] = None) -> sv.Detections:
    """
    Converts a set of masks to a marks (sv.Detections) object.

    Args:
        masks (np.ndarray): A 3D numpy array with shape `(N, H, W)`, where `N` is the
            number of masks, `H` is the height, and `W` is the width.
        labels (Optional[List[int]]): A list of label IDs for the markers. Default 1-indexing.

    Returns:
        sv.Detections: An object containing the masks and their bounding box
            coordinates.
    """
    labels = list(range(1, len(masks)+1)) if labels is None else labels
    if len(masks) == 0:
        marks = sv.Detections.empty()
        marks.mask = np.empty((0, 0, 0), dtype=bool)
        return marks
    return sv.Detections(
        mask=masks,
        xyxy=sv.mask_to_xyxy(masks=masks),
        class_id=np.asarray(labels),
    )


def refine_marks(
    marks: sv.Detections,
    maximum_hole_area: float = 0.01,
    maximum_island_area: float = 0.01,
    minimum_mask_area: float = 0.02,
    maximum_mask_area: float = 1.0
) -> sv.Detections:
    """
    Refines a set of masks by removing small islands and holes, and filtering by mask
    area.

    Args:
        marks (sv.Detections): An object containing the masks and their bounding box
            coordinates.
        maximum_hole_area (float): The maximum relative area of holes to be filled in
            each mask.
        maximum_island_area (float): The maximum relative area of islands to be removed
            from each mask.
        minimum_mask_area (float): The minimum relative area for a mask to be retained.
        maximum_mask_area (float): The maximum relative area for a mask to be retained.

    Returns:
        sv.Detections: An object containing the refined masks and their bounding box
            coordinates.
    """
    result_masks = []
    for mask in marks.mask:
        mask = adjust_mask_features_by_relative_area(
            mask=mask,
            area_threshold=maximum_island_area,
            feature_type=FeatureType.ISLAND)
        mask = adjust_mask_features_by_relative_area(
            mask=mask,
            area_threshold=maximum_hole_area,
            feature_type=FeatureType.HOLE)
        if np.any(mask):
            result_masks.append(mask)
    result_masks = np.array(result_masks)
    result_masks = filter_masks_by_relative_area(
        masks=result_masks,
        minimum_area=minimum_mask_area,
        maximum_area=maximum_mask_area)
    return sv.Detections(
        mask=result_masks,
        xyxy=sv.mask_to_xyxy(masks=result_masks)
    )
