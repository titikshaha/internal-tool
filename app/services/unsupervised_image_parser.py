"""
Unsupervised image parser for floor-plan-like drawings.

This module avoids labeled annotations by using:
- ORB local descriptors
- OpenCV k-means visual codebook ("training")
- Contour-based geometric segmentation for inference

The parser emits structured JSON suitable for downstream processing.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pytesseract

try:
    import cv2
    import numpy as np
    CV_STACK_ERROR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    CV_STACK_ERROR = str(exc)


DEFAULT_MODEL_PATH = os.path.join("app", "models", "unsupervised_codebook.json")


def _ensure_cv_stack() -> None:
    """Fail with clear message if OpenCV/NumPy binary stack is unavailable."""
    if cv2 is None or np is None:
        raise ValueError(
            "OpenCV/NumPy is unavailable for unsupervised image parsing. "
            f"Import error: {CV_STACK_ERROR}"
        )


@dataclass
class TrainStats:
    image_count: int
    descriptor_count: int
    clusters: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "image_count": self.image_count,
            "descriptor_count": self.descriptor_count,
            "clusters": self.clusters,
        }


def _load_gray(path: str) -> np.ndarray:
    _ensure_cv_stack()
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def _collect_descriptors(image_paths: List[str]) -> Tuple[np.ndarray, int]:
    _ensure_cv_stack()
    orb = cv2.ORB_create(nfeatures=2000)
    all_desc: List[np.ndarray] = []
    used_images = 0

    for path in image_paths:
        gray = _load_gray(path)
        _, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) == 0:
            continue
        all_desc.append(descriptors.astype(np.float32))
        used_images += 1

    if not all_desc:
        raise ValueError("No descriptors found in training images.")

    return np.vstack(all_desc), used_images


def train_codebook(
    image_dir: str,
    output_path: str = DEFAULT_MODEL_PATH,
    clusters: int = 32,
) -> Dict[str, Any]:
    """
    Train an unsupervised visual codebook from unlabeled images and persist it as JSON.
    """
    if clusters < 2:
        raise ValueError("clusters must be >= 2")
    if not os.path.isdir(image_dir):
        raise ValueError(f"Training directory not found: {image_dir}")

    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if not image_paths:
        raise ValueError(f"No images found in: {image_dir}")

    descriptors, used_images = _collect_descriptors(image_paths)
    cluster_count = min(clusters, len(descriptors))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(
        descriptors,
        cluster_count,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    histogram = np.bincount(labels.flatten(), minlength=cluster_count).astype(int).tolist()
    payload = {
        "model_type": "orb_bovw",
        "cluster_count": int(cluster_count),
        "centers": centers.tolist(),
        "histogram": histogram,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    stats = TrainStats(
        image_count=used_images,
        descriptor_count=int(len(descriptors)),
        clusters=int(cluster_count),
    )
    return {
        "trained": True,
        "model_path": output_path,
        "stats": stats.to_dict(),
    }


def _extract_text_blocks(gray: np.ndarray) -> List[Dict[str, Any]]:
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    blocks: List[Dict[str, Any]] = []
    for i, text in enumerate(data.get("text", [])):
        token = (text or "").strip()
        if not token:
            continue
        conf = float(data["conf"][i]) if str(data["conf"][i]).strip() not in {"", "-1"} else -1.0
        if conf >= 0 and conf < 30:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        blocks.append(
            {
                "text": token,
                "confidence": conf,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
            }
        )
    return blocks


def _extract_shapes(gray: np.ndarray) -> List[Dict[str, Any]]:
    _ensure_cv_stack()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes: List[Dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 250:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)
        kind = "polygon"
        if len(approx) == 4:
            kind = "quad"
        elif len(approx) > 6:
            kind = "complex"

        shapes.append(
            {
                "type": kind,
                "area_px": float(area),
                "perimeter_px": float(perimeter),
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "vertices": int(len(approx)),
            }
        )

    shapes.sort(key=lambda s: s["area_px"], reverse=True)
    return shapes


def _descriptor_histogram(gray: np.ndarray, centers: np.ndarray) -> List[int]:
    _ensure_cv_stack()
    orb = cv2.ORB_create(nfeatures=2000)
    _, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or len(descriptors) == 0:
        return [0] * len(centers)

    desc = descriptors.astype(np.float32)
    distances = np.linalg.norm(desc[:, None, :] - centers[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    return np.bincount(nearest, minlength=len(centers)).astype(int).tolist()


def parse_image_to_json(
    image_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Parse an image and return unlabeled structure/text JSON.
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}. Train first.")

    with open(model_path, "r", encoding="utf-8") as fh:
        model = json.load(fh)

    centers = np.array(model.get("centers", []), dtype=np.float32)
    if centers.ndim != 2 or centers.shape[0] == 0:
        raise ValueError("Invalid model: centers missing or malformed.")

    gray = _load_gray(image_path)
    h, w = gray.shape[:2]
    shapes = _extract_shapes(gray)
    text_blocks = _extract_text_blocks(gray)
    descriptor_hist = _descriptor_histogram(gray, centers)

    return {
        "file_type": "image",
        "image_path": image_path,
        "canvas": {"width": int(w), "height": int(h)},
        "model": {
            "type": model.get("model_type", "unknown"),
            "cluster_count": int(model.get("cluster_count", len(centers))),
        },
        "features": {
            "descriptor_histogram": descriptor_hist,
            "shape_count": len(shapes),
            "text_count": len(text_blocks),
        },
        "elements": {
            "shapes": shapes,
            "text_blocks": text_blocks,
        },
    }
    _ensure_cv_stack()
    _ensure_cv_stack()
