"""
Phase 1 supervised pipeline scaffold.

This module provides:
- dataset loading + validation for image/annotation pairs
- lightweight supervised "training" artifact generation
- lightweight inference for unlabeled images using learned class templates

It is intentionally simple so the project can evolve to a deep model
without changing API contracts.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


DEFAULT_SUPERVISED_MODEL_PATH = os.path.join("app", "models", "supervised_baseline.json")
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class LabeledObject:
    label: str
    bbox: Tuple[float, float, float, float]  # x, y, w, h in pixels


@dataclass
class LabeledSample:
    file_name: str
    width: int
    height: int
    objects: List[LabeledObject]


def _validate_bbox(bbox: Any, width: int, height: int) -> Tuple[float, float, float, float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bbox must be a 4-element list: [x, y, w, h]")

    x, y, w, h = bbox
    try:
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
    except (TypeError, ValueError):
        raise ValueError("bbox values must be numeric")

    if w <= 0 or h <= 0:
        raise ValueError("bbox width/height must be positive")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("bbox must lie inside image bounds")

    return (x, y, w, h)


def _load_annotation(annotation_path: str) -> LabeledSample:
    with open(annotation_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    required = {"file_name", "width", "height", "objects"}
    missing = required.difference(data.keys())
    if missing:
        raise ValueError(f"Missing required keys {sorted(missing)} in {annotation_path}")

    file_name = str(data["file_name"])
    width = int(data["width"])
    height = int(data["height"])
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be > 0")

    objects_raw = data["objects"]
    if not isinstance(objects_raw, list):
        raise ValueError("objects must be a list")

    objects: List[LabeledObject] = []
    for idx, obj in enumerate(objects_raw):
        if not isinstance(obj, dict):
            raise ValueError(f"objects[{idx}] must be a dict")
        label = str(obj.get("label", "")).strip()
        if not label:
            raise ValueError(f"objects[{idx}].label is required")
        bbox = _validate_bbox(obj.get("bbox"), width, height)
        objects.append(LabeledObject(label=label, bbox=bbox))

    return LabeledSample(file_name=file_name, width=width, height=height, objects=objects)


def _resolve_image_path(dataset_dir: str, file_name: str) -> str:
    candidates = [
        os.path.join(dataset_dir, "images", file_name),
        os.path.join(dataset_dir, file_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTS:
                return path
    raise ValueError(f"Image file not found for annotation file_name='{file_name}'")


def load_supervised_dataset(dataset_dir: str) -> List[LabeledSample]:
    """
    Load labeled dataset from:
    - {dataset_dir}/annotations/*.json
    - images resolved from {dataset_dir}/images or {dataset_dir}
    """
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    ann_dir = os.path.join(dataset_dir, "annotations")
    if not os.path.isdir(ann_dir):
        raise ValueError(f"Missing annotations directory: {ann_dir}")

    annotation_paths = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    if not annotation_paths:
        raise ValueError(f"No annotation files found in: {ann_dir}")

    dataset: List[LabeledSample] = []
    for ann_path in annotation_paths:
        sample = _load_annotation(ann_path)
        _resolve_image_path(dataset_dir, sample.file_name)
        dataset.append(sample)

    return dataset


def validate_supervised_dataset(dataset_dir: str) -> Dict[str, Any]:
    """
    Validate supervised dataset and return diagnostics without training.
    """
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    ann_dir = os.path.join(dataset_dir, "annotations")
    if not os.path.isdir(ann_dir):
        raise ValueError(f"Missing annotations directory: {ann_dir}")

    annotation_paths = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    if not annotation_paths:
        raise ValueError(f"No annotation files found in: {ann_dir}")

    issues: List[str] = []
    valid_samples = 0
    total_objects = 0
    class_distribution: Dict[str, int] = {}

    for ann_path in annotation_paths:
        try:
            sample = _load_annotation(ann_path)
            _resolve_image_path(dataset_dir, sample.file_name)
            valid_samples += 1
            total_objects += len(sample.objects)
            for obj in sample.objects:
                class_distribution[obj.label] = class_distribution.get(obj.label, 0) + 1
        except Exception as exc:
            issues.append(f"{os.path.basename(ann_path)}: {exc}")

    return {
        "valid": len(issues) == 0,
        "dataset_dir": dataset_dir,
        "annotation_files": len(annotation_paths),
        "valid_samples": valid_samples,
        "invalid_samples": len(annotation_paths) - valid_samples,
        "total_objects": total_objects,
        "class_distribution": class_distribution,
        "issues": issues,
    }


def train_supervised_model(
    dataset_dir: str,
    output_path: str = DEFAULT_SUPERVISED_MODEL_PATH,
    min_samples: int = 1,
) -> Dict[str, Any]:
    """
    Train a lightweight supervised baseline model from labeled annotations.
    """
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")

    dataset = load_supervised_dataset(dataset_dir)
    if len(dataset) < min_samples:
        raise ValueError(
            f"Not enough labeled samples: found {len(dataset)}, requires at least {min_samples}"
        )

    class_counts: Dict[str, int] = {}
    class_aspect_ratios: Dict[str, List[float]] = {}
    class_area_ratios: Dict[str, List[float]] = {}
    object_count = 0

    for sample in dataset:
        image_area = float(sample.width * sample.height)
        for obj in sample.objects:
            x, y, w, h = obj.bbox
            object_count += 1
            class_counts[obj.label] = class_counts.get(obj.label, 0) + 1
            class_aspect_ratios.setdefault(obj.label, []).append(w / h)
            class_area_ratios.setdefault(obj.label, []).append((w * h) / image_area)

    if object_count == 0:
        raise ValueError("No labeled objects found in annotations")

    classes: Dict[str, Dict[str, float]] = {}
    for label, count in class_counts.items():
        classes[label] = {
            "count": float(count),
            "aspect_ratio_mean": float(np.mean(class_aspect_ratios[label])),
            "area_ratio_mean": float(np.mean(class_area_ratios[label])),
        }

    dominant_label = max(class_counts.items(), key=lambda kv: kv[1])[0]
    model_payload = {
        "model_type": "supervised_baseline_v1",
        "dataset_dir": dataset_dir,
        "sample_count": len(dataset),
        "object_count": object_count,
        "class_count": len(classes),
        "dominant_label": dominant_label,
        "classes": classes,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(model_payload, fh)

    return {
        "trained": True,
        "model_path": output_path,
        "stats": {
            "sample_count": len(dataset),
            "object_count": object_count,
            "class_count": len(classes),
            "dominant_label": dominant_label,
        },
    }


def _connected_components(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Extract connected component bounding boxes from binary image.
    Returns list of (x, y, w, h).
    """
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=np.uint8)
    bboxes: List[Tuple[int, int, int, int]] = []

    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0 or visited[y, x] == 1:
                continue

            stack = [(x, y)]
            visited[y, x] = 1
            min_x = max_x = x
            min_y = max_y = y
            count = 0

            while stack:
                cx, cy = stack.pop()
                count += 1
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy

                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    if binary[ny, nx] == 0 or visited[ny, nx] == 1:
                        continue
                    visited[ny, nx] = 1
                    stack.append((nx, ny))

            if count < 25:
                continue
            bx = min_x
            by = min_y
            bw = max_x - min_x + 1
            bh = max_y - min_y + 1
            bboxes.append((bx, by, bw, bh))

    return bboxes


def _choose_label(model: Dict[str, Any], bbox: Tuple[int, int, int, int], image_area: float) -> str:
    classes = model.get("classes", {})
    if not classes:
        return model.get("dominant_label", "object")

    _, _, w, h = bbox
    aspect_ratio = float(w) / max(float(h), 1.0)
    area_ratio = (float(w) * float(h)) / max(image_area, 1.0)

    best_label = model.get("dominant_label", "object")
    best_score = float("inf")
    for label, stats in classes.items():
        ar = float(stats.get("aspect_ratio_mean", 1.0))
        aa = float(stats.get("area_ratio_mean", 0.01))
        score = abs(aspect_ratio - ar) + abs(area_ratio - aa)
        if score < best_score:
            best_score = score
            best_label = label

    return best_label


def parse_image_supervised(
    image_path: str,
    model_path: str = DEFAULT_SUPERVISED_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Run lightweight supervised inference on a single image.
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}. Train first.")

    with open(model_path, "r", encoding="utf-8") as fh:
        model = json.load(fh)

    image = Image.open(image_path).convert("L")
    arr = np.array(image)
    h, w = arr.shape

    # Foreground = dark structures, typical for floor plans on white background.
    threshold = np.percentile(arr, 35)
    binary = (arr <= threshold).astype(np.uint8)

    bboxes = _connected_components(binary)
    image_area = float(w * h)
    detections: List[Dict[str, Any]] = []
    for bbox in bboxes:
        x, y, bw, bh = bbox
        area_ratio = (bw * bh) / max(image_area, 1.0)
        if area_ratio < 0.002:
            continue
        label = _choose_label(model, bbox, image_area)
        detections.append(
            {
                "label": label,
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "score": 0.5,
            }
        )

    return {
        "file_type": "image",
        "image_path": image_path,
        "model": {
            "type": model.get("model_type", "unknown"),
            "path": model_path,
        },
        "canvas": {"width": int(w), "height": int(h)},
        "detections": detections,
        "summary": {
            "count": len(detections),
            "labels": sorted(list({d["label"] for d in detections})),
        },
    }
