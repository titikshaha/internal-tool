"""Tests for phase-1 supervised dataset/training/inference scaffold."""

import json
import os
import tempfile

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app
from app.services.supervised_pipeline import (
    parse_image_supervised,
    train_supervised_model,
    validate_supervised_dataset,
)

client = TestClient(app)


def _create_labeled_dataset(root: str) -> str:
    images_dir = os.path.join(root, "images")
    annotations_dir = os.path.join(root, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    image_path = os.path.join(images_dir, "sample_1.png")
    img = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((30, 40, 140, 190), outline="black", width=3)
    draw.rectangle((180, 60, 280, 200), outline="black", width=3)
    img.save(image_path)

    annotation = {
        "file_name": "sample_1.png",
        "width": 320,
        "height": 240,
        "objects": [
            {"label": "room", "bbox": [30, 40, 110, 150]},
            {"label": "room", "bbox": [180, 60, 100, 140]},
        ],
    }
    ann_path = os.path.join(annotations_dir, "sample_1.json")
    with open(ann_path, "w", encoding="utf-8") as fh:
        json.dump(annotation, fh)

    return image_path


def test_train_and_infer_supervised_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = _create_labeled_dataset(tmpdir)
        model_path = os.path.join(tmpdir, "supervised_model.json")

        train_result = train_supervised_model(
            dataset_dir=tmpdir,
            output_path=model_path,
            min_samples=1,
        )
        assert train_result["trained"] is True
        assert os.path.exists(model_path)
        assert train_result["stats"]["class_count"] >= 1

        parsed = parse_image_supervised(image_path, model_path=model_path)
        assert parsed["file_type"] == "image"
        assert "detections" in parsed
        assert isinstance(parsed["detections"], list)


def test_api_train_supervised_route():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_labeled_dataset(tmpdir)
        model_path = os.path.join(tmpdir, "api_supervised_model.json")

        response = client.post(
            "/model/train-supervised",
            json={
                "dataset_dir": tmpdir,
                "output_path": model_path,
                "min_samples": 1,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["trained"] is True
        assert os.path.exists(model_path)


def test_api_supervised_parse_missing_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = _create_labeled_dataset(tmpdir)
        missing_model = os.path.join(tmpdir, "does_not_exist.json")

        with open(image_path, "rb") as fh:
            response = client.post(
                "/parse/image-supervised",
                params={"model_path": missing_model},
                files={"file": ("sample_1.png", fh.read(), "image/png")},
            )
        assert response.status_code == 400
        assert "Model file not found" in response.json()["detail"]


def test_validate_supervised_dataset_reports_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_labeled_dataset(tmpdir)
        report = validate_supervised_dataset(tmpdir)
        assert report["valid"] is True
        assert report["valid_samples"] == 1
        assert report["invalid_samples"] == 0
        assert report["total_objects"] == 2
        assert report["class_distribution"].get("room") == 2


def test_validate_supervised_dataset_reports_issues():
    with tempfile.TemporaryDirectory() as tmpdir:
        images_dir = os.path.join(tmpdir, "images")
        annotations_dir = os.path.join(tmpdir, "annotations")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        bad_annotation = {
            "file_name": "missing.png",
            "width": 320,
            "height": 240,
            "objects": [{"label": "room", "bbox": [0, 0, 50, 50]}],
        }
        with open(os.path.join(annotations_dir, "bad.json"), "w", encoding="utf-8") as fh:
            json.dump(bad_annotation, fh)

        report = validate_supervised_dataset(tmpdir)
        assert report["valid"] is False
        assert report["invalid_samples"] == 1
        assert len(report["issues"]) == 1


def test_api_validate_supervised_dataset_route():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_labeled_dataset(tmpdir)
        response = client.post(
            "/model/validate-supervised-dataset",
            json={"dataset_dir": tmpdir},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["valid"] is True
        assert body["valid_samples"] == 1
