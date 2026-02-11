"""Tests for unlabeled image training/parsing pipeline."""

import os
import tempfile

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
from fastapi.testclient import TestClient

from app.main import app
from app.services.unsupervised_image_parser import parse_image_to_json, train_codebook

client = TestClient(app)


def _make_synthetic_plan(path: str, with_text: bool = True) -> None:
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # Outer wall rectangle
    cv2.rectangle(image, (40, 40), (360, 320), (0, 0, 0), 3)
    # Internal wall
    cv2.line(image, (200, 40), (200, 320), (0, 0, 0), 2)

    if with_text:
        cv2.putText(
            image,
            "ROOM",
            (75, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(path, image)


def test_train_and_parse_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        img1 = os.path.join(tmpdir, "plan_1.png")
        img2 = os.path.join(tmpdir, "plan_2.png")
        model_path = os.path.join(tmpdir, "model.json")

        _make_synthetic_plan(img1, with_text=True)
        _make_synthetic_plan(img2, with_text=False)

        train_result = train_codebook(tmpdir, output_path=model_path, clusters=8)
        assert train_result["trained"] is True
        assert os.path.exists(model_path)
        assert train_result["stats"]["image_count"] >= 1
        assert train_result["stats"]["clusters"] >= 2

        parsed = parse_image_to_json(img1, model_path=model_path)
        assert parsed["file_type"] == "image"
        assert "elements" in parsed
        assert "shapes" in parsed["elements"]
        assert len(parsed["features"]["descriptor_histogram"]) == parsed["model"]["cluster_count"]
        assert parsed["features"]["shape_count"] >= 1


def test_api_train_unsupervised_route():
    with tempfile.TemporaryDirectory() as tmpdir:
        img = os.path.join(tmpdir, "plan.png")
        model_path = os.path.join(tmpdir, "trained.json")
        _make_synthetic_plan(img, with_text=True)

        response = client.post(
            "/model/train-unsupervised",
            json={"image_dir": tmpdir, "output_path": model_path, "clusters": 6},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["trained"] is True
        assert os.path.exists(model_path)


def test_api_parse_unsupervised_route_requires_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        img = os.path.join(tmpdir, "plan.png")
        _make_synthetic_plan(img, with_text=True)

        with open(img, "rb") as fh:
            response = client.post(
                "/parse/image-unsupervised",
                params={"model_path": os.path.join(tmpdir, "missing.json")},
                files={"file": ("plan.png", fh.read(), "image/png")},
            )

        assert response.status_code == 400
        assert "Model file not found" in response.json()["detail"]
