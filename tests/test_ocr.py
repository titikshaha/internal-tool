"""Tests for OCR extraction."""

import pytest
from app.services.ocr_extractor import (
    extract_dimensions_from_text,
    extract_room_labels_from_text
)


class TestDimensionExtraction:
    """Test dimension parsing from OCR text."""

    def test_metric_mm(self):
        text = "Wall length 4500mm"
        dims = extract_dimensions_from_text(text)
        assert len(dims) >= 1
        assert any(d['numeric_value'] == 4500 for d in dims)

    def test_bare_number(self):
        # Test bare number detection (common in floor plans)
        text = "Length 3500"
        dims = extract_dimensions_from_text(text)
        assert len(dims) >= 1
        assert any(d['numeric_value'] == 3500 for d in dims)

    def test_metric_with_space(self):
        text = "Length: 2500 mm"
        dims = extract_dimensions_from_text(text)
        assert len(dims) >= 1
        assert any(d['numeric_value'] == 2500 for d in dims)

    def test_imperial_feet_inches(self):
        text = "Door width 2'-6\""
        dims = extract_dimensions_from_text(text)
        assert len(dims) >= 1
        # 2'6" = 30 inches = 762mm
        assert any(abs(d['numeric_value'] - 762) < 10 for d in dims)

    def test_multiple_dimensions(self):
        text = "Kitchen 4500mm x 3200mm"
        dims = extract_dimensions_from_text(text)
        assert len(dims) >= 2

    def test_filters_unrealistic_values(self):
        text = "Page 1 of 50"  # 50 should be filtered out
        dims = extract_dimensions_from_text(text)
        assert not any(d['numeric_value'] == 50 for d in dims)


class TestRoomExtraction:
    """Test room label extraction from OCR text."""

    def test_basic_rooms(self):
        text = "Kitchen Living Room Bedroom"
        rooms = extract_room_labels_from_text(text)
        assert len(rooms) >= 3

    def test_compound_rooms(self):
        text = "Master Bedroom En-Suite"
        rooms = extract_room_labels_from_text(text)
        assert any('master' in r['name'].lower() for r in rooms)
        assert any('en-suite' in r['name'].lower() or 'ensuite' in r['name'].lower() for r in rooms)

    def test_case_insensitive(self):
        text = "KITCHEN kitchen Kitchen"
        rooms = extract_room_labels_from_text(text)
        # Should deduplicate
        kitchen_count = sum(1 for r in rooms if 'kitchen' in r['name'].lower())
        assert kitchen_count == 1

    def test_no_false_positives(self):
        text = "Drawing Number: 12345 Scale: 1:100"
        rooms = extract_room_labels_from_text(text)
        assert len(rooms) == 0
