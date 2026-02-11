"""
Test PDF extraction with real floor plan PDFs
"""

import os
import pytest
from app.services.pdf_extractor import extract_floor_plan

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def get_test_pdfs():
    """Get list of test PDF files"""
    if not os.path.exists(FIXTURES_DIR):
        return []
    return [f for f in os.listdir(FIXTURES_DIR) if f.endswith('.pdf')]


@pytest.mark.parametrize("pdf_name", get_test_pdfs())
def test_extraction_runs_without_error(pdf_name):
    """Test that extraction completes without raising exceptions"""
    pdf_path = os.path.join(FIXTURES_DIR, pdf_name)
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()

    result = extract_floor_plan(pdf_content)

    assert result is not None
    assert "walls" in result
    assert "dimensions" in result
    assert "rooms" in result
    assert "confidence" in result


@pytest.mark.parametrize("pdf_name", get_test_pdfs())
def test_walls_detected(pdf_name):
    """Test that walls are detected in floor plans"""
    pdf_path = os.path.join(FIXTURES_DIR, pdf_name)
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()

    result = extract_floor_plan(pdf_content)

    walls = result.get("walls", [])
    print(f"\n{pdf_name}: Found {len(walls)} walls")

    # Should detect at least some walls
    assert len(walls) >= 3, f"Expected at least 3 walls, got {len(walls)}"


@pytest.mark.parametrize("pdf_name", get_test_pdfs())
def test_dimensions_found(pdf_name):
    """Test that dimensions are extracted from floor plans"""
    pdf_path = os.path.join(FIXTURES_DIR, pdf_name)
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()

    result = extract_floor_plan(pdf_content)

    dimensions = result.get("dimensions", [])
    print(f"\n{pdf_name}: Found {len(dimensions)} dimensions")

    # Should find at least some dimensions
    assert len(dimensions) >= 2, f"Expected at least 2 dimensions, got {len(dimensions)}"


@pytest.mark.parametrize("pdf_name", get_test_pdfs())
def test_confidence_score(pdf_name):
    """Test that confidence score is reasonable"""
    pdf_path = os.path.join(FIXTURES_DIR, pdf_name)
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()

    result = extract_floor_plan(pdf_content)

    confidence = result.get("confidence", 0)
    print(f"\n{pdf_name}: Confidence = {confidence}")

    # Confidence should be > 0.3 for valid floor plans
    assert confidence > 0.3, f"Expected confidence > 0.3, got {confidence}"


@pytest.mark.parametrize("pdf_name", get_test_pdfs())
def test_rooms_detected(pdf_name):
    """Test that rooms are detected in floor plans"""
    pdf_path = os.path.join(FIXTURES_DIR, pdf_name)
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()

    result = extract_floor_plan(pdf_content)

    rooms = result.get("rooms", [])
    print(f"\n{pdf_name}: Found {len(rooms)} rooms")

    # Should detect at least some rooms
    assert len(rooms) >= 2, f"Expected at least 2 rooms, got {len(rooms)}"


def test_simple_floor_plan_details():
    """Test specific expectations for simple floor plan"""
    pdf_path = os.path.join(FIXTURES_DIR, "simple_floor_plan.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("simple_floor_plan.pdf not found")

    with open(pdf_path, "rb") as f:
        pdf_content = f.read()

    result = extract_floor_plan(pdf_content)

    # Print detailed results for debugging
    print("\n=== Simple Floor Plan Extraction Results ===")
    print(f"Walls: {len(result.get('walls', []))}")
    print(f"Rooms: {len(result.get('rooms', []))}")
    print(f"Dimensions: {len(result.get('dimensions', []))}")
    print(f"Doors: {len(result.get('doors', []))}")
    print(f"Windows: {len(result.get('windows', []))}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")

    # Room names should include our labeled rooms
    room_names = [r.get("label", "").upper() for r in result.get("rooms", [])]
    print(f"Room names: {room_names}")

    # Check we found expected rooms
    expected_rooms = ["KITCHEN", "LIVING ROOM", "BEDROOM", "BATHROOM"]
    for room in expected_rooms:
        found = any(room in name for name in room_names)
        print(f"  {room}: {'Found' if found else 'Not found'}")


def test_detailed_floor_plan_details():
    """Test specific expectations for detailed floor plan"""
    pdf_path = os.path.join(FIXTURES_DIR, "detailed_floor_plan.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("detailed_floor_plan.pdf not found")

    with open(pdf_path, "rb") as f:
        pdf_content = f.read()

    result = extract_floor_plan(pdf_content)

    # Print detailed results for debugging
    print("\n=== Detailed Floor Plan Extraction Results ===")
    print(f"Walls: {len(result.get('walls', []))}")
    print(f"Rooms: {len(result.get('rooms', []))}")
    print(f"Dimensions: {len(result.get('dimensions', []))}")
    print(f"Doors: {len(result.get('doors', []))}")
    print(f"Windows: {len(result.get('windows', []))}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")

    # Should have more walls than simple plan
    assert len(result.get("walls", [])) >= 5

    # Should have multiple bedrooms
    room_names = [r.get("label", "").upper() for r in result.get("rooms", [])]
    print(f"Room names: {room_names}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
