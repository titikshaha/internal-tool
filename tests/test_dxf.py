"""Tests for DXF extraction."""

import pytest
import ezdxf
import tempfile
import os
from app.services.dxf_extractor import extract_from_dxf


@pytest.fixture
def simple_dxf_file():
    """Create a simple DXF file for testing."""
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Add walls
    msp.add_line((0, 0), (5000, 0), dxfattribs={'layer': 'Walls'})
    msp.add_line((5000, 0), (5000, 4000), dxfattribs={'layer': 'Walls'})
    msp.add_line((5000, 4000), (0, 4000), dxfattribs={'layer': 'Walls'})
    msp.add_line((0, 4000), (0, 0), dxfattribs={'layer': 'Walls'})

    # Add room label
    msp.add_text("Kitchen", dxfattribs={'layer': 'Text', 'height': 200}).set_placement((2500, 2000))

    # Add dimension
    msp.add_text("5000mm", dxfattribs={'layer': 'Dims', 'height': 150}).set_placement((2500, -200))

    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
        doc.saveas(f.name)
        yield f.name

    os.unlink(f.name)


@pytest.fixture
def complex_dxf_file():
    """Create a more complex DXF file with polylines."""
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Add polyline walls
    points = [(0, 0), (6000, 0), (6000, 5000), (0, 5000), (0, 0)]
    msp.add_lwpolyline(points, dxfattribs={'layer': 'A-Walls'}, close=True)

    # Add multiple rooms
    msp.add_text("Living Room", dxfattribs={'height': 200}).set_placement((3000, 2500))
    msp.add_text("Kitchen", dxfattribs={'height': 200}).set_placement((1000, 4000))
    msp.add_text("Bedroom", dxfattribs={'height': 200}).set_placement((5000, 4000))

    # Add door arc (90 degree swing)
    msp.add_arc(center=(2000, 0), radius=800, start_angle=0, end_angle=90)

    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
        doc.saveas(f.name)
        yield f.name

    os.unlink(f.name)


class TestDXFExtraction:
    """Test DXF file extraction."""

    def test_extract_walls(self, simple_dxf_file):
        result = extract_from_dxf(simple_dxf_file)
        assert len(result['walls']) == 4

    def test_extract_rooms(self, simple_dxf_file):
        result = extract_from_dxf(simple_dxf_file)
        assert len(result['rooms']) >= 1
        assert any('kitchen' in r['name'].lower() for r in result['rooms'])

    def test_extract_dimensions(self, simple_dxf_file):
        result = extract_from_dxf(simple_dxf_file)
        assert len(result['dimensions']) >= 1
        assert any(d['numeric_value'] == 5000 for d in result['dimensions'])

    def test_confidence_score(self, simple_dxf_file):
        result = extract_from_dxf(simple_dxf_file)
        assert 0 < result['extraction_confidence'] <= 1

    def test_polyline_walls(self, complex_dxf_file):
        result = extract_from_dxf(complex_dxf_file)
        assert len(result['walls']) >= 4

    def test_multiple_rooms(self, complex_dxf_file):
        result = extract_from_dxf(complex_dxf_file)
        assert len(result['rooms']) >= 3

    def test_door_detection(self, complex_dxf_file):
        result = extract_from_dxf(complex_dxf_file)
        assert len(result['doors']) >= 1

    def test_layer_names_returned(self, simple_dxf_file):
        result = extract_from_dxf(simple_dxf_file)
        assert 'layer_names' in result
        assert len(result['layer_names']) > 0

    def test_invalid_file(self):
        result = extract_from_dxf('/nonexistent/file.dxf')
        assert 'error' in result
        assert result['extraction_confidence'] == 0
