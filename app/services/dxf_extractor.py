"""
DXF file parser for extracting floor plan data.
Uses ezdxf to read AutoCAD DXF files.
"""

import ezdxf
from ezdxf.entities import Line, LWPolyline, Polyline, Text, MText, Insert
from typing import List, Dict, Any, Tuple
import re
import math


def extract_from_dxf(file_path: str) -> Dict[str, Any]:
    """
    Extract floor plan data from a DXF file.
    Returns walls, dimensions, rooms, doors, windows.
    """
    try:
        doc = ezdxf.readfile(file_path)
    except Exception as e:
        return {
            'error': f'Failed to read DXF file: {str(e)}',
            'walls': [],
            'dimensions': [],
            'rooms': [],
            'doors': [],
            'windows': [],
            'extraction_confidence': 0.0
        }

    msp = doc.modelspace()

    # Extract data
    walls = extract_walls(msp)
    dimensions = extract_dimensions(msp, doc)
    rooms = extract_rooms(msp)
    doors = extract_doors(msp)
    windows = extract_windows(msp)

    # Calculate confidence based on what we found
    confidence = calculate_confidence(walls, dimensions, rooms)

    return {
        'walls': walls,
        'dimensions': dimensions,
        'rooms': rooms,
        'doors': doors,
        'windows': windows,
        'layer_names': get_layer_names(doc),
        'extraction_confidence': confidence,
        'file_type': 'dxf'
    }


def extract_walls(msp) -> List[Dict[str, Any]]:
    """Extract wall lines from DXF modelspace."""
    walls = []

    # Common wall layer names
    wall_layers = ['wall', 'walls', 'a-wall', 'a-walls', 'arch-wall',
                   'interior', 'exterior', 'partition', 'structure']

    # Extract LINE entities
    for entity in msp.query('LINE'):
        layer = entity.dxf.layer.lower()

        # Check if it's on a wall layer or if layer contains 'wall'
        is_wall_layer = any(wl in layer for wl in wall_layers) or 'wall' in layer

        start = entity.dxf.start
        end = entity.dxf.end
        length = math.sqrt(
            (end.x - start.x) ** 2 +
            (end.y - start.y) ** 2
        )

        # Only include lines likely to be walls (> 100mm)
        if length > 100:
            walls.append({
                'start': {'x': round(start.x, 2), 'y': round(start.y, 2)},
                'end': {'x': round(end.x, 2), 'y': round(end.y, 2)},
                'length': round(length, 2),
                'layer': entity.dxf.layer,
                'is_wall_layer': is_wall_layer
            })

    # Extract LWPOLYLINE entities (more common for walls)
    for entity in msp.query('LWPOLYLINE'):
        layer = entity.dxf.layer.lower()
        is_wall_layer = any(wl in layer for wl in wall_layers) or 'wall' in layer

        points = list(entity.get_points())
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            length = math.sqrt(
                (end[0] - start[0]) ** 2 +
                (end[1] - start[1]) ** 2
            )

            if length > 100:
                walls.append({
                    'start': {'x': round(start[0], 2), 'y': round(start[1], 2)},
                    'end': {'x': round(end[0], 2), 'y': round(end[1], 2)},
                    'length': round(length, 2),
                    'layer': entity.dxf.layer,
                    'is_wall_layer': is_wall_layer
                })

        # Handle closed polylines
        if entity.closed and len(points) > 2:
            start = points[-1]
            end = points[0]
            length = math.sqrt(
                (end[0] - start[0]) ** 2 +
                (end[1] - start[1]) ** 2
            )
            if length > 100:
                walls.append({
                    'start': {'x': round(start[0], 2), 'y': round(start[1], 2)},
                    'end': {'x': round(end[0], 2), 'y': round(end[1], 2)},
                    'length': round(length, 2),
                    'layer': entity.dxf.layer,
                    'is_wall_layer': is_wall_layer
                })

    return walls


def extract_dimensions(msp, doc) -> List[Dict[str, Any]]:
    """Extract dimension annotations from DXF."""
    dimensions = []

    # Get DIMENSION entities
    for entity in msp.query('DIMENSION'):
        try:
            # Get the measurement value
            measurement = entity.dxf.get('actual_measurement', None)
            if measurement is None:
                # Try to calculate from geometry
                measurement = entity.get_measurement()

            if measurement and measurement > 0:
                dimensions.append({
                    'value': f'{measurement:.0f}mm',
                    'numeric_value': round(measurement, 2),
                    'unit': 'mm',
                    'type': 'dimension_entity'
                })
        except Exception:
            continue

    # Also extract from TEXT and MTEXT (often dimensions are just text)
    dimension_pattern = re.compile(r'(\d{3,5})\s*(mm)?|(\d{1,3}[.,]\d{1,2})\s*m(?!m)', re.IGNORECASE)

    for entity in msp.query('TEXT MTEXT'):
        try:
            if hasattr(entity.dxf, 'text'):
                text = entity.dxf.text
            elif hasattr(entity, 'text'):
                text = entity.text
            else:
                continue

            matches = dimension_pattern.finditer(text)
            for match in matches:
                try:
                    if match.group(1):
                        value = float(match.group(1))
                    elif match.group(3):
                        value = float(match.group(3).replace(',', '.')) * 1000
                    else:
                        continue

                    if value is not None and 100 <= value <= 50000:
                        dimensions.append({
                            'value': text.strip(),
                            'numeric_value': value,
                            'unit': 'mm',
                            'type': 'text_label'
                        })
                except (ValueError, TypeError):
                    continue
        except Exception:
            continue

    # Remove duplicates
    seen = set()
    unique = []
    for d in dimensions:
        key = round(d['numeric_value'], -1)  # Round to nearest 10
        if key not in seen:
            seen.add(key)
            unique.append(d)

    return unique


def extract_rooms(msp) -> List[Dict[str, Any]]:
    """Extract room labels from text entities."""
    rooms = []

    room_keywords = [
        'kitchen', 'bedroom', 'bathroom', 'living', 'lounge', 'dining',
        'hallway', 'hall', 'corridor', 'entrance', 'utility', 'storage',
        'garage', 'office', 'study', 'en-suite', 'ensuite', 'wc', 'toilet',
        'cloakroom', 'pantry', 'laundry', 'master', 'guest', 'family',
        'sitting', 'conservatory', 'porch', 'landing', 'stairs', 'attic',
        'basement', 'workshop', 'store', 'reception', 'lobby', 'foyer',
        'shower', 'wet room', 'boot room'
    ]

    for entity in msp.query('TEXT MTEXT'):
        try:
            if hasattr(entity.dxf, 'text'):
                text = entity.dxf.text
            elif hasattr(entity, 'text'):
                text = entity.text
            else:
                continue

            text_lower = text.lower().strip()

            for keyword in room_keywords:
                if keyword in text_lower:
                    # Get position
                    if hasattr(entity.dxf, 'insert'):
                        pos = entity.dxf.insert
                        position = {'x': round(pos.x, 2), 'y': round(pos.y, 2)}
                    else:
                        position = None

                    rooms.append({
                        'name': text.strip().title(),
                        'position': position,
                        'layer': entity.dxf.layer
                    })
                    break
        except Exception:
            continue

    return rooms


def extract_doors(msp) -> List[Dict[str, Any]]:
    """Extract door blocks/symbols from DXF."""
    doors = []

    # Doors are usually block references
    door_keywords = ['door', 'dr', 'ent', 'swing']

    for entity in msp.query('INSERT'):
        try:
            block_name = entity.dxf.name.lower()
            if any(kw in block_name for kw in door_keywords):
                pos = entity.dxf.insert
                doors.append({
                    'position': {'x': round(pos.x, 2), 'y': round(pos.y, 2)},
                    'block_name': entity.dxf.name,
                    'rotation': round(entity.dxf.rotation, 2) if hasattr(entity.dxf, 'rotation') else 0
                })
        except Exception:
            continue

    # Also look for arcs (door swings)
    for entity in msp.query('ARC'):
        try:
            # Door swing arcs are typically 90 degrees
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            angle_diff = abs(end_angle - start_angle)

            if 85 <= angle_diff <= 95:  # ~90 degree arc
                center = entity.dxf.center
                doors.append({
                    'position': {'x': round(center.x, 2), 'y': round(center.y, 2)},
                    'type': 'swing_arc',
                    'radius': round(entity.dxf.radius, 2)
                })
        except Exception:
            continue

    return doors


def extract_windows(msp) -> List[Dict[str, Any]]:
    """Extract window blocks/symbols from DXF."""
    windows = []

    window_keywords = ['window', 'win', 'wdw', 'glazing']

    for entity in msp.query('INSERT'):
        try:
            block_name = entity.dxf.name.lower()
            if any(kw in block_name for kw in window_keywords):
                pos = entity.dxf.insert
                windows.append({
                    'position': {'x': round(pos.x, 2), 'y': round(pos.y, 2)},
                    'block_name': entity.dxf.name
                })
        except Exception:
            continue

    return windows


def get_layer_names(doc) -> List[str]:
    """Get all layer names in the DXF file."""
    return [layer.dxf.name for layer in doc.layers]


def calculate_confidence(walls, dimensions, rooms) -> float:
    """Calculate extraction confidence score."""
    score = 0.0

    # Walls found
    if len(walls) > 0:
        score += 0.3
    if len(walls) > 10:
        score += 0.1
    if len(walls) > 50:
        score += 0.1

    # Dimensions found
    if len(dimensions) > 0:
        score += 0.2
    if len(dimensions) > 5:
        score += 0.1

    # Rooms found
    if len(rooms) > 0:
        score += 0.15
    if len(rooms) > 3:
        score += 0.05

    return min(score, 1.0)
