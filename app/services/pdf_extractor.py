"""
QS Parser - Complete PDF Floor Plan Extractor
Following CONSTRUCTION-AI-COMPLETE-SYSTEM-SPEC.md exactly.

TOOLS USED:
- PyMuPDF (fitz): PDF parsing, vector graphics extraction
- pdfplumber: Text extraction with coordinates
- Tesseract (pytesseract): OCR fallback
- NetworkX: cycle_basis for finding enclosed rooms
- Shapely: Polygon area calculation
- OpenCV: Image preprocessing for OCR
- NumPy: Coordinate calculations

NEVER calculates floor area from wall perimeter.
"""

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
import networkx as nx
import numpy as np
import cv2
import re
import math
import os
import tempfile
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from shapely.geometry import Polygon, MultiPoint, LineString, Point
from shapely.ops import unary_union


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class WallSegment:
    start: Tuple[float, float]
    end: Tuple[float, float]
    length_pts: float  # Length in PDF points
    length_m: float    # Length in meters (after scale)
    thickness: float = 0.0

    def to_dict(self) -> dict:
        return {
            "start": {"x": self.start[0], "y": self.start[1]},
            "end": {"x": self.end[0], "y": self.end[1]},
            "length": self.length_pts,
            "length_m": self.length_m,
            "thickness": self.thickness
        }


@dataclass
class Room:
    name: str
    polygon: Optional[Polygon]
    area_m2: float
    center: Tuple[float, float]
    source: str  # 'polygon', 'label', 'stated'

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.name,
            "area_m2": self.area_m2,
            "center": {"x": self.center[0], "y": self.center[1]},
            "source": self.source
        }


@dataclass
class PageData:
    page_num: int
    page_type: str  # 'floor_plan', 'elevation', 'section', 'other'
    floor_name: str  # 'Ground Floor', 'First Floor', etc.
    scale: int       # 50, 100, 200, etc.
    walls: List[WallSegment]
    rooms: List[Room]
    dimensions: List[dict]
    floor_area_m2: float
    floor_area_source: str
    wall_height_m: Optional[float]
    text: str


# ============================================
# CONSTANTS
# ============================================

# Room keywords for label detection
ROOM_KEYWORDS = [
    'kitchen', 'bedroom', 'bathroom', 'living', 'dining',
    'hall', 'hallway', 'corridor', 'utility', 'storage',
    'garage', 'office', 'study', 'en-suite', 'ensuite',
    'wc', 'toilet', 'shower', 'bath', 'lounge', 'sitting',
    'family', 'breakfast', 'pantry', 'laundry', 'closet',
    'wardrobe', 'landing', 'stairs', 'porch', 'entrance',
    'reception', 'drawing', 'master', 'guest', 'kids',
    'kit', 'bed', 'bth', 'liv', 'din', 'lobby', 'ground',
    'first', 'floor', 'room', 'area', 'space'
]

# Floor name patterns
FLOOR_PATTERNS = [
    (r'ground\s*floor', 'Ground Floor'),
    (r'first\s*floor', 'First Floor'),
    (r'second\s*floor', 'Second Floor'),
    (r'basement', 'Basement'),
    (r'attic', 'Attic'),
    (r'roof\s*plan', 'Roof Plan'),
    (r'elevation', 'Elevation'),
    (r'section', 'Section'),
]

# Minimum wall length in PDF points (filter noise)
MIN_WALL_LENGTH_PTS = 30  # About 0.5m at 1:100 scale

# Snap tolerance for graph building (PDF points)
SNAP_TOLERANCE = 10

# OCR configurations for difficult floor plans
OCR_CONFIGS = [
    "--oem 3 --psm 6",
    "--oem 3 --psm 11",
    "--oem 3 --psm 12",
]


# ============================================
# MAIN EXTRACTION FUNCTION
# ============================================

def extract_from_pdf(file_path: str) -> Dict[str, Any]:
    """
    Extract floor plan data from a PDF file.

    ARCHITECTURE (from spec):
    Stage 1: Document Analysis
    Stage 2: Per-Page Extraction
    Stage 3: Elevation/Section Extraction
    Stage 4: Aggregation

    Returns structured JSON with floor areas, rooms, walls, etc.
    """
    print(f"\n{'='*60}")
    print(f"QS PARSER - EXTRACTING: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    # Open with both PyMuPDF and pdfplumber
    fitz_doc = fitz.open(file_path)
    plumber_doc = pdfplumber.open(file_path)

    all_pages: List[PageData] = []
    all_text = []
    ocr_used = False

    # ========================================
    # STAGE 1 & 2: Per-page processing
    # ========================================
    for page_num in range(len(fitz_doc)):
        print(f"\n--- Processing Page {page_num + 1} ---")

        fitz_page = fitz_doc[page_num]
        plumber_page = plumber_doc.pages[page_num] if page_num < len(plumber_doc.pages) else None

        # Extract text using all methods
        text, used_ocr = extract_text_all_methods(fitz_page, plumber_page)
        all_text.append(text)
        if used_ocr:
            ocr_used = True

        print(f"  Text extracted: {len(text)} chars (OCR: {used_ocr})")

        # Classify page type
        page_type, floor_name = classify_page(text)
        print(f"  Page type: {page_type}, Floor: {floor_name}")

        # Detect scale
        scale = detect_scale_from_text(text)
        print(f"  Scale detected: 1:{scale}")

        # Extract walls (PyMuPDF get_drawings)
        walls = extract_walls_from_vectors(fitz_page, scale)
        print(f"  Walls extracted: {len(walls)}")

        # Extract dimensions
        dimensions = extract_dimensions_from_text(text, fitz_page)
        print(f"  Dimensions found: {len(dimensions)}")

        # Extract room labels with positions
        room_labels = extract_room_labels(fitz_page, plumber_page, text)
        print(f"  Room labels found: {len(room_labels)}")

        # Find enclosed rooms using NetworkX cycle_basis
        room_polygons = find_room_polygons_networkx(walls, scale)
        print(f"  Room polygons found: {len(room_polygons)}")

        # Match labels to polygons
        rooms = match_labels_to_polygons(room_labels, room_polygons, scale)
        print(f"  Rooms matched: {len(rooms)}")

        # Calculate floor area (PRIORITY ORDER - NEVER from perimeter)
        floor_area_m2, area_source = calculate_floor_area(
            text, rooms, walls, room_polygons, scale
        )
        print(f"  Floor area: {floor_area_m2} m² (source: {area_source})")

        # Extract wall height (for elevations/sections)
        wall_height = None
        if page_type in ['elevation', 'section']:
            wall_height = extract_wall_height(text)
            print(f"  Wall height: {wall_height}m")

        page_data = PageData(
            page_num=page_num,
            page_type=page_type,
            floor_name=floor_name,
            scale=scale,
            walls=walls,
            rooms=rooms,
            dimensions=dimensions,
            floor_area_m2=floor_area_m2,
            floor_area_source=area_source,
            wall_height_m=wall_height,
            text=text
        )
        all_pages.append(page_data)

    fitz_doc.close()
    plumber_doc.close()

    # ========================================
    # STAGE 4: Aggregation
    # ========================================
    result = aggregate_results(all_pages, all_text, ocr_used, file_path)

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"Total floor area: {result['summary']['total_floor_area_m2']} m²")
    print(f"Floors: {result['summary']['floors']}")
    print(f"Rooms: {result['summary']['total_rooms']}")
    print(f"{'='*60}\n")

    return result


def extract_from_image(file_path: str) -> Dict[str, Any]:
    """
    Extract floor plan data from a single image by wrapping it into a one-page PDF.
    Reuses the PDF pipeline so output schema stays consistent.
    """
    image = Image.open(file_path).convert("RGB")
    width, height = image.size

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        tmp_pdf_path = tmp_pdf.name

    try:
        doc = fitz.open()
        page = doc.new_page(width=width, height=height)
        page.insert_image(fitz.Rect(0, 0, width, height), filename=file_path)
        doc.save(tmp_pdf_path)
        doc.close()

        result = extract_from_pdf(tmp_pdf_path)
        result["file_type"] = "image"
        result["source_format"] = os.path.splitext(file_path)[1].lower().lstrip(".")
        return result
    finally:
        if os.path.exists(tmp_pdf_path):
            os.unlink(tmp_pdf_path)


# ============================================
# STAGE 1: TEXT EXTRACTION (USE ALL THREE METHODS)
# ============================================

def extract_text_all_methods(fitz_page, plumber_page) -> Tuple[str, bool]:
    """
    Extract text using PyMuPDF, pdfplumber, and OCR fallback.
    Returns (combined_text, ocr_was_used).
    """
    text_parts = []
    ocr_used = False

    # Method 1: PyMuPDF
    try:
        fitz_text = fitz_page.get_text("text")
        if fitz_text.strip():
            text_parts.append(fitz_text)
    except Exception as e:
        print(f"  PyMuPDF text extraction failed: {e}")

    # Method 2: pdfplumber
    if plumber_page:
        try:
            plumber_text = plumber_page.extract_text() or ""
            if plumber_text.strip():
                text_parts.append(plumber_text)
        except Exception as e:
            print(f"  pdfplumber text extraction failed: {e}")

    # Method 3: OCR fallback if limited text
    combined_so_far = " ".join(text_parts)
    if len(combined_so_far.strip()) < 100:
        try:
            ocr_text = extract_text_ocr(fitz_page)
            if ocr_text.strip():
                text_parts.append(ocr_text)
                ocr_used = True
        except Exception as e:
            print(f"  OCR extraction failed: {e}")

    return " ".join(text_parts), ocr_used


def extract_text_ocr(fitz_page) -> str:
    """
    Extract text using Tesseract OCR.
    Uses 2x resolution for quality OCR.
    """
    # Render page to image at 2x resolution for better OCR
    matrix = fitz.Matrix(2, 2)
    pix = fitz_page.get_pixmap(matrix=matrix)

    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Convert to grayscale and generate multiple preprocessing variants.
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    ocr_inputs = build_ocr_variants(cv_img)

    best_text = ""
    for ocr_img in ocr_inputs:
        for config in OCR_CONFIGS:
            try:
                candidate = pytesseract.image_to_string(ocr_img, config=config)
            except Exception:
                continue
            if len(candidate) > len(best_text):
                best_text = candidate

    return best_text


def build_ocr_variants(gray: np.ndarray) -> List[np.ndarray]:
    """
    Build OCR image variants to improve text capture on scans and noisy drawings.
    """
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )
    kernel = np.ones((2, 2), np.uint8)
    morph_close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)

    return [gray, denoised, otsu, adaptive, morph_close]


# ============================================
# PAGE CLASSIFICATION
# ============================================

def classify_page(text: str) -> Tuple[str, str]:
    """
    Classify page as floor_plan, elevation, section, or other.
    Returns (page_type, floor_name).
    """
    text_lower = text.lower()

    # Check for elevation/section first
    if 'elevation' in text_lower and 'floor' not in text_lower:
        return 'elevation', 'Elevation'
    if 'section' in text_lower and 'floor' not in text_lower:
        return 'section', 'Section'

    # Check for explicit floor patterns
    for pattern, name in FLOOR_PATTERNS:
        if re.search(pattern, text_lower):
            if 'elevation' not in name.lower() and 'section' not in name.lower():
                return 'floor_plan', name
            else:
                return name.lower().replace(' ', '_'), name

    # Floor plan indicators (room names, area annotations, scale)
    floor_plan_indicators = [
        'bedroom', 'kitchen', 'bathroom', 'living', 'dining', 'hall',
        'utility', 'wc', 'toilet', 'shower', 'lounge', 'study',
        'sqm', 'sq m', 'm²', 'm2',  # Area annotations
        'scale', '1:',  # Scale indicators
        'space', 'room', 'area'
    ]

    indicator_count = sum(1 for kw in floor_plan_indicators if kw in text_lower)

    if indicator_count >= 2:
        # Check if it mentions specific floor
        if 'first' in text_lower or 'upper' in text_lower:
            return 'floor_plan', 'First Floor'
        elif 'second' in text_lower:
            return 'floor_plan', 'Second Floor'
        elif 'ground' in text_lower or 'lower' in text_lower:
            return 'floor_plan', 'Ground Floor'
        else:
            return 'floor_plan', 'Ground Floor'  # Default to ground if floor plan detected

    # If has any single strong indicator
    if any(kw in text_lower for kw in ['bedroom', 'kitchen', 'living', 'bathroom']):
        return 'floor_plan', 'Unknown Floor'

    return 'other', 'Unknown'


# ============================================
# SCALE DETECTION
# ============================================

def detect_scale_from_text(text: str) -> int:
    """
    Detect drawing scale from text (e.g., "Scale 1:100").
    Returns scale factor (e.g., 100 for 1:100).
    """
    # Pattern: "1:50", "Scale 1:100", "1 : 200", etc.
    patterns = [
        r'(?:scale\s*)?1\s*:\s*(\d+)',
        r'scale\s+(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            scale = int(match.group(1))
            if scale in [20, 50, 100, 200, 500]:  # Valid architectural scales
                return scale

    # Default to 1:100 (common for floor plans)
    return 100


# ============================================
# WALL EXTRACTION (PYMUPDF get_drawings)
# ============================================

def extract_walls_from_vectors(fitz_page, scale: int) -> List[WallSegment]:
    """
    Extract wall segments from vector graphics using PyMuPDF.
    Filters hatching and annotation lines.
    """
    walls = []
    all_lines = []

    # Get all vector drawings
    drawings = fitz_page.get_drawings()

    for path in drawings:
        width = path.get("width") or 0
        color = path.get("color")

        # Skip very thin lines (annotations, hatching)
        # But don't be too aggressive - some CAD exports have thin wall lines

        for item in path.get("items", []):
            if item[0] == "l":  # Line segment
                p1, p2 = item[1], item[2]

                # Calculate length in PDF points
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                length_pts = math.sqrt(dx * dx + dy * dy)

                # Skip very short lines
                if length_pts < MIN_WALL_LENGTH_PTS:
                    continue

                # Skip point objects (start == end)
                if abs(dx) < 1 and abs(dy) < 1:
                    continue

                # Calculate angle for filtering
                angle = math.degrees(math.atan2(dy, dx)) % 180

                all_lines.append({
                    "p1": (p1.x, p1.y),
                    "p2": (p2.x, p2.y),
                    "length_pts": length_pts,
                    "width": width,
                    "angle": angle
                })

            elif item[0] == "re":  # Rectangle
                rect = item[1]
                w, h = rect.width, rect.height

                # Skip tiny rectangles
                if w < MIN_WALL_LENGTH_PTS and h < MIN_WALL_LENGTH_PTS:
                    continue

                # Add rectangle edges as walls
                edges = [
                    ((rect.x0, rect.y0), (rect.x1, rect.y0), w),
                    ((rect.x1, rect.y0), (rect.x1, rect.y1), h),
                    ((rect.x1, rect.y1), (rect.x0, rect.y1), w),
                    ((rect.x0, rect.y1), (rect.x0, rect.y0), h),
                ]
                for p1, p2, length in edges:
                    if length >= MIN_WALL_LENGTH_PTS:
                        all_lines.append({
                            "p1": p1,
                            "p2": p2,
                            "length_pts": length,
                            "width": width,
                            "angle": 0 if p1[1] == p2[1] else 90
                        })

    # Filter hatching patterns (many parallel lines at same angle)
    all_lines = filter_hatching_patterns(all_lines)

    # Raster fallback for scanned PDFs or flattened drawings with limited vectors.
    if len(all_lines) < 5:
        all_lines.extend(extract_walls_from_raster(fitz_page))

    # Convert to WallSegment objects
    for line in all_lines:
        length_m = pts_to_meters(line["length_pts"], scale)

        walls.append(WallSegment(
            start=line["p1"],
            end=line["p2"],
            length_pts=line["length_pts"],
            length_m=length_m,
            thickness=line["width"]
        ))

    return walls


def extract_walls_from_raster(fitz_page) -> List[dict]:
    """
    Detect probable wall line segments from rendered page image using Canny + HoughLinesP.
    Returns line dicts in PDF point coordinates.
    """
    matrix = fitz.Matrix(2, 2)
    pix = fitz_page.get_pixmap(matrix=matrix, alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=40,
        maxLineGap=8
    )

    if lines is None:
        return []

    scale_down = 2.0
    result = []
    for line in lines[:2000]:
        x1, y1, x2, y2 = line[0]
        p1 = (x1 / scale_down, y1 / scale_down)
        p2 = (x2 / scale_down, y2 / scale_down)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length_pts = math.sqrt(dx * dx + dy * dy)
        if length_pts < MIN_WALL_LENGTH_PTS:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180
        result.append({
            "p1": p1,
            "p2": p2,
            "length_pts": length_pts,
            "width": 0.0,
            "angle": angle
        })

    # Keep the longest lines to reduce image-noise detections.
    result.sort(key=lambda x: x["length_pts"], reverse=True)
    return result[:500]


def filter_hatching_patterns(lines: List[dict]) -> List[dict]:
    """
    Filter out hatching patterns (many parallel lines at same angle).
    Hatching typically has 10+ lines at nearly identical angles and spacing.
    """
    if len(lines) < 20:
        return lines

    # Group lines by angle (rounded to nearest 5 degrees)
    angle_groups = {}
    for line in lines:
        angle_key = round(line["angle"] / 5) * 5
        if angle_key not in angle_groups:
            angle_groups[angle_key] = []
        angle_groups[angle_key].append(line)

    # If any angle has too many lines, it's likely hatching
    filtered = []
    for angle, group in angle_groups.items():
        if len(group) > 50:
            # Likely hatching - keep only longest 20%
            group.sort(key=lambda x: x["length_pts"], reverse=True)
            filtered.extend(group[:max(10, len(group) // 5)])
        else:
            filtered.extend(group)

    return filtered


# ============================================
# ROOM DETECTION (NETWORKX cycle_basis)
# ============================================

def find_room_polygons_networkx(walls: List[WallSegment], scale: int) -> List[Polygon]:
    """
    Find enclosed rooms using NetworkX cycle detection and Shapely.
    This is the CORRECT way to find room areas.
    """
    if not walls:
        return []

    # Build graph of wall connections
    G = nx.Graph()

    def snap_point(x: float, y: float) -> Tuple[int, int]:
        """Snap coordinates to grid to handle small gaps."""
        return (
            round(x / SNAP_TOLERANCE) * SNAP_TOLERANCE,
            round(y / SNAP_TOLERANCE) * SNAP_TOLERANCE
        )

    # Add edges for each wall
    for wall in walls:
        p1 = snap_point(wall.start[0], wall.start[1])
        p2 = snap_point(wall.end[0], wall.end[1])

        if p1 != p2:  # Skip zero-length after snapping
            G.add_edge(p1, p2)

    print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_edges() < 3:
        return []

    # Find cycles (enclosed spaces)
    try:
        cycles = nx.cycle_basis(G)
        print(f"    Cycles found: {len(cycles)}")
    except Exception as e:
        print(f"    cycle_basis failed: {e}")
        return []

    polygons = []
    for cycle in cycles:
        if len(cycle) < 3:
            continue

        try:
            # Create Shapely polygon
            poly = Polygon(cycle)

            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix self-intersections

            if not poly.is_valid or poly.is_empty:
                continue

            # Calculate area in m²
            area_m2 = polygon_area_m2(poly, scale)

            # Filter realistic room sizes (2-200 m²)
            if 2 < area_m2 < 200:
                polygons.append(poly)
                print(f"    Valid room polygon: {area_m2:.1f} m²")

        except Exception as e:
            continue

    return polygons


def polygon_area_m2(poly: Polygon, scale: int) -> float:
    """Calculate polygon area in m² given scale."""
    # PDF points to meters: pts / 72 (inches) * 0.0254 (m/inch) * scale
    scale_factor = (1 / 72) * 0.0254 * scale
    area_m2 = poly.area * (scale_factor ** 2)
    return area_m2


def pts_to_meters(pts: float, scale: int) -> float:
    """Convert PDF points to meters given scale."""
    # pts / 72 = inches, * 0.0254 = meters, * scale = real meters
    return pts / 72 * 0.0254 * scale


# ============================================
# ROOM LABEL EXTRACTION
# ============================================

def extract_room_labels(fitz_page, plumber_page, text: str) -> List[dict]:
    """
    Extract room labels with positions using pdfplumber words.
    """
    labels = []

    # Method 1: pdfplumber words (best for positions)
    if plumber_page:
        try:
            words = plumber_page.extract_words()
            for word in words:
                word_text = word.get("text", "").strip()
                word_lower = word_text.lower()

                # Check if contains room keyword
                for keyword in ROOM_KEYWORDS:
                    if keyword in word_lower or word_lower in keyword:
                        x0 = word.get("x0", 0)
                        x1 = word.get("x1", 0)
                        y0 = word.get("top", 0)
                        y1 = word.get("bottom", 0)

                        labels.append({
                            "text": word_text,
                            "position": {
                                "x": (x0 + x1) / 2,
                                "y": (y0 + y1) / 2
                            },
                            "source": "pdfplumber"
                        })
                        break
        except Exception as e:
            print(f"    pdfplumber word extraction failed: {e}")

    # Method 2: PyMuPDF text dict (backup)
    try:
        text_dict = fitz_page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    span_text = span["text"].strip()
                    span_lower = span_text.lower()
                    bbox = span["bbox"]

                    for keyword in ROOM_KEYWORDS:
                        if keyword in span_lower:
                            # Check if already found by pdfplumber
                            already_found = any(
                                abs(l["position"]["x"] - (bbox[0]+bbox[2])/2) < 20 and
                                abs(l["position"]["y"] - (bbox[1]+bbox[3])/2) < 20
                                for l in labels
                            )
                            if not already_found:
                                labels.append({
                                    "text": span_text,
                                    "position": {
                                        "x": (bbox[0] + bbox[2]) / 2,
                                        "y": (bbox[1] + bbox[3]) / 2
                                    },
                                    "source": "fitz"
                                })
                            break
    except Exception as e:
        print(f"    PyMuPDF text dict failed: {e}")

    return labels


def match_labels_to_polygons(
    labels: List[dict],
    polygons: List[Polygon],
    scale: int
) -> List[Room]:
    """
    Match room labels to their corresponding polygons.
    """
    rooms = []
    used_polygons = set()

    for label in labels:
        label_point = Point(label["position"]["x"], label["position"]["y"])

        # Find polygon containing this label
        best_poly = None
        best_idx = -1

        for idx, poly in enumerate(polygons):
            if idx in used_polygons:
                continue
            if poly.contains(label_point) or poly.distance(label_point) < 20:
                best_poly = poly
                best_idx = idx
                break

        area_m2 = 0.0
        if best_poly:
            area_m2 = polygon_area_m2(best_poly, scale)
            used_polygons.add(best_idx)

        rooms.append(Room(
            name=label["text"],
            polygon=best_poly,
            area_m2=round(area_m2, 1),
            center=(label["position"]["x"], label["position"]["y"]),
            source="polygon" if best_poly else "label"
        ))

    # Add unlabeled polygons as "Room"
    for idx, poly in enumerate(polygons):
        if idx not in used_polygons:
            centroid = poly.centroid
            area_m2 = polygon_area_m2(poly, scale)
            rooms.append(Room(
                name="Room",
                polygon=poly,
                area_m2=round(area_m2, 1),
                center=(centroid.x, centroid.y),
                source="polygon"
            ))

    return rooms


# ============================================
# DIMENSION EXTRACTION
# ============================================

def extract_dimensions_from_text(text: str, fitz_page) -> List[dict]:
    """
    Extract dimension annotations from text.
    """
    dimensions = []

    # Patterns for dimensions
    patterns = [
        (r'(\d+(?:\.\d+)?)\s*mm', 'mm'),
        (r'(\d+(?:\.\d+)?)\s*m(?!\w)', 'm'),
        (r'(\d+(?:\.\d+)?)\s*cm', 'cm'),
        (r'(\d{4,5})(?:\s|$|[^\d])', 'mm'),  # 4-5 digit numbers likely mm
    ]

    for pattern, unit in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = float(match.group(1))

                # Convert to mm
                if unit == 'm':
                    value_mm = value * 1000
                elif unit == 'cm':
                    value_mm = value * 10
                else:
                    value_mm = value

                # Filter realistic dimensions (100mm to 50000mm)
                if 100 <= value_mm <= 50000:
                    dimensions.append({
                        "value": match.group(0),
                        "numeric_value": value_mm,
                        "unit": unit
                    })
            except:
                continue

    return dimensions


# ============================================
# FLOOR AREA CALCULATION (PRIORITY ORDER)
# ============================================

def calculate_floor_area(
    text: str,
    rooms: List[Room],
    walls: List[WallSegment],
    room_polygons: List[Polygon],
    scale: int
) -> Tuple[float, str]:
    """
    Calculate floor area using PRIORITY ORDER.
    NEVER calculates from wall perimeter.

    Priority 1: Stated area in text
    Priority 2: Sum of room polygon areas (Shapely)
    Priority 3: Convex hull of wall points (Shapely)
    Priority 4: Estimate from room count (last resort)
    """

    # PRIORITY 1: Find stated area in text
    stated_area = find_stated_area(text)
    if stated_area and stated_area > 10:
        print(f"    Using stated area: {stated_area} m²")
        return stated_area, "stated"

    # PRIORITY 2: Sum of room polygon areas (Shapely)
    if room_polygons:
        total_polygon_area = sum(polygon_area_m2(p, scale) for p in room_polygons)
        if total_polygon_area > 10:
            print(f"    Using polygon sum: {total_polygon_area:.1f} m²")
            return round(total_polygon_area, 1), "calculated_from_rooms"

    # Also try room areas if polygons were matched
    room_areas = [r.area_m2 for r in rooms if r.area_m2 > 0]
    if room_areas and sum(room_areas) > 10:
        print(f"    Using room areas sum: {sum(room_areas):.1f} m²")
        return round(sum(room_areas), 1), "calculated_from_rooms"

    # PRIORITY 3: Convex hull of wall points (Shapely)
    if walls:
        try:
            all_points = []
            for wall in walls:
                all_points.append(wall.start)
                all_points.append(wall.end)

            if len(all_points) >= 3:
                multi_point = MultiPoint(all_points)
                hull = multi_point.convex_hull

                if hasattr(hull, 'area') and hull.area > 0:
                    hull_area_m2 = polygon_area_m2(hull, scale)

                    # Apply 85% factor for internal area
                    internal_area = hull_area_m2 * 0.85

                    if 10 < internal_area < 1000:
                        print(f"    Using convex hull: {internal_area:.1f} m²")
                        return round(internal_area, 1), "calculated_from_bounds"
        except Exception as e:
            print(f"    Convex hull calculation failed: {e}")

    # PRIORITY 4: Estimate from room count (last resort)
    room_count = len(rooms) if rooms else 0
    if room_count > 0:
        estimated = room_count * 15  # 15m² average per room
        print(f"    Using room count estimate: {estimated} m²")
        return estimated, "estimated"

    return 0.0, "unknown"


def find_stated_area(text: str) -> Optional[float]:
    """
    Look for explicitly stated floor area in text.
    Prefers areas labeled as "SPACE" or "FLOOR AREA" over individual room areas.
    If no floor-level area found, sums individual room areas.
    """
    # First, look for explicitly labeled floor/space areas
    floor_patterns = [
        r'SPACE\s+(\d+(?:\.\d+)?)\s*SQM',  # "SPACE 68SQM"
        r'SPACE\s+(\d+(?:\.\d+)?)\s*SQ\s*M',
        r'(?:floor\s*area|gfa|gia|nfa)[:\s]+(\d+(?:\.\d+)?)\s*(?:m²|m2|sqm|sq\.?\s*m)',
        r'(?:total|gross)\s*(?:area)?[:\s]+(\d+(?:\.\d+)?)\s*(?:m²|m2|sqm)',
    ]

    floor_areas = []
    for pattern in floor_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                area = float(match.group(1))
                # Floor areas should be 20-200 m² for residential
                if 20 < area < 200:
                    floor_areas.append(area)
            except:
                continue

    # If we found explicit floor areas, use the smallest (single floor)
    if floor_areas:
        # Filter out areas > 100 which might be total GFA
        single_floor = [a for a in floor_areas if 20 < a < 100]
        if single_floor:
            return max(single_floor)  # Return largest single floor area found
        return min(floor_areas)

    # Fallback: look for any area annotations
    all_patterns = [
        r'(\d+(?:\.\d+)?)\s*SQM',
        r'(\d+(?:\.\d+)?)\s*SQ\s*M',
        r'(\d+(?:\.\d+)?)\s*m2',
        r'(\d+(?:\.\d+)?)\s*m²',
    ]

    all_areas = []
    for pattern in all_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                area = float(match.group(1))
                if 1 < area < 500:  # Include small room areas (1+ m²)
                    all_areas.append(area)
            except:
                continue

    if all_areas:
        # Filter to floor-sized areas (30-100 m² typical for single floor)
        floor_sized = [a for a in all_areas if 30 < a < 100]
        if floor_sized:
            return max(floor_sized)  # Return largest floor-sized area

        # If no floor-sized areas, look for areas 20-200 range
        mid_sized = [a for a in all_areas if 20 < a < 200]
        if mid_sized:
            return max(mid_sized)

        # NEW: If only individual room areas found (< 30 m² each), sum them
        # This handles floor plans with labeled rooms but no floor-level area
        room_sized = [a for a in all_areas if 2 < a < 30]

        # Deduplicate - OCR often finds same text twice (PyMuPDF + pdfplumber)
        # Keep unique values only
        unique_room_sized = list(set(room_sized))

        if len(unique_room_sized) >= 3:  # At least 3 unique rooms
            total = sum(unique_room_sized)
            # Sanity check: total should be reasonable floor area (20-150 m²)
            if 20 < total < 150:
                print(f"    Summing {len(unique_room_sized)} unique room areas: {unique_room_sized} = {total:.1f} m²")
                return round(total, 1)

    return None


# ============================================
# WALL HEIGHT EXTRACTION
# ============================================

def extract_wall_height(text: str) -> Optional[float]:
    """
    Extract wall height from elevation/section text.
    Looks for F.F.L., ceiling height, floor-to-floor, etc.
    """
    patterns = [
        r'(?:ceiling|wall)\s*(?:height|ht)[:\s]+(\d+(?:\.\d+)?)\s*(?:m|mm)',
        r'(?:ffl|f\.f\.l\.?)[:\s]+(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:m|mm)\s*(?:ceiling|floor.to.floor)',
        r'(?:height|ht)[:\s]+(\d+(?:\.\d+)?)\s*(?:m|mm)',
        r'(?:2[,.])?(\d{3})\s*(?:mm)?(?:\s|$)',  # 2675, 2.675, 2,675
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1).replace(',', '.'))
                # Convert mm to m if needed
                if value > 100:
                    value = value / 1000
                if 2.0 < value < 5.0:  # Realistic height
                    return value
            except:
                continue

    return None


# ============================================
# RESULT AGGREGATION
# ============================================

def aggregate_results(
    pages: List[PageData],
    all_text: List[str],
    ocr_used: bool,
    file_path: str
) -> Dict[str, Any]:
    """
    Aggregate results from all pages into final output.
    """
    # Separate floor plans from elevations
    floor_plans = [p for p in pages if p.page_type == 'floor_plan']
    elevations = [p for p in pages if p.page_type in ['elevation', 'section']]

    # Get wall height from elevations
    wall_height = None
    for elev in elevations:
        if elev.wall_height_m:
            wall_height = elev.wall_height_m
            break

    # Calculate totals
    total_floor_area = sum(p.floor_area_m2 for p in floor_plans)
    floor_breakdown = [
        {"name": p.floor_name, "area_m2": p.floor_area_m2}
        for p in floor_plans
    ]

    # Collect all rooms
    all_rooms = []
    for page in floor_plans:
        for room in page.rooms:
            all_rooms.append(room.to_dict())

    # Collect all walls
    all_walls = []
    for page in floor_plans:
        for wall in page.walls:
            all_walls.append(wall.to_dict())

    # Collect all dimensions
    all_dimensions = []
    for page in pages:
        all_dimensions.extend(page.dimensions)

    # Detect scale (use most common or first)
    scales = [p.scale for p in pages if p.scale]
    detected_scale = max(set(scales), key=scales.count) if scales else 100

    # Calculate confidence
    confidence = calculate_confidence(
        len(all_walls),
        len(all_dimensions),
        len(all_rooms),
        total_floor_area > 0
    )

    return {
        "summary": {
            "total_floor_area_m2": round(total_floor_area, 1),
            "floors": len(floor_plans),
            "floor_breakdown": floor_breakdown,
            "total_rooms": len(all_rooms),
            "wall_height_m": wall_height,
            "scale": f"1:{detected_scale}",
            "confidence": confidence
        },
        "walls": all_walls,
        "rooms": all_rooms,
        "dimensions": all_dimensions,
        "doors": [],  # TODO: door detection
        "windows": [],  # TODO: window detection
        "page_count": len(pages),
        "raw_text": all_text,
        "extraction_confidence": confidence / 100,
        "confidence": confidence / 100,
        "ocr_used": ocr_used,
        "file_type": "pdf",
        "filename": os.path.basename(file_path)
    }


def calculate_confidence(
    wall_count: int,
    dimension_count: int,
    room_count: int,
    has_area: bool
) -> int:
    """Calculate confidence score (0-100)."""
    score = 0

    # Walls: up to 30
    if wall_count > 30:
        score += 30
    elif wall_count > 15:
        score += 20
    elif wall_count > 5:
        score += 10

    # Dimensions: up to 25
    if dimension_count > 10:
        score += 25
    elif dimension_count > 5:
        score += 15
    elif dimension_count > 0:
        score += 5

    # Rooms: up to 25
    if room_count > 5:
        score += 25
    elif room_count > 2:
        score += 15
    elif room_count > 0:
        score += 5

    # Floor area: 20
    if has_area:
        score += 20

    return min(score, 100)


# ============================================
# LEGACY INTERFACE (for API compatibility)
# ============================================

def extract_floor_plan(pdf_content: bytes) -> dict:
    """
    Legacy interface for extracting from bytes.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_content)
        tmp_path = tmp.name

    try:
        result = extract_from_pdf(tmp_path)

        # Add legacy fields
        result["page_width"] = 0
        result["page_height"] = 0
        result["scale_factor"] = int(result["summary"]["scale"].split(":")[1])

        return result
    finally:
        os.unlink(tmp_path)


def extract_all_pages(pdf_content: bytes) -> List[dict]:
    """
    Legacy interface for multi-page extraction.
    """
    result = extract_floor_plan(pdf_content)
    return [result]  # Return as single-item list for compatibility


# ============================================
# COMMAND LINE TESTING
# ============================================

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    result = extract_from_pdf(pdf_path)

    print("\n" + "="*60)
    print("EXTRACTION RESULT")
    print("="*60)
    print(json.dumps(result["summary"], indent=2))
