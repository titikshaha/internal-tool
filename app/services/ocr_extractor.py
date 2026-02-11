"""
OCR extraction for PDFs with text rendered as curves.
Uses Tesseract to read text from page images.
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
from typing import List, Dict, Any, Optional


def pdf_page_to_image(page: fitz.Page, dpi: int = 200) -> Image.Image:
    """Convert a PDF page to a PIL Image for OCR processing."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))


def extract_text_with_ocr(page: fitz.Page) -> str:
    """Extract all text from a PDF page using OCR."""
    image = pdf_page_to_image(page)
    text = pytesseract.image_to_string(image)
    return text


def extract_dimensions_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract dimension values from OCR text.
    Handles formats: 4500, 4500mm, 4.5m, 4500 mm, 4'-6", etc.
    """
    dimensions = []

    # Pattern for metric dimensions
    metric_patterns = [
        # 4500mm or 4500 mm
        r'(\d{3,5})\s*mm',
        # 4.5m or 4.5 m or 4,5m
        r'(\d{1,3}[.,]\d{1,2})\s*m(?!m)',
        # Bare numbers 3-5 digits (likely mm)
        r'(?<![.,\d])(\d{3,5})(?![.,\d])',
    ]

    # Pattern for imperial dimensions
    imperial_patterns = [
        # 4'-6" or 4' 6" or 4'6"
        r"(\d{1,2})'[\s-]*(\d{1,2})\"",
        # 4' or 4 ft
        r"(\d{1,2})'\s*(?![\d\"])",
    ]

    # Extract metric
    for pattern in metric_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                value = match.group(1).replace(',', '.')

                # Convert to mm
                if 'm' in text[match.end():match.end()+2].lower() and 'mm' not in text[match.end():match.end()+3].lower():
                    # It's meters, convert to mm
                    numeric_value = float(value) * 1000
                else:
                    numeric_value = float(value)

                # Filter out unrealistic values (less than 100mm or more than 50000mm)
                if numeric_value is not None and 100 <= numeric_value <= 50000:
                    dimensions.append({
                        'value': match.group(0),
                        'numeric_value': numeric_value,
                        'unit': 'mm',
                        'source': 'ocr'
                    })
            except (ValueError, TypeError):
                continue

    # Extract imperial
    for pattern in imperial_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            try:
                if len(match.groups()) == 2:
                    feet = int(match.group(1))
                    inches = int(match.group(2))
                    numeric_value = (feet * 12 + inches) * 25.4  # Convert to mm
                else:
                    feet = int(match.group(1))
                    numeric_value = feet * 12 * 25.4

                if numeric_value is not None and 100 <= numeric_value <= 50000:
                    dimensions.append({
                        'value': match.group(0),
                        'numeric_value': numeric_value,
                        'unit': 'mm',
                        'source': 'ocr'
                    })
            except (ValueError, TypeError):
                continue

    # Remove duplicates (same numeric value)
    seen = set()
    unique_dimensions = []
    for dim in dimensions:
        if dim['numeric_value'] not in seen:
            seen.add(dim['numeric_value'])
            unique_dimensions.append(dim)

    return unique_dimensions


def extract_room_labels_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract room names from OCR text."""
    room_keywords = [
        'kitchen', 'bedroom', 'bathroom', 'living room', 'living', 'lounge',
        'dining room', 'dining', 'hallway', 'hall', 'corridor', 'entrance',
        'utility', 'storage', 'garage', 'office', 'study', 'en-suite', 'ensuite',
        'en suite', 'wc', 'toilet', 'cloakroom', 'pantry', 'laundry',
        'master bedroom', 'guest bedroom', 'family room', 'sitting room',
        'conservatory', 'porch', 'vestibule', 'landing', 'stairs', 'stairwell',
        'attic', 'basement', 'cellar', 'workshop', 'store', 'plant room',
        'reception', 'lobby', 'foyer', 'closet', 'wardrobe', 'dressing room',
        'shower room', 'wet room', 'boot room', 'mud room'
    ]

    rooms = []
    text_lower = text.lower()

    for keyword in room_keywords:
        if keyword in text_lower:
            # Find the actual case-preserved version
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            match = pattern.search(text)
            if match:
                rooms.append({
                    'name': match.group(0).title(),
                    'source': 'ocr'
                })

    # Remove duplicates
    seen = set()
    unique_rooms = []
    for room in rooms:
        if room['name'].lower() not in seen:
            seen.add(room['name'].lower())
            unique_rooms.append(room)

    return unique_rooms


def process_pdf_with_ocr(pdf_path: str) -> Dict[str, Any]:
    """
    Process a PDF using OCR when normal text extraction fails.
    Returns dimensions and room labels found.
    """
    doc = fitz.open(pdf_path)

    all_dimensions = []
    all_rooms = []
    all_text = []

    for page_num, page in enumerate(doc):
        # Run OCR on page
        ocr_text = extract_text_with_ocr(page)
        all_text.append(ocr_text)

        # Extract dimensions
        dimensions = extract_dimensions_from_text(ocr_text)
        for dim in dimensions:
            dim['page'] = page_num + 1
        all_dimensions.extend(dimensions)

        # Extract room labels
        rooms = extract_room_labels_from_text(ocr_text)
        for room in rooms:
            room['page'] = page_num + 1
        all_rooms.extend(rooms)

    doc.close()

    return {
        'dimensions': all_dimensions,
        'rooms': all_rooms,
        'raw_text': all_text,
        'ocr_used': True
    }
