"""
Create test floor plan PDFs for extraction testing
"""

import fitz  # PyMuPDF


def create_simple_floor_plan():
    """
    Create a simple floor plan PDF with:
    - Outer walls (rectangle)
    - Internal walls
    - Room labels
    - Dimensions
    """
    # Create A4 document
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 in points

    # Drawing settings
    wall_color = (0, 0, 0)  # Black
    wall_width = 2.0  # Thick lines for walls
    thin_width = 0.5  # Thin lines for dimensions

    # Scale: Let's say 1 point = 10mm (so 595pt = ~6m wide)
    # We'll draw a small apartment: ~10m x 8m

    # Offset from page edge
    ox, oy = 50, 100

    # Outer walls (10m x 8m = 1000pt x 800pt at our scale... but that's too big)
    # Let's use a smaller scale: 1pt = 20mm, so 500pt = 10m
    scale = 0.5  # points per mm (so 1m = 500pt... still too big)

    # Actually let's just draw in points and label dimensions
    # Outer boundary: 400 x 300 points
    outer_w, outer_h = 400, 300

    # Draw outer walls (thick rectangle)
    outer_rect = fitz.Rect(ox, oy, ox + outer_w, oy + outer_h)
    page.draw_rect(outer_rect, color=wall_color, width=wall_width)

    # Internal walls
    # Vertical wall at 1/3 width (creates hallway/rooms)
    wall1_x = ox + 150
    page.draw_line(
        fitz.Point(wall1_x, oy),
        fitz.Point(wall1_x, oy + outer_h),
        color=wall_color, width=wall_width
    )

    # Horizontal wall in left section (creates 2 rooms)
    wall2_y = oy + 150
    page.draw_line(
        fitz.Point(ox, wall2_y),
        fitz.Point(wall1_x, wall2_y),
        color=wall_color, width=wall_width
    )

    # Horizontal wall in right section (bathroom)
    wall3_y = oy + 200
    page.draw_line(
        fitz.Point(wall1_x, wall3_y),
        fitz.Point(ox + outer_w, wall3_y),
        color=wall_color, width=wall_width
    )

    # Room labels
    font_size = 10

    # Kitchen (top left)
    page.insert_text(
        fitz.Point(ox + 30, oy + 80),
        "KITCHEN",
        fontsize=font_size
    )

    # Living Room (bottom left)
    page.insert_text(
        fitz.Point(ox + 20, oy + 230),
        "LIVING ROOM",
        fontsize=font_size
    )

    # Bedroom (top right)
    page.insert_text(
        fitz.Point(wall1_x + 60, oy + 100),
        "BEDROOM 1",
        fontsize=font_size
    )

    # Bathroom (bottom right)
    page.insert_text(
        fitz.Point(wall1_x + 70, oy + 250),
        "BATHROOM",
        fontsize=font_size
    )

    # Dimensions
    dim_offset = 20

    # Top dimension (total width)
    page.insert_text(
        fitz.Point(ox + outer_w/2 - 20, oy - 10),
        "8000mm",
        fontsize=8
    )

    # Left dimension (total height)
    page.insert_text(
        fitz.Point(ox - 40, oy + outer_h/2),
        "6000mm",
        fontsize=8
    )

    # Kitchen width
    page.insert_text(
        fitz.Point(ox + 50, oy + 140),
        "3000mm",
        fontsize=7
    )

    # Bedroom width
    page.insert_text(
        fitz.Point(wall1_x + 80, oy + 190),
        "5000mm",
        fontsize=7
    )

    # Scale notation
    page.insert_text(
        fitz.Point(ox, oy + outer_h + 50),
        "Scale 1:100",
        fontsize=9
    )

    # Title
    page.insert_text(
        fitz.Point(ox + 100, 50),
        "SAMPLE FLOOR PLAN - APARTMENT 1A",
        fontsize=14
    )

    # Door arcs (simple representation)
    # Door in kitchen wall
    door_center = fitz.Point(wall1_x - 30, wall2_y)
    page.draw_circle(door_center, 20, color=(0.5, 0.5, 0.5), width=thin_width)

    # Save
    output_path = "tests/fixtures/simple_floor_plan.pdf"
    doc.save(output_path)
    doc.close()
    print(f"Created: {output_path}")
    return output_path


def create_detailed_floor_plan():
    """
    Create a more detailed floor plan with multiple rooms
    """
    doc = fitz.open()
    page = doc.new_page(width=842, height=595)  # A4 Landscape

    wall_color = (0, 0, 0)
    wall_width = 2.5

    # Larger apartment layout
    ox, oy = 50, 50

    # Main outer walls - 3 bedroom apartment
    outer_w, outer_h = 700, 450
    outer_rect = fitz.Rect(ox, oy, ox + outer_w, oy + outer_h)
    page.draw_rect(outer_rect, color=wall_color, width=wall_width)

    # Hallway (central corridor)
    hallway_y1 = oy + 180
    hallway_y2 = oy + 270

    # Left section walls (2 bedrooms)
    left_div = ox + 250
    page.draw_line(fitz.Point(left_div, oy), fitz.Point(left_div, hallway_y1),
                   color=wall_color, width=wall_width)
    page.draw_line(fitz.Point(left_div, hallway_y2), fitz.Point(left_div, oy + outer_h),
                   color=wall_color, width=wall_width)

    # Bedroom divider
    bed_div_x = ox + 125
    page.draw_line(fitz.Point(bed_div_x, oy), fitz.Point(bed_div_x, hallway_y1),
                   color=wall_color, width=wall_width)

    # Right section (living, kitchen, bathroom)
    right_div = ox + 500
    page.draw_line(fitz.Point(right_div, hallway_y2), fitz.Point(right_div, oy + outer_h),
                   color=wall_color, width=wall_width)

    # Bathroom wall
    bath_y = hallway_y2 + 90
    page.draw_line(fitz.Point(left_div, bath_y), fitz.Point(right_div, bath_y),
                   color=wall_color, width=wall_width)

    # Kitchen divider
    kitchen_x = right_div + 100
    page.draw_line(fitz.Point(kitchen_x, oy), fitz.Point(kitchen_x, hallway_y1),
                   color=wall_color, width=wall_width)

    # Hallway walls
    page.draw_line(fitz.Point(ox, hallway_y1), fitz.Point(left_div, hallway_y1),
                   color=wall_color, width=wall_width)
    page.draw_line(fitz.Point(ox, hallway_y2), fitz.Point(ox + outer_w, hallway_y2),
                   color=wall_color, width=wall_width)
    page.draw_line(fitz.Point(left_div, hallway_y1), fitz.Point(ox + outer_w, hallway_y1),
                   color=wall_color, width=wall_width)

    # Room labels
    rooms = [
        ("BEDROOM 1", ox + 30, oy + 90),
        ("BEDROOM 2", bed_div_x + 30, oy + 90),
        ("MASTER BEDROOM", ox + 50, hallway_y2 + 80),
        ("EN-SUITE", left_div + 30, bath_y + 40),
        ("BATHROOM", left_div + 30, hallway_y2 + 40),
        ("LIVING ROOM", right_div + 50, hallway_y2 + 80),
        ("KITCHEN", right_div + 30, oy + 90),
        ("UTILITY", kitchen_x + 20, oy + 90),
        ("HALLWAY", left_div + 80, hallway_y1 + 50),
    ]

    for name, x, y in rooms:
        page.insert_text(fitz.Point(x, y), name, fontsize=9)

    # Dimensions
    dims = [
        ("14000mm", ox + outer_w/2 - 30, oy - 15),  # Total width
        ("9000mm", ox - 35, oy + outer_h/2),  # Total height
        ("5000mm", ox + 60, hallway_y1 - 10),  # Left section
        ("5000mm", left_div + 100, hallway_y1 - 10),  # Middle
        ("4000mm", right_div + 60, hallway_y1 - 10),  # Right
        ("3600mm", ox + 30, oy + 160),  # Bedroom depth
        ("3.6m", bed_div_x + 30, oy + 160),  # Bedroom 2
    ]

    for text, x, y in dims:
        page.insert_text(fitz.Point(x, y), text, fontsize=8)

    # Scale and title
    page.insert_text(fitz.Point(ox, oy + outer_h + 30), "Scale 1:50", fontsize=10)
    page.insert_text(fitz.Point(ox + 200, 30), "3-BEDROOM APARTMENT - UNIT 2B", fontsize=14)

    output_path = "tests/fixtures/detailed_floor_plan.pdf"
    doc.save(output_path)
    doc.close()
    print(f"Created: {output_path}")
    return output_path


if __name__ == "__main__":
    create_simple_floor_plan()
    create_detailed_floor_plan()
    print("\nTest PDFs created in tests/fixtures/")
