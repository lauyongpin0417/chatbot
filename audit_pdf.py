"""
PDF Structure Audit Tool for RAG

This tool audits every PDF page and detects:

1. NORMAL
   Normal native text page.

2. TABLE
   Traditional/native PDF table.

3. FLOW_TABLE
   Visual workflow / process table / matrix-like layout
   made from native PDF text + vector shapes.

4. IMAGE_CONTENT
   Page contains an embedded/image block that cannot be
   extracted as native text.

   This is especially important for pages like:

       Native text
       +
       IMAGE-BASED FLOW TABLE
       +
       Native text

5. IMAGE_TABLE_CANDIDATE
   Image content that looks likely to contain structured
   table-like information based on its surrounding layout.

6. IMAGE_FLOW_CANDIDATE
   Image content surrounded by workflow-like/native text
   structure. This is only a CANDIDATE, not a guarantee.

7. OCR
   Page has little/no native text and contains image content.
   Likely scanned page.

8. VISION
   Complex visual content candidate.

9. REVIEW
   Ambiguous page.

IMPORTANT:
This script does NOT perform OCR or Vision.

It only identifies pages that require special processing.

Usage:

    python audit_pdf.py "knowledge_docs\\RMS_USER_MANUALv7.pdf"

Install:

    pip install pymupdf
"""

import sys
import pymupdf


# ============================================================
# CONFIGURATION
# ============================================================

MIN_NATIVE_TEXT_CHARS = 20

# Image occupying >= 15% of page
# is already considered significant.
IMAGE_CONTENT_RATIO = 0.15

# Image occupying >= 30% of page
# is considered large.
LARGE_IMAGE_RATIO = 0.30

# Image occupying >= 55% of page
# is considered very large.
VERY_LARGE_IMAGE_RATIO = 0.55

# Traditional table detection
TABLE_SCORE_THRESHOLD = 60

# Flow table detection
FLOW_TABLE_SCORE_THRESHOLD = 60

# OCR detection
OCR_SCORE_THRESHOLD = 70

# Vision detection
VISION_SCORE_THRESHOLD = 70

# Spatial analysis
COORD_TOLERANCE = 12

FLOW_MIN_COLUMNS = 4
FLOW_MIN_ROWS = 3

# Vector analysis
DRAWING_THRESHOLD = 20


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_percentage(value):
    return f"{value * 100:.1f}%"


def get_page_area(page):
    return abs(page.rect)


def rect_area(rect):
    return abs(rect.width * rect.height)


# ============================================================
# TEXT ANALYSIS
# ============================================================

def analyze_text(page):
    """
    Analyse native/extractable text.
    """

    text = page.get_text("text").strip()

    char_count = len(text)

    blocks = page.get_text("blocks")

    text_blocks = []

    for block in blocks:

        if len(block) < 5:
            continue

        block_text = block[4].strip()

        if block_text:
            text_blocks.append(block_text)

    lines = []

    for block_text in text_blocks:

        for line in block_text.splitlines():

            line = line.strip()

            if line:
                lines.append(line)

    short_lines = [
        line
        for line in lines
        if 1 <= len(line) <= 35
    ]

    if lines:

        avg_line_length = (
            sum(len(line) for line in lines)
            / len(lines)
        )

    else:

        avg_line_length = 0

    short_line_ratio = (
        len(short_lines) / len(lines)
        if lines
        else 0
    )

    return {
        "text": text,
        "char_count": char_count,
        "block_count": len(text_blocks),
        "line_count": len(lines),
        "short_line_count": len(short_lines),
        "short_line_ratio": short_line_ratio,
        "avg_line_length": avg_line_length,
    }


# ============================================================
# IMAGE BLOCK ANALYSIS
# ============================================================

def analyze_image_blocks(page):
    """
    IMPORTANT:

    This checks BOTH:

        1. page.get_images()
        2. page.get_text("dict") image blocks

    The second method is important because a PDF may contain
    an image block that is not straightforwardly reported by
    page.get_images().

    This is designed to catch pages such as:

        Native text
        +
        Image-based flow/table
        +
        Native text
    """

    page_area = get_page_area(page)

    image_rects = []

    # --------------------------------------------------------
    # METHOD 1
    # page.get_images()
    # --------------------------------------------------------

    try:

        images = page.get_images(full=True)

    except Exception:

        images = []

    for image in images:

        xref = image[0]

        try:

            rects = page.get_image_rects(xref)

            for rect in rects:

                if rect_area(rect) > 0:
                    image_rects.append({
                        "rect": rect,
                        "source": "get_images",
                        "xref": xref,
                    })

        except Exception:

            continue

    # --------------------------------------------------------
    # METHOD 2
    # page.get_text("dict")
    #
    # Image blocks have:
    #
    #     block["type"] == 1
    # --------------------------------------------------------

    try:

        page_dict = page.get_text("dict")

        blocks = page_dict.get(
            "blocks",
            []
        )

    except Exception:

        blocks = []

    for block in blocks:

        if block.get("type") != 1:
            continue

        bbox = block.get("bbox")

        if not bbox:
            continue

        rect = pymupdf.Rect(bbox)

        if rect_area(rect) <= 0:
            continue

        # Avoid counting the same image twice.
        duplicate = False

        for existing in image_rects:

            existing_rect = existing["rect"]

            intersection = (
                existing_rect
                & rect
            )

            if (
                rect_area(intersection)
                > 0.90 * min(
                    rect_area(existing_rect),
                    rect_area(rect)
                )
            ):

                duplicate = True
                break

        if not duplicate:

            image_rects.append({
                "rect": rect,
                "source": "text_dict",
                "xref": None,
            })

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    image_infos = []

    largest_ratio = 0.0

    total_image_area = 0.0

    for item in image_rects:

        rect = item["rect"]

        area = rect_area(rect)

        if page_area <= 0:
            continue

        ratio = area / page_area

        total_image_area += area

        largest_ratio = max(
            largest_ratio,
            ratio
        )

        image_infos.append({
            "rect": rect,
            "ratio": ratio,
            "source": item["source"],
        })

    total_image_ratio = (
        total_image_area / page_area
        if page_area > 0
        else 0
    )

    return {
        "image_count": len(image_infos),
        "largest_image_ratio": largest_ratio,
        "total_image_ratio": total_image_ratio,
        "images": image_infos,
        "get_images_count": len(images),
        "dict_image_block_count": sum(
            1
            for block in blocks
            if block.get("type") == 1
        ),
    }


# ============================================================
# VECTOR / DRAWING ANALYSIS
# ============================================================

def analyze_drawings(page):

    try:

        drawings = page.get_drawings()

    except Exception:

        drawings = []

    horizontal_lines = 0
    vertical_lines = 0
    rectangles = 0
    filled_shapes = 0

    for drawing in drawings:

        rect = drawing.get("rect")

        if not rect:
            continue

        width = abs(rect.width)
        height = abs(rect.height)

        # Rectangle-like shape
        if (
            width > 10
            and height > 10
        ):

            rectangles += 1

        # Horizontal line
        if (
            width > 20
            and height <= 3
        ):

            horizontal_lines += 1

        # Vertical line
        if (
            height > 20
            and width <= 3
        ):

            vertical_lines += 1

        # Filled object
        fill = drawing.get("fill")

        if fill is not None:

            filled_shapes += 1

    return {
        "drawing_count": len(drawings),
        "horizontal_lines": horizontal_lines,
        "vertical_lines": vertical_lines,
        "rectangles": rectangles,
        "filled_shapes": filled_shapes,
    }


# ============================================================
# SPATIAL TEXT ANALYSIS
# ============================================================

def cluster_coordinates(
    values,
    tolerance=COORD_TOLERANCE
):

    if not values:
        return []

    values = sorted(values)

    clusters = [
        [values[0]]
    ]

    for value in values[1:]:

        current = clusters[-1]

        average = (
            sum(current)
            / len(current)
        )

        if abs(
            value - average
        ) <= tolerance:

            current.append(value)

        else:

            clusters.append(
                [value]
            )

    return [
        sum(cluster) / len(cluster)
        for cluster in clusters
    ]


def analyze_spatial_text(page):

    try:

        words = page.get_text("words")

    except Exception:

        words = []

    if not words:

        return {
            "word_count": 0,
            "x_columns": 0,
            "y_rows": 0,
            "column_score": 0,
            "row_score": 0,
        }

    x_positions = []
    y_positions = []

    for word in words:

        if len(word) < 5:
            continue

        x0, y0, x1, y1 = word[:4]

        x_positions.append(x0)
        y_positions.append(y0)

    x_clusters = cluster_coordinates(
        x_positions
    )

    y_clusters = cluster_coordinates(
        y_positions
    )

    column_count = len(x_clusters)

    row_count = len(y_clusters)

    # --------------------------------------------------------
    # Column score
    # --------------------------------------------------------

    if column_count >= 8:

        column_score = 40

    elif column_count >= 6:

        column_score = 30

    elif column_count >= FLOW_MIN_COLUMNS:

        column_score = 20

    else:

        column_score = 0

    # --------------------------------------------------------
    # Row score
    # --------------------------------------------------------

    if row_count >= 10:

        row_score = 40

    elif row_count >= 6:

        row_score = 30

    elif row_count >= FLOW_MIN_ROWS:

        row_score = 20

    else:

        row_score = 0

    return {
        "word_count": len(words),
        "x_columns": column_count,
        "y_rows": row_count,
        "column_score": column_score,
        "row_score": row_score,
    }


# ============================================================
# NATIVE TABLE DETECTION
# ============================================================

def detect_native_tables(page):

    result = {
        "supported": False,
        "lines_tables": 0,
        "text_tables": 0,
        "total_tables": 0,
    }

    if not hasattr(page, "find_tables"):

        return result

    result["supported"] = True

    # --------------------------------------------------------
    # Lines strategy
    # --------------------------------------------------------

    try:

        result_lines = page.find_tables(
            strategy="lines"
        )

        result["lines_tables"] = len(
            result_lines.tables
        )

    except Exception:

        pass

    # --------------------------------------------------------
    # Text strategy
    # --------------------------------------------------------

    try:

        result_text = page.find_tables(
            strategy="text"
        )

        result["text_tables"] = len(
            result_text.tables
        )

    except Exception:

        pass

    result["total_tables"] = max(
        result["lines_tables"],
        result["text_tables"]
    )

    return result


# ============================================================
# TRADITIONAL TABLE SCORE
# ============================================================

def calculate_table_score(
    text_info,
    drawing_info,
    native_table_info
):

    score = 0
    reasons = []

    # Native table
    if native_table_info["total_tables"] > 0:

        score += 70

        reasons.append(
            "native PDF table detected "
            f"({native_table_info['total_tables']})"
        )

    # Lines
    total_lines = (
        drawing_info["horizontal_lines"]
        + drawing_info["vertical_lines"]
    )

    if total_lines >= 6:

        score += 20

        reasons.append(
            f"table-like lines ({total_lines})"
        )

    elif total_lines >= 3:

        score += 10

        reasons.append(
            f"some table-like lines ({total_lines})"
        )

    # Rectangles
    if drawing_info["rectangles"] >= 10:

        score += 15

        reasons.append(
            f"many rectangles "
            f"({drawing_info['rectangles']})"
        )

    elif drawing_info["rectangles"] >= 5:

        score += 8

        reasons.append(
            f"multiple rectangles "
            f"({drawing_info['rectangles']})"
        )

    # Short lines
    if (
        text_info["line_count"] >= 8
        and text_info["short_line_ratio"] >= 0.55
    ):

        score += 10

        reasons.append(
            "short structured text"
        )

    return min(score, 100), reasons


# ============================================================
# FLOW TABLE SCORE
# ============================================================

def calculate_flow_table_score(
    text_info,
    spatial_info,
    drawing_info
):

    score = 0
    reasons = []

    columns = spatial_info["x_columns"]
    rows = spatial_info["y_rows"]

    # --------------------------------------------------------
    # Multiple columns
    # --------------------------------------------------------

    if columns >= 8:

        score += 30

        reasons.append(
            f"many aligned columns ({columns})"
        )

    elif columns >= 6:

        score += 25

        reasons.append(
            f"multiple aligned columns ({columns})"
        )

    elif columns >= 4:

        score += 15

        reasons.append(
            f"multiple aligned columns ({columns})"
        )

    # --------------------------------------------------------
    # Multiple rows
    # --------------------------------------------------------

    if rows >= 10:

        score += 25

        reasons.append(
            f"many aligned rows ({rows})"
        )

    elif rows >= 6:

        score += 20

        reasons.append(
            f"multiple aligned rows ({rows})"
        )

    elif rows >= 3:

        score += 10

        reasons.append(
            f"multiple aligned rows ({rows})"
        )

    # --------------------------------------------------------
    # Text blocks
    # --------------------------------------------------------

    if text_info["block_count"] >= 8:

        score += 10

        reasons.append(
            f"many text blocks "
            f"({text_info['block_count']})"
        )

    # --------------------------------------------------------
    # Vector shapes
    # --------------------------------------------------------

    if drawing_info["drawing_count"] >= 40:

        score += 30

        reasons.append(
            f"many vector shapes "
            f"({drawing_info['drawing_count']})"
        )

    elif drawing_info["drawing_count"] >= 20:

        score += 20

        reasons.append(
            f"many vector shapes "
            f"({drawing_info['drawing_count']})"
        )

    elif drawing_info["drawing_count"] >= 10:

        score += 10

        reasons.append(
            f"multiple vector shapes "
            f"({drawing_info['drawing_count']})"
        )

    # Filled shapes
    if drawing_info["filled_shapes"] >= 8:

        score += 15

        reasons.append(
            f"filled visual shapes "
            f"({drawing_info['filled_shapes']})"
        )

    return min(score, 100), reasons


# ============================================================
# IMAGE CONTENT ANALYSIS
# ============================================================

def calculate_image_content_score(
    text_info,
    image_info
):

    score = 0
    reasons = []

    largest = (
        image_info["largest_image_ratio"]
    )

    total = (
        image_info["total_image_ratio"]
    )

    # --------------------------------------------------------
    # Image exists
    # --------------------------------------------------------

    if image_info["image_count"] > 0:

        score += 30

        reasons.append(
            f"image block detected "
            f"({image_info['image_count']})"
        )

    # --------------------------------------------------------
    # Significant image
    # --------------------------------------------------------

    if largest >= VERY_LARGE_IMAGE_RATIO:

        score += 40

        reasons.append(
            f"very large image "
            f"({safe_percentage(largest)})"
        )

    elif largest >= LARGE_IMAGE_RATIO:

        score += 30

        reasons.append(
            f"large image "
            f"({safe_percentage(largest)})"
        )

    elif largest >= IMAGE_CONTENT_RATIO:

        score += 20

        reasons.append(
            f"significant image "
            f"({safe_percentage(largest)})"
        )

    # --------------------------------------------------------
    # Total image area
    # --------------------------------------------------------

    if total >= 0.60:

        score += 20

        reasons.append(
            f"image content covers "
            f"{safe_percentage(total)} of page"
        )

    elif total >= 0.30:

        score += 10

        reasons.append(
            f"multiple image content "
            f"({safe_percentage(total)} page)"
        )

    # --------------------------------------------------------
    # Native text also exists
    # --------------------------------------------------------

    if text_info["char_count"] >= 100:

        reasons.append(
            "native text also exists"
        )

    return min(score, 100), reasons


# ============================================================
# IMAGE + TEXT CLASSIFICATION
# ============================================================

def classify_image_content(
    text_info,
    image_info,
    spatial_info,
    drawing_info
):

    if image_info["image_count"] == 0:

        return None

    largest = (
        image_info["largest_image_ratio"]
    )

    # --------------------------------------------------------
    # Is there meaningful native text?
    # --------------------------------------------------------

    has_native_text = (
        text_info["char_count"]
        >= MIN_NATIVE_TEXT_CHARS
    )

    # --------------------------------------------------------
    # Image + lots of surrounding text
    #
    # This is the important case for page 7.
    # --------------------------------------------------------

    if has_native_text:

        # ----------------------------------------------------
        # Strong structured surrounding layout
        # ----------------------------------------------------

        if (
            spatial_info["x_columns"] >= 4
            and spatial_info["y_rows"] >= 3
        ):

            return "IMAGE_FLOW_CANDIDATE"

        # ----------------------------------------------------
        # Large image + lots of structured text
        # ----------------------------------------------------

        if (
            largest >= LARGE_IMAGE_RATIO
            and text_info["block_count"] >= 5
        ):

            return "IMAGE_CONTENT"

        # ----------------------------------------------------
        # Large image with normal surrounding text
        # ----------------------------------------------------

        if largest >= IMAGE_CONTENT_RATIO:

            return "IMAGE_CONTENT"

    # --------------------------------------------------------
    # Image only / almost image-only
    # --------------------------------------------------------

    if (
        not has_native_text
        and largest >= VERY_LARGE_IMAGE_RATIO
    ):

        return "OCR"

    # --------------------------------------------------------
    # Image exists but small
    # --------------------------------------------------------

    if largest >= IMAGE_CONTENT_RATIO:

        return "IMAGE_CONTENT"

    return None


# ============================================================
# OCR SCORE
# ============================================================

def calculate_ocr_score(
    text_info,
    image_info
):

    score = 0
    reasons = []

    char_count = text_info["char_count"]

    largest = (
        image_info["largest_image_ratio"]
    )

    # No text
    if char_count < MIN_NATIVE_TEXT_CHARS:

        score += 60

        reasons.append(
            "little/no native text"
        )

    # Very little text
    elif char_count < 100:

        score += 20

        reasons.append(
            "very little native text"
        )

    # Image
    if image_info["image_count"] > 0:

        score += 20

        reasons.append(
            "image content present"
        )

    # Huge image
    if largest >= VERY_LARGE_IMAGE_RATIO:

        score += 30

        reasons.append(
            "very large image"
        )

    return min(score, 100), reasons


# ============================================================
# VISION SCORE
# ============================================================

def calculate_vision_score(
    image_info,
    drawing_info,
    image_class
):

    score = 0
    reasons = []

    largest = (
        image_info["largest_image_ratio"]
    )

    # --------------------------------------------------------
    # Large visual content
    # --------------------------------------------------------

    if largest >= VERY_LARGE_IMAGE_RATIO:

        score += 40

        reasons.append(
            "very large visual content"
        )

    elif largest >= LARGE_IMAGE_RATIO:

        score += 25

        reasons.append(
            "large visual content"
        )

    # --------------------------------------------------------
    # Vector shapes
    # --------------------------------------------------------

    if drawing_info["drawing_count"] >= 40:

        score += 30

        reasons.append(
            "many vector drawings"
        )

    elif drawing_info["drawing_count"] >= 20:

        score += 20

        reasons.append(
            "many vector drawings"
        )

    # --------------------------------------------------------
    # Image content
    # --------------------------------------------------------

    if image_class in {
        "IMAGE_CONTENT",
        "IMAGE_FLOW_CANDIDATE",
    }:

        score += 20

        reasons.append(
            "embedded visual content"
        )

    return min(score, 100), reasons


# ============================================================
# FINAL PAGE CLASSIFICATION
# ============================================================

def classify_page(
    text_info,
    image_info,
    table_score,
    flow_table_score,
    image_class,
    ocr_score,
    vision_score
):

    # ========================================================
    # 1. IMAGE + NATIVE TEXT
    #
    # This gets priority over NORMAL.
    # ========================================================

    if image_class == "IMAGE_FLOW_CANDIDATE":

        return "IMAGE_FLOW_CANDIDATE"

    if image_class == "IMAGE_CONTENT":

        return "IMAGE_CONTENT"

    # ========================================================
    # 2. OCR
    # ========================================================

    if (
        ocr_score >= OCR_SCORE_THRESHOLD
        and text_info["char_count"]
        < MIN_NATIVE_TEXT_CHARS
    ):

        return "OCR"

    # ========================================================
    # 3. FLOW TABLE
    # ========================================================

    if (
        flow_table_score
        >= FLOW_TABLE_SCORE_THRESHOLD
    ):

        return "FLOW_TABLE"

    # ========================================================
    # 4. NORMAL TABLE
    # ========================================================

    if (
        table_score
        >= TABLE_SCORE_THRESHOLD
    ):

        return "TABLE"

    # ========================================================
    # 5. VISION
    # ========================================================

    if (
        vision_score
        >= VISION_SCORE_THRESHOLD
    ):

        return "VISION"

    # ========================================================
    # 6. REVIEW
    # ========================================================

    if (
        table_score >= 40
        or flow_table_score >= 40
        or ocr_score >= 40
        or vision_score >= 40
    ):

        return "REVIEW"

    # ========================================================
    # 7. NORMAL
    # ========================================================

    return "NORMAL"


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage: python audit_pdf.py <path_to_pdf>"
        )

        sys.exit(1)

    pdf_path = sys.argv[1]

    print("=" * 105)
    print("PDF STRUCTURE AUDIT FOR RAG")
    print("=" * 105)

    print(
        f"File: {pdf_path}"
    )

    print()

    try:

        doc = pymupdf.open(
            pdf_path
        )

    except Exception as e:

        print(
            f"ERROR: Cannot open PDF: {e}"
        )

        sys.exit(1)

    total_pages = len(doc)

    results = []

    # ========================================================
    # PROCESS EVERY PAGE
    # ========================================================

    for i, page in enumerate(doc):

        page_num = i + 1

        # ----------------------------------------------------
        # Analyse page
        # ----------------------------------------------------

        text_info = analyze_text(
            page
        )

        spatial_info = analyze_spatial_text(
            page
        )

        image_info = analyze_image_blocks(
            page
        )

        drawing_info = analyze_drawings(
            page
        )

        native_table_info = (
            detect_native_tables(
                page
            )
        )

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        table_score, table_reasons = (
            calculate_table_score(
                text_info,
                drawing_info,
                native_table_info
            )
        )

        flow_table_score, flow_reasons = (
            calculate_flow_table_score(
                text_info,
                spatial_info,
                drawing_info
            )
        )

        image_score, image_reasons = (
            calculate_image_content_score(
                text_info,
                image_info
            )
        )

        image_class = (
            classify_image_content(
                text_info,
                image_info,
                spatial_info,
                drawing_info
            )
        )

        ocr_score, ocr_reasons = (
            calculate_ocr_score(
                text_info,
                image_info
            )
        )

        vision_score, vision_reasons = (
            calculate_vision_score(
                image_info,
                drawing_info,
                image_class
            )
        )

        # ----------------------------------------------------
        # Final classification
        # ----------------------------------------------------

        classification = classify_page(
            text_info,
            image_info,
            table_score,
            flow_table_score,
            image_class,
            ocr_score,
            vision_score
        )

        results.append({

            "page": page_num,

            "text": text_info,

            "spatial": spatial_info,

            "image": image_info,

            "drawing": drawing_info,

            "native_table": native_table_info,

            "table_score": table_score,

            "table_reasons": table_reasons,

            "flow_table_score": flow_table_score,

            "flow_reasons": flow_reasons,

            "image_score": image_score,

            "image_reasons": image_reasons,

            "image_class": image_class,

            "ocr_score": ocr_score,

            "ocr_reasons": ocr_reasons,

            "vision_score": vision_score,

            "vision_reasons": vision_reasons,

            "classification": classification,
        })

    doc.close()

    # ========================================================
    # SUMMARY
    # ========================================================

    print("=" * 105)
    print("SUMMARY")
    print("=" * 105)

    print(
        f"Total pages: {total_pages}"
    )

    print()

    classifications = {}

    for result in results:

        category = result[
            "classification"
        ]

        classifications[category] = (
            classifications.get(
                category,
                0
            ) + 1
        )

    categories = [
        "NORMAL",
        "TABLE",
        "FLOW_TABLE",
        "IMAGE_CONTENT",
        "IMAGE_FLOW_CANDIDATE",
        "OCR",
        "VISION",
        "REVIEW",
    ]

    for category in categories:

        print(
            f"{category:<25}: "
            f"{classifications.get(category, 0)}"
        )

    # ========================================================
    # IMAGE DETECTION SUMMARY
    # ========================================================

    print()
    print("=" * 105)
    print("IMAGE CONTENT DETECTION")
    print("=" * 105)

    image_pages = [
        result
        for result in results
        if result["image"]["image_count"] > 0
    ]

    print(
        f"Pages containing detected image blocks: "
        f"{len(image_pages)}"
    )

    if image_pages:

        print()

        for result in image_pages:

            print(
                f"Page {result['page']:>3} | "
                f"{result['classification']:<25} | "
                f"Images={result['image']['image_count']:>2} | "
                f"get_images={result['image']['get_images_count']:>2} | "
                f"dict_blocks={result['image']['dict_image_block_count']:>2} | "
                f"Largest="
                f"{safe_percentage(result['image']['largest_image_ratio'])} | "
                f"Total="
                f"{safe_percentage(result['image']['total_image_ratio'])}"
            )

    # ========================================================
    # IMAGE FLOW CANDIDATES
    # ========================================================

    print()
    print("=" * 105)
    print("IMAGE-BASED FLOW / STRUCTURED VISUAL CANDIDATES")
    print("=" * 105)

    image_flow_pages = [
        result
        for result in results
        if result["classification"]
        == "IMAGE_FLOW_CANDIDATE"
    ]

    if image_flow_pages:

        for result in image_flow_pages:

            print(
                f"Page {result['page']:>3} | "
                f"Image="
                f"{safe_percentage(result['image']['largest_image_ratio'])} | "
                f"Text={result['text']['char_count']:>5} | "
                f"Blocks={result['text']['block_count']:>3} | "
                f"Columns={result['spatial']['x_columns']:>3} | "
                f"Rows={result['spatial']['y_rows']:>3}"
            )

            print(
                "       "
                "This page contains native text plus "
                "image content with structured layout."
            )

    else:

        print(
            "No strong image-flow candidates detected."
        )

    # ========================================================
    # ALL IMAGE CONTENT PAGES
    # ========================================================

    print()
    print("=" * 105)
    print("ALL IMAGE CONTENT PAGES")
    print("=" * 105)

    if image_pages:

        print(
            [
                result["page"]
                for result in image_pages
            ]
        )

    else:

        print(
            "No image content detected."
        )

    # ========================================================
    # TRADITIONAL TABLES
    # ========================================================

    print()
    print("=" * 105)
    print("TRADITIONAL TABLE CANDIDATES")
    print("=" * 105)

    table_pages = [
        result
        for result in results
        if result["classification"]
        == "TABLE"
    ]

    if table_pages:

        for result in table_pages:

            print(
                f"Page {result['page']:>3} | "
                f"TABLE={result['table_score']:>3} | "
                f"Text={result['text']['char_count']:>5} | "
                f"Blocks={result['text']['block_count']:>3} | "
                f"Drawings={result['drawing']['drawing_count']:>3}"
            )

            if result["table_reasons"]:

                print(
                    "       "
                    + "; ".join(
                        result["table_reasons"]
                    )
                )

    else:

        print(
            "No traditional table candidates."
        )

    # ========================================================
    # FLOW TABLES
    # ========================================================

    print()
    print("=" * 105)
    print("NATIVE FLOW / VISUAL TABLE CANDIDATES")
    print("=" * 105)

    flow_pages = [
        result
        for result in results
        if result["classification"]
        == "FLOW_TABLE"
    ]

    if flow_pages:

        for result in flow_pages:

            print(
                f"Page {result['page']:>3} | "
                f"FLOW={result['flow_table_score']:>3} | "
                f"Columns={result['spatial']['x_columns']:>3} | "
                f"Rows={result['spatial']['y_rows']:>3} | "
                f"Drawings={result['drawing']['drawing_count']:>3}"
            )

    else:

        print(
            "No native flow-table candidates."
        )

    # ========================================================
    # OCR
    # ========================================================

    print()
    print("=" * 105)
    print("OCR CANDIDATES")
    print("=" * 105)

    ocr_pages = [
        result
        for result in results
        if result["classification"]
        == "OCR"
    ]

    if ocr_pages:

        for result in ocr_pages:

            print(
                f"Page {result['page']:>3} | "
                f"OCR={result['ocr_score']:>3} | "
                f"Text={result['text']['char_count']:>5} | "
                f"Images={result['image']['image_count']:>2} | "
                f"Largest="
                f"{safe_percentage(result['image']['largest_image_ratio'])}"
            )

    else:

        print(
            "No strong OCR candidates."
        )

    # ========================================================
    # VISION
    # ========================================================

    print()
    print("=" * 105)
    print("VISION CANDIDATES")
    print("=" * 105)

    vision_pages = [
        result
        for result in results
        if result["classification"]
        == "VISION"
    ]

    if vision_pages:

        for result in vision_pages:

            print(
                f"Page {result['page']:>3} | "
                f"VISION={result['vision_score']:>3} | "
                f"Images={result['image']['image_count']:>2} | "
                f"Drawings="
                f"{result['drawing']['drawing_count']:>3}"
            )

    else:

        print(
            "No strong Vision candidates."
        )

    # ========================================================
    # MANUAL REVIEW
    # ========================================================

    print()
    print("=" * 105)
    print("PAGES RECOMMENDED FOR MANUAL REVIEW")
    print("=" * 105)

    review_categories = {
        "TABLE",
        "FLOW_TABLE",
        "IMAGE_CONTENT",
        "IMAGE_FLOW_CANDIDATE",
        "OCR",
        "VISION",
        "REVIEW",
    }

    review_pages = [
        result
        for result in results
        if result["classification"]
        in review_categories
    ]

    if review_pages:

        for result in review_pages:

            print(
                f"Page {result['page']:>3} | "
                f"{result['classification']:<25} | "
                f"TABLE={result['table_score']:>3} | "
                f"FLOW={result['flow_table_score']:>3} | "
                f"IMAGE={result['image_score']:>3} | "
                f"OCR={result['ocr_score']:>3}"
            )

    else:

        print(
            "No pages require manual review."
        )

    # ========================================================
    # RAG RECOMMENDATION
    # ========================================================

    print()
    print("=" * 105)
    print("RAG PROCESSING RECOMMENDATION")
    print("=" * 105)

    print(
        "NORMAL"
    )
    print(
        "  -> Native text extraction"
    )

    print()
    print(
        "TABLE"
    )
    print(
        "  -> Table-aware extraction"
    )

    print()
    print(
        "FLOW_TABLE"
    )
    print(
        "  -> Preserve spatial structure"
    )

    print()
    print(
        "IMAGE_CONTENT"
    )
    print(
        "  -> Inspect image content"
    )

    print()
    print(
        "IMAGE_FLOW_CANDIDATE"
    )
    print(
        "  -> Treat image as important visual content"
    )
    print(
        "  -> OCR/Vision or image-aware extraction"
    )

    print()
    print(
        "OCR"
    )
    print(
        "  -> Local Tesseract OCR"
    )

    print()
    print(
        "VISION"
    )
    print(
        "  -> Visual/Vision processing"
    )

    print()
    print(
        "REVIEW"
    )
    print(
        "  -> Manual inspection"
    )

    print()
    print("=" * 105)
    print("IMPORTANT")
    print("=" * 105)

    print(
        "IMAGE_FLOW_CANDIDATE does NOT guarantee that the "
        "image is a flowchart."
    )

    print(
        "It means the page contains image content together "
        "with structured native text/layout."
    )

    print(
        "For your RAG pipeline, these pages should NOT be "
        "treated as normal text-only pages."
    )

    print("=" * 105)


if __name__ == "__main__":
    main()