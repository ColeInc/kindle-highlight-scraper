import cv2
import pytesseract
import pyautogui
import os
import json
import datetime
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

# Configure pytesseract (path to Tesseract executable may vary)
if os.name == 'nt':  # Windows
    # Adjust path as needed
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:  # macOS/Linux
    # For macOS with Homebrew
    pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
    # or '/usr/bin/tesseract' for Linux

# Directories
SCREENSHOT_DIR = "screenshots"
OUTPUT_DIR = "extracted_highlights"
DEBUG_DIR = "debug_images"  # For saving debug images

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)  # Create debug directory

# ------------------- Extraction Phase -------------------


def extract_next_page():
    """Simulates pressing the right-arrow key."""
    pyautogui.press('right')
    time.sleep(0.5)  # wait for page change


def extract_screenshot(page_num):
    """Captures and saves a screenshot."""
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    screenshot_path = os.path.join(
        SCREENSHOT_DIR, f"screenshot_{timestamp}_page{page_num}.png")
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)
    return screenshot_path


def extraction_orchestrater(pages_to_capture=5):
    """Main extraction orchestrator."""
    paths = []

    # Add a timeout before starting the extraction process
    print("Waiting 2 seconds before starting extraction...")
    time.sleep(2)
    for page_num in range(1, pages_to_capture + 1):
        extract_next_page()
        screenshot_path = extract_screenshot(page_num)
        paths.append(screenshot_path)
        print(f"Captured screenshot: {screenshot_path}")
    return paths

# ------------------- Processing Phase -------------------


def detect_highlights(image, color_lower, color_upper):
    """Detects highlighted areas based on color range."""
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, color_lower, color_upper)
    return mask


def detect_columns(image):
    """Detect columns in a book page using improved algorithm."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get text regions
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Apply morphology to connect text within columns
    kernel = np.ones((5, 50), np.uint8)  # Horizontal kernel
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Sum pixel values along vertical axis to get horizontal profile
    h_profile = np.sum(morph, axis=0)

    # Normalize and smooth the profile
    if np.max(h_profile) > 0:
        h_profile = h_profile / np.max(h_profile)
    h_profile_smooth = np.convolve(h_profile, np.ones(100)/100, mode='same')

    # Save horizontal profile for debugging
    plt_h = np.zeros((300, len(h_profile_smooth)), dtype=np.uint8)
    for i in range(len(h_profile_smooth)):
        cv2.line(plt_h, (i, 300), (i, 300 -
                 int(h_profile_smooth[i] * 300)), 255, 1)

    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    profile_path = os.path.join(DEBUG_DIR, f"h_profile_{timestamp}.png")
    cv2.imwrite(profile_path, plt_h)

    # Find valleys (potential column separators)
    width = image.shape[1]

    # Use adaptive threshold to find valleys
    # Calculate mean and standard deviation, and use them to set threshold
    non_zero_profile = h_profile_smooth[h_profile_smooth > 0.05]
    if len(non_zero_profile) > 0:
        mean_density = np.mean(non_zero_profile)
        std_density = np.std(non_zero_profile)
        # Set threshold below mean (valleys are lower than peaks)
        valley_threshold = max(0.15, mean_density - 1.5 * std_density)
    else:
        # Default threshold if calculation fails
        valley_threshold = 0.15

    print(f"Valley threshold: {valley_threshold}")

    # Find significant valleys
    valleys = []
    in_valley = False
    valley_start = 0
    min_val = 1.0
    min_idx = 0

    # Ignore the edges of the image (10% on each side)
    edge_margin = int(width * 0.1)

    for i in range(edge_margin, width - edge_margin):
        if h_profile_smooth[i] < valley_threshold:
            if not in_valley:
                valley_start = i
                min_val = h_profile_smooth[i]
                min_idx = i
                in_valley = True
            elif h_profile_smooth[i] < min_val:
                min_val = h_profile_smooth[i]
                min_idx = i
        else:
            if in_valley:
                # Found valley end
                valley_end = i
                # Use the lowest point of the valley
                valleys.append(min_idx)
                in_valley = False

    # Handle case where last column extends to edge
    if in_valley:
        valleys.append(min_idx)

    # Filter out valleys that are too close to each other
    if len(valleys) > 1:
        filtered_valleys = [valleys[0]]
        for i in range(1, len(valleys)):
            if valleys[i] - filtered_valleys[-1] > width * 0.15:  # Minimum 15% of width apart
                filtered_valleys.append(valleys[i])
        valleys = filtered_valleys

    # Based on valleys, determine column boundaries
    columns = []

    # Check if we actually found any significant valleys
    if len(valleys) == 0:
        # Single column
        columns.append((0, width))
    else:
        # First column starts at 0
        columns.append((0, valleys[0]))

        # Middle columns
        for i in range(len(valleys)-1):
            columns.append((valleys[i], valleys[i+1]))

        # Last column ends at page width
        columns.append((valleys[-1], width))

    # Apply minimum width filter (at least 10% of page width)
    min_col_width = width // 10
    filtered_columns = []
    for col_start, col_end in columns:
        if col_end - col_start >= min_col_width:
            filtered_columns.append((col_start, col_end))

    # If all columns were filtered out, fall back to single column
    if not filtered_columns:
        filtered_columns = [(0, width)]

    # Save debug image with column boundaries
    debug_img = image.copy()
    for idx, (col_start, col_end) in enumerate(filtered_columns):
        # Green line for start, red line for end
        cv2.line(debug_img, (col_start, 0),
                 (col_start, image.shape[0]), (0, 255, 0), 2)
        cv2.line(debug_img, (col_end, 0),
                 (col_end, image.shape[0]), (0, 0, 255), 2)

        # Add column number label
        cv2.putText(debug_img, f"Col {idx+1}", (col_start + 20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Draw threshold line on profile plot for debugging
    threshold_img = plt_h.copy()
    threshold_y = 300 - int(valley_threshold * 300)
    cv2.line(threshold_img, (0, threshold_y), (width, threshold_y), 128, 2)
    cv2.imwrite(os.path.join(
        DEBUG_DIR, f"threshold_{timestamp}.png"), threshold_img)

    # Save final debug image
    debug_path = os.path.join(DEBUG_DIR, f"columns_{timestamp}.png")
    cv2.imwrite(debug_path, debug_img)

    print(f"Detected {len(filtered_columns)} columns in the image")
    for i, (start, end) in enumerate(filtered_columns):
        print(f"  Column {i+1}: {start}-{end} (width: {end-start}px)")

    return filtered_columns


def extract_full_column_text(image, col_start, col_end):
    """Extract all text from a column, regardless of highlighting."""
    # Extract the column region
    col_img = image[:, col_start:col_end].copy()

    # Preprocess the image for better OCR
    gray = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Apply mild denoising
    denoised = cv2.fastNlMeansDenoising(contrast, None, 10, 7, 21)

    # Use adaptiveThreshold for better results with text
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Upsample image for better OCR - double the size
    height, width = binary.shape
    upscaled = cv2.resize(binary, (width*2, height*2),
                          interpolation=cv2.INTER_CUBIC)

    # Save debug image
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    debug_path = os.path.join(
        DEBUG_DIR, f"col_binary_{timestamp}_{col_start}.png")
    cv2.imwrite(debug_path, binary)

    # Convert to PIL image for Tesseract
    pil_img = Image.fromarray(upscaled)

    # Extract text with Tesseract using column-specific settings
    # Using PSM 6 (Assume a single uniform block of text) for better results with book columns
    config = '--oem 3 --psm 6 -l eng --dpi 300'
    text = pytesseract.image_to_string(pil_img, config=config)

    # Split text into lines and store with vertical position
    lines_with_positions = []

    # Use tesseract to get bounding boxes for each line
    boxes = pytesseract.image_to_data(
        pil_img, config=config, output_type=pytesseract.Output.DICT)

    prev_block_num = -1
    line_text = ""
    line_y = 0

    # Process box data to extract lines with positions
    for i in range(len(boxes['text'])):
        # Skip empty entries
        if boxes['text'][i].strip() == '':
            continue

        # New line or new block
        if boxes['block_num'][i] != prev_block_num or boxes['line_num'][i] != boxes['line_num'][i-1] if i > 0 else True:
            # Save previous line if it exists
            if line_text:
                # Scale the y-position back to original image size
                original_y = line_y // 2  # Scale back from upscaled image
                lines_with_positions.append((original_y, line_text.strip()))
                line_text = ""

            # Start new line
            line_text = boxes['text'][i]
            line_y = boxes['top'][i]
            prev_block_num = boxes['block_num'][i]
        else:
            # Continue current line
            line_text += " " + boxes['text'][i]

    # Add the last line if it exists
    if line_text:
        original_y = line_y // 2
        lines_with_positions.append((original_y, line_text.strip()))

    # Save line positions for debugging
    with open(os.path.join(DEBUG_DIR, f"lines_{timestamp}_{col_start}.txt"), 'w') as f:
        for y, line in lines_with_positions:
            f.write(f"Y: {y} - {line}\n")

    return lines_with_positions


def get_highlighted_regions(image, highlight_mask):
    """Get bounding boxes of highlighted regions."""
    # Dilate the mask to connect nearby highlighted areas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    dilated_mask = cv2.dilate(highlight_mask, kernel, iterations=1)

    # Find contours in the dilated mask
    contours, _ = cv2.findContours(
        dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Skip very small regions
        if w < 30 or h < 15:
            continue
        regions.append((x, y, w, h))

    return regions


def match_text_to_highlights(lines_with_positions, highlight_regions):
    """Match extracted text lines to highlight regions more accurately."""
    if not lines_with_positions or not highlight_regions:
        return []

    # Sort lines by vertical position
    lines_with_positions.sort(key=lambda x: x[0])

    # Create debug information
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    with open(os.path.join(DEBUG_DIR, f"highlight_matching_{timestamp}.txt"), 'w') as f:
        f.write("Line positions:\n")
        for y, line in lines_with_positions:
            f.write(f"Y: {y} - {line}\n")

        f.write("\nHighlight regions:\n")
        for i, (x, y, w, h) in enumerate(highlight_regions):
            f.write(
                f"Region {i+1}: x={x}, y={y}, w={w}, h={h}, bottom={y+h}\n")

    # Calculate average line height
    if len(lines_with_positions) > 1:
        line_heights = []
        for i in range(1, len(lines_with_positions)):
            height = lines_with_positions[i][0] - lines_with_positions[i-1][0]
            if height > 0:
                line_heights.append(height)

        avg_line_height = np.mean(line_heights) if line_heights else 20
    else:
        avg_line_height = 20  # Default value

    # Match highlighted regions to lines
    highlighted_lines = []

    for x, y, w, h in highlight_regions:
        region_top = y
        region_bottom = y + h

        # Debug: log region bounds
        with open(os.path.join(DEBUG_DIR, f"highlight_matching_{timestamp}.txt"), 'a') as f:
            f.write(
                f"\nMatching for region at y={y}, height={h} (bottom={region_bottom}):\n")

        # Find all lines that overlap with this region
        matching_lines = []
        for line_y, line_text in lines_with_positions:
            # Estimate line height and calculate its bottom
            line_bottom = line_y + avg_line_height

            # Check if line overlaps with highlight region
            # A line overlaps if:
            # 1. The line's top is between the region's top and bottom
            # 2. The line's bottom is between the region's top and bottom
            # 3. The line completely contains the region
            if (region_top <= line_y <= region_bottom) or \
               (region_top <= line_bottom <= region_bottom) or \
               (line_y <= region_top and line_bottom >= region_bottom):
                matching_lines.append(line_text)

                # Debug: log matched line
                with open(os.path.join(DEBUG_DIR, f"highlight_matching_{timestamp}.txt"), 'a') as f:
                    f.write(
                        f"  MATCHED: Y={line_y}, Bottom={line_bottom} - {line_text}\n")
            else:
                # Debug: log rejected line
                with open(os.path.join(DEBUG_DIR, f"highlight_matching_{timestamp}.txt"), 'a') as f:
                    f.write(
                        f"  REJECTED: Y={line_y}, Bottom={line_bottom} - {line_text}\n")

        # Add all matching lines
        highlighted_lines.extend(matching_lines)

    # Remove duplicates while preserving order
    seen = set()
    result = []
    for line in highlighted_lines:
        if line not in seen:
            seen.add(line)
            result.append(line)

    return result


def process_highlighted_column(image, highlight_mask, col_start, col_end, color):
    """Process a column with highlights using the hybrid approach."""
    # Extract column region and mask
    col_img = image[:, col_start:col_end].copy()
    col_mask = highlight_mask[:, col_start:col_end].copy()

    # Save debug image of mask
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    debug_mask_path = os.path.join(
        DEBUG_DIR, f"col_mask_{timestamp}_{col_start}.png")
    cv2.imwrite(debug_mask_path, col_mask)

    # Get all text from the column with positions
    lines_with_positions = extract_full_column_text(
        col_img, 0, col_end - col_start)

    # Get highlighted regions in column coordinates
    highlighted_regions = get_highlighted_regions(col_img, col_mask)

    # Draw the detected highlights on debug image
    debug_img = col_img.copy()
    for i, (x, y, w, h) in enumerate(highlighted_regions):
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_img, f"{i+1}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Draw each detected line
    for i, (y, line) in enumerate(lines_with_positions):
        # Draw a line at the position
        cv2.line(debug_img, (0, y), (col_img.shape[1], y), (255, 0, 0), 1)
        # Add text index
        cv2.putText(debug_img, f"L{i+1}", (5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    debug_regions_path = os.path.join(
        DEBUG_DIR, f"col_regions_{timestamp}_{col_start}.png")
    cv2.imwrite(debug_regions_path, debug_img)

    # Match text lines to highlighted regions
    highlighted_lines = match_text_to_highlights(
        lines_with_positions, highlighted_regions)

    # Create highlight objects
    highlights = []
    for line in highlighted_lines:
        if line.strip():
            highlight = {
                "text": line.strip(),
                "color": color,
                "heading": is_chapter_title(line)
            }
            highlights.append(highlight)

    return highlights


def is_chapter_title(text):
    """Heuristic to detect chapter titles."""
    return text.lower().startswith("chapter") or text.isupper()


def process_single_screenshot(screenshot_path, highlights):
    """Processes a single screenshot to extract highlighted text."""
    image = cv2.imread(screenshot_path)
    if image is None:
        print(f"Error: Could not read image at {screenshot_path}")
        return

    # Save original image for reference
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    cv2.imwrite(os.path.join(DEBUG_DIR, f"original_{timestamp}.png"), image)

    # Define HSV color ranges for highlights (adjust as necessary)
    blue_lower, blue_upper = np.array([85, 50, 150]), np.array([115, 150, 255])
    red_lower1, red_upper1 = np.array([0, 20, 200]), np.array(
        [10, 80, 255])  # Adjusted for lighter pink
    red_lower2, red_upper2 = np.array([160, 20, 200]), np.array(
        [179, 80, 255])  # Adjusted for lighter pink

    # Create masks
    blue_mask = detect_highlights(image, blue_lower, blue_upper)
    red_mask1 = detect_highlights(image, red_lower1, red_upper1)
    red_mask2 = detect_highlights(image, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Save debug images of masks
    cv2.imwrite(os.path.join(
        DEBUG_DIR, f"blue_mask_{timestamp}.png"), blue_mask)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"red_mask_{timestamp}.png"), red_mask)

    # Detect columns in the image
    columns = detect_columns(image)

    # Fall back to single column if detection failed
    if not columns:
        width = image.shape[1]
        columns = [(0, width)]
        print("Column detection failed, falling back to single column")

    try:
        # Process each column
        all_blue_highlights = []
        all_red_highlights = []

        for col_start, col_end in columns:
            # Process blue highlights
            blue_highlights = process_highlighted_column(
                image, blue_mask, col_start, col_end, "blue")
            all_blue_highlights.extend(blue_highlights)

            # Process red highlights
            red_highlights = process_highlighted_column(
                image, red_mask, col_start, col_end, "red")
            all_red_highlights.extend(red_highlights)

        # Print results
        print(f"Found {len(all_blue_highlights)} blue highlighted texts")
        print(f"Found {len(all_red_highlights)} red highlighted texts")

        for i, h in enumerate(all_blue_highlights):
            print(f"Blue highlight {i+1}: {h['text']}")
            highlights.append(h)

        for i, h in enumerate(all_red_highlights):
            print(f"Red highlight {i+1}: {h['text']}")
            highlights.append(h)

    except Exception as e:
        print(f"Error processing highlights in {screenshot_path}: {str(e)}")
        import traceback
        traceback.print_exc()


def screenshot_data_orchestrator():
    """Main orchestrator for processing screenshots."""
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    highlights = []  # Single list for all highlights

    # Process a specific file first
    specific_file = os.path.join(
        SCREENSHOT_DIR, "screenshot_2025.03.29_15.52.37_page2.png")
    if os.path.exists(specific_file):
        print(f"Processing specific file: {specific_file}")
        process_single_screenshot(specific_file, highlights)

    # UNCOMMENT DIS STUFF:
    # Get all screenshot files from the directory and sort by creation time
    # screenshot_paths = [os.path.join(SCREENSHOT_DIR, f) for f in os.listdir(
    #     SCREENSHOT_DIR) if f.endswith('.png')]
    # # Sort by creation time
    # screenshot_paths.sort(key=lambda x: os.path.getctime(x))

    # Process all screenshots
    # for screenshot in screenshot_paths:
    #     print(f"Processing {screenshot}")
    #     process_single_screenshot(screenshot, highlights)

    # Save to JSON with updated timestamp format
    output_path = os.path.join(OUTPUT_DIR, f"highlights_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(highlights, f, indent=2)

    print(f"Highlights saved to {output_path}")
    print(f"Extracted {len(highlights)} highlights")


# ------------------- Main Function -------------------


def main(pages_to_capture=5):
    """Main orchestrator function to trigger all."""
    # screenshot_paths = extraction_orchestrater(pages_to_capture)
    screenshot_data_orchestrator()


if __name__ == "__main__":
    # Adjust number of pages as required
    main(pages_to_capture=5)
