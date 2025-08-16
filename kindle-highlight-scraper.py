import cv2
import pyautogui
import os
import json
import datetime
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import logging
from typing import Dict, List, Any

# Setup logging with detailed formatting FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kindle_scraper.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Starting Kindle Highlight Scraper...")
logger.info("Loading environment variables from .env file")
load_dotenv()

# Configure Gemini API
logger.info("Configuring Gemini API connection")
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

logger.info("Gemini API key loaded successfully")
genai.configure(api_key=api_key)
logger.info("Gemini API configured successfully")

# Directories
SCREENSHOT_DIR = "screenshots"
OUTPUT_DIR = "extracted_highlights"
DEBUG_DIR = "debug_images"  # For saving debug images

logger.info(f"Setting up directories: {SCREENSHOT_DIR}, {OUTPUT_DIR}, {DEBUG_DIR}")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)  # Create debug directory
logger.info("All directories created successfully")

# ------------------- Extraction Phase -------------------


def extract_next_page():
    """Simulates pressing the right-arrow key."""
    logger.debug("Simulating right arrow key press for next page")
    pyautogui.press('right')
    time.sleep(0.5)  # wait for page change
    logger.debug("Page turn completed")


def extract_screenshot(page_num):
    """Captures and saves a screenshot."""
    logger.info(f"Capturing screenshot for page {page_num}")
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    screenshot_path = os.path.join(
        SCREENSHOT_DIR, f"screenshot_{timestamp}_page{page_num}.png")
    logger.debug(f"Screenshot will be saved to: {screenshot_path}")
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)
    logger.info(f"Screenshot saved successfully: {screenshot_path}")
    return screenshot_path


def extraction_orchestrater(pages_to_capture=5):
    """Main extraction orchestrator."""
    logger.info(f"Starting screenshot extraction for {pages_to_capture} pages")
    paths = []

    # Add a timeout before starting the extraction process
    logger.info("Waiting 2 seconds before starting extraction process...")
    print("Waiting 2 seconds before starting extraction...")
    time.sleep(2)
    
    for page_num in range(1, pages_to_capture + 1):
        logger.info(f"Processing page {page_num} of {pages_to_capture}")
        extract_next_page()
        screenshot_path = extract_screenshot(page_num)
        paths.append(screenshot_path)
        logger.info(f"Successfully captured page {page_num}: {screenshot_path}")
        print(f"Captured screenshot: {screenshot_path}")
    
    logger.info(f"Screenshot extraction completed. Total pages captured: {len(paths)}")
    return paths

# ------------------- Processing Phase -------------------

# Define the schema for Gemini API response
GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "extracted_text": {
            "type": "string",
            "description": "Complete text content from the image"
        },
        "columns_detected": {
            "type": "integer",
            "description": "Number of columns detected in the layout (1 or 2)"
        },
        "blue_highlights": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Array of complete blue highlighted text segments, with multi-line highlights grouped together"
        },
        "red_highlights": {
            "type": "array", 
            "items": {"type": "string"},
            "description": "Array of complete red/pink highlighted text segments, with multi-line highlights grouped together"
        }
    },
    "required": ["extracted_text", "columns_detected", "blue_highlights", "red_highlights"]
}


def extract_text_with_gemini(image_path: str) -> Dict[str, Any]:
    """
    Extract text and highlights from an image using Gemini 2.5 Flash API.
    
    Args:
        image_path: Path to the image file to process
        
    Returns:
        Dictionary containing extracted text, blue highlights, and red highlights
    """
    try:
        logger.info(f"🔍 Starting Gemini analysis for image: {os.path.basename(image_path)}")
        
        # Validate image file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check file size
        file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
        logger.debug(f"Image file size: {file_size:.2f} MB")
        
        # Load and prepare the image
        logger.debug("Loading image file...")
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        
        # Convert image to PIL format for Gemini
        logger.debug("Converting image to PIL format...")
        pil_image = Image.open(image_path)
        logger.debug(f"Image dimensions: {pil_image.size[0]}x{pil_image.size[1]} pixels")
        
        # Initialize Gemini model
        logger.info("🤖 Initializing Gemini 2.0 Flash model...")
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Create the prompt
        logger.debug("Preparing AI prompt for highlight detection...")
        prompt = """
        Analyze this book/document page image and extract highlighted text with proper grouping and ordering.

        CRITICAL INSTRUCTIONS:

        1. COLUMN DETECTION:
           - Determine if this is a single-column or two-column layout
           - Set "columns_detected" to 1 or 2 accordingly

        2. HIGHLIGHT GROUPING:
           - Group consecutive highlighted lines into single entries
           - If a highlight spans multiple lines, combine them into ONE complete text segment
           - Do NOT split multi-line highlights into separate array items
           - Example: If 3 consecutive lines are highlighted blue, return as 1 blue highlight entry

        3. READING ORDER (VERY IMPORTANT):
           - For single-column: Order highlights from top to bottom as they appear
           - For two-column: Read LEFT column completely from top to bottom, THEN right column top to bottom
           - The arrays should reflect the exact reading order someone would follow when reading the page
           - Maintain chronological order as if reading the document naturally

        4. TEXT PROCESSING:
           - Join multi-line highlights with spaces, preserving sentence structure
           - Remove excessive whitespace but maintain readability
           - Keep punctuation and capitalization intact

        5. COLOR DETECTION:
           - Blue highlights: Any blue, cyan, or blue-tinted highlighting
           - Red highlights: Any red, pink, magenta, or red-tinted highlighting
           - Be generous in color detection - include lighter shades

        Return JSON with:
        - extracted_text: Complete page text
        - columns_detected: 1 or 2
        - blue_highlights: Array of complete blue highlight segments in reading order
        - red_highlights: Array of complete red highlight segments in reading order

        If no highlights found in a color, return empty array for that field.
        """
        
        # Generate content with schema
        logger.info("📤 Sending image to Gemini API for analysis...")
        response = model.generate_content(
            [prompt, pil_image],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=GEMINI_RESPONSE_SCHEMA
            )
        )
        
        logger.info("📥 Received response from Gemini API")
        logger.debug(f"Response length: {len(response.text)} characters")
        
        # Parse the response
        logger.debug("Parsing JSON response...")
        result = json.loads(response.text)
        
        # Log detailed results
        blue_count = len(result.get('blue_highlights', []))
        red_count = len(result.get('red_highlights', []))
        text_length = len(result.get('extracted_text', ''))
        columns_detected = result.get('columns_detected', 'unknown')
        
        logger.info(f"✅ Gemini analysis completed successfully:")
        logger.info(f"   📄 Columns detected: {columns_detected}")
        logger.info(f"   📘 Blue highlights found: {blue_count}")
        logger.info(f"   📕 Red highlights found: {red_count}")
        logger.info(f"   📄 Total text extracted: {text_length} characters")
        
        if blue_count > 0:
            for i, highlight in enumerate(result.get('blue_highlights', [])[:3], 1):  # Show first 3
                logger.debug(f"   Blue highlight {i}: {highlight[:100]}{'...' if len(highlight) > 100 else ''}")
        
        if red_count > 0:
            for i, highlight in enumerate(result.get('red_highlights', [])[:3], 1):  # Show first 3
                logger.debug(f"   Red highlight {i}: {highlight[:100]}{'...' if len(highlight) > 100 else ''}")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse Gemini response as JSON: {str(e)}")
        logger.debug(f"Raw response: {response.text if 'response' in locals() else 'No response received'}")
        return {
            "extracted_text": "",
            "blue_highlights": [],
            "red_highlights": []
        }
    except Exception as e:
        logger.error(f"❌ Error processing image with Gemini: {str(e)}")
        logger.debug(f"Exception type: {type(e).__name__}")
        # Return empty result on error
        return {
            "extracted_text": "",
            "blue_highlights": [],
            "red_highlights": []
        }


def is_chapter_title(text):
    """Heuristic to detect chapter titles."""
    return text.lower().startswith("chapter") or text.isupper()


# Legacy OCR functions removed - now using Gemini API for text extraction


# ------------------- Legacy Functions (deprecated) -------------------
# The following functions are no longer used with Gemini API integration
# but are kept for reference. They can be removed in future cleanup.

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


def process_single_screenshot(screenshot_path, highlights):
    """Processes a single screenshot to extract highlighted text using Gemini API."""
    logger.info(f"🖼️  Processing screenshot: {os.path.basename(screenshot_path)}")
    
    if not os.path.exists(screenshot_path):
        logger.error(f"❌ Screenshot not found: {screenshot_path}")
        return

    try:
        logger.debug(f"Screenshot file path: {screenshot_path}")
        
        # Extract text and highlights using Gemini
        logger.info("⚡ Calling Gemini API for text extraction...")
        gemini_result = extract_text_with_gemini(screenshot_path)
        
        blue_highlights = gemini_result.get('blue_highlights', [])
        red_highlights = gemini_result.get('red_highlights', [])
        
        logger.info(f"📊 Processing results from Gemini:")
        logger.info(f"   Blue highlights to process: {len(blue_highlights)}")
        logger.info(f"   Red highlights to process: {len(red_highlights)}")
        
        processed_blue = 0
        processed_red = 0
        
        # Process blue highlights
        logger.debug("Processing blue highlights...")
        for i, highlight_text in enumerate(blue_highlights, 1):
            if highlight_text.strip():
                is_heading = is_chapter_title(highlight_text)
                highlight = {
                    "text": highlight_text.strip(),
                    "color": "blue",
                    "heading": is_heading
                }
                highlights.append(highlight)
                processed_blue += 1
                logger.debug(f"   Blue #{i}: {'[HEADING] ' if is_heading else ''}{highlight_text[:50]}{'...' if len(highlight_text) > 50 else ''}")

        # Process red highlights
        logger.debug("Processing red highlights...")
        for i, highlight_text in enumerate(red_highlights, 1):
            if highlight_text.strip():
                is_heading = is_chapter_title(highlight_text)
                highlight = {
                    "text": highlight_text.strip(),
                    "color": "red", 
                    "heading": is_heading
                }
                highlights.append(highlight)
                processed_red += 1
                logger.debug(f"   Red #{i}: {'[HEADING] ' if is_heading else ''}{highlight_text[:50]}{'...' if len(highlight_text) > 50 else ''}")

        logger.info(f"✅ Screenshot processing completed:")
        logger.info(f"   📘 Blue highlights processed: {processed_blue}")
        logger.info(f"   📕 Red highlights processed: {processed_red}")
        logger.info(f"   📚 Total highlights added: {processed_blue + processed_red}")
        
        print(f"Found {len(blue_highlights)} blue highlighted texts")
        print(f"Found {len(red_highlights)} red highlighted texts")

    except Exception as e:
        logger.error(f"❌ Error processing screenshot {os.path.basename(screenshot_path)}: {str(e)}")
        logger.debug(f"Full error details: {type(e).__name__}: {str(e)}")
        print(f"Error processing highlights in {screenshot_path}: {str(e)}")


def screenshot_data_orchestrator():
    """Main orchestrator for processing screenshots."""
    logger.info("🚀 Starting screenshot processing orchestrator")
    
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    highlights = []  # Single list for all highlights
    
    logger.info(f"📅 Session timestamp: {timestamp}")
    logger.info(f"📁 Screenshot directory: {SCREENSHOT_DIR}")
    logger.info(f"💾 Output directory: {OUTPUT_DIR}")

    # Check if screenshot directory exists and has files
    if not os.path.exists(SCREENSHOT_DIR):
        logger.error(f"❌ Screenshot directory does not exist: {SCREENSHOT_DIR}")
        print(f"Error: Screenshot directory not found: {SCREENSHOT_DIR}")
        return
    
    # Get all screenshot files
    logger.info("📂 Scanning for screenshot files...")
    all_files = os.listdir(SCREENSHOT_DIR)
    screenshot_files = [f for f in all_files if f.endswith('.png')]
    
    logger.info(f"📊 Found {len(screenshot_files)} PNG files in screenshot directory")
    if len(screenshot_files) == 0:
        logger.warning("⚠️  No PNG files found in screenshot directory")
        print("Warning: No screenshot files found to process")
        return

    # Process a specific file first (if it exists)
    specific_file = os.path.join(SCREENSHOT_DIR, "screenshot_2025.03.29_15.52.37_page2.png")
    if os.path.exists(specific_file):
        logger.info(f"🎯 Processing specific test file: {os.path.basename(specific_file)}")
        print(f"Processing specific file: {specific_file}")
        process_single_screenshot(specific_file, highlights)
        logger.info(f"✅ Specific file processing completed. Current highlights: {len(highlights)}")
    else:
        logger.info("ℹ️  Specific test file not found, will process all files")

    # Get all screenshot files from the directory and sort by creation time
    logger.info("📋 Preparing to process all screenshot files...")
    screenshot_paths = [os.path.join(SCREENSHOT_DIR, f) for f in screenshot_files]
    
    # Sort by creation time
    logger.debug("🔄 Sorting files by creation time...")
    screenshot_paths.sort(key=lambda x: os.path.getctime(x))
    
    logger.info(f"📑 Will process {len(screenshot_paths)} files in chronological order:")
    for i, path in enumerate(screenshot_paths, 1):
        logger.debug(f"   {i}. {os.path.basename(path)}")

    # Process all screenshots
    logger.info("🔄 Starting batch processing of all screenshots...")
    for i, screenshot in enumerate(screenshot_paths, 1):
        logger.info(f"📄 Processing file {i}/{len(screenshot_paths)}: {os.path.basename(screenshot)}")
        print(f"Processing {screenshot}")
        
        initial_count = len(highlights)
        process_single_screenshot(screenshot, highlights)
        new_highlights = len(highlights) - initial_count
        
        logger.info(f"   ✅ File {i} completed. New highlights: {new_highlights}, Total: {len(highlights)}")
        print(f"   Added {new_highlights} highlights. Total: {len(highlights)}")

    # Save to JSON with updated timestamp format
    logger.info("💾 Saving results to JSON file...")
    output_path = os.path.join(OUTPUT_DIR, f"highlights_{timestamp}.json")
    
    logger.debug(f"Output file path: {output_path}")
    logger.debug(f"Total highlights to save: {len(highlights)}")
    
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(highlights, f, indent=2, ensure_ascii=False)
        
        file_size = os.path.getsize(output_path) / 1024  # KB
        logger.info(f"✅ Successfully saved highlights to: {output_path}")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   📝 Total highlights extracted: {len(highlights)}")
        logger.info(f"   📁 Output file size: {file_size:.2f} KB")
        logger.info(f"   📄 Files processed: {len(screenshot_paths)}")
        
        # Count highlights by color and headings
        blue_count = sum(1 for h in highlights if h.get('color') == 'blue')
        red_count = sum(1 for h in highlights if h.get('color') == 'red')
        heading_count = sum(1 for h in highlights if h.get('heading', False))
        
        logger.info(f"   📘 Blue highlights: {blue_count}")
        logger.info(f"   📕 Red highlights: {red_count}")
        logger.info(f"   📚 Chapter headings: {heading_count}")
        
        # Log some sample highlights for verification
        if blue_count > 0:
            logger.debug("📘 Sample blue highlights:")
            for i, h in enumerate([h for h in highlights if h.get('color') == 'blue'][:2], 1):
                text_preview = h.get('text', '')[:80] + ('...' if len(h.get('text', '')) > 80 else '')
                logger.debug(f"   {i}. {text_preview}")
        
        if red_count > 0:
            logger.debug("📕 Sample red highlights:")
            for i, h in enumerate([h for h in highlights if h.get('color') == 'red'][:2], 1):
                text_preview = h.get('text', '')[:80] + ('...' if len(h.get('text', '')) > 80 else '')
                logger.debug(f"   {i}. {text_preview}")
        
        print(f"Highlights saved to {output_path}")
        print(f"Extracted {len(highlights)} highlights")
        
    except Exception as e:
        logger.error(f"❌ Failed to save highlights to file: {str(e)}")
        print(f"Error saving highlights: {str(e)}")
        
    logger.info("🎉 Screenshot processing orchestrator completed!")


# ------------------- Main Function -------------------


def main():
    """Main orchestrator function to trigger screenshot processing."""
    logger.info("=" * 80)
    logger.info("🚀 KINDLE HIGHLIGHT SCRAPER - STARTING APPLICATION")
    logger.info("=" * 80)
    
    try:
        screenshot_data_orchestrator()
        logger.info("🎉 Application completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Application interrupted by user")
        print("\nApplication interrupted by user")
        
    except Exception as e:
        logger.error(f"❌ Application failed with error: {str(e)}")
        logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
        print(f"Application error: {str(e)}")
        
    finally:
        logger.info("🔚 Application shutdown complete")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
