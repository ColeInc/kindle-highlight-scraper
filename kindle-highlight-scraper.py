import cv2
import pytesseract
import pyautogui
import os
import json
import datetime
from PIL import Image
import numpy as np
import time

# Configure pytesseract (path to Tesseract executable may vary)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update if necessary

# Directories
SCREENSHOT_DIR = "screenshots"
OUTPUT_DIR = "extracted_highlights"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- Extraction Phase -------------------

def extract_next_page():
    """Simulates pressing the right-arrow key."""
    pyautogui.press('right')
    time.sleep(0.5)  # wait for page change

def extract_screenshot(page_num):
    """Captures and saves a screenshot."""
    screenshot_path = os.path.join(SCREENSHOT_DIR, f"screenshot_{page_num}.png")
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)
    return screenshot_path

def orchestrate_extraction(pages_to_capture=5):
    """Main extraction orchestrator."""
    paths = []
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

def ocr_text_from_mask(image, mask):
    """Extracts text from masked areas using OCR."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    texts = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = image[y:y+h, x:x+w]
        pil_img = Image.fromarray(cropped)
        text = pytesseract.image_to_string(pil_img).strip()
        if text:
            texts.append((y, text))  # Keep vertical position to preserve order
    texts.sort(key=lambda item: item[0])  # Sort by vertical position
    return [text for _, text in texts]

def is_chapter_title(text):
    """Heuristic to detect chapter titles."""
    return text.lower().startswith("chapter") or text.isupper()

def process_single_screenshot(screenshot_path, highlights):
    """Processes a single screenshot to extract highlighted text."""
    image = cv2.imread(screenshot_path)

    # Define HSV color ranges for highlights (adjust as necessary)
    blue_lower, blue_upper = np.array([100, 150, 0]), np.array([140, 255, 255])
    red_lower1, red_upper1 = np.array([0, 150, 0]), np.array([10, 255, 255])
    red_lower2, red_upper2 = np.array([160, 150, 0]), np.array([179, 255, 255])

    # Masks
    blue_mask = detect_highlights(image, blue_lower, blue_upper)
    red_mask1 = detect_highlights(image, red_lower1, red_upper1)
    red_mask2 = detect_highlights(image, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Extract texts with positions
    blue_texts = ocr_text_from_mask(image, blue_mask)
    red_texts = ocr_text_from_mask(image, red_mask)

    # Tagging highlights with colors preserving reading order
    combined_highlights = []
    for y, text in blue_texts:
        combined_highlights.append((y, 'blue', text))
    for y, text in red_texts:
        combined_highlights.append((y, 'red', text))

    # Sort by vertical position (y-coordinate) to maintain reading order
    combined_highlights.sort(key=lambda item: item[0])

    # Process and append highlights to the main list
    for _, color, text in combined_highlights:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            highlight = {
                "text": line,
                "color": color,
                "heading": is_chapter_title(line)
            }
            highlights.append(highlight)


def process_highlights(texts):
    """Splits texts on gaps, marks chapter headings."""
    processed = []
    for text in texts:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            highlight = {
                "text": line,
                "heading": is_chapter_title(line)
            }
            processed.append(highlight)
    return processed

def process_screenshot_data(screenshot_paths):
    """Main orchestrator for processing screenshots."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    blue_highlights, red_highlights = [], []

    for screenshot in screenshot_paths:
        print(f"Processing {screenshot}")
        process_single_screenshot(screenshot, blue_highlights, red_highlights)

    # Save to JSON
    blue_output_path = os.path.join(OUTPUT_DIR, f"blue_highlights_{timestamp}.json")
    red_output_path = os.path.join(OUTPUT_DIR, f"red_highlights_{timestamp}.json")

    with open(blue_output_path, "w") as bf:
        json.dump(blue_highlights, bf, indent=2)
    with open(red_output_path, "w") as rf:
        json.dump(red_highlights, rf, indent=2)

    print(f"Blue highlights saved to {blue_output_path}")
    print(f"Red highlights saved to {red_output_path}")

# ------------------- Main Function -------------------

def main(pages_to_capture=5):
    """Main orchestrator function to trigger all."""
    screenshot_paths = orchestrate_extraction(pages_to_capture)
    process_screenshot_data(screenshot_paths)

if __name__ == "__main__":
    # Adjust number of pages as required
    main(pages_to_capture=5)
