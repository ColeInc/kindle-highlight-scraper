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

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def ocr_text_from_mask(image, mask):
    """Extracts text from masked areas using OCR."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(
        dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    if image is None:
        print(f"Error: Could not read image at {screenshot_path}")
        return

    # Define HSV color ranges for highlights (adjust as necessary)
    blue_lower, blue_upper = np.array([85, 50, 150]), np.array([115, 150, 255])
    red_lower1, red_upper1 = np.array([0, 150, 0]), np.array([10, 255, 255])
    red_lower2, red_upper2 = np.array([160, 150, 0]), np.array([179, 255, 255])

    # Masks
    blue_mask = detect_highlights(image, blue_lower, blue_upper)
    red_mask1 = detect_highlights(image, red_lower1, red_upper1)
    red_mask2 = detect_highlights(image, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    try:
        # Extract texts with positions
        blue_texts = ocr_text_from_mask(image, blue_mask)
        red_texts = ocr_text_from_mask(image, red_mask)

        # Tagging highlights with colors preserving reading order
        combined_highlights = []

        # Handle case where OCR didn't return y-coordinates
        if blue_texts and not isinstance(blue_texts[0], tuple):
            print(
                f"Warning: Blue highlights found but no position data. Processing text only.")
            for text in blue_texts:
                if text.strip():
                    highlight = {
                        "text": text.strip(),
                        "color": "blue",
                        "heading": is_chapter_title(text)
                    }
                    highlights.append(highlight)
        else:
            for y, text in blue_texts:
                combined_highlights.append((y, 'blue', text))

        if red_texts and not isinstance(red_texts[0], tuple):
            print(
                f"Warning: Red highlights found but no position data. Processing text only.")
            for text in red_texts:
                if text.strip():
                    highlight = {
                        "text": text.strip(),
                        "color": "red",
                        "heading": is_chapter_title(text)
                    }
                    highlights.append(highlight)
        else:
            for y, text in red_texts:
                combined_highlights.append((y, 'red', text))

        # Sort and process only if we have position data
        if combined_highlights:
            combined_highlights.sort(key=lambda item: item[0])
            for _, color, text in combined_highlights:
                lines = [line.strip()
                         for line in text.split("\n") if line.strip()]
                for line in lines:
                    highlight = {
                        "text": line,
                        "color": color,
                        "heading": is_chapter_title(line)
                    }
                    highlights.append(highlight)

    except Exception as e:
        print(f"Error processing highlights in {screenshot_path}: {str(e)}")
        print(f"Blue texts found: {len(blue_texts)}")
        print(f"Red texts found: {len(red_texts)}")


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


def screenshot_data_orchestrator():
    """Main orchestrator for processing screenshots."""
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    highlights = []  # Single list for all highlights

    # Get all screenshot files from the directory and sort by creation time
    screenshot_paths = [os.path.join(SCREENSHOT_DIR, f) for f in os.listdir(
        SCREENSHOT_DIR) if f.endswith('.png')]
    # Sort by creation time
    screenshot_paths.sort(key=lambda x: os.path.getctime(x))

    for screenshot in screenshot_paths:
        print(f"Processing {screenshot}")
        process_single_screenshot(screenshot, highlights)

    # Save to JSON with updated timestamp format
    output_path = os.path.join(OUTPUT_DIR, f"highlights_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(highlights, f, indent=2)

    print(f"Highlights saved to {output_path}")

# ------------------- Main Function -------------------


def main(pages_to_capture=5):
    """Main orchestrator function to trigger all."""
    # screenshot_paths = extraction_orchestrater(pages_to_capture)
    screenshot_data_orchestrator()


if __name__ == "__main__":
    # Adjust number of pages as required
    main(pages_to_capture=5)
