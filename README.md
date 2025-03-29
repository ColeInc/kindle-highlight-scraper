# screenshot highlight extraction script

this script automates the extraction of highlighted text (red or blue highlights) from a kindle. it will flick through each page, take a screenshot, and store all screenshots in a directory. then, it outputs all highlighted text into a structured json file, marking the color and detecting chapter headings automatically.

## overall script flow:

- it simulates keyboard presses (right arrow) to flip through pages of a digital book.
- after each flip, it takes a screenshot of the page and saves it.
- each saved screenshot is processed individually.
- during processing, the script uses image processing (opencv) to detect red or blue highlights.
- text within highlighted areas is extracted using optical character recognition (OCR) with tesseract.
- extracted highlights are saved in chronological order, clearly tagging each snippet with its highlight color (red or blue), and marking headings.

## tools and approaches used:

- **OCR**: tesseract (pytesseract) for reliable text extraction.
- **image processing**: opencv (cv2) to identify highlights based on color.
- **automation**: pyautogui for simulating page turns and capturing screenshots.

## setting up your book for extraction:

to ensure accurate text extraction:

- highlight important text clearly using red or blue colors only.
- highlight chapter titles to ensure they're correctly identified. a heading is detected if it starts with the word "chapter" or is fully uppercase.
- avoid overlapping or unclear highlights; maintain slight gaps between separate highlights to ensure proper text separation in the output.
- screenshots should clearly show highlighted text without obstruction (e.g., avoid cursor placement or pop-ups).

following this setup will yield structured and easily readable output from the script.

