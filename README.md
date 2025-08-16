# Kindle Highlight Extraction Script

This script automates the extraction of highlighted text (red or blue highlights) from Kindle pages using advanced AI-powered text recognition. It processes screenshots of book pages and outputs structured JSON files containing the highlighted content.

## Features

- **AI-Powered Text Extraction**: Uses Google's Gemini 2.5 Flash for accurate text and highlight detection
- **Automated Screenshot Processing**: Processes existing screenshots to extract highlighted text
- **Intelligent Color Detection**: Automatically identifies red/pink and blue highlights
- **Chapter Detection**: Automatically marks headings and chapter titles
- **JSON Output**: Structured data format for easy integration with other tools

## Setup and Installation

### Prerequisites

1. **Python 3.7+**
2. **Google AI API Key**: Get your API key from [Google AI Studio](https://aistudio.google.com/)

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install opencv-python pyautogui pillow numpy matplotlib google-generativeai python-dotenv
```

### Environment Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## Usage

### Processing Screenshots

The script processes screenshots from the `screenshots/` directory:

```bash
python kindle-highlight-scraper.py
```

### Output

- **JSON Files**: Saved in `extracted_highlights/` with timestamp naming
- **Log Files**: Detailed logs saved to `kindle_scraper.log` 
- **Console Output**: Real-time progress with emojis and status updates
- **Structured Format**: Each highlight includes text, color, and heading detection

### Logging Levels

The application provides comprehensive logging at multiple levels:

- **INFO**: 📊 Progress updates, file processing status, statistics
- **DEBUG**: 🔍 Detailed API calls, file operations, processing steps  
- **ERROR**: ❌ Error conditions and exception handling
- **WARNING**: ⚠️ Non-fatal issues and warnings

View detailed logs in the `kindle_scraper.log` file for troubleshooting.

### Output Format

```json
[
  {
    "text": "This is highlighted text",
    "color": "red" | "blue",
    "heading": true | false
  }
]
```

## Architecture

- **Gemini Integration**: Uses Google's Gemini 2.5 Flash model for vision and text extraction
- **Schema-based Output**: Ensures consistent JSON structure using response schemas  
- **Error Handling**: Comprehensive logging and graceful error recovery
- **Legacy Support**: Previous OCR functions preserved for reference

## Tools and Technologies

- **AI Vision**: Google Gemini 2.5 Flash for text and highlight detection
- **Image Processing**: OpenCV for basic image operations
- **Automation**: PyAutoGUI for screenshot capture
- **Environment Management**: python-dotenv for configuration

## Chapter Detection

Text is automatically marked as a heading if it:
- Starts with the word "chapter" (case-insensitive)
- Is written in all uppercase letters

