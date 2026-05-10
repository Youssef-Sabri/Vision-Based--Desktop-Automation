# Vision-Based Desktop Automation

A desktop automation script for Windows that uses a Vision-Language Model (VLM) to visually locate and interact with desktop icons.

## Features

- **ScreenSeekeR Grounding Engine**: Implements the official recursive architecture from arXiv:2504.07981. Uses a Planner/Grounder workflow, Box Dilation, and Gaussian Centrality scoring to bypass popups and dynamically locate icons without hardcoded coordinates.
- **Pixel Caching**: Speeds up automation by skipping the complex AI check if the icon hasn't moved between automation loops.
- **API Integration**: Fetches dummy data to process during the automation loop.
- **Robustness**: Includes automatic retries, threading, Non-Maximum Suppression (NMS), and an emergency kill switch (`Esc` key).

## Setup

1. Install requirements (requires Python 3.10+):
   ```powershell
   uv sync
   ```

2. Add your Google Gemini API key to a `.env` file:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

3. Ensure there is a visible "Notepad" shortcut on your desktop.

## Usage

### Main Automation Loop
Fetches dummy posts, uses the ScreenSeekeR algorithm to locate Notepad, types the content, saves the files, and closes the window.
```powershell
uv run python src/main.py
```

### Diagnostic Testing
Generates annotated screenshots to verify that the ScreenSeekeR AI can accurately locate the target icon, bypassing visual obstructions.
```powershell
uv run python generate_screenshots.py
```

## Project Structure

- `src/main.py`: Main entry point and automation loop.
- `src/grounding.py`: The ScreenSeekeR Visual Engine (Planner, Grounder, Recursive Search).
- `src/automation.py`: Controls the mouse and keyboard actions.
- `src/api.py`: Fetches data from external REST APIs.
- `src/utils.py`: Shared logging and retry utilities.
