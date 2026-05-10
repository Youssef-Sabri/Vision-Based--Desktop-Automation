# Vision-Based Desktop Automation

A robust desktop automation engine for Windows that utilizes Vision-Language Models (VLM) for dynamic UI grounding. It locates application icons visually, eliminating the need for hardcoded coordinates.

This project implements the core methodologies outlined in the **ScreenSeekeR** paper (arXiv:2504.07981) to achieve production-grade accuracy and speed.

## 🚀 Features

- **Dynamic Grounding**: Implements a Cascaded Visual Search with a dual Planner/Grounder LLM architecture.
- **ScreenSeekeR Compliance**: Utilizes Box Dilation, Non-Maximum Suppression (NMS), and Gaussian Centrality Scoring for pixel-perfect precision.
- **Smart Pixel Caching**: Bypasses AI inference if the target icon hasn't moved, ensuring near-instant execution for repetitive tasks.
- **Resilient Workflow**: Automatically fetches data from APIs (with failover handling) and processes it into local text files.
- **Fail-Safe Mechanisms**: Includes an instant "Kill Switch" (`Esc` key) and automated retry logic for robust desktop management.

## 🛠️ Setup

1. **Prerequisites**: Windows 10/11, Python 3.10+, and `uv` package manager.
2. **Environment**: Create a `.env` file and add your Google Gemini API Key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```
3. **Install Dependencies**:
   ```powershell
   uv sync
   ```

## 💻 Usage

### Main Automation
Runs the full workflow: clears the desktop, fetches posts, visually locates Notepad, types content, and saves files to an output directory.
```powershell
uv run python src/main.py
```

### Scenario Testing
Validates the grounding engine by generating annotated screenshots for evaluation.
```powershell
uv run python generate_screenshots.py
```

## 📂 Project Structure

- `src/main.py`: Main orchestration and desktop management.
- `src/grounding.py`: The visual search engine (ScreenSeekeR Implementation).
- `src/automation.py`: Keyboard/mouse interaction, pixel-caching, and app launching.
- `src/api.py`: Data fetching with automatic failovers.
- `src/utils.py`: Logging and execution retry utilities.
- `temp/`: Directory where intermediate visual crops are saved (if `SAVE_DEBUG` is enabled).
- `debugging/`: Directory for annotated grounding reports.
