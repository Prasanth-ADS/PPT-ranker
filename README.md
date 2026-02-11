# Hackathon PPT Ranker

A production-grade system to automate the scoring and ranking of hackathon presentations using AI.

## Prerequisites

1. **Python 3.10+**
2. **Tesseract OCR**:
   - Windows: Download and install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
   - Add `C:\Program Files\Tesseract-OCR` to your System PATH variables.
3. **Hardware**:
   - 16GB RAM recommended (for DeepSeek-7B model).
   - GPU recommended but CPU supported (slower).

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   # Optional Tuning
   BATCH_SIZE=5
   ENABLE_CACHE=true
   ```

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Folder Structure
- `src/`: Core logic modules
- `downloads/`: Cached PPT/PDF files
- `output/`: Final ranking results
- `data/`: Input spreadsheets (if running locally)
