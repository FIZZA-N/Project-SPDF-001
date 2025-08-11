## Universal Document Text Extractor

A free, offline desktop app to extract text from PDFs and images (JPEG/PNG/BMP/TIFF), using native PDF text extraction and Tesseract OCR fallback. Built with Python and Tkinter.

### Features
- Editable and scanned PDF support (native text first, OCR fallback)
- Image preprocessing for better OCR (grayscale, denoise, Otsu threshold, deskew)
- Simple GUI to open files, view text, and save as .txt
- Works fully offline with free, open-source tools

### Requirements
- Python 3.8+
- Tesseract OCR
  - Windows: install from `https://github.com/UB-Mannheim/tesseract/wiki`
  - macOS: `brew install tesseract`
  - Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`
- Poppler (for PDF → image OCR fallback)
  - Windows: install Poppler for Windows and add its `bin` to PATH or set `POPPLER_PATH`
  - macOS: `brew install poppler`
  - Linux: `sudo apt-get install poppler-utils`

### Quick Start (Windows PowerShell)
```powershell
cd D:\development\Project-SPDF-001
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional: set explicit paths if not on PATH
# $env:TESSERACT_CMD = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# $env:POPPLER_PATH = 'C:\\path\\to\\poppler-xx\\Library\\bin'

python DocumentTextExtractor\document_extractor.py
```

### Notes
- On Windows, the app will try common Tesseract locations automatically. If not found, set `TESSERACT_CMD`.
- For OCR on PDFs, Poppler is required by `pdf2image`. Set `POPPLER_PATH` to its `bin` directory if not in PATH.
- Large PDFs are processed page-by-page for memory safety.

### Build an executable
```powershell
pip install pyinstaller
pyinstaller --onefile --windowed --name "DocumentExtractor" DocumentTextExtractor\document_extractor.py
```

### Alternative: Web app (Streamlit)
If you prefer a web UI, create `app.py` with Streamlit and reuse the processing class.

### Troubleshooting
- If you see errors mentioning `pdfinfo`/Poppler: install Poppler and set `POPPLER_PATH`.
- If you see errors mentioning Tesseract: install Tesseract and set `TESSERACT_CMD`.