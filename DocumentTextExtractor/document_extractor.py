import os
import sys
import platform
import threading
from typing import Callable, Optional, List

import numpy as np
import cv2
from PIL import Image
import pytesseract
import pdfplumber
from PyPDF2 import PdfReader
from pdf2image import convert_from_path

# Tkinter imports
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import scrolledtext


def try_configure_tesseract_explicit_path() -> Optional[str]:
    """
    Try to set pytesseract command path automatically on Windows.
    Returns the path if set, else None.
    """
    explicit_path = os.environ.get("TESSERACT_CMD")
    if explicit_path and os.path.isfile(explicit_path):
        pytesseract.pytesseract.tesseract_cmd = explicit_path
        return explicit_path

    if platform.system().lower().startswith("win"):
        common_paths = [
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        ]
        for path in common_paths:
            if os.path.isfile(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return path
    return None


class DocumentProcessor:
    def __init__(self, tesseract_cmd: Optional[str] = None, poppler_path: Optional[str] = None) -> None:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            try_configure_tesseract_explicit_path()

        self.poppler_path = poppler_path or os.environ.get("POPPLER_PATH")

    # --------------- Image Preprocessing ---------------
    def _to_grayscale(self, image_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    def _threshold_otsu(self, gray: np.ndarray) -> np.ndarray:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        # Median blur is robust for salt-and-pepper noise
        return cv2.medianBlur(gray, 3)

    def _deskew(self, gray_or_bin: np.ndarray) -> np.ndarray:
        # Attempt lightweight deskew using moments
        image = gray_or_bin.copy()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Invert for better text foreground
        _, bin_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_inv = cv2.bitwise_not(bin_img)
        coords = np.column_stack(np.where(bin_inv > 0))
        if coords.size == 0:
            return gray_or_bin
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.5:
            return gray_or_bin
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray_or_bin, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def preprocess_image(self, image_bgr: np.ndarray, apply_deskew: bool = True) -> np.ndarray:
        gray = self._to_grayscale(image_bgr)
        denoised = self._denoise(gray)
        if apply_deskew:
            denoised = self._deskew(denoised)
        bin_img = self._threshold_otsu(denoised)
        return bin_img

    # --------------- OCR ---------------
    def ocr_numpy_image(self, image: np.ndarray, lang: str = "eng") -> str:
        # Convert to PIL for pytesseract
        if len(image.shape) == 2:
            pil_img = Image.fromarray(image)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        config = "--oem 3 --psm 6"
        return pytesseract.image_to_string(pil_img, lang=lang, config=config)

    # --------------- File handlers ---------------
    def process_image_file(self, filepath: str, lang: str = "eng") -> str:
        image_bgr = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError("Unable to read image file.")
        preprocessed = self.preprocess_image(image_bgr)
        return self.ocr_numpy_image(preprocessed, lang=lang)

    def _extract_text_from_pdf_native(self, filepath: str) -> str:
        texts: List[str] = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    texts.append(page_text)
        return "\n\n".join(texts).strip()

    def _ocr_pdf_page(self, filepath: str, page_index_zero_based: int, dpi: int = 300, lang: str = "eng") -> str:
        images = convert_from_path(
            filepath,
            dpi=dpi,
            first_page=page_index_zero_based + 1,
            last_page=page_index_zero_based + 1,
            poppler_path=self.poppler_path,
        )
        if not images:
            return ""
        pil_img = images[0]
        image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        preprocessed = self.preprocess_image(image_bgr)
        return self.ocr_numpy_image(preprocessed, lang=lang)

    def process_pdf_file(
        self,
        filepath: str,
        lang: str = "eng",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> str:
        # Try native extraction first
        native_text = self._extract_text_from_pdf_native(filepath)
        if native_text and len(native_text) >= 50:
            return native_text

        # Fallback to page-by-page OCR
        try:
            reader = PdfReader(filepath)
            total_pages = len(reader.pages)
        except Exception as exc:
            raise RuntimeError(f"Failed to read PDF: {exc}")

        ocr_texts: List[str] = []
        for page_idx in range(total_pages):
            if progress_callback:
                progress_callback(page_idx + 1, total_pages, f"OCR page {page_idx + 1}/{total_pages}")
            try:
                page_text = self._ocr_pdf_page(filepath, page_idx, dpi=300, lang=lang)
            except Exception as exc:
                # Provide a helpful hint if poppler is missing
                if "poppler" in str(exc).lower() or "pdfinfo" in str(exc).lower():
                    raise RuntimeError(
                        "PDF to image conversion failed. On Windows, install Poppler and set POPPLER_PATH to its 'bin' directory."
                    ) from exc
                raise
            ocr_texts.append(page_text.strip())

        return ("\n\n".join(ocr_texts)).strip()

    def process_file(
        self,
        filepath: str,
        lang: str = "eng",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> str:
        if not os.path.isfile(filepath):
            raise FileNotFoundError("File does not exist.")

        ext = os.path.splitext(filepath)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            if progress_callback:
                progress_callback(0, 1, "Running OCR on image...")
            return self.process_image_file(filepath, lang=lang)
        elif ext == ".pdf":
            return self.process_pdf_file(filepath, lang=lang, progress_callback=progress_callback)
        else:
            raise RuntimeError("Unsupported file type. Supported: PDF, JPG, JPEG, PNG, BMP, TIF, TIFF")


class DocumentExtractorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Universal Document Text Extractor")
        self.root.geometry("900x650")

        # Processor
        self.processor = DocumentProcessor()

        # UI Elements
        top_frame = tk.Frame(root)
        top_frame.pack(fill=tk.X, padx=10, pady=8)

        self.open_button = tk.Button(top_frame, text="Open File", command=self.open_file)
        self.open_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(top_frame, text="Save Text", command=self.save_text, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(8, 0))

        self.clear_button = tk.Button(top_frame, text="Clear", command=self.clear_text)
        self.clear_button.pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(top_frame, textvariable=self.status_var, anchor="w")
        self.status_label.pack(side=tk.LEFT, padx=16)

        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 11))
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Track current processing thread
        self._worker: Optional[threading.Thread] = None

    # ---------- UI Actions ----------
    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def open_file(self) -> None:
        filetypes = (
            ("Supported documents", "*.pdf;*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"),
            ("PDF files", "*.pdf"),
            ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"),
            ("All files", "*.*"),
        )
        filepath = filedialog.askopenfilename(title="Select a document", filetypes=filetypes)
        if not filepath:
            return

        self.text_area.delete("1.0", tk.END)
        self.save_button.config(state=tk.DISABLED)
        self.open_button.config(state=tk.DISABLED)
        self.set_status("Processing...")

        def run() -> None:
            try:
                result = self.processor.process_file(filepath, progress_callback=self._on_progress)
                self.root.after(0, self._on_complete, result)
            except Exception as exc:
                self.root.after(0, self._on_error, exc)

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        # Schedule UI-safe update
        self.root.after(0, lambda: self.set_status(message))

    def _on_complete(self, text: str) -> None:
        self.text_area.insert(tk.END, text or "")
        self.text_area.see(tk.END)
        self.save_button.config(state=tk.NORMAL)
        self.open_button.config(state=tk.NORMAL)
        self.set_status("Done")

    def _on_error(self, exc: Exception) -> None:
        self.open_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        self.set_status("Error")
        message = str(exc)
        extra_hint = ""
        if "Poppler" in message or "POPPLER_PATH" in message or "pdfinfo" in message:
            extra_hint = (
                "\n\nHint: Install Poppler for Windows and set POPPLER_PATH to its 'bin' folder."
            )
        if "Tesseract" in message or "tesseract" in message:
            extra_hint += (
                "\nHint: Install Tesseract OCR and ensure 'tesseract.exe' is in PATH or set TESSERACT_CMD."
            )
        messagebox.showerror("Processing failed", f"{message}{extra_hint}")

    def save_text(self) -> None:
        text = self.text_area.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Nothing to save", "There is no extracted text to save.")
            return
        filepath = filedialog.asksaveasfilename(
            title="Save extracted text",
            defaultextension=".txt",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
        )
        if not filepath:
            return
        try:
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            messagebox.showinfo("Saved", "Extracted text saved successfully.")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def clear_text(self) -> None:
        self.text_area.delete("1.0", tk.END)
        self.set_status("Cleared")


def main() -> None:
    root = tk.Tk()
    app = DocumentExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


