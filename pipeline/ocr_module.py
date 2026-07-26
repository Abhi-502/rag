"""
ocr_module.py
=============
Diagram step:  Blood Report Image  ->  OCR (PaddleOCR)

Job of this file: turn an image file into plain text.

We try PaddleOCR first (as shown in the diagram). If it isn't installed,
we fall back to pytesseract so the project still runs on a lighter setup.
Both paths return the exact same shape of data, so nothing downstream
needs to know which engine was actually used.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class OcrLine:
    """One line of recognized text and how confident the OCR engine was."""
    text: str
    confidence: float


def run_ocr(image_path: str) -> List[OcrLine]:
    """
    Read an image from disk and return a list of OcrLine objects.

    This is the single function the rest of the pipeline calls — it hides
    which OCR engine is actually doing the work.
    """
    try:
        return _run_paddleocr(image_path)
    except ImportError:
        print("[ocr_module] PaddleOCR not installed, falling back to Tesseract.")
        return _run_tesseract(image_path)


def _run_paddleocr(image_path: str) -> List[OcrLine]:
    """Use PaddleOCR (matches the diagram box: 'OCR (PaddleOCR)')."""
    from paddleocr import PaddleOCR  # imported lazily so it's optional

    ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    result = ocr_engine.ocr(image_path, cls=True)

    lines: List[OcrLine] = []
    # PaddleOCR returns: [[ [box, (text, confidence)], ... ]]
    for page in result:
        for _box, (text, confidence) in page:
            lines.append(OcrLine(text=text.strip(), confidence=float(confidence)))
    return lines


def _run_tesseract(image_path: str) -> List[OcrLine]:
    """Lightweight fallback if PaddleOCR isn't available in the environment."""
    import pytesseract
    from PIL import Image

    image = Image.open(image_path)
    # image_to_data gives per-line confidence, similar shape to PaddleOCR's output
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    lines: List[OcrLine] = []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        if not text:
            continue
        # Tesseract gives confidence as 0-100, PaddleOCR gives 0-1 — normalize.
        confidence = max(float(conf), 0) / 100.0
        lines.append(OcrLine(text=text, confidence=confidence))
    return lines


def ocr_lines_to_text(lines: List[OcrLine]) -> str:
    """Flatten OCR lines into a single block of text for the next steps."""
    return "\n".join(line.text for line in lines)
