from pathlib import Path
from datetime import datetime, UTC
import json

import pytesseract
from PIL import Image, ImageOps, ImageFilter
from pdf2image import convert_from_path

# Tesseract path for your PC
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\38760\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Leave this as None for now.
# If you get a Poppler error later, we will set the path.
POPPLER_PATH = r"C:\Users\38760\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Improve OCR accuracy by cleaning the image.
    """
    gray = ImageOps.grayscale(image)
    sharp = gray.filter(ImageFilter.SHARPEN)
    bw = sharp.point(lambda x: 0 if x < 150 else 255, "1")
    return bw


def save_ocr_result(data, filename="ocr_result.json"):
    """
    Save OCR result to JSON file.
    """
    output_dir = Path("data/raw/ocr")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def ocr_image(image_path: str):
    """
    Perform OCR on a single image.
    """
    image = Image.open(image_path)

    raw_text = pytesseract.image_to_string(image)

    processed = preprocess_image(image)
    processed_text = pytesseract.image_to_string(processed)

    result = {
        "source": image_path,
        "file_name": Path(image_path).name,
        "type": "ocr_image",
        "extraction_timestamp": datetime.now(UTC).isoformat(),
        "raw_text": raw_text,
        "processed_text": processed_text
    }

    return result


def ocr_scanned_pdf(pdf_path: str, poppler_path: str | None = POPPLER_PATH):
    """
    Convert scanned PDF pages to images, preprocess them,
    run OCR page by page, and store metadata per page.
    """
    pages = convert_from_path(pdf_path, poppler_path=poppler_path)

    results = []

    for i, page_image in enumerate(pages, start=1):
        raw_text = pytesseract.image_to_string(page_image)

        processed = preprocess_image(page_image)
        processed_text = pytesseract.image_to_string(processed)

        page_result = {
            "source": pdf_path,
            "file_name": Path(pdf_path).name,
            "type": "ocr_pdf",
            "page_number": i,
            "extraction_timestamp": datetime.now(UTC).isoformat(),
            "raw_text": raw_text,
            "processed_text": processed_text
        }

        results.append(page_result)

    return results


if __name__ == "__main__":
    # IMAGE OCR TEST
    image_path = "data/raw/images/test_scan.png"
    image_result = ocr_image(image_path)
    save_ocr_result(image_result, "ocr_image_result.json")

    print("IMAGE OCR RAW TEXT:")
    print(image_result["raw_text"])
    print("\nIMAGE OCR PROCESSED TEXT:")
    print(image_result["processed_text"])

    # PDF OCR TEST
    pdf_path = "data/raw/scanned/test_scan.pdf"
    pdf_results = ocr_scanned_pdf(pdf_path)
    save_ocr_result(pdf_results, "ocr_pdf_result.json")

    print(f"\nPDF OCR completed for {len(pdf_results)} page(s).")
    for page in pdf_results:
        print(f"\n--- PAGE {page['page_number']} PROCESSED TEXT ---")
        print(page["processed_text"][:1000])