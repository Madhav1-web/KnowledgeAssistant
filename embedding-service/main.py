import base64
import gc
import io
import statistics
import tempfile
from typing import List
import os
os.environ["FLAGS_use_mkldnn"] = "0"
import cv2
import numpy as np
import pdf2image
# import pytesseract  # Tesseract path (commented — kept for low-confidence fallback)
from paddleocr import PaddleOCR
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder

app = FastAPI()

INSTRUCTION = "Instruct: Given a question, retrieve relevant passages that answer it\nQuery: "

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
paddle_ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True,
    # Use mobile detection model instead of the default server model — much lower RAM usage
    text_detection_model_name="PP-OCRv5_mobile_det",
)


class EmbedRequest(BaseModel):
    texts: List[str]
    is_query: bool = False


class OcrRequest(BaseModel):
    pdf_bytes_b64: str
    page_index: int  # 0-based page number to OCR


class RerankRequest(BaseModel):
    query: str
    passages: List[str]


class TableExtractRequest(BaseModel):
    pdf_bytes_b64: str
    page_index: int  # 0-based page number


def deskew(img_array: np.ndarray) -> tuple[np.ndarray, float]:
    """Detect and correct document skew. Returns (corrected_array, angle_degrees)."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return img_array, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angles in [-90, 0); remap to [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return img_array, angle
    h, w = img_array.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


@app.post("/embed")
def embed(req: EmbedRequest):
    texts = [INSTRUCTION + t for t in req.texts] if req.is_query else req.texts
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,  # L2 normalize → inner product = cosine sim
        batch_size=8,
    )
    return {"embeddings": embeddings.tolist()}


@app.post("/rerank")
def rerank(req: RerankRequest):
    pairs = [[req.query, p] for p in req.passages]
    scores = reranker.predict(pairs)
    return {"scores": scores.tolist()}


@app.post("/ocr")
def ocr(req: OcrRequest):
    try:
        print("We in the ocr endpoint, attempting to decode base64 PDF data...")
        raw = base64.b64decode(req.pdf_bytes_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 input")

    # first_page/last_page in pdf2image are 1-based
    page_1based = req.page_index + 1
    try:
        print(f"We in the ocr endpoint, attempting to convert page {page_1based} of PDF to image...")
        # Use a temp dir so Poppler writes the image to disk instead of keeping it in memory,
        # and reduce DPI from 200→150 to cut the image footprint by ~44%.
        with tempfile.TemporaryDirectory() as tmp_dir:
            images = pdf2image.convert_from_bytes(
                raw,
                dpi=400,
                first_page=page_1based,
                last_page=page_1based,
                output_folder=tmp_dir,
                fmt="jpeg",
                jpegopt={"quality": 85},
            )
            if not images:
                return {"text": "", "confidence": 0.0}

            # --- PaddleOCR path ---
            img_array = np.array(images[0])
            # Release the PIL image immediately; we only need the numpy array
            images[0].close()
            del images
            raw_bytes_freed = raw  # let go of the raw PDF bytes too
            del raw_bytes_freed

            img_array, skew_angle = deskew(img_array)
            print(f"  [deskew] corrected angle: {skew_angle:.2f}°")
            img_h, img_w = int(img_array.shape[0]), int(img_array.shape[1])
            result = paddle_ocr.ocr(img_array)
            del img_array
            gc.collect()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF to image conversion failed: {e}")

    words = []
    ocr_lines = []

    def _bbox_to_rect(poly):
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        return {"x": min(xs), "y": min(ys), "width": max(xs) - min(xs), "height": max(ys) - min(ys)}

    if result:
        for page_result in result:
            # New PaddleX / PP-OCRv5 format: dict with rec_texts, rec_scores, dt_polys
            if isinstance(page_result, dict):
                rec_texts  = page_result.get("rec_texts", [])
                rec_scores = page_result.get("rec_scores", [])
                dt_polys   = page_result.get("dt_polys", [])
                for idx, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                    if not text.strip():
                        continue
                    words.append((text, float(score)))
                    rect = _bbox_to_rect(dt_polys[idx]) if idx < len(dt_polys) else {"x": 0, "y": 0, "width": 0, "height": 0}
                    ocr_lines.append({**rect, "text": text, "confidence": float(score)})
            # Old PaddleOCR format: list of [bbox, [text, confidence]]
            elif isinstance(page_result, list):
                for word_info in page_result:
                    bbox = word_info[0] if word_info else None
                    part = word_info[1]
                    if isinstance(part, (list, tuple)) and len(part) >= 2:
                        text, confidence = part[0], part[1]
                    elif isinstance(part, (list, tuple)) and len(part) == 1:
                        text, confidence = part[0], 1.0
                    elif isinstance(part, str):
                        text, confidence = part, 1.0
                    else:
                        continue
                    if not text.strip():
                        continue
                    words.append((text, float(confidence)))
                    rect = _bbox_to_rect(bbox) if bbox else {"x": 0, "y": 0, "width": 0, "height": 0}
                    ocr_lines.append({**rect, "text": text, "confidence": float(confidence)})

    text = " ".join(w for w, _ in words)
    confidence = round(statistics.mean(c * 100 for _, c in words), 2) if words else 0.0

    # --- TESSERACT path (commented — re-enable as fallback when confidence < threshold) ---
    # data = pytesseract.image_to_data(
    #     images[0], output_type=pytesseract.Output.DICT, lang="eng"
    # )
    # # conf == -1 for non-text rows; filter those out along with empty strings
    # words = [
    #     (data["text"][i], int(data["conf"][i]))
    #     for i in range(len(data["text"]))
    #     if int(data["conf"][i]) > 0 and data["text"][i].strip()
    # ]
    # text = " ".join(w for w, _ in words)
    # confidence = round(statistics.mean(c for _, c in words), 2) if words else 0.0

    return {"text": text, "confidence": confidence, "lines": ocr_lines, "image_height": img_h, "image_width": img_w, "skew_angle": round(float(skew_angle), 2)}


@app.post("/extract-tables")
def extract_tables(req: TableExtractRequest):
    try:
        import pdfplumber
        raw = base64.b64decode(req.pdf_bytes_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 input or pdfplumber unavailable")

    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            if req.page_index >= len(pdf.pages):
                raise HTTPException(status_code=400, detail=f"Page index {req.page_index} out of range")
            page = pdf.pages[req.page_index]
            found = page.find_tables()
            result = []
            for i, tbl_finder in enumerate(found):
                tbl_data = tbl_finder.extract()
                bbox = tbl_finder.bbox  # (x0, top, x1, bottom)
                rows = [
                    " | ".join(cell or "" for cell in row)
                    for row in (tbl_data or [])
                ]
                result.append({
                    "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                    "text": "\n".join(rows),
                })
        return {"tables": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Table extraction failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "dims": model.get_sentence_embedding_dimension(), "reranker": "bge-reranker-v2-m3"}

