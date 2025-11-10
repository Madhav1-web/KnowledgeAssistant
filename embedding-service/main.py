import base64
import gc
import statistics
import tempfile
from typing import List
import os
os.environ["FLAGS_use_mkldnn"] = "0"
import numpy as np
import pdf2image
# import pytesseract  # Tesseract path (commented — kept for low-confidence fallback)
from paddleocr import PaddleOCR
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

INSTRUCTION = "Instruct: Given a question, retrieve relevant passages that answer it\nQuery: "

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
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


@app.post("/embed")
def embed(req: EmbedRequest):
    texts = [INSTRUCTION + t for t in req.texts] if req.is_query else req.texts
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,  # L2 normalize → inner product = cosine sim
        batch_size=8,
    )
    return {"embeddings": embeddings.tolist()}


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
                dpi=150,
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

            result = paddle_ocr.ocr(img_array)
            del img_array
            gc.collect()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF to image conversion failed: {e}")

    words = []
    if result:
        for page_result in result:
            # New PaddleX / PP-OCRv5 format: each element is a dict with
            # 'rec_texts' and 'rec_scores' lists.
            if isinstance(page_result, dict):
                rec_texts = page_result.get("rec_texts", [])
                rec_scores = page_result.get("rec_scores", [])
                for text, score in zip(rec_texts, rec_scores):
                    if text.strip():
                        words.append((text, float(score)))
            # Old PaddleOCR format: list of [bbox, [text, confidence]] lines.
            elif isinstance(page_result, list):
                for word_info in page_result:
                    part = word_info[1]
                    if isinstance(part, (list, tuple)) and len(part) >= 2:
                        text, confidence = part[0], part[1]
                    elif isinstance(part, (list, tuple)) and len(part) == 1:
                        text, confidence = part[0], 1.0
                    elif isinstance(part, str):
                        text, confidence = part, 1.0
                    else:
                        continue
                    if text.strip():
                        words.append((text, float(confidence)))

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

    return {"text": text, "confidence": confidence}


@app.get("/health")
def health():
    return {"status": "ok", "dims": model.get_sentence_embedding_dimension()}
