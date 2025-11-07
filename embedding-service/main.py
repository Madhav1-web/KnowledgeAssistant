import base64
import statistics
from typing import List

import pdf2image
import pytesseract
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

INSTRUCTION = "Instruct: Given a question, retrieve relevant passages that answer it\nQuery: "

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)


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
        raw = base64.b64decode(req.pdf_bytes_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 input")

    # first_page/last_page in pdf2image are 1-based
    page_1based = req.page_index + 1
    try:
        images = pdf2image.convert_from_bytes(
            raw, dpi=200, first_page=page_1based, last_page=page_1based
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF to image conversion failed: {e}")

    if not images:
        return {"text": "", "confidence": 0.0}

    data = pytesseract.image_to_data(
        images[0], output_type=pytesseract.Output.DICT, lang="eng"
    )
    # conf == -1 for non-text rows; filter those out along with empty strings
    words = [
        (data["text"][i], int(data["conf"][i]))
        for i in range(len(data["text"]))
        if int(data["conf"][i]) > 0 and data["text"][i].strip()
    ]
    text = " ".join(w for w, _ in words)
    confidence = round(statistics.mean(c for _, c in words), 2) if words else 0.0

    return {"text": text, "confidence": confidence}


@app.get("/health")
def health():
    return {"status": "ok", "dims": model.get_sentence_embedding_dimension()}
