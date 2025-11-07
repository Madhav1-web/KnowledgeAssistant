import { Injectable, Logger, OnModuleInit } from '@nestjs/common';

const PYTHON_SERVICE_URL = 'http://localhost:8000';

@Injectable()
export class EmbeddingService implements OnModuleInit {
  private readonly logger = new Logger(EmbeddingService.name);

  async onModuleInit() {
    let ready = false;
    for (let i = 0; i < 30; i++) {
      try {
        const res = await fetch(`${PYTHON_SERVICE_URL}/health`);
        if (res.ok) {
          const body = (await res.json()) as { dims: number };
          this.logger.log(`Python embedding service ready. Dims: ${body.dims}`);
          ready = true;
          break;
        }
      } catch {
        this.logger.log(`Waiting for Python embedding service... (${i + 1}/30)`);
        await new Promise((r) => setTimeout(r, 2000));
      }
    }
    if (!ready) throw new Error('Python embedding service did not start in time.');
  }

  async getEmbedding(text: string, isQuery = false): Promise<number[]> {
    const res = await fetch(`${PYTHON_SERVICE_URL}/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts: [text], is_query: isQuery }),
    });
    const body = (await res.json()) as { embeddings: number[][] };
    return body.embeddings[0];
  }

  async getOcrPage(pdfBuffer: Buffer, pageIndex: number): Promise<{ text: string; confidence: number }> {
    const res = await fetch(`${PYTHON_SERVICE_URL}/ocr`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pdf_bytes_b64: pdfBuffer.toString('base64'),
        page_index: pageIndex,
      }),
    });
    if (!res.ok) {
      throw new Error(`OCR service error ${res.status}: ${await res.text()}`);
    }
    const body = (await res.json()) as { text: string; confidence: number };
    return body;
  }
}
