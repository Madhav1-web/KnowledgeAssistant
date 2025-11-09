/* eslint-disable prettier/prettier */
import { Injectable } from '@nestjs/common';
import pdfParse from 'pdf-parse';
import { chunkText } from '../../common/utils/chunk.util';
import { EmbeddingService } from '../embedding/embedding.service';
import { VectorService } from '../vector/vector.service';

@Injectable()
export class IngestionService {
  constructor(
    private embeddingService: EmbeddingService,
    private vectorService: VectorService,
  ) {}

  async processFile(file: Express.Multer.File) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`[Ingestion] Processing file: ${file.originalname} (${(file.size / 1024).toFixed(1)} KB)`);

    // Stage 1: PDF parse
    const parsed = await pdfParse(file.buffer);
    const text = parsed.text;
    const wordCount = text.trim().split(/\s+/).length;
    console.log(`\n[Ingestion] Stage 1 — PDF parsed`);
    console.log(`  Pages: ${parsed.numpages} | Characters: ${text.length} | Words: ~${wordCount}`);
    console.log(`  Text preview: "${text.slice(0, 200).replace(/\n/g, ' ')}..."`);

    // Stage 2: Chunking
    const chunks = chunkText(text);
    console.log(`\n[Ingestion] Stage 2 — Text chunked`);
    console.log(`  Total chunks: ${chunks.length}`);
    chunks.forEach((c, i) => console.log(`  chunk[${i}]: ${c.length} chars`));

    // Stage 3: Embeddings
    console.log(`\n[Ingestion] Stage 3 — Generating embeddings (${chunks.length} chunks, sequential)`);
    const embeddings: number[][] = [];
    for (const chunk of chunks) {
      embeddings.push(await this.embeddingService.getEmbedding(chunk));
    }
    console.log(`[Ingestion] All embeddings done. Each vector: ${embeddings[0]?.length ?? 0} dimensions`);

    // Stage 4: Store
    console.log(`\n[Ingestion] Stage 4 — Storing in vector store`);
    this.vectorService.store(chunks, embeddings);

    console.log(`\n[Ingestion] Done! File: ${file.originalname}`);
    console.log(`${'='.repeat(60)}\n`);

    return {
      message: 'Document processed successfully',
      stats: {
        filename: file.originalname,
        pages: parsed.numpages,
        characters: text.length,
        words: wordCount,
        chunks: chunks.length,
        embeddingDimensions: embeddings[0]?.length ?? 0,
      },
    };
  }
}