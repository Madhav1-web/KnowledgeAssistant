/* eslint-disable prettier/prettier */
import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';
import { ChunkedDocument, StoredDocument } from '../../common/types/chunking.types';

const INDEX_PATH = path.resolve('./data/vectors.faiss');
const DOCS_PATH = path.resolve('./data/documents.json');
const DIMS = 1024; // Qwen3-Embedding-0.6B output dimensions (via sentence-transformers)

@Injectable()
export class VectorService implements OnModuleInit {
  private readonly logger = new Logger(VectorService.name);
  private index: any; // faiss-node IndexFlatIP
  private documents: StoredDocument[] = [];

  async onModuleInit() {
    const faissModule = await import('faiss-node'); // native addon — dynamic import for safety
    const faiss = (faissModule as any).default ?? faissModule;
    fs.mkdirSync('./data', { recursive: true });

    if (fs.existsSync(INDEX_PATH) && fs.existsSync(DOCS_PATH)) {
      this.index = faiss.IndexFlatIP.read(INDEX_PATH);
      const raw = JSON.parse(fs.readFileSync(DOCS_PATH, 'utf-8'));
      // migration guard: wrap legacy plain-string entries in StoredDocument shape
      this.documents = (raw as (string | StoredDocument)[]).map((entry, i) =>
        typeof entry === 'string'
          ? { text: entry, type: 'text' as const, pageNumber: 0, chunkIndex: i, metadata: { strategy: 'fixed' as const } }
          : entry,
      );
      this.logger.log(`Loaded ${this.documents.length} vectors from disk.`);
    } else {
      this.index = new faiss.IndexFlatIP(DIMS);
      this.logger.log('Created new FAISS IndexFlatIP (1024 dims).');
    }
  }

  store(chunks: ChunkedDocument[], embeddings: number[][]): void {
    this.index.add(embeddings.flat()); // faiss-node takes flat numeric array
    this.documents.push(
      ...chunks.map((c) => ({
        text: c.text,
        type: c.type,
        pageNumber: c.pageNumber,
        chunkIndex: c.chunkIndex,
        boundingBox: c.boundingBox,
        metadata: c.metadata,
      })),
    );
    this.persist();
    this.logger.log(`Stored ${chunks.length} chunks. Total: ${this.documents.length}`);
  }

  // Returns plain text strings — preserves existing query API contract
  search(queryEmbedding: number[], k = 3): string[] {
    return this.searchWithScores(queryEmbedding, k).map((r) => r.chunk.text);
  }

  searchWithScores(queryEmbedding: number[], k = 3): { chunk: StoredDocument; score: number }[] {
    if (this.documents.length === 0) return [];
    const { labels, distances } = this.index.search(
      queryEmbedding,
      Math.min(k, this.documents.length),
    );
    return labels.map((idx: number, i: number) => ({
      chunk: this.documents[idx],
      score: distances[i],
    }));
  }

  private persist(): void {
    this.index.write(INDEX_PATH);
    fs.writeFileSync(DOCS_PATH, JSON.stringify(this.documents));
  }

  getState() {
    return {
      totalDocs: this.documents.length,
      embeddingDimensions: DIMS,
      indexSize: this.index.ntotal(),
      docs: this.documents.map((doc, i) => ({
        index: i,
        preview: doc.text.slice(0, 120).replace(/\n/g, ' '),
        charCount: doc.text.length,
        type: doc.type,
        pageNumber: doc.pageNumber,
        strategy: doc.metadata.strategy,
      })),
    };
  }
}
