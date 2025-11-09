/* eslint-disable prettier/prettier */
import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';

const INDEX_PATH = path.resolve('./data/vectors.faiss');
const DOCS_PATH = path.resolve('./data/documents.json');
const DIMS = 1024; // Qwen3-Embedding-0.6B output dimensions (via sentence-transformers)

@Injectable()
export class VectorService implements OnModuleInit {
  private readonly logger = new Logger(VectorService.name);
  private index: any; // faiss-node IndexFlatIP
  private documents: string[] = [];

  async onModuleInit() {
    const faissModule = await import('faiss-node'); // native addon — dynamic import for safety
    const faiss = (faissModule as any).default ?? faissModule;
    fs.mkdirSync('./data', { recursive: true });

    if (fs.existsSync(INDEX_PATH) && fs.existsSync(DOCS_PATH)) {
      this.index = faiss.IndexFlatIP.read(INDEX_PATH);
      this.documents = JSON.parse(fs.readFileSync(DOCS_PATH, 'utf-8'));
      this.logger.log(`Loaded ${this.documents.length} vectors from disk.`);
    } else {
      this.index = new faiss.IndexFlatIP(DIMS);
      this.logger.log('Created new FAISS IndexFlatIP (768 dims).');
    }
  }

  store(chunks: string[], embeddings: number[][]): void {
    this.index.add(embeddings.flat()); // faiss-node takes flat numeric array
    this.documents.push(...chunks);
    this.persist();
    this.logger.log(`Stored ${chunks.length} chunks. Total: ${this.documents.length}`);
  }

  search(queryEmbedding: number[], k = 3): string[] {
    return this.searchWithScores(queryEmbedding, k).map(r => r.chunk);
  }

  searchWithScores(queryEmbedding: number[], k = 3): { chunk: string; score: number }[] {
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
        preview: doc.slice(0, 120).replace(/\n/g, ' '),
        charCount: doc.length,
      })),
    };
  }
}
