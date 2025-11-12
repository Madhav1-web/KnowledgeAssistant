/* eslint-disable prettier/prettier */
import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';
import { ChunkedDocument, StoredDocument } from '../../common/types/chunking.types';

const INDEX_PATH = path.resolve('./data/vectors.faiss');
const DOCS_PATH = path.resolve('./data/documents.json');
const DIMS = 1024; // Qwen3-Embedding-0.6B output dimensions (via sentence-transformers)

export interface ScoredResult {
  chunk: StoredDocument;
  score: number;
  docIndex: number;
}

@Injectable()
export class VectorService implements OnModuleInit {
  private readonly logger = new Logger(VectorService.name);
  private index: any; // faiss-node IndexFlatIP
  private documents: StoredDocument[] = [];

  // BM25 index structures
  private bm25TermFreqs = new Map<string, Map<number, number>>(); // term → docIdx → tf
  private bm25DocFreq = new Map<string, number>();                 // term → df
  private bm25DocLengths: number[] = [];
  private bm25AvgDocLength = 1;

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

    this.buildBm25Index();
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
    this.buildBm25Index();
    this.logger.log(`Stored ${chunks.length} chunks. Total: ${this.documents.length}`);
  }

  // Returns plain text strings — preserves existing query API contract
  search(queryEmbedding: number[], k = 3): string[] {
    return this.searchWithScores(queryEmbedding, k).map((r) => r.chunk.text);
  }

  searchWithScores(queryEmbedding: number[], k = 3): ScoredResult[] {
    if (this.documents.length === 0) return [];
    const { labels, distances } = this.index.search(
      queryEmbedding,
      Math.min(k, this.documents.length),
    );
    return labels.map((idx: number, i: number) => ({
      chunk: this.documents[idx],
      score: distances[i],
      docIndex: idx,
    }));
  }

  searchBm25(query: string, k = 20): ScoredResult[] {
    if (this.documents.length === 0) return [];

    const k1 = 1.5;
    const b = 0.75;
    const N = this.documents.length;
    const queryTerms = this.tokenize(query);

    const scores = new Float64Array(N);

    for (const term of queryTerms) {
      const df = this.bm25DocFreq.get(term) ?? 0;
      if (df === 0) continue;

      const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);
      const termDocMap = this.bm25TermFreqs.get(term)!;

      for (const [docIdx, tf] of termDocMap) {
        const docLen = this.bm25DocLengths[docIdx];
        const norm = tf + k1 * (1 - b + b * docLen / this.bm25AvgDocLength);
        scores[docIdx] += idf * (tf * (k1 + 1)) / norm;
      }
    }

    const results: ScoredResult[] = [];
    for (let i = 0; i < N; i++) {
      if (scores[i] > 0) results.push({ chunk: this.documents[i], score: scores[i], docIndex: i });
    }
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, k);
  }

  private tokenize(text: string): string[] {
    return text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean);
  }

  private buildBm25Index(): void {
    this.bm25TermFreqs.clear();
    this.bm25DocFreq.clear();
    this.bm25DocLengths = [];

    for (let docIdx = 0; docIdx < this.documents.length; docIdx++) {
      const tokens = this.tokenize(this.documents[docIdx].text);
      this.bm25DocLengths.push(tokens.length);

      const termFreq = new Map<string, number>();
      for (const token of tokens) {
        termFreq.set(token, (termFreq.get(token) ?? 0) + 1);
      }

      for (const [term, freq] of termFreq) {
        if (!this.bm25TermFreqs.has(term)) this.bm25TermFreqs.set(term, new Map());
        this.bm25TermFreqs.get(term)!.set(docIdx, freq);
        this.bm25DocFreq.set(term, (this.bm25DocFreq.get(term) ?? 0) + 1);
      }
    }

    this.bm25AvgDocLength =
      this.bm25DocLengths.length > 0
        ? this.bm25DocLengths.reduce((a, b) => a + b, 0) / this.bm25DocLengths.length
        : 1;
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
