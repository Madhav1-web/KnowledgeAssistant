import { Injectable } from '@nestjs/common';

@Injectable()
export class VectorService {
  private documents: string[] = [];
  private embeddings: number[][] = [];

  store(chunks: string[], vectors: number[][]) {
    this.documents.push(...chunks);
    this.embeddings.push(...vectors);
  }

  search(queryEmbedding: number[], k = 3): string[] {
    const scores = this.embeddings.map((emb, index) => ({
      index,
      score: cosineSimilarity(queryEmbedding, emb),
    }));

    scores.sort((a, b) => b.score - a.score);

    return scores.slice(0, k).map((s) => this.documents[s.index]);
  }
}

function cosineSimilarity(a: number[], b: number[]) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (magA * magB);
}
