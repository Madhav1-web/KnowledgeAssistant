import {
  Controller,
  Get,
  Post,
  Query,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import type { StrategyName } from '../../common/utils/chunking-strategy.factory';
import { IngestionService } from './ingestion.service';
import { VectorService, ScoredResult } from '../vector/vector.service';
import { EmbeddingService } from '../embedding/embedding.service';

// Reciprocal Rank Fusion — merges semantic and BM25 ranked lists into one ordering.
// k=60 is the standard constant that dampens the effect of high ranks.
function reciprocalRankFusion(
  semanticResults: ScoredResult[],
  bm25Results: ScoredResult[],
  k = 60,
): ScoredResult[] {
  const scores = new Map<number, { result: ScoredResult; rrfScore: number }>();

  for (let rank = 0; rank < semanticResults.length; rank++) {
    const r = semanticResults[rank];
    const entry = scores.get(r.docIndex) ?? { result: r, rrfScore: 0 };
    entry.rrfScore += 1 / (k + rank + 1);
    scores.set(r.docIndex, entry);
  }

  for (let rank = 0; rank < bm25Results.length; rank++) {
    const r = bm25Results[rank];
    const entry = scores.get(r.docIndex) ?? { result: r, rrfScore: 0 };
    entry.rrfScore += 1 / (k + rank + 1);
    scores.set(r.docIndex, entry);
  }

  return [...scores.values()]
    .sort((a, b) => b.rrfScore - a.rrfScore)
    .map(({ result }) => result);
}

@Controller('documents')
export class IngestionController {
  constructor(
    private readonly ingestionService: IngestionService,
    private readonly vectorService: VectorService,
    private readonly embeddingService: EmbeddingService,
  ) {}

  @Post('upload')
  @UseInterceptors(FileInterceptor('file'))
  async uploadFile(
    @UploadedFile() file: Express.Multer.File,
    @Query('strategy') strategy?: StrategyName,
  ) {
    return await this.ingestionService.processFile(file, strategy);
  }

  @Get('debug')
  getDebug() {
    return this.vectorService.getState();
  }

  @Get('search')
  async search(@Query('q') query: string) {
    if (!query) return { error: 'Provide a query param: ?q=your+question' };
    console.log(`\n[Search] Query: "${query}"`);

    const t0 = performance.now();

    const queryEmbedding = await this.embeddingService.getEmbedding(
      query,
      true,
    );
    const tEmbedded = performance.now();

    // Semantic retrieval (FAISS cosine similarity)
    const semanticCandidates = this.vectorService.searchWithScores(
      queryEmbedding,
      20,
    );
    const tFaiss = performance.now();

    // Lexical retrieval (BM25 keyword matching)
    const bm25Candidates = this.vectorService.searchBm25(query, 20);
    const tBm25 = performance.now();

    // Merge both lists with Reciprocal Rank Fusion, take top 20 for reranking
    const merged = reciprocalRankFusion(
      semanticCandidates,
      bm25Candidates,
    ).slice(0, 20);

    const rerankScores = await this.embeddingService.rerank(
      query,
      merged.map((r) => r.chunk.text),
    );
    const tReranked = performance.now();

    const results = merged
      .map((r, i) => ({ ...r, rerankScore: rerankScores[i] }))
      .sort((a, b) => b.rerankScore - a.rerankScore)
      .slice(0, 3);

    const timings = {
      embedMs: Math.round(tEmbedded - t0),
      faissMs: Math.round(tFaiss - tEmbedded),
      bm25Ms: Math.round(tBm25 - tFaiss),
      rerankMs: Math.round(tReranked - tBm25),
      totalMs: Math.round(tReranked - t0),
    };
    console.log(`[Search] timings:`, timings);

    return { query, timings, results };
  }
}
