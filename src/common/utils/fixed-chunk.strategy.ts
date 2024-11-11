/* eslint-disable prettier/prettier */
import { ChunkedDocument, ChunkingInput, ChunkingStrategy } from '../types/chunking.types';
import { chunkText } from './chunk.util';

export class FixedChunkStrategy implements ChunkingStrategy {
  readonly name = 'fixed' as const;

  constructor(
    private readonly chunkSize = 500,
    private readonly overlap = 100,
  ) {}

  chunk(input: ChunkingInput): ChunkedDocument[] {
    const rawChunks = chunkText(input.fullText, this.chunkSize, this.overlap);
    return rawChunks.map((text, i) => ({
      text,
      type: 'text' as const,
      pageNumber: 0,
      chunkIndex: i,
      metadata: {
        strategy: 'fixed' as const,
        sourceFile: input.sourceFile,
      },
    }));
  }
}
