import { Injectable } from '@nestjs/common';
import * as pdfParse from 'pdf-parse';
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
    const data = await pdfParse(file.buffer);
    const text = data.text;

    const chunks = chunkText(text);

    const embeddings = await Promise.all(
      chunks.map(chunk => this.embeddingService.getEmbedding(chunk)),
    );

    this.vectorService.store(chunks, embeddings);

    return { message: 'Document processed successfully' };
  }
}