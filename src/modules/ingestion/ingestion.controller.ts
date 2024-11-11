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
import { VectorService } from '../vector/vector.service';
import { EmbeddingService } from '../embedding/embedding.service';

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
    const queryEmbedding = await this.embeddingService.getEmbedding(query, true);
    const results = this.vectorService.searchWithScores(queryEmbedding, 3);
    return { query, results };
  }
}
