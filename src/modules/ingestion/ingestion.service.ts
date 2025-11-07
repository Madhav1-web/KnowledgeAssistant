/* eslint-disable prettier/prettier */
import { HttpException, HttpStatus, Injectable } from '@nestjs/common';
import pdfParse from 'pdf-parse';
import { chunkText } from '../../common/utils/chunk.util';
import {
  betterOf,
  computeQualityScore,
  getThresholds,
  needsPageOcr,
} from '../../common/utils/quality-score.util';
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

    const thresholds = getThresholds();

    // Stage 1: PDF parse — collect per-page texts via pagerender callback
    const perPageTexts: string[] = [];
    const parsed = await pdfParse(file.buffer, {
      pagerender: (pageData: any) =>
        pageData.getTextContent().then((content: any) => {
          const pageText = (content.items as any[]).map((item: any) => item.str).join(' ');
          perPageTexts.push(pageText);
          return pageText;
        }),
    });

    console.log(`\n[Ingestion] Stage 1 — PDF parsed`);
    console.log(`  Pages: ${parsed.numpages} | Characters: ${parsed.text.length} | Words: ~${parsed.text.trim().split(/\s+/).length}`);
    console.log(`  Text preview: "${parsed.text.slice(0, 200).replace(/\n/g, ' ')}..."`);

    // Stage 1.5: Per-page parsability check + selective OCR + betterOf merge
    console.log(`\n[Ingestion] Stage 1.5 — Per-page parsability check`);
    console.log(`  Thresholds: charsPerPage=${thresholds.charsPerPage}, alphaRatio=${thresholds.alphaRatio}`);

    const mergedPages: string[] = [];
    const ocrConfidences: number[] = [];
    let pagesOcrd = 0;

    for (let i = 0; i < perPageTexts.length; i++) {
      const pageText = perPageTexts[i] ?? '';

      if (needsPageOcr(pageText, thresholds)) {
        console.log(`  Page ${i + 1}: sparse/garbled (${pageText.length} chars) — triggering OCR`);
        try {
          const ocrResult = await this.embeddingService.getOcrPage(file.buffer, i);
          const chosen = betterOf(pageText, ocrResult.text);
          mergedPages.push(chosen);
          if (ocrResult.confidence > 0) ocrConfidences.push(ocrResult.confidence);
          pagesOcrd++;
          console.log(`  Page ${i + 1}: OCR done (conf=${ocrResult.confidence.toFixed(1)}%) — kept ${chosen === ocrResult.text ? 'OCR' : 'pdf-parse'} text`);
        } catch (err) {
          console.warn(`  Page ${i + 1}: OCR failed (${(err as Error).message}) — falling back to pdf-parse text`);
          mergedPages.push(pageText);
        }
      } else {
        mergedPages.push(pageText);
      }
    }

    // If pagerender didn't fire (some PDF types) fall back to parsed.text
    const finalText = perPageTexts.length > 0
      ? mergedPages.join('\n\n')
      : parsed.text;

    const avgOcrConf = ocrConfidences.length > 0
      ? ocrConfidences.reduce((a, b) => a + b, 0) / ocrConfidences.length
      : undefined;

    console.log(`  Summary: ${pagesOcrd}/${parsed.numpages} pages OCR'd | final text length: ${finalText.length}`);

    // Stage 1.6: Quality scoring
    const quality = computeQualityScore(finalText, parsed.numpages, avgOcrConf, thresholds);

    const bar = (score: number, width = 20) => {
      const filled = Math.round(score * width);
      return `[${'█'.repeat(filled)}${' '.repeat(width - filled)}] ${(score * 100).toFixed(1).padStart(5)}%`;
    };
    const s = quality.signals;
    console.log(`\n[Ingestion] Stage 1.6 — Quality scoring`);
    console.log(`  ┌─────────────────────────────────────────────────┐`);
    console.log(`  │  File   : ${file.originalname}`);
    console.log(`  │  Score  : ${quality.combinedScore.toFixed(3)}   Label: ${quality.label}   OCR: ${quality.usedOcr}`);
    console.log(`  ├─────────────────────────────────────────────────┤`);
    console.log(`  │  textLength   ${bar(s.textLengthScore)}`);
    console.log(`  │  weirdChar    ${bar(s.weirdCharScore)}`);
    console.log(`  │  wordValidity ${bar(s.wordValidityScore)}`);
    console.log(`  │  structure    ${bar(s.structureScore)}`);
    if (s.ocrConfidenceScore !== undefined) {
      console.log(`  │  ocrConf      ${bar(s.ocrConfidenceScore)}`);
    }
    console.log(`  └─────────────────────────────────────────────────┘`);

    if (quality.label === 'Bad Doc') {
      if (thresholds.skipBadDocs) {
        console.error(`  *** Bad Doc — QUALITY_SKIP_BAD_DOCS=true → rejecting document ***`);
        throw new HttpException(
          {
            message: 'Document quality too low to ingest. Set QUALITY_SKIP_BAD_DOCS=false to process anyway.',
            quality: {
              label: quality.label,
              combinedScore: quality.combinedScore,
              signals: quality.signals,
            },
          },
          HttpStatus.UNPROCESSABLE_ENTITY,
        );
      }
      console.warn(`  *** WARNING: Bad Doc (score=${quality.combinedScore}) — processing with lowConfidence flag ***`);
    }

    // Stage 2: Chunking
    const chunks = chunkText(finalText);
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
        characters: finalText.length,
        words: finalText.trim().split(/\s+/).length,
        chunks: chunks.length,
        embeddingDimensions: embeddings[0]?.length ?? 0,
      },
      quality: {
        label: quality.label,
        combinedScore: quality.combinedScore,
        usedOcr: quality.usedOcr,
        ocrPagesCount: pagesOcrd,
        signals: quality.signals,
        lowConfidence: quality.label === 'Bad Doc',
      },
    };
  }
}
