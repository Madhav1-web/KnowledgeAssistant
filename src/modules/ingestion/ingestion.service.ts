/* eslint-disable prettier/prettier */
import * as fs from 'fs';
import * as path from 'path';
import { HttpException, HttpStatus, Injectable } from '@nestjs/common';
import pdfParse from 'pdf-parse';
import { PdfTextItem } from '../../common/types/chunking.types';
import { createChunkingStrategy, StrategyName } from '../../common/utils/chunking-strategy.factory';
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

  async processFile(file: Express.Multer.File, strategyOverride?: StrategyName) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`[Ingestion] Processing file: ${file.originalname} (${(file.size / 1024).toFixed(1)} KB)`);

    const thresholds = getThresholds();

    // Stage 1: PDF parse — collect per-page texts AND bounding box items
    const perPageTexts: string[] = [];
    const perPageItems: PdfTextItem[][] = [];

    const parsed = await pdfParse(file.buffer, {
      pagerender: (pageData: any) =>
        pageData.getTextContent().then((content: any) => {
          const rawItems = content.items as any[];
          const pdfItems: PdfTextItem[] = rawItems.map((item) => ({
            str: item.str ?? '',
            transform: Array.isArray(item.transform) ? item.transform : [1, 0, 0, 12, 0, 0],
            width: item.width ?? 0,
            height: item.height ?? 0,
            fontName: item.fontName,
            fontSize: Math.abs(
              (item.transform?.[3] ?? item.transform?.[0] ?? 12) as number,
            ),
          }));
          perPageItems.push(pdfItems);
          const pageText = pdfItems.map((i) => i.str).join(' ');
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

    type OcrPageResult = { text: string; confidence: number; lines: { text: string; x: number; y: number; width: number; height: number; confidence: number }[]; imageHeight: number; imageWidth: number; skewAngle: number };
    const ocrResultsPerPage: (OcrPageResult | null)[] = new Array(perPageTexts.length).fill(null);

    for (let i = 0; i < perPageTexts.length; i++) {
      const pageText = perPageTexts[i] ?? '';

      if (needsPageOcr(pageText, thresholds)) {
        console.log(`  Page ${i + 1}: sparse/garbled (${pageText.length} chars) — triggering OCR`);
        try {
          const ocrResult = await this.embeddingService.getOcrPage(file.buffer, i);
          ocrResultsPerPage[i] = ocrResult;
          const chosen = betterOf(pageText, ocrResult.text);
          mergedPages.push(chosen);
          if (ocrResult.confidence > 0) ocrConfidences.push(ocrResult.confidence);
          pagesOcrd++;
          console.log(`  Page ${i + 1}: OCR done (conf=${ocrResult.confidence.toFixed(1)}%) — kept ${chosen === ocrResult.text ? 'OCR' : 'pdf-parse'} text`);

          // Patch perPageItems with synthetic PdfTextItems built from OCR bounding boxes.
          // OCR coords: Y=0 is top, increases downward.
          // Layout chunker expects PDF coords: Y=0 is bottom, increases upward.
          // Flip: pdfY = imageHeight - (ocrY + lineHeight)
          if (ocrResult.lines.length > 0 && ocrResult.imageHeight > 0) {
            perPageItems[i] = ocrResult.lines.map((line) => {
              const pdfY = ocrResult.imageHeight - (line.y + line.height);
              const fontSize = Math.max(line.height, 1);
              return {
                str: line.text,
                transform: [fontSize, 0, 0, fontSize, line.x, pdfY],
                width: line.width,
                height: line.height,
                fontSize,
              };
            });
            console.log(`  Page ${i + 1}: injected ${perPageItems[i].length} synthetic layout items from OCR`);
          }
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

    // Stage 2: Chunking — strategy selected via env var or per-request override
    const strategy = createChunkingStrategy(strategyOverride);
    console.log(`\n[Ingestion] Stage 2 — Chunking (strategy: ${strategy.name})`);

    const chunkedDocs = strategy.chunk({
      fullText: finalText,
      pageItems: perPageItems.length > 0 ? perPageItems : undefined,
      pageCount: parsed.numpages,
      sourceFile: file.originalname,
    });

    console.log(`  Total chunks: ${chunkedDocs.length}`);
    const typeCounts = { text: 0, table: 0, figure: 0 };
    chunkedDocs.forEach((c, i) => {
      typeCounts[c.type]++;
      console.log(`  chunk[${i}]: ${c.text.length} chars | type=${c.type} | page=${c.pageNumber}`);
    });
    console.log(`  Content types — text: ${typeCounts.text}, table: ${typeCounts.table}, figure: ${typeCounts.figure}`);

    // Debug: comprehensive dump of every processing stage
    const debugDir = path.join(process.cwd(), 'chunk-debug');
    fs.mkdirSync(debugDir, { recursive: true });
    const safeName = file.originalname.replace(/[^a-z0-9_.-]/gi, '_');
    const debugPath = path.join(debugDir, `${safeName}_debug.txt`);
    const DIV  = '━'.repeat(80);
    const DIV2 = '─'.repeat(60);
    const DIV3 = '· · · · · · · · · · · · · · · · · · · · · · · · · ·';

    const lines: string[] = [];
    const w = (...strs: string[]) => lines.push(...strs);

    // ── FILE INFO ─────────────────────────────────────────────────────────────
    w(DIV, `FILE: ${file.originalname}`, `SIZE: ${(file.size / 1024).toFixed(1)} KB`, `PAGES: ${parsed.numpages}`, `STRATEGY: ${strategy.name}`, DIV, '');

    // ── PER-PAGE BREAKDOWN ────────────────────────────────────────────────────
    for (let pg = 0; pg < parsed.numpages; pg++) {
      w(DIV2, `PAGE ${pg + 1} / ${parsed.numpages}`, DIV2, '');

      // Raw pdf-parse text
      w('── RAW PDF-PARSE TEXT ──', perPageTexts[pg] ?? '(none)', '', DIV3, '');

      // Raw PdfTextItems
      const items = perPageItems[pg] ?? [];
      w(`── RAW ITEMS (${items.length} total) ──`);
      items.forEach((item, idx) => {
        const [, , , , x, y] = item.transform;
        w(`  item[${idx}] x=${x?.toFixed(1)} y=${y?.toFixed(1)} w=${item.width?.toFixed(1)} h=${item.height?.toFixed(1)} fs=${item.fontSize?.toFixed(1)} font="${item.fontName ?? '?'}" str="${item.str}"`);
      });
      w('', DIV3, '');

      // OCR result (if this page was OCR'd)
      const ocr = ocrResultsPerPage[pg];
      if (ocr) {
        const skewLabel = Math.abs(ocr.skewAngle) < 0.5
          ? 'none (<0.5°)'
          : `${ocr.skewAngle > 0 ? '+' : ''}${ocr.skewAngle.toFixed(2)}° — deskew applied`;
        w(`── OCR RESULT (conf=${ocr.confidence.toFixed(1)}% | image ${ocr.imageWidth}×${ocr.imageHeight} | skew: ${skewLabel}) ──`);
        w(`  OCR text: ${ocr.text}`);
        w('');
        w(`  OCR lines (${ocr.lines.length}):`);
        ocr.lines.forEach((ln, idx) => {
          w(`    line[${idx}] x=${ln.x.toFixed(1)} y=${ln.y.toFixed(1)} w=${ln.width.toFixed(1)} h=${ln.height.toFixed(1)} conf=${ln.confidence.toFixed(3)} → "${ln.text}"`);
        });
      } else {
        w('── OCR: not triggered for this page ──');
      }
      w('', DIV3, '');

      // Merged page text (after betterOf decision)
      w('── MERGED PAGE TEXT (chosen by betterOf) ──', mergedPages[pg] ?? '(none)', '', DIV2, '', '');
    }

    // ── FINAL COMBINED TEXT ───────────────────────────────────────────────────
    w(DIV, 'FINAL COMBINED TEXT (all pages joined)', DIV, finalText, '', DIV, '', '');

    // ── CHUNKS ────────────────────────────────────────────────────────────────
    w(DIV, `CHUNKS — total: ${chunkedDocs.length}  strategy: ${strategy.name}`, DIV, '');
    chunkedDocs.forEach((c, i) => {
      const bb = c.boundingBox
        ? `x=${c.boundingBox.x.toFixed(1)} y=${c.boundingBox.y.toFixed(1)} w=${c.boundingBox.width.toFixed(1)} h=${c.boundingBox.height.toFixed(1)}`
        : 'no bounding box';
      const meta = Object.entries(c.metadata).map(([k, v]) => `${k}=${v}`).join(' | ');
      w(
        `┌─ CHUNK ${i} ${'─'.repeat(Math.max(0, 70 - String(i).length))}`,
        `│  type     : ${c.type}`,
        `│  page     : ${c.pageNumber}`,
        `│  chars    : ${c.text.length}`,
        `│  bbox     : ${bb}`,
        `│  metadata : ${meta}`,
        `│`,
        `│  TEXT:`,
        ...c.text.split('\n').map(l => `│    ${l}`),
        `└${'─'.repeat(72)}`,
        '',
      );
    });

    fs.writeFileSync(debugPath, lines.join('\n'), 'utf8');
    console.log(`  Full debug dump written → ${debugPath}`);

    // Stage 3: Embeddings
    console.log(`\n[Ingestion] Stage 3 — Generating embeddings (${chunkedDocs.length} chunks, sequential)`);
    const embeddings: number[][] = [];
    for (const doc of chunkedDocs) {
      embeddings.push(await this.embeddingService.getEmbedding(doc.text));
    }
    console.log(`[Ingestion] All embeddings done. Each vector: ${embeddings[0]?.length ?? 0} dimensions`);

    // Stage 4: Store
    console.log(`\n[Ingestion] Stage 4 — Storing in vector store`);
    this.vectorService.store(chunkedDocs, embeddings);

    console.log(`\n[Ingestion] Done! File: ${file.originalname}`);
    console.log(`${'='.repeat(60)}\n`);

    return {
      message: 'Document processed successfully',
      stats: {
        filename: file.originalname,
        pages: parsed.numpages,
        characters: finalText.length,
        words: finalText.trim().split(/\s+/).length,
        chunks: chunkedDocs.length,
        embeddingDimensions: embeddings[0]?.length ?? 0,
        strategy: strategy.name,
        contentTypes: typeCounts,
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
