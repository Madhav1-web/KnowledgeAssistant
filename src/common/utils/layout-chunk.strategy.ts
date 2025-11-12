/* eslint-disable prettier/prettier */
import {
  ChunkedDocument,
  ChunkingInput,
  ChunkingStrategy,
  ContentType,
  LayoutBlock,
  LayoutLine,
  PdfTextItem,
} from '../types/chunking.types';
import { FixedChunkStrategy } from './fixed-chunk.strategy';

export interface LayoutChunkOptions {
  yToleranceFactor?: number; // yTolerance = factor × avg font size of page items — default 0.4
  gapMultiplier?: number;    // gap > gapMultiplier * avgLineHeight → block boundary — default 1.5
  maxChunkChars?: number;    // merge text blocks until this limit — default 600
  overlapChars?: number;     // chars of previous chunk to carry into next — default 150
  columnTolerance?: number;  // X gap threshold to detect a new column — default 20
  minTableRows?: number;     // min distinct Y rows to call a block a table — default 2
  minTableCols?: number;     // min distinct X columns to call a block a table — default 3
}

const FIGURE_CAPTION_RE = /^(figure|fig\.|chart|diagram|illustration|image|graph)\s*\d*/i;

export class LayoutChunkStrategy implements ChunkingStrategy {
  readonly name = 'layout' as const;

  private yToleranceFactor: number;
  private gapMultiplier: number;
  private maxChunkChars: number;
  private overlapChars: number;
  private columnTolerance: number;
  private minTableRows: number;
  private minTableCols: number;

  constructor(options: LayoutChunkOptions = {}) {
    this.yToleranceFactor = options.yToleranceFactor ?? 0.4;
    this.gapMultiplier = options.gapMultiplier ?? 1.5;
    this.maxChunkChars = options.maxChunkChars ?? 600;
    this.overlapChars = options.overlapChars ?? 150;
    this.columnTolerance = options.columnTolerance ?? 20;
    this.minTableRows = options.minTableRows ?? 2;
    this.minTableCols = options.minTableCols ?? 3;
  }

  chunk(input: ChunkingInput): ChunkedDocument[] {
    if (!input.pageItems || input.pageItems.length === 0) {
      console.warn('[LayoutChunk] No pageItems available — falling back to fixed chunking');
      return new FixedChunkStrategy().chunk(input);
    }

    const allBlocks: LayoutBlock[] = [];

    console.log(`[LayoutChunk] ${input.pageItems.length} page(s) of items received`);
    for (let pageIdx = 0; pageIdx < input.pageItems.length; pageIdx++) {
      const items = input.pageItems[pageIdx];
      console.log(`  pg${pageIdx}: ${items?.length ?? 0} raw items`);
      if (!items || items.length === 0) continue;

      // Phase A: group items into lines by Y proximity
      const lines = this.groupIntoLines(items, pageIdx);

      // Phase B: group lines into blocks by vertical gap
      const blocks = this.groupIntoBlocks(lines, pageIdx);

      // Phase C: classify each block
      for (const block of blocks) {
        this.classifyBlock(block);
      }

      allBlocks.push(...blocks);
    }

    if (allBlocks.length === 0) {
      console.warn('[LayoutChunk] All page items were empty/whitespace — falling back to fixed chunking');
      return new FixedChunkStrategy().chunk(input);
    }

    // Phase D: assemble final chunks
    return this.assembleChunks(allBlocks, input.sourceFile);
  }

  // ── Phase A ──────────────────────────────────────────────────────────────

  private groupIntoLines(items: PdfTextItem[], pageNumber: number): LayoutLine[] {
    const lines: LayoutLine[] = [];
    const nonEmpty = items.filter((i) => i.str.trim());
    console.log(`  [groupIntoLines] pg${pageNumber} — total items: ${items.length}, non-empty: ${nonEmpty.length}`);
    items.slice(0, 20).forEach((item, idx) => {
      const x = Array.isArray(item.transform) ? item.transform[4] : '?';
      const y = Array.isArray(item.transform) ? item.transform[5] : '?';
      console.log(`    raw[${idx}] x=${typeof x === 'number' ? x.toFixed(1) : x} y=${typeof y === 'number' ? y.toFixed(1) : y} str="${item.str}"`);
    });
    if (items.length > 20) console.log(`    ... (${items.length - 20} more items not shown)`);

    // Compute per-page effective yTolerance as a fraction of average font size
    const fontSizes = items
      .filter((i) => i.str.trim() && Array.isArray(i.transform) && i.transform.length >= 6)
      .map((i) => Math.abs(i.transform[3] ?? i.transform[0] ?? 12));
    const avgFontSize = fontSizes.length > 0
      ? fontSizes.reduce((a, b) => a + b, 0) / fontSizes.length
      : 12;
    const effectiveYTolerance = this.yToleranceFactor * avgFontSize;
    console.log(`  [groupIntoLines] pg${pageNumber} — avgFontSize=${avgFontSize.toFixed(1)} yToleranceFactor=${this.yToleranceFactor} effectiveYTolerance=${effectiveYTolerance.toFixed(1)}`);

    for (const item of items) {
      if (!item.str.trim()) continue;
      if (!Array.isArray(item.transform) || item.transform.length < 6) continue;

      const x = item.transform[4];
      const y = item.transform[5];
      const fontSize = Math.abs(item.transform[3] ?? item.transform[0] ?? 12);

      console.log(`  [item] pg${pageNumber} x=${x.toFixed(1)} y=${y.toFixed(1)} fs=${fontSize.toFixed(1)} "${item.str}"`);

      // find existing line within Y tolerance
      const existing = lines.find((l) => Math.abs(l.y - y) <= effectiveYTolerance);
      if (existing) {
        existing.items.push({ ...item, fontSize });
        // update representative Y as mean
        existing.y = existing.items.reduce((sum, it) => sum + it.transform[5], 0) / existing.items.length;
        existing.text = existing.items
          .sort((a, b) => a.transform[4] - b.transform[4])
          .map((i) => i.str)
          .join(' ');
        existing.x = Math.min(existing.x, x);
        existing.width = Math.max(...existing.items.map((i) => i.transform[4] + (i.width ?? 0))) - existing.x;
        existing.avgFontSize =
          existing.items.reduce((sum, i) => sum + (i.fontSize ?? 12), 0) / existing.items.length;
      } else {
        lines.push({
          text: item.str,
          x,
          y,
          width: item.width ?? 0,
          avgFontSize: fontSize,
          pageNumber,
          items: [{ ...item, fontSize }],
        });
      }
    }

    // PDF Y origin is bottom-left; sort descending Y = top-to-bottom reading order
    const sorted = lines.sort((a, b) => b.y - a.y);
    console.log(`  [lines] pg${pageNumber} — ${sorted.length} lines after grouping:`);
    sorted.forEach((l, i) => {
      console.log(`    line[${i}] y=${l.y.toFixed(1)} x=${l.x.toFixed(1)} fs=${l.avgFontSize.toFixed(1)} "${l.text.slice(0, 80)}"`);
    });
    return sorted;
  }

  // ── Phase B ──────────────────────────────────────────────────────────────

  private groupIntoBlocks(lines: LayoutLine[], pageNumber: number): LayoutBlock[] {
    if (lines.length === 0) return [];

    // compute avg line height from consecutive line Y differences
    let totalGap = 0;
    let gapCount = 0;
    for (let i = 0; i < lines.length - 1; i++) {
      const gap = lines[i].y - lines[i + 1].y;
      if (gap > 0) { totalGap += gap; gapCount++; }
    }
    const avgLineHeight = gapCount > 0 ? totalGap / gapCount : 14;
    const gapThreshold = this.gapMultiplier * avgLineHeight;
    console.log(`  [blocks] pg${pageNumber} — avgLineHeight=${avgLineHeight.toFixed(1)} gapThreshold=${gapThreshold.toFixed(1)}`);

    const blocks: LayoutBlock[] = [];
    let current: LayoutLine[] = [lines[0]];

    for (let i = 1; i < lines.length; i++) {
      const gap = lines[i - 1].y - lines[i].y;
      console.log(`    gap[${i-1}→${i}]=${gap.toFixed(1)} ${gap > gapThreshold ? '← BLOCK BOUNDARY' : ''}`);
      if (gap > gapThreshold) {
        blocks.push(this.makeBlock(current, pageNumber));
        current = [lines[i]];
      } else {
        current.push(lines[i]);
      }
    }
    if (current.length > 0) blocks.push(this.makeBlock(current, pageNumber));

    return blocks;
  }

  private makeBlock(lines: LayoutLine[], pageNumber: number): LayoutBlock {
    const text = lines.map((l) => l.text).join(' ');
    const minX = Math.min(...lines.map((l) => l.x));
    const maxX = Math.max(...lines.map((l) => l.x + l.width));
    const minY = Math.min(...lines.map((l) => l.y));
    const maxY = Math.max(...lines.map((l) => l.y));
    return {
      lines,
      text,
      pageNumber,
      boundingBox: { x: minX, y: minY, width: maxX - minX, height: maxY - minY },
      detectedType: 'text',
      confidence: 0,
    };
  }

  // ── Phase C ──────────────────────────────────────────────────────────────

  private classifyBlock(block: LayoutBlock): void {
    // Figure detection: caption pattern or sparse-text large area
    const trimmed = block.text.trim();
    const wordCount = trimmed.split(/\s+/).filter(Boolean).length;
    const area = block.boundingBox.width * block.boundingBox.height;

    if (FIGURE_CAPTION_RE.test(trimmed)) {
      block.detectedType = 'figure';
      block.confidence = 0.9;
      return;
    }

    if (wordCount < 10 && area > 50_000) {
      block.detectedType = 'figure';
      block.confidence = 0.7;
      return;
    }

    // Table detection: check for grid pattern across all items in block
    const allItems = block.lines.flatMap((l) => l.items);
    if (allItems.length < this.minTableRows * this.minTableCols) {
      block.detectedType = 'text';
      block.confidence = 0.8;
      return;
    }

    const xPositions = allItems.map((i) => i.transform[4]).sort((a, b) => a - b);
    const yPositions = [...new Set(block.lines.map((l) => Math.round(l.y)))];

    // cluster X positions into columns
    const columnClusters: number[][] = [];
    let currentCluster: number[] = [xPositions[0]];
    for (let i = 1; i < xPositions.length; i++) {
      if (xPositions[i] - xPositions[i - 1] > this.columnTolerance) {
        columnClusters.push(currentCluster);
        currentCluster = [xPositions[i]];
      } else {
        currentCluster.push(xPositions[i]);
      }
    }
    columnClusters.push(currentCluster);

    const colCount = columnClusters.length;
    const rowCount = yPositions.length;

    if (colCount >= this.minTableCols && rowCount >= this.minTableRows) {
      block.detectedType = 'table';
      // confidence scales with how regular the grid is
      block.confidence = Math.min(0.95, 0.5 + (colCount / 10) + (rowCount / 20));
      return;
    }

    block.detectedType = 'text';
    block.confidence = 0.8;
  }

  // ── Phase D ──────────────────────────────────────────────────────────────

  private assembleChunks(blocks: LayoutBlock[], sourceFile: string): ChunkedDocument[] {
    const chunks: ChunkedDocument[] = [];
    let chunkIndex = 0;

    let accText = '';
    let accBlocks: LayoutBlock[] = [];
    let accAvgFontSize = 0;
    let overlapCarry = ''; // tail of previous text chunk to prepend for context continuity

    const flushAccumulated = () => {
      if (!accText.trim()) return;
      const firstBlock = accBlocks[0];
      const trimmed = accText.trim();
      chunks.push({
        text: trimmed,
        type: 'text',
        pageNumber: firstBlock?.pageNumber ?? 0,
        chunkIndex: chunkIndex++,
        boundingBox: firstBlock?.boundingBox,
        metadata: { strategy: 'layout', sourceFile },
      });
      // carry the last overlapChars into the next text chunk to avoid boundary cuts
      overlapCarry = trimmed.length > this.overlapChars
        ? trimmed.slice(-this.overlapChars)
        : '';
      accText = '';
      accBlocks = [];
      accAvgFontSize = 0;
    };

    for (const block of blocks) {
      const type: ContentType = block.detectedType;

      if (type !== 'text') {
        // non-text blocks: flush any accumulated text, then emit as own chunk
        flushAccumulated();
        overlapCarry = ''; // don't carry text context across a table/figure boundary
        if (block.text.trim()) {
          chunks.push({
            text: block.text.trim(),
            type,
            pageNumber: block.pageNumber,
            chunkIndex: chunkIndex++,
            boundingBox: block.boundingBox,
            metadata: {
              strategy: 'layout',
              sourceFile,
              ...(type === 'figure' ? { figureCaption: block.text.trim() } : {}),
              ...(type === 'table' ? { tableId: `p${block.pageNumber}-t${chunkIndex}` } : {}),
            },
          });
        }
        continue;
      }

      // text block: check for heading (font size jump forces a boundary)
      const blockFontSize =
        block.lines.reduce((sum, l) => sum + l.avgFontSize, 0) / (block.lines.length || 1);
      const isHeading =
        accAvgFontSize > 0 &&
        blockFontSize > accAvgFontSize * 1.2 &&
        block.text.length < 100;

      if (isHeading || accText.length + block.text.length > this.maxChunkChars) {
        flushAccumulated();
      }

      // start fresh accumulation: prepend overlap from previous chunk if available
      if (!accText && overlapCarry) {
        accText = overlapCarry + ' ' + block.text;
        overlapCarry = '';
      } else {
        accText += (accText ? ' ' : '') + block.text;
      }
      accBlocks.push(block);
      accAvgFontSize =
        accBlocks.reduce(
          (sum, b) => sum + b.lines.reduce((s, l) => s + l.avgFontSize, 0) / (b.lines.length || 1),
          0,
        ) / accBlocks.length;
    }

    flushAccumulated();

    return chunks;
  }
}
