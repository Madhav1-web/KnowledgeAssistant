/* eslint-disable prettier/prettier */

export interface PdfTextItem {
  str: string;
  transform: number[]; // [scaleX, skewX, skewY, scaleY, x, y]
  width: number;
  height: number;
  fontName?: string;
  fontSize?: number; // derived from |transform[3]| or |transform[0]|
}

export interface LayoutLine {
  text: string;
  x: number;
  y: number;
  width: number;
  avgFontSize: number;
  pageNumber: number;
  items: PdfTextItem[];
}

export interface LayoutBlock {
  lines: LayoutLine[];
  text: string;
  pageNumber: number;
  boundingBox: { x: number; y: number; width: number; height: number };
  detectedType: ContentType;
  confidence: number;
}

export type ContentType = 'text' | 'table' | 'figure';

export interface ChunkedDocument {
  text: string;
  type: ContentType;
  pageNumber: number;
  chunkIndex: number;
  boundingBox?: { x: number; y: number; width: number; height: number };
  metadata: {
    strategy: 'fixed' | 'layout';
    sourceFile?: string;
    tableId?: string;
    figureCaption?: string;
  };
}

export interface ChunkingStrategy {
  name: 'fixed' | 'layout';
  chunk(input: ChunkingInput): ChunkedDocument[];
}

export interface ChunkingInput {
  fullText: string;
  pageItems?: PdfTextItem[][];
  pageCount: number;
  sourceFile: string;
}

export interface StoredDocument {
  text: string;
  type: ContentType;
  pageNumber: number;
  chunkIndex: number;
  boundingBox?: ChunkedDocument['boundingBox'];
  metadata: ChunkedDocument['metadata'];
}
