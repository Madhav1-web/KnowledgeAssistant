/* eslint-disable prettier/prettier */

// ---------------------------------------------------------------------------
// Quality thresholds — all configurable via environment variables
// ---------------------------------------------------------------------------

export interface QualityThresholds {
  charsPerPage: number;  // min chars/page before OCR is considered
  alphaRatio: number;    // min fraction of alpha words before OCR is considered
  good: number;          // combined score ≥ this → Good Doc
  decent: number;        // combined score ≥ this → Decent Doc (else Bad Doc)
  skipBadDocs: boolean;  // if true, caller should reject Bad Docs rather than embed them
}

export function getThresholds(): QualityThresholds {
  return {
    charsPerPage: Number(process.env.QUALITY_THRESHOLD_CHARS_PER_PAGE ?? 100),
    alphaRatio:   Number(process.env.QUALITY_THRESHOLD_ALPHA_RATIO   ?? 0.30),
    good:         Number(process.env.QUALITY_THRESHOLD_GOOD          ?? 0.70),
    decent:       Number(process.env.QUALITY_THRESHOLD_DECENT        ?? 0.40),
    skipBadDocs:  (process.env.QUALITY_SKIP_BAD_DOCS ?? 'false') === 'true',
  };
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface QualitySignals {
  textLengthScore: number;
  weirdCharScore: number;
  wordValidityScore: number;
  structureScore: number;
  ocrConfidenceScore?: number;  // only present when at least one page used OCR
}

export interface QualityResult {
  signals: QualitySignals;
  combinedScore: number;
  label: 'Good Doc' | 'Decent Doc' | 'Bad Doc';
  usedOcr: boolean;
}

// ---------------------------------------------------------------------------
// Shared helper — word validity ratio
// ---------------------------------------------------------------------------

const ALPHA_WORD = /^[a-zA-Z]{2,}$/;

export function wordValidityRatio(text: string): number {
  const tokens = text.trim().split(/\s+/).filter((t) => t.length > 0);
  if (tokens.length === 0) return 0;
  return tokens.filter((t) => ALPHA_WORD.test(t)).length / tokens.length;
}

// ---------------------------------------------------------------------------
// Per-page OCR decision (AND logic — avoids false triggers on tables/code)
// ---------------------------------------------------------------------------

export function needsPageOcr(pageText: string, thresholds: QualityThresholds): boolean {
  // If the page has enough characters it's likely machine-readable — skip OCR
  if (pageText.length >= thresholds.charsPerPage) return false;
  // Only trigger OCR if text is ALSO garbled (low alpha ratio)
  return wordValidityRatio(pageText) < thresholds.alphaRatio;
}

// ---------------------------------------------------------------------------
// Intelligent merge — keep whichever version (pdf-parse vs OCR) scores higher
// ---------------------------------------------------------------------------

export function betterOf(pdfText: string, ocrText: string): string {
  return wordValidityRatio(pdfText) >= wordValidityRatio(ocrText) ? pdfText : ocrText;
}

// ---------------------------------------------------------------------------
// Individual signal computations (each returns 0–1)
// ---------------------------------------------------------------------------

function computeTextLengthScore(text: string, numPages: number): number {
  if (numPages === 0 || text.length === 0) return 0;
  const charsPerPage = text.length / numPages;
  return Math.min(1, charsPerPage / 300);  // 300 chars/page = baseline decent
}

function computeWeirdCharScore(text: string): number {
  if (text.length === 0) return 0;
  // C0 control chars (excluding tab=0x09, newline=0x0A, carriage-return=0x0D)
  const controlMatches  = (text.match(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g) ?? []).length;
  // Unicode replacement character — indicates encoding corruption
  const replacementMatches = (text.match(/\uFFFD/g) ?? []).length;
  // 4+ consecutive non-alphabetic non-space characters (symbol noise)
  const symbolRunMatches = (text.match(/[^a-zA-Z\s]{4,}/g) ?? []).length;

  const weirdCount = controlMatches + replacementMatches + symbolRunMatches;
  return Math.max(0, 1 - weirdCount / text.length);
}

function computeWordValidityScore(text: string): number {
  return wordValidityRatio(text);
}

function computeStructureScore(text: string): number {
  if (text.length === 0) return 0;

  // Sub-signal A: sentence ending density — ~10 endings per 1000 chars = score 1.0
  const sentenceEndings = (text.match(/[.?!]/g) ?? []).length;
  const per1000 = (sentenceEndings / text.length) * 1000;
  const sentenceScore = Math.min(1, per1000 / 10);

  // Sub-signal B: newlines present (structured text has paragraphs / line breaks)
  const newlineScore = text.includes('\n') ? 1 : 0;

  // Sub-signal C: average word length in [2, 10] is normal English prose;
  // OCR garbage tends toward 1-char tokens or very long runs
  const tokens = text.trim().split(/\s+/).filter((t) => t.length > 0);
  const avgLen = tokens.length > 0
    ? tokens.reduce((sum, t) => sum + t.length, 0) / tokens.length
    : 0;
  const wordLenScore = avgLen >= 2 && avgLen <= 10
    ? 1 - Math.abs(avgLen - 5) / 5  // peak at avgLen=5, falls off toward boundaries
    : 0;

  return (sentenceScore + newlineScore + wordLenScore) / 3;
}

// ---------------------------------------------------------------------------
// Combined score + label
// ---------------------------------------------------------------------------

function computeCombinedScore(signals: QualitySignals): number {
  const { textLengthScore, weirdCharScore, wordValidityScore, structureScore, ocrConfidenceScore } = signals;

  if (ocrConfidenceScore !== undefined) {
    // OCR path: Tesseract confidence is the most authoritative signal
    return (
      0.15 * textLengthScore +
      0.20 * weirdCharScore +
      0.20 * wordValidityScore +
      0.10 * structureScore +
      0.35 * ocrConfidenceScore
    );
  }

  // Native pdf-parse path
  return (
    0.25 * textLengthScore +
    0.30 * weirdCharScore +
    0.30 * wordValidityScore +
    0.15 * structureScore
  );
}

function getLabel(score: number, t: QualityThresholds): 'Good Doc' | 'Decent Doc' | 'Bad Doc' {
  if (score >= t.good)   return 'Good Doc';
  if (score >= t.decent) return 'Decent Doc';
  return 'Bad Doc';
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

/**
 * Compute a quality score for extracted (or OCR'd) document text.
 *
 * @param text          Final merged text (after per-page OCR decisions)
 * @param numPages      Total page count from pdf-parse
 * @param avgOcrConf    Average Tesseract confidence (0–100) across OCR'd pages,
 *                      or undefined if no pages were OCR'd
 * @param thresholds    Threshold config (from getThresholds())
 */
export function computeQualityScore(
  text: string,
  numPages: number,
  avgOcrConf: number | undefined,
  thresholds: QualityThresholds,
): QualityResult {
  const usedOcr = avgOcrConf !== undefined;

  const signals: QualitySignals = {
    textLengthScore:   computeTextLengthScore(text, numPages),
    weirdCharScore:    computeWeirdCharScore(text),
    wordValidityScore: computeWordValidityScore(text),
    structureScore:    computeStructureScore(text),
    ...(usedOcr && { ocrConfidenceScore: avgOcrConf / 100 }),
  };

  const combinedScore = Math.round(computeCombinedScore(signals) * 1000) / 1000;

  return {
    signals,
    combinedScore,
    label: getLabel(combinedScore, thresholds),
    usedOcr,
  };
}
