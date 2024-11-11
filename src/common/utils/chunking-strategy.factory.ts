/* eslint-disable prettier/prettier */
import { ChunkingStrategy } from '../types/chunking.types';
import { FixedChunkStrategy } from './fixed-chunk.strategy';
import { LayoutChunkOptions, LayoutChunkStrategy } from './layout-chunk.strategy';

export type StrategyName = 'fixed' | 'layout';

export function createChunkingStrategy(
  override?: StrategyName,
  layoutOptions?: LayoutChunkOptions,
): ChunkingStrategy {
  const name: StrategyName =
    override ?? ((process.env.CHUNKING_STRATEGY ?? 'fixed') as StrategyName);

  switch (name) {
    case 'layout':
      return new LayoutChunkStrategy(layoutOptions);
    case 'fixed':
    default:
      return new FixedChunkStrategy();
  }
}
