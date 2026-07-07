import type { LoadedDataset } from '../../store/useDatasetStore'
import type { ProcessedDataset } from './pipelineExecutor'
import type { PipelineConfig } from '../compiler/pipelineGraph'

/** Conservative limits — browser JSON + XGBoost worker memory */
export const TRAINING_LIMITS = {
  maxInputRows: 100_000,
  maxOutputRows: 200_000,
  maxFeatures: 2_000,
  maxCells: 20_000_000,
  maxJsonBytes: 48 * 1024 * 1024,
  fetchTimeoutMs: 15 * 60 * 1000,
} as const

export type PayloadCheckResult = {
  ok: true
  estimatedJsonBytes: number
  rows: number
  features: number
  cells: number
} | {
  ok: false
  error: string
  hint: string
}

export function estimateOneHotFeatureCount(
  rowCount: number,
  targetColumn: string,
  sampleRow: Record<string, unknown>,
  colsToEncode: string[],
  rows: Record<string, unknown>[],
): number {
  let cols = Object.keys(sampleRow).filter((k) => k !== targetColumn).length
  for (const colName of colsToEncode) {
    const uniq = new Set(rows.map((r) => String(r[colName]))).size
    cols += Math.max(0, uniq - 1)
    if (cols > TRAINING_LIMITS.maxFeatures) return cols
  }
  void rowCount
  return cols
}

export function checkInputDatasetSize(dataset: LoadedDataset): PayloadCheckResult {
  const rows = dataset.rows.length
  if (rows > TRAINING_LIMITS.maxInputRows) {
    return {
      ok: false,
      error: `Dataset has ${rows.toLocaleString()} rows (max ${TRAINING_LIMITS.maxInputRows.toLocaleString()}).`,
      hint: 'Add a Filter node in the pipeline or import a smaller CSV.',
    }
  }
  return {
    ok: true,
    estimatedJsonBytes: 0,
    rows,
    features: dataset.columns.length,
    cells: rows * dataset.columns.length,
  }
}

export function checkPipelineExpansion(
  rowCount: number,
  targetColumn: string,
  rows: Record<string, unknown>[],
  pc: PipelineConfig,
  dataset: LoadedDataset,
): PayloadCheckResult | null {
  if (rows.length === 0) return null
  const sampleRow = rows[0]

  if (pc.oneHotEncode) {
    const colsToEncode = pc.oneHotColumns.length > 0
      ? pc.oneHotColumns
      : dataset.columns.filter((c) => c.type === 'string' && c.name !== targetColumn).map((c) => c.name)
    const projected = estimateOneHotFeatureCount(rowCount, targetColumn, sampleRow, colsToEncode, rows)
    if (projected > TRAINING_LIMITS.maxFeatures) {
      return {
        ok: false,
        error: `One-hot encoding would create ~${projected.toLocaleString()} features (max ${TRAINING_LIMITS.maxFeatures.toLocaleString()}).`,
        hint: 'Encode fewer columns, use Ordinal Encode instead, or drop high-cardinality categoricals.',
      }
    }
  }

  if (rowCount > TRAINING_LIMITS.maxOutputRows) {
    return {
      ok: false,
      error: `${rowCount.toLocaleString()} rows after pipeline (max ${TRAINING_LIMITS.maxOutputRows.toLocaleString()}).`,
      hint: 'Add filters or use a smaller source dataset.',
    }
  }

  return null
}

export function validateProcessedDataset(data: ProcessedDataset): PayloadCheckResult {
  const trainRows = data.X_train.length
  const valRows = data.X_val.length
  const features = data.featureCount
  const totalRows = trainRows + valRows
  const cells = totalRows * features

  if (features === 0) {
    return {
      ok: false,
      error: 'Pipeline produced zero features.',
      hint: 'Add numeric features or one-hot encode categoricals in Dataset → Pipeline.',
    }
  }
  if (features > TRAINING_LIMITS.maxFeatures) {
    return {
      ok: false,
      error: `${features.toLocaleString()} features (max ${TRAINING_LIMITS.maxFeatures.toLocaleString()}).`,
      hint: 'Use Select K Best, drop columns, or avoid one-hot on high-cardinality fields.',
    }
  }
  if (totalRows > TRAINING_LIMITS.maxOutputRows) {
    return {
      ok: false,
      error: `${totalRows.toLocaleString()} train+val rows (max ${TRAINING_LIMITS.maxOutputRows.toLocaleString()}).`,
      hint: 'Filter rows in the pipeline or reduce dataset size.',
    }
  }
  if (cells > TRAINING_LIMITS.maxCells) {
    return {
      ok: false,
      error: `Matrix too large (${cells.toLocaleString()} cells, max ${TRAINING_LIMITS.maxCells.toLocaleString()}).`,
      hint: 'Reduce rows or features — this size can crash the browser tab.',
    }
  }

  const estimatedJsonBytes = cells * 12 + totalRows * 8 + 4096
  if (estimatedJsonBytes > TRAINING_LIMITS.maxJsonBytes) {
    return {
      ok: false,
      error: `Estimated payload ~${Math.round(estimatedJsonBytes / 1024 / 1024)} MB (max ${Math.round(TRAINING_LIMITS.maxJsonBytes / 1024 / 1024)} MB).`,
      hint: 'Smaller datasets train reliably in the browser. Filter rows or features first.',
    }
  }

  return { ok: true, estimatedJsonBytes, rows: totalRows, features, cells }
}

export function safeStringifyTrainingPayload(
  data: ProcessedDataset,
  config: Record<string, unknown>,
): { ok: true; body: string; bytes: number } | { ok: false; error: string; hint: string } {
  try {
    const body = JSON.stringify({ data, config })
    const bytes = new Blob([body]).size
    if (bytes > TRAINING_LIMITS.maxJsonBytes) {
      return {
        ok: false,
        error: `Serialized payload is ${Math.round(bytes / 1024 / 1024)} MB (max ${Math.round(TRAINING_LIMITS.maxJsonBytes / 1024 / 1024)} MB).`,
        hint: 'Reduce rows or features before training.',
      }
    }
    return { ok: true, body, bytes }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    return {
      ok: false,
      error: `Failed to serialize training data${msg ? `: ${msg}` : ''}.`,
      hint: 'Dataset is too large for the browser. Filter rows/features in the pipeline.',
    }
  }
}

export async function pingTrainingBackend(apiBase: string): Promise<{ ok: true } | { ok: false; error: string; hint: string }> {
  try {
    const res = await fetch(`${apiBase}/api/health`, {
      signal: AbortSignal.timeout(5000),
    })
    if (!res.ok) {
      return {
        ok: false,
        error: `Backend returned HTTP ${res.status}.`,
        hint: 'Restart with: npm run dev:backend',
      }
    }
    return { ok: true }
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Network error'
    return {
      ok: false,
      error: `Cannot reach backend (${msg}).`,
      hint: 'Start the API: npm run dev:backend (port 8000)',
    }
  }
}
