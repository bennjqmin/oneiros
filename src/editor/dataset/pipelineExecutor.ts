/**
 * Executes the dataset preprocessing pipeline entirely in the browser.
 * Walks pipeline nodes in topological order, applies each transformation,
 * then returns train/val arrays ready to POST to the backend.
 */

import type { AppNode, AppEdge } from '../../types/graph'
import type { LoadedDataset } from '../../store/useDatasetStore'
import { parsePipelineNodes } from '../compiler/pipelineGraph'

// ── Output type ───────────────────────────────────────────────────────────────

export interface ProcessedDataset {
  X_train: number[][]
  y_train: number[]
  X_val: number[][]
  y_val: number[]
  featureNames: string[]
  featureCount: number
  classNames: string[]
  classCount: number
  trainSamples: number
  valSamples: number
  datasetName: string
}

export type PipelineResult = { ok: true; data: ProcessedDataset } | { ok: false; error: string }

// ── Topological sort ──────────────────────────────────────────────────────────

function seededShuffle<T>(arr: T[], seed: number): T[] {
  const out = [...arr]
  let s = seed
  const rand = () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return (s >>> 0) / 0xffffffff
  }
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

function minMaxNorm(X: number[][]): number[][] {
  if (X.length === 0) return X
  const nf = X[0].length
  const mins = Array(nf).fill(Infinity)
  const maxs = Array(nf).fill(-Infinity)
  for (const row of X) {
    for (let j = 0; j < nf; j++) {
      if (row[j] < mins[j]) mins[j] = row[j]
      if (row[j] > maxs[j]) maxs[j] = row[j]
    }
  }
  return X.map((row) =>
    row.map((v, j) => {
      const r = maxs[j] - mins[j]
      return r === 0 ? 0 : (v - mins[j]) / r
    }),
  )
}

function zscoreNorm(X: number[][]): number[][] {
  if (X.length === 0) return X
  const nf = X[0].length
  const means = Array(nf).fill(0)
  for (const row of X) {
    for (let j = 0; j < nf; j++) means[j] += row[j]
  }
  for (let j = 0; j < nf; j++) means[j] /= X.length
  const stds = Array(nf).fill(0)
  for (const row of X) {
    for (let j = 0; j < nf; j++) stds[j] += (row[j] - means[j]) ** 2
  }
  for (let j = 0; j < nf; j++) stds[j] = Math.sqrt(stds[j] / X.length) || 1
  return X.map((row) => row.map((v, j) => (v - means[j]) / stds[j]))
}

function encodeLabels(values: unknown[]): { encoded: number[]; classes: string[] } {
  const classes = [...new Set(values.map(String))].sort()
  const classMap = new Map(classes.map((c, i) => [c, i]))
  return { encoded: values.map((v) => classMap.get(String(v)) ?? 0), classes }
}

function expandOneHot(
  rows: Record<string, unknown>[],
  columnName: string,
  classes: string[],
): { newRows: Record<string, unknown>[]; newColNames: string[] } {
  const newColNames = classes.map((c) => `${columnName}_${c}`)
  const newRows = rows.map((row) => {
    const out = { ...row }
    for (const c of classes) {
      out[`${columnName}_${c}`] = String(row[columnName]) === c ? 1 : 0
    }
    delete out[columnName]
    return out
  })
  return { newRows, newColNames }
}

// ── Additional transform helpers ──────────────────────────────────────────────

function colMean(X: number[][], j: number): number {
  return X.reduce((s, r) => s + r[j], 0) / X.length
}
function colMedian(X: number[][], j: number): number {
  const sorted = X.map((r) => r[j]).sort((a, b) => a - b)
  const m = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[m - 1] + sorted[m]) / 2 : sorted[m]
}
function colStd(X: number[][], j: number, mean: number): number {
  const variance = X.reduce((s, r) => s + (r[j] - mean) ** 2, 0) / X.length
  return Math.sqrt(variance) || 1
}

// ── Main executor ─────────────────────────────────────────────────────────────

export function executePipeline(
  dataset: LoadedDataset,
  targetColumn: string,
  pipelineNodes: AppNode[],
  pipelineEdges: AppEdge[],
  options?: { maxRows?: number; task?: 'classification' | 'regression' },
): PipelineResult {
  if (!targetColumn) return { ok: false, error: 'No target column selected.' }
  if (!dataset.columns.find((c) => c.name === targetColumn)) {
    return { ok: false, error: `Target column "${targetColumn}" not found in dataset.` }
  }

  const fullRowCount = dataset.rows.length
  const previewRows =
    options?.maxRows != null && fullRowCount > options.maxRows
      ? dataset.rows.slice(0, options.maxRows)
      : dataset.rows
  const scaleToFull = previewRows.length > 0 ? fullRowCount / previewRows.length : 1

  const { filters, config: pc } = parsePipelineNodes(pipelineNodes, pipelineEdges)

  let rows = [...previewRows]

  for (const f of filters) {
    rows = rows.filter((row) => {
      const v = Number(row[f.column])
      if (f.operator === '>') return v > f.value
      if (f.operator === '<') return v < f.value
      if (f.operator === '>=') return v >= f.value
      if (f.operator === '<=') return v <= f.value
      if (f.operator === '==') return v === f.value
      if (f.operator === '!=') return v !== f.value
      return true
    })
  }

  // ── Apply row-level transforms ────────────────────────────────────────────

  if (pc.dropDuplicates) {
    const seen = new Set<string>()
    rows = rows.filter((row) => {
      const key = JSON.stringify(row)
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
  }

  // Drop named columns
  if (pc.dropColumns.length > 0) {
    rows = rows.map((row) => {
      const out = { ...row }
      for (const col of pc.dropColumns) delete out[col]
      return out
    })
  }

  if (rows.length < 4) {
    return { ok: false, error: 'Dataset has too few rows after filtering/deduplication.' }
  }

  // Shuffle
  if (pc.shuffle) rows = seededShuffle(rows, pc.shuffleSeed)

  // Ordinal encode before one-hot
  if (pc.ordinalEncode) {
    const colsToEncode = pc.ordinalEncodeColumns.length > 0
      ? pc.ordinalEncodeColumns
      : dataset.columns.filter((c) => c.type === 'string' && c.name !== targetColumn).map((c) => c.name)
    for (const colName of colsToEncode) {
      const uniq = [...new Set(rows.map((r) => String(r[colName])))].sort()
      const map = new Map(uniq.map((v, i) => [v, i]))
      rows = rows.map((row) => ({ ...row, [colName]: map.get(String(row[colName])) ?? 0 }))
    }
  }

  // One-hot encode categorical feature columns
  if (pc.oneHotEncode) {
    const colsToEncode = pc.oneHotColumns.length > 0
      ? pc.oneHotColumns
      : dataset.columns.filter((c) => c.type === 'string' && c.name !== targetColumn).map((c) => c.name)
    for (const colName of colsToEncode) {
      const uniq = [...new Set(rows.map((r) => String(r[colName])))]
      const result = expandOneHot(rows, colName, uniq)
      rows = result.newRows
    }
  }

  // Determine feature columns (all remaining numeric cols except target)
  const sampleRow = rows[0]
  let featureNames = Object.keys(sampleRow).filter(
    (k) => k !== targetColumn && typeof sampleRow[k] === 'number',
  )

  if (featureNames.length === 0) {
    return {
      ok: false,
      error: 'No numeric feature columns found. Make sure features are numeric (not the target column).',
    }
  }

  // Extract X (numeric features only)
  let X = rows.map((row) => featureNames.map((f) => Number(row[f] ?? 0)))

  // Extract y (target)
  const rawTargets = rows.map((r) => r[targetColumn])
  const task = options?.task ?? 'classification'
  let y: number[]
  let classNames: string[]
  if (task === 'regression') {
    y = rawTargets.map((v) => {
      const n = Number(v)
      return Number.isFinite(n) ? n : NaN
    })
    classNames = []
    const bad = y.filter((v) => !Number.isFinite(v)).length
    if (bad > 0) {
      return {
        ok: false,
        error: `${bad} row(s) have non-numeric target values. Regression requires numeric targets.`,
      }
    }
  } else {
    const encoded = encodeLabels(rawTargets)
    y = encoded.encoded
    classNames = encoded.classes
  }

  // ── Apply column-level transforms ─────────────────────────────────────────

  // FillNaN (replace NaN / Infinity)
  if (pc.fillNaN) {
    const fillVals = featureNames.map((_, j) => {
      if (pc.fillNaNStrategy === 'median') return colMedian(X, j)
      if (pc.fillNaNStrategy === 'constant') return pc.fillNaNConstant
      return colMean(X, j) // mean (default)
    })
    X = X.map((row) =>
      row.map((v, j) => (!isFinite(v) || isNaN(v) ? fillVals[j] : v))
    )
  }

  // Log transform (log1p)
  if (pc.logTransform) {
    const idxs = pc.logTransformColumns.length > 0
      ? pc.logTransformColumns.map((c) => featureNames.indexOf(c)).filter((i) => i >= 0)
      : featureNames.map((_, i) => i)
    X = X.map((row) => row.map((v, j) => idxs.includes(j) ? Math.log1p(Math.max(0, v)) : v))
  }

  // Clip outliers
  if (pc.clipOutliers) {
    const stats = featureNames.map((_, j) => {
      const m = colMean(X, j)
      const s = colStd(X, j, m)
      return { lo: m - pc.clipStdFactor * s, hi: m + pc.clipStdFactor * s }
    })
    X = X.map((row) => row.map((v, j) => Math.max(stats[j].lo, Math.min(stats[j].hi, v))))
  }

  // Bin column
  if (pc.binColumnEnabled && pc.binColumn) {
    const ci = featureNames.indexOf(pc.binColumn)
    if (ci >= 0) {
      const vals = X.map((r) => r[ci])
      const mn = Math.min(...vals), mx = Math.max(...vals)
      if (pc.binStrategy === 'quantile') {
        const sorted = [...vals].sort((a, b) => a - b)
        const q = featureNames.map((_, i) => sorted[Math.floor(i / featureNames.length * sorted.length)] ?? mn)
        X = X.map((row) => {
          const bin = q.findIndex((b) => row[ci] <= b)
          return row.map((v, j) => j === ci ? (bin < 0 ? pc.binCount - 1 : bin) : v)
        })
      } else {
        const step = (mx - mn) / pc.binCount || 1
        X = X.map((row) => row.map((v, j) => j === ci ? Math.min(pc.binCount - 1, Math.floor((v - mn) / step)) : v))
      }
    }
  }

  // Normalise feature matrix (min-max)
  if (pc.normalize) {
    const colsToNorm = pc.normalizeColumns.length > 0
      ? pc.normalizeColumns.map((c) => featureNames.indexOf(c)).filter((i) => i >= 0)
      : null
    if (colsToNorm === null) {
      X = pc.normalizeMethod === 'zscore' ? zscoreNorm(X) : minMaxNorm(X)
    } else {
      const full = pc.normalizeMethod === 'zscore' ? zscoreNorm(X) : minMaxNorm(X)
      X = X.map((row, ri) => row.map((v, ci) => (colsToNorm.includes(ci) ? full[ri][ci] : v)))
    }
  }

  // Standard scaler (z-score per column)
  if (pc.standardScaler) {
    const idxs = pc.standardScalerColumns.length > 0
      ? pc.standardScalerColumns.map((c) => featureNames.indexOf(c)).filter((i) => i >= 0)
      : featureNames.map((_, i) => i)
    const stats = idxs.map((j) => { const m = colMean(X, j); return { m, s: colStd(X, j, m) } })
    X = X.map((row) => row.map((v, j) => {
      const pos = idxs.indexOf(j)
      return pos >= 0 ? (v - stats[pos].m) / stats[pos].s : v
    }))
  }

  // Select K best features by variance
  if (pc.selectKBest && pc.selectK < featureNames.length) {
    const variances = featureNames.map((_, j) => {
      const m = colMean(X, j)
      return X.reduce((s, r) => s + (r[j] - m) ** 2, 0) / X.length
    })
    const ranked = variances.map((v, i) => ({ i, v })).sort((a, b) => b.v - a.v).slice(0, pc.selectK)
    const keep = new Set(ranked.map((r) => r.i))
    featureNames = featureNames.filter((_, i) => keep.has(i))
    X = X.map((row) => row.filter((_, i) => keep.has(i)))
  }

  // Balance classes (oversample minority)
  if (pc.balanceClasses) {
    const classCounts = new Map<number, number[]>()
    y.forEach((cls, i) => { if (!classCounts.has(cls)) classCounts.set(cls, []); classCounts.get(cls)!.push(i) })
    const maxCount = Math.max(...[...classCounts.values()].map((v) => v.length))
    const newX: number[][] = [...X]
    const newY: number[] = [...y]
    for (const [cls, idxs] of classCounts) {
      const need = maxCount - idxs.length
      for (let i = 0; i < need; i++) {
        const src = idxs[i % idxs.length]
        newX.push(X[src])
        newY.push(cls)
      }
    }
    X = newX
    // y is already the labels array — overwrite for balance
    y.length = 0
    for (const v of newY) y.push(v)
  }

  // Split
  const splitIdx = Math.floor(X.length * pc.trainRatio)
  const X_train = X.slice(0, splitIdx)
  const y_train = y.slice(0, splitIdx)
  const X_val = X.slice(splitIdx)
  const y_val = y.slice(splitIdx)

  if (X_val.length === 0) {
    return { ok: false, error: 'Validation set is empty. Lower the train ratio or add more data.' }
  }

  const trainSamples =
    scaleToFull > 1 ? Math.round(X_train.length * scaleToFull) : X_train.length
  const valSamples =
    scaleToFull > 1 ? Math.round(X_val.length * scaleToFull) : X_val.length

  return {
    ok: true,
    data: {
      X_train,
      y_train,
      X_val,
      y_val,
      featureNames,
      featureCount: featureNames.length,
      classNames,
      classCount: classNames.length,
      trainSamples,
      valSamples,
      datasetName: dataset.name,
    },
  }
}
