import type { AppNode, AppEdge } from '../../types/graph'
import type { LoadedDataset } from '../../store/useDatasetStore'
import type { CompileResult } from './types'
import { parsePipelineNodes, type FilterStep, type PipelineConfig } from './pipelineGraph'

function pyStr(s: string): string {
  return JSON.stringify(s)
}

function pyFilterExpr(f: FilterStep): string {
  const col = `df[${pyStr(f.column)}]`
  const val = f.value
  switch (f.operator) {
    case '>': return `${col} > ${val}`
    case '<': return `${col} < ${val}`
    case '>=': return `${col} >= ${val}`
    case '<=': return `${col} <= ${val}`
    case '==': return `${col} == ${val}`
    case '!=': return `${col} != ${val}`
    default: return 'True'
  }
}

function stringFeatureCols(dataset: LoadedDataset, targetColumn: string): string[] {
  return dataset.columns.filter((c) => c.type === 'string' && c.name !== targetColumn).map((c) => c.name)
}

export interface PipelineCodegenOptions {
  dataset: LoadedDataset
  targetColumn: string
  filters: FilterStep[]
  config: PipelineConfig
  task?: 'classification' | 'regression'
  /** When true, stop after split and expose arrays (for XGBoost script). */
  forTraining?: boolean
}

export function generatePipelinePython(options: PipelineCodegenOptions): { lines: string[]; imports: Set<string> } {
  const { dataset, targetColumn, filters, config, task = 'classification', forTraining = false } = options
  const lines: string[] = []
  const imports = new Set<string>(['import numpy as np', 'import pandas as pd'])

  const strCols = stringFeatureCols(dataset, targetColumn)
  const autoStr = strCols.map(pyStr).join(', ')

  lines.push(`TARGET = ${pyStr(targetColumn)}`)
  lines.push(`TRAIN_RATIO = ${config.trainRatio}`)
  lines.push('')

  if (filters.length > 0) {
    lines.push('# Row filters')
    for (const f of filters) {
      lines.push(`df = df[${pyFilterExpr(f)}]`)
    }
    lines.push('')
  }

  if (config.dropDuplicates) {
    lines.push('df = df.drop_duplicates()')
    lines.push('')
  }

  if (config.dropColumns.length > 0) {
    lines.push(`df = df.drop(columns=[${config.dropColumns.map(pyStr).join(', ')}])`)
    lines.push('')
  }

  if (config.shuffle) {
    lines.push(`df = df.sample(frac=1, random_state=${config.shuffleSeed}).reset_index(drop=True)`)
    lines.push('')
  }

  if (config.ordinalEncode) {
    imports.add('from sklearn.preprocessing import LabelEncoder')
    lines.push('# Ordinal encoding')
    if (config.ordinalEncodeColumns.length > 0) {
      for (const col of config.ordinalEncodeColumns) {
        lines.push(`_le = LabelEncoder()`)
        lines.push(`df[${pyStr(col)}] = _le.fit_transform(df[${pyStr(col)}].astype(str))`)
      }
    } else {
      lines.push(`for _col in [${autoStr}]:`)
      lines.push('    _le = LabelEncoder()')
      lines.push('    df[_col] = _le.fit_transform(df[_col].astype(str))')
    }
    lines.push('')
  }

  if (config.oneHotEncode) {
    lines.push('# One-hot encoding')
    if (config.oneHotColumns.length > 0) {
      lines.push(`df = pd.get_dummies(df, columns=[${config.oneHotColumns.map(pyStr).join(', ')}], dtype=float)`)
    } else {
      lines.push(`df = pd.get_dummies(df, columns=[${autoStr}], dtype=float)`)
    }
    lines.push('')
  }

  lines.push('feature_cols = [c for c in df.columns if c != TARGET and pd.api.types.is_numeric_dtype(df[c])]')
  lines.push('if not feature_cols:')
  lines.push('    raise ValueError("No numeric feature columns. Encode categoricals or check the pipeline.")')
  lines.push('X = df[feature_cols].astype(float).values')
  lines.push('y_raw = df[TARGET]')
  lines.push('')

  if (task === 'regression') {
    lines.push('y = pd.to_numeric(y_raw, errors="coerce").values')
    lines.push('if not np.isfinite(y).all():')
    lines.push('    raise ValueError("Regression requires numeric targets.")')
  } else {
    imports.add('from sklearn.preprocessing import LabelEncoder')
    lines.push('_target_le = LabelEncoder()')
    lines.push('y = _target_le.fit_transform(y_raw.astype(str))')
    lines.push('class_names = list(_target_le.classes_)')
  }
  lines.push('')

  if (config.fillNaN) {
    imports.add('from sklearn.impute import SimpleImputer')
    const strat = config.fillNaNStrategy === 'median' ? 'median'
      : config.fillNaNStrategy === 'constant' ? 'constant' : 'mean'
    lines.push('# Fill NaN / Inf')
    if (strat === 'constant') {
      lines.push(`_imputer = SimpleImputer(strategy="constant", fill_value=${config.fillNaNConstant})`)
    } else {
      lines.push(`_imputer = SimpleImputer(strategy=${pyStr(strat)})`)
    }
    lines.push('X = _imputer.fit_transform(X)')
    lines.push('')
  } else {
    lines.push('X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)')
    lines.push('')
  }

  if (config.logTransform) {
    lines.push('# Log transform (log1p)')
    if (config.logTransformColumns.length > 0) {
      for (const col of config.logTransformColumns) {
        lines.push(`if ${pyStr(col)} in feature_cols:`)
        lines.push(`    _j = feature_cols.index(${pyStr(col)})`)
        lines.push('    X[:, _j] = np.log1p(np.maximum(0, X[:, _j]))')
      }
    } else {
      lines.push('X = np.log1p(np.maximum(0, X))')
    }
    lines.push('')
  }

  if (config.clipOutliers) {
    lines.push(`# Clip outliers (±${config.clipStdFactor} std)`)
    lines.push('for _j in range(X.shape[1]):')
    lines.push('    _m, _s = X[:, _j].mean(), X[:, _j].std() or 1.0')
    lines.push(`    _lo, _hi = _m - ${config.clipStdFactor} * _s, _m + ${config.clipStdFactor} * _s`)
    lines.push('    X[:, _j] = np.clip(X[:, _j], _lo, _hi)')
    lines.push('')
  }

  if (config.binColumnEnabled && config.binColumn) {
    lines.push('# Bin column')
    lines.push(`if ${pyStr(config.binColumn)} in feature_cols:`)
    lines.push(`    _j = feature_cols.index(${pyStr(config.binColumn)})`)
    if (config.binStrategy === 'quantile') {
      lines.push(`    _edges = np.quantile(X[:, _j], np.linspace(0, 1, ${config.binCount + 1}))`)
      lines.push('    X[:, _j] = np.digitize(X[:, _j], _edges[1:-1], right=True)')
    } else {
      lines.push(`    _mn, _mx = X[:, _j].min(), X[:, _j].max()`)
      lines.push(`    _step = (_mx - _mn) / ${config.binCount} or 1.0`)
      lines.push(`    X[:, _j] = np.minimum(${config.binCount - 1}, np.floor((X[:, _j] - _mn) / _step))`)
    }
    lines.push('')
  }

  if (config.normalize) {
    if (config.normalizeMethod === 'zscore') {
      imports.add('from sklearn.preprocessing import StandardScaler')
      lines.push('# Normalize (z-score)')
      if (config.normalizeColumns.length > 0) {
        lines.push('_norm_idxs = [feature_cols.index(c) for c in [')
        lines.push(`    ${config.normalizeColumns.map(pyStr).join(', ')}`)
        lines.push('] if c in feature_cols]')
        lines.push('_scaler = StandardScaler()')
        lines.push('X[:, _norm_idxs] = _scaler.fit_transform(X[:, _norm_idxs])')
      } else {
        lines.push('_scaler = StandardScaler()')
        lines.push('X = _scaler.fit_transform(X)')
      }
    } else {
      imports.add('from sklearn.preprocessing import MinMaxScaler')
      lines.push('# Normalize (min-max)')
      if (config.normalizeColumns.length > 0) {
        lines.push('_norm_idxs = [feature_cols.index(c) for c in [')
        lines.push(`    ${config.normalizeColumns.map(pyStr).join(', ')}`)
        lines.push('] if c in feature_cols]')
        lines.push('_scaler = MinMaxScaler()')
        lines.push('X[:, _norm_idxs] = _scaler.fit_transform(X[:, _norm_idxs])')
      } else {
        lines.push('_scaler = MinMaxScaler()')
        lines.push('X = _scaler.fit_transform(X)')
      }
    }
    lines.push('')
  }

  if (config.standardScaler) {
    imports.add('from sklearn.preprocessing import StandardScaler')
    lines.push('# Standard scaler')
    if (config.standardScalerColumns.length > 0) {
      lines.push('_ss_idxs = [feature_cols.index(c) for c in [')
      lines.push(`    ${config.standardScalerColumns.map(pyStr).join(', ')}`)
      lines.push('] if c in feature_cols]')
      lines.push('_ss = StandardScaler()')
      lines.push('X[:, _ss_idxs] = _ss.fit_transform(X[:, _ss_idxs])')
    } else {
      lines.push('_ss = StandardScaler()')
      lines.push('X = _ss.fit_transform(X)')
    }
    lines.push('')
  }

  if (config.selectKBest) {
    lines.push('# Select K best features by variance')
    lines.push(`_k = min(${config.selectK}, X.shape[1])`)
    lines.push('_vars = X.var(axis=0)')
    lines.push('_keep = np.argsort(_vars)[::-1][:_k]')
    lines.push('X = X[:, _keep]')
    lines.push('feature_cols = [feature_cols[i] for i in _keep]')
    lines.push('')
  }

  if (config.balanceClasses && task === 'classification') {
    imports.add('from imblearn.over_sampling import RandomOverSampler')
    lines.push('# Balance classes (oversample minority)')
    lines.push('_ros = RandomOverSampler(random_state=42)')
    lines.push('X, y = _ros.fit_resample(X, y)')
    lines.push('')
  }

  lines.push('# Train / validation split')
  lines.push('_split = int(len(X) * TRAIN_RATIO)')
  lines.push('X_train, X_val = X[:_split], X[_split:]')
  lines.push('y_train, y_val = y[:_split], y[_split:]')
  lines.push('if len(X_val) == 0:')
  lines.push('    raise ValueError("Validation set is empty. Lower train ratio or add more rows.")')
  lines.push('')

  if (!forTraining) {
    lines.push('print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples, Features: {len(feature_cols)}")')
    if (task === 'classification') {
      lines.push('print(f"Classes: {class_names}")')
    }
  }

  return { lines, imports }
}

export function compilePipeline(
  dataset: LoadedDataset | null,
  targetColumn: string | null,
  pipelineNodes: AppNode[],
  pipelineEdges: AppEdge[],
  options?: { task?: 'classification' | 'regression' },
): CompileResult {
  const errors: string[] = []
  const warnings: string[] = []

  if (!dataset) {
    return { code: '', errors: ['No dataset loaded.'], warnings, filename: 'pipeline.py', label: 'Generated Pipeline' }
  }
  if (!targetColumn) {
    return { code: '', errors: ['No target column selected.'], warnings, filename: 'pipeline.py', label: 'Generated Pipeline' }
  }
  if (!dataset.columns.find((c) => c.name === targetColumn)) {
    return { code: '', errors: [`Target column "${targetColumn}" not found.`], warnings, filename: 'pipeline.py', label: 'Generated Pipeline' }
  }

  const sourceCount = pipelineNodes.filter((n) => n.type === 'datasetSource').length
  if (sourceCount === 0) {
    errors.push('Pipeline has no Source node.')
  }

  const { filters, config } = parsePipelineNodes(pipelineNodes, pipelineEdges)
  if (errors.length > 0) {
    return { code: '', errors, warnings, filename: 'pipeline.py', label: 'Generated Pipeline' }
  }

  const task = options?.task ?? 'classification'
  const dataPath = dataset.name.replace(/\\/g, '/')
  const { lines, imports } = generatePipelinePython({
    dataset, targetColumn, filters, config, task, forTraining: false,
  })

  const importLines = [...imports].sort()
  const code = [
    '"""',
    'Generated by Oneiros — dataset preprocessing pipeline',
    `Dataset: ${dataset.name}`,
    `Target: ${targetColumn}`,
    '"""',
    '',
    ...importLines,
    '',
    `DATA_PATH = ${pyStr(dataPath)}  # update path to your CSV/JSON export`,
    '',
    'def load_data(path: str) -> pd.DataFrame:',
    '    if path.lower().endswith(".json"):',
    '        return pd.read_json(path)',
    '    return pd.read_csv(path)',
    '',
    'def main():',
    '    df = load_data(DATA_PATH)',
    '    print(f"Loaded {len(df)} rows from {DATA_PATH}")',
    '',
    ...lines.map((l) => (l ? `    ${l}` : '')),
    '',
    'if __name__ == "__main__":',
    '    main()',
    '',
  ].join('\n')

  const safeName = dataset.name.replace(/\.[^.]+$/, '').replace(/\s+/g, '_').toLowerCase()
  return {
    code,
    errors,
    warnings,
    filename: `${safeName}_pipeline.py`,
    label: 'Generated Pipeline',
  }
}

/** Pipeline preprocessing block for embedding in XGBoost scripts. */
export function pipelinePreprocessBlock(
  dataset: LoadedDataset,
  targetColumn: string,
  pipelineNodes: AppNode[],
  pipelineEdges: AppEdge[],
  task: 'classification' | 'regression',
): { lines: string[]; imports: Set<string>; errors: string[] } {
  const { filters, config } = parsePipelineNodes(pipelineNodes, pipelineEdges)
  const { lines, imports } = generatePipelinePython({
    dataset, targetColumn, filters, config, task, forTraining: true,
  })
  return { lines, imports, errors: [] }
}
