import { t } from '../../theme/tokens'
import { Handle, Position } from '@xyflow/react'
import type { NodeProps, Node } from '@xyflow/react'

// ── Category colour palette ───────────────────────────────────────────────────

const C = {
  source: { border: '#0ea5e9', header: 'rgba(14,165,233,0.1)', text: '#7dd3fc', handle: '#0ea5e9' },
  transform: { border: '#6366f1', header: 'rgba(99,102,241,0.1)', text: '#a5b4fc', handle: '#6366f1' },
  output: { border: t.success, header: 'rgba(16,185,129,0.1)', text: '#6ee7b7', handle: t.success },
}

// ── Shared components ─────────────────────────────────────────────────────────

function DSNode({
  label,
  color,
  selected = false,
  hasTarget = true,
  hasSource = true,
  children,
}: {
  label: string
  color: typeof C.source
  selected?: boolean
  hasTarget?: boolean
  hasSource?: boolean
  children?: React.ReactNode
}) {
  return (
    <div
      style={{
        width: 196,
        background: t.bgElevated,
        borderRadius: 9,
        border: `1px solid ${selected ? color.border : t.borderDefault}`,
        boxShadow: selected
          ? `0 0 0 1px ${color.border}30, 0 6px 20px rgba(0,0,0,0.4)`
          : '0 3px 12px rgba(0,0,0,0.35)',
        overflow: 'hidden',
        transition: 'border-color 0.15s, box-shadow 0.15s',
      }}
    >
      {hasTarget && (
        <Handle type="target" position={Position.Left}
          style={{ width: 9, height: 9, background: t.bgElevated, border: `2px solid ${color.handle}`, left: -5, borderRadius: '50%' }} />
      )}
      <div style={{ background: color.header, borderBottom: `1px solid ${t.borderDefault}`, padding: '7px 11px', display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: color.text }}>{label}</span>
        <span style={{ marginLeft: 'auto', width: 5, height: 5, borderRadius: '50%', background: color.border }} />
      </div>
      {children && <div style={{ padding: '8px 11px' }}>{children}</div>}
      {hasSource && (
        <Handle type="source" position={Position.Right}
          style={{ width: 9, height: 9, background: color.handle, border: `2px solid ${color.handle}80`, right: -5, borderRadius: '50%' }} />
      )}
    </div>
  )
}

function Row({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
      <span style={{ color: t.textMuted }}>{label}</span>
      <span style={{ color: t.textPrimary, fontWeight: 500 }}>{value}</span>
    </div>
  )
}

// ── Source node ───────────────────────────────────────────────────────────────

type SourceData = Node<{ label: string; datasetName: string }, 'datasetSource'>

export function DatasetSourceNode({ data, selected }: NodeProps<SourceData>) {
  return (
    <DSNode label="Dataset Source" color={C.source} selected={selected} hasTarget={false}>
      <Row label="File" value={data.datasetName || '—'} />
    </DSNode>
  )
}

// ── Normalize node ────────────────────────────────────────────────────────────

type NormalizeData = Node<{ label: string; method: string; columns: string }, 'normalizeNode'>

export function NormalizeNode({ data, selected }: NodeProps<NormalizeData>) {
  return (
    <DSNode label="Normalize" color={C.transform} selected={selected}>
      <Row label="Method" value={data.method || 'min-max'} />
      <Row label="Columns" value={data.columns || 'all numeric'} />
    </DSNode>
  )
}

// ── One-hot encode node ───────────────────────────────────────────────────────

type OneHotData = Node<{ label: string; columns: string }, 'oneHotEncodeNode'>

export function OneHotEncodeNode({ data, selected }: NodeProps<OneHotData>) {
  return (
    <DSNode label="One-Hot Encode" color={C.transform} selected={selected}>
      <Row label="Columns" value={data.columns || 'all categorical'} />
    </DSNode>
  )
}

// ── Shuffle node ──────────────────────────────────────────────────────────────

type ShuffleData = Node<{ label: string; seed: number }, 'shuffleNode'>

export function ShuffleNode({ data, selected }: NodeProps<ShuffleData>) {
  return (
    <DSNode label="Shuffle" color={C.transform} selected={selected}>
      <Row label="Seed" value={data.seed ?? 42} />
    </DSNode>
  )
}

// ── Split node ────────────────────────────────────────────────────────────────

type SplitData = Node<{ label: string; trainRatio: number; valRatio: number; testRatio: number }, 'splitNode'>

export function SplitNode({ data, selected }: NodeProps<SplitData>) {
  const train = data.trainRatio ?? 0.7
  const val = data.valRatio ?? 0.15
  const test = data.testRatio ?? 0.15
  return (
    <DSNode label="Split" color={C.output} selected={selected} hasSource={false}>
      <Row label="Train" value={`${Math.round(train * 100)}%`} />
      <Row label="Val" value={`${Math.round(val * 100)}%`} />
      <Row label="Test" value={`${Math.round(test * 100)}%`} />
    </DSNode>
  )
}

// ── Filter node ───────────────────────────────────────────────────────────────

type FilterData = Node<{ label: string; column: string; operator: string; value: string }, 'filterNode'>

export function FilterNode({ data, selected }: NodeProps<FilterData>) {
  return (
    <DSNode label="Filter" color={C.transform} selected={selected}>
      <Row label="Column" value={data.column || '—'} />
      <Row label="Rule" value={`${data.operator || '>'} ${data.value || '0'}`} />
    </DSNode>
  )
}

// ── FillNaN node ──────────────────────────────────────────────────────────────

type FillNaNData = Node<{ label: string; strategy: string; constant: number }, 'fillNaNNode'>

export function FillNaNNode({ data, selected }: NodeProps<FillNaNData>) {
  return (
    <DSNode label="Fill NaN" color={C.transform} selected={selected}>
      <Row label="Strategy" value={data.strategy || 'mean'} />
      {(data.strategy || 'mean') === 'constant' && <Row label="Value" value={data.constant ?? 0} />}
    </DSNode>
  )
}

// ── DropColumns node ──────────────────────────────────────────────────────────

type DropColumnsData = Node<{ label: string; columns: string }, 'dropColumnsNode'>

export function DropColumnsNode({ data, selected }: NodeProps<DropColumnsData>) {
  const cols = data.columns || '—'
  return (
    <DSNode label="Drop Columns" color={C.transform} selected={selected}>
      <Row label="Columns" value={cols.length > 24 ? cols.slice(0, 24) + '…' : cols} />
    </DSNode>
  )
}

// ── ClipOutliers node ─────────────────────────────────────────────────────────

type ClipData = Node<{ label: string; stdFactor: number }, 'clipOutliersNode'>

export function ClipOutliersNode({ data, selected }: NodeProps<ClipData>) {
  return (
    <DSNode label="Clip Outliers" color={C.transform} selected={selected}>
      <Row label="Threshold" value={`±${data.stdFactor ?? 3}σ`} />
    </DSNode>
  )
}

// ── StandardScaler node ───────────────────────────────────────────────────────

type StdScalerData = Node<{ label: string; columns: string }, 'standardScalerNode'>

export function StandardScalerNode({ data, selected }: NodeProps<StdScalerData>) {
  return (
    <DSNode label="Standard Scaler" color={C.transform} selected={selected}>
      <Row label="Columns" value={data.columns || 'all numeric'} />
    </DSNode>
  )
}

// ── LogTransform node ─────────────────────────────────────────────────────────

type LogTransformData = Node<{ label: string; columns: string }, 'logTransformNode'>

export function LogTransformNode({ data, selected }: NodeProps<LogTransformData>) {
  return (
    <DSNode label="Log Transform" color={C.transform} selected={selected}>
      <Row label="Columns" value={data.columns || 'all numeric'} />
      <Row label="Func" value="log1p(x)" />
    </DSNode>
  )
}

// ── BinColumn node ────────────────────────────────────────────────────────────

type BinColumnData = Node<{ label: string; column: string; bins: number; strategy: string }, 'binColumnNode'>

export function BinColumnNode({ data, selected }: NodeProps<BinColumnData>) {
  return (
    <DSNode label="Bin Column" color={C.transform} selected={selected}>
      <Row label="Column" value={data.column || '—'} />
      <Row label="Bins" value={data.bins ?? 5} />
      <Row label="Strategy" value={data.strategy || 'equal-width'} />
    </DSNode>
  )
}

// ── SelectKBest node ──────────────────────────────────────────────────────────

type SelectKBestData = Node<{ label: string; k: number }, 'selectKBestNode'>

export function SelectKBestNode({ data, selected }: NodeProps<SelectKBestData>) {
  return (
    <DSNode label="Select K Best" color={C.transform} selected={selected}>
      <Row label="Top K" value={data.k ?? 10} />
      <Row label="Score" value="variance" />
    </DSNode>
  )
}

// ── OrdinalEncode node ────────────────────────────────────────────────────────

type OrdinalEncodeData = Node<{ label: string; columns: string }, 'ordinalEncodeNode'>

export function OrdinalEncodeNode({ data, selected }: NodeProps<OrdinalEncodeData>) {
  return (
    <DSNode label="Ordinal Encode" color={C.transform} selected={selected}>
      <Row label="Columns" value={data.columns || 'all categorical'} />
    </DSNode>
  )
}

// ── BalanceClasses node ───────────────────────────────────────────────────────

type BalanceData = Node<{ label: string }, 'balanceClassesNode'>

export function BalanceClassesNode({ selected }: NodeProps<BalanceData>) {
  return (
    <DSNode label="Balance Classes" color={C.transform} selected={selected}>
      <Row label="Method" value="oversample" />
    </DSNode>
  )
}

// ── DropDuplicates node ───────────────────────────────────────────────────────

type DropDupData = Node<{ label: string }, 'dropDuplicatesNode'>

export function DropDuplicatesNode({ selected }: NodeProps<DropDupData>) {
  return (
    <DSNode label="Drop Duplicates" color={C.transform} selected={selected}>
      <Row label="Keep" value="first" />
    </DSNode>
  )
}

// ── Node type registry for XYFlow ─────────────────────────────────────────────

export const datasetNodeTypes = {
  datasetSource:      DatasetSourceNode,
  normalizeNode:      NormalizeNode,
  oneHotEncodeNode:   OneHotEncodeNode,
  shuffleNode:        ShuffleNode,
  splitNode:          SplitNode,
  filterNode:         FilterNode,
  fillNaNNode:        FillNaNNode,
  dropColumnsNode:    DropColumnsNode,
  clipOutliersNode:   ClipOutliersNode,
  standardScalerNode: StandardScalerNode,
  logTransformNode:   LogTransformNode,
  binColumnNode:      BinColumnNode,
  selectKBestNode:    SelectKBestNode,
  ordinalEncodeNode:  OrdinalEncodeNode,
  balanceClassesNode: BalanceClassesNode,
  dropDuplicatesNode: DropDuplicatesNode,
}

// ── Palette definition (used by the pipeline toolbar) ────────────────────────

export interface DatasetNodeDef {
  type: string
  label: string
  description: string
  color: string
  defaultData: Record<string, unknown>
}

export const datasetNodeDefs: DatasetNodeDef[] = [
  // ── Missing data ─────────────────────────────────────────────────────────
  {
    type: 'fillNaNNode',
    label: 'Fill NaN',
    description: 'Impute missing values',
    color: t.warning,
    defaultData: { label: 'Fill NaN', strategy: 'mean', constant: 0 },
  },
  {
    type: 'dropDuplicatesNode',
    label: 'Drop Duplicates',
    description: 'Remove identical rows',
    color: t.warning,
    defaultData: { label: 'Drop Duplicates' },
  },
  // ── Feature transforms ───────────────────────────────────────────────────
  {
    type: 'normalizeNode',
    label: 'Normalize',
    description: 'Min-max scale to [0, 1]',
    color: '#6366f1',
    defaultData: { label: 'Normalize', method: 'min-max', columns: '' },
  },
  {
    type: 'standardScalerNode',
    label: 'Standard Scaler',
    description: 'Z-score (mean=0, std=1)',
    color: '#6366f1',
    defaultData: { label: 'Standard Scaler', columns: '' },
  },
  {
    type: 'logTransformNode',
    label: 'Log Transform',
    description: 'Apply log1p to numeric cols',
    color: '#6366f1',
    defaultData: { label: 'Log Transform', columns: '' },
  },
  {
    type: 'clipOutliersNode',
    label: 'Clip Outliers',
    description: 'Clamp beyond N std devs',
    color: '#6366f1',
    defaultData: { label: 'Clip Outliers', stdFactor: 3 },
  },
  {
    type: 'binColumnNode',
    label: 'Bin Column',
    description: 'Discretise continuous column',
    color: '#6366f1',
    defaultData: { label: 'Bin Column', column: '', bins: 5, strategy: 'equal-width' },
  },
  // ── Encoding ─────────────────────────────────────────────────────────────
  {
    type: 'oneHotEncodeNode',
    label: 'One-Hot Encode',
    description: 'Encode categorical columns',
    color: '#6366f1',
    defaultData: { label: 'One-Hot Encode', columns: '' },
  },
  {
    type: 'ordinalEncodeNode',
    label: 'Ordinal Encode',
    description: 'Map categories → integers',
    color: '#6366f1',
    defaultData: { label: 'Ordinal Encode', columns: '' },
  },
  // ── Selection / sampling ─────────────────────────────────────────────────
  {
    type: 'dropColumnsNode',
    label: 'Drop Columns',
    description: 'Remove named columns',
    color: '#6366f1',
    defaultData: { label: 'Drop Columns', columns: '' },
  },
  {
    type: 'selectKBestNode',
    label: 'Select K Best',
    description: 'Keep top K by variance',
    color: '#6366f1',
    defaultData: { label: 'Select K Best', k: 10 },
  },
  {
    type: 'balanceClassesNode',
    label: 'Balance Classes',
    description: 'Oversample minority class',
    color: '#6366f1',
    defaultData: { label: 'Balance Classes' },
  },
  {
    type: 'shuffleNode',
    label: 'Shuffle',
    description: 'Randomise row order',
    color: '#6366f1',
    defaultData: { label: 'Shuffle', seed: 42 },
  },
  {
    type: 'filterNode',
    label: 'Filter',
    description: 'Filter rows by condition',
    color: '#6366f1',
    defaultData: { label: 'Filter', column: '', operator: '>', value: '0' },
  },
  // ── Output ───────────────────────────────────────────────────────────────
  {
    type: 'splitNode',
    label: 'Split',
    description: 'Train / val / test split',
    color: t.success,
    defaultData: { label: 'Split', trainRatio: 0.7, valRatio: 0.15, testRatio: 0.15 },
  },
]
