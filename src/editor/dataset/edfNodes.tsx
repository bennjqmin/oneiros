import { t } from '../../theme/tokens'
import { Handle, Position } from '@xyflow/react'
import type { NodeProps, Node } from '@xyflow/react'

// ── EDF category palette ──────────────────────────────────────────────────────

const C = {
  source:    { border: '#8b5cf6', header: 'rgba(139,92,246,0.12)', text: '#c4b5fd', handle: '#8b5cf6' },
  signal:    { border: '#6366f1', header: 'rgba(99,102,241,0.10)', text: '#a5b4fc', handle: '#6366f1' },
  analysis:  { border: '#06b6d4', header: 'rgba(6,182,212,0.10)',  text: '#67e8f9', handle: '#06b6d4' },
  output:    { border: t.success, header: 'rgba(16,185,129,0.10)', text: '#6ee7b7', handle: t.success },
}

// ── Shared shell ──────────────────────────────────────────────────────────────

function EDFNode({
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
    <div style={{
      width: 200,
      background: t.bgElevated,
      borderRadius: 9,
      border: `1px solid ${selected ? color.border : t.borderDefault}`,
      boxShadow: selected
        ? `0 0 0 1px ${color.border}40, 0 6px 20px rgba(0,0,0,0.4)`
        : '0 3px 12px rgba(0,0,0,0.35)',
      overflow: 'hidden',
      transition: 'border-color 0.15s, box-shadow 0.15s',
    }}>
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

// ── EDF Source ────────────────────────────────────────────────────────────────

type EDFSourceData = Node<{ label: string; datasetName: string; channels?: number; sfreq?: number; duration?: number }, 'edfSourceNode'>

export function EDFSourceNode({ data, selected }: NodeProps<EDFSourceData>) {
  return (
    <EDFNode label="EDF Source" color={C.source} selected={selected} hasTarget={false}>
      <Row label="File" value={data.datasetName || '—'} />
      {data.channels != null && <Row label="Channels" value={data.channels} />}
      {data.sfreq    != null && <Row label="Sampling" value={`${data.sfreq} Hz`} />}
      {data.duration != null && <Row label="Duration" value={`${data.duration.toFixed(1)} s`} />}
    </EDFNode>
  )
}

// ── Bandpass Filter ───────────────────────────────────────────────────────────

type BandpassData = Node<{ label: string; lo: number; hi: number }, 'bandpassNode'>

export function BandpassNode({ data, selected }: NodeProps<BandpassData>) {
  return (
    <EDFNode label="Bandpass Filter" color={C.signal} selected={selected}>
      <Row label="Low"  value={`${data.lo ?? 0.5} Hz`} />
      <Row label="High" value={`${data.hi ?? 40} Hz`} />
    </EDFNode>
  )
}

// ── Notch Filter ──────────────────────────────────────────────────────────────

type NotchData = Node<{ label: string; freq: number }, 'notchFilterNode'>

export function NotchFilterNode({ data, selected }: NodeProps<NotchData>) {
  return (
    <EDFNode label="Notch Filter" color={C.signal} selected={selected}>
      <Row label="Freq" value={`${data.freq ?? 50} Hz`} />
    </EDFNode>
  )
}

// ── Resample ──────────────────────────────────────────────────────────────────

type ResampleData = Node<{ label: string; sfreq: number }, 'edfResampleNode'>

export function EDFResampleNode({ data, selected }: NodeProps<ResampleData>) {
  return (
    <EDFNode label="Resample" color={C.signal} selected={selected}>
      <Row label="Target" value={`${data.sfreq ?? 128} Hz`} />
    </EDFNode>
  )
}

// ── Pick Channels ─────────────────────────────────────────────────────────────

type PickChannelsData = Node<{ label: string; channels: string }, 'pickChannelsNode'>

export function PickChannelsNode({ data, selected }: NodeProps<PickChannelsData>) {
  const ch = data.channels || 'all'
  const display = ch.length > 20 ? ch.slice(0, 20) + '…' : ch
  return (
    <EDFNode label="Pick Channels" color={C.analysis} selected={selected}>
      <Row label="Channels" value={display} />
    </EDFNode>
  )
}

// ── Epoch ─────────────────────────────────────────────────────────────────────

type EpochData = Node<{
  label: string
  mode: 'fixed' | 'event'
  window: number
  overlap: number
  tmin: number
  tmax: number
  eventId: number | null
}, 'epochNode'>

export function EpochNode({ data, selected }: NodeProps<EpochData>) {
  const mode = data.mode ?? 'fixed'
  return (
    <EDFNode label="Epoch" color={C.analysis} selected={selected}>
      <Row label="Mode" value={mode} />
      {mode === 'fixed' ? (
        <>
          <Row label="Window" value={`${data.window ?? 1} s`} />
          <Row label="Overlap" value={`${data.overlap ?? 0} s`} />
        </>
      ) : (
        <>
          <Row label="tmin"    value={`${data.tmin ?? -0.2} s`} />
          <Row label="tmax"    value={`${data.tmax ?? 0.8} s`} />
          {data.eventId != null && <Row label="Event ID" value={data.eventId} />}
        </>
      )}
    </EDFNode>
  )
}

// ── ICA Artefact Removal ──────────────────────────────────────────────────────

type ICAData = Node<{ label: string; nComponents: number; eogChannel: string }, 'icaNode'>

export function ICANode({ data, selected }: NodeProps<ICAData>) {
  return (
    <EDFNode label="ICA" color={C.analysis} selected={selected} hasSource={false}>
      <Row label="Components" value={data.nComponents ?? 15} />
      <Row label="EOG ch"     value={data.eogChannel || 'none'} />
    </EDFNode>
  )
}

// ── EDF Epoch Output (final node) ─────────────────────────────────────────────

type EpochOutputData = Node<{ label: string }, 'edfEpochOutputNode'>

export function EDFEpochOutputNode({ selected }: NodeProps<EpochOutputData>) {
  return (
    <EDFNode label="Epoch Output" color={C.output} selected={selected} hasSource={false}>
      <div style={{ fontSize: 10, color: t.textMuted, lineHeight: 1.5 }}>
        Flatten epochs → tabular training data
      </div>
    </EDFNode>
  )
}

// ── Registry ──────────────────────────────────────────────────────────────────

export const edfNodeTypes = {
  edfSourceNode:      EDFSourceNode,
  bandpassNode:       BandpassNode,
  notchFilterNode:    NotchFilterNode,
  edfResampleNode:    EDFResampleNode,
  pickChannelsNode:   PickChannelsNode,
  epochNode:          EpochNode,
  icaNode:            ICANode,
  edfEpochOutputNode: EDFEpochOutputNode,
}

// ── Palette ───────────────────────────────────────────────────────────────────

export interface EDFNodeDef {
  type: string
  label: string
  description: string
  color: string
  defaultData: Record<string, unknown>
}

export const edfNodeDefs: EDFNodeDef[] = [
  {
    type: 'bandpassNode',
    label: 'Bandpass Filter',
    description: 'Pass frequencies in a band',
    color: '#6366f1',
    defaultData: { label: 'Bandpass', lo: 0.5, hi: 40 },
  },
  {
    type: 'notchFilterNode',
    label: 'Notch Filter',
    description: 'Remove power-line noise',
    color: '#6366f1',
    defaultData: { label: 'Notch Filter', freq: 50 },
  },
  {
    type: 'edfResampleNode',
    label: 'Resample',
    description: 'Change sampling frequency',
    color: '#6366f1',
    defaultData: { label: 'Resample', sfreq: 128 },
  },
  {
    type: 'pickChannelsNode',
    label: 'Pick Channels',
    description: 'Select a subset of channels',
    color: '#06b6d4',
    defaultData: { label: 'Pick Channels', channels: '' },
  },
  {
    type: 'epochNode',
    label: 'Epoch',
    description: 'Segment signal into windows',
    color: '#06b6d4',
    defaultData: { label: 'Epoch', mode: 'fixed', window: 1, overlap: 0, tmin: -0.2, tmax: 0.8, eventId: null },
  },
  {
    type: 'icaNode',
    label: 'ICA',
    description: 'Remove artefacts (eye/ECG)',
    color: '#06b6d4',
    defaultData: { label: 'ICA', nComponents: 15, eogChannel: '' },
  },
  {
    type: 'edfEpochOutputNode',
    label: 'Epoch Output',
    description: 'Export epochs to training',
    color: t.success,
    defaultData: { label: 'Epoch Output' },
  },
]
