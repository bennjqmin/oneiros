import { t } from '../../theme/tokens'
import { Handle, Position } from '@xyflow/react'
import type { NodeProps } from '@xyflow/react'
import type { NodeData } from '../nodeTypes'

// ── CV Augmentation palette (teal / cyan) ─────────────────────────────────────

const C = {
  source:  { border: '#0ea5e9', header: 'rgba(14,165,233,0.12)',  text: '#7dd3fc', handle: '#0ea5e9' },
  spatial: { border: '#06b6d4', header: 'rgba(6,182,212,0.11)',   text: '#67e8f9', handle: '#06b6d4' },
  color:   { border: t.warning, header: 'rgba(245,158,11,0.10)',  text: '#fcd34d', handle: t.warning },
  noise:   { border: '#8b5cf6', header: 'rgba(139,92,246,0.10)',  text: '#c4b5fd', handle: '#8b5cf6' },
  norm:    { border: t.success, header: 'rgba(16,185,129,0.10)',  text: '#6ee7b7', handle: t.success },
}

// ── Shared shell ──────────────────────────────────────────────────────────────

function AugNode({
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
      width: 210,
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
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 3 }}>
      <span style={{ fontSize: 10, color: t.textMuted }}>{label}</span>
      <span style={{ fontSize: 11, color: t.textPrimary, fontVariantNumeric: 'tabular-nums' }}>{value}</span>
    </div>
  )
}

// ── Nodes ─────────────────────────────────────────────────────────────────────

export function AugSourceNode({ data, selected }: NodeProps<NodeData & Record<string, unknown>>) {
  const name = String(data.name ?? 'Image Dataset')
  const total = Number(data.totalImages ?? 0)
  const classes = Number(data.classCount ?? '?')
  const shape = data.inputShape as number[] | undefined
  const shapeStr = shape ? `${shape[0]}×${shape[1]}×${shape[2]}` : '?'
  return (
    <AugNode label="Image Dataset" color={C.source} selected={selected} hasTarget={false}>
      <div style={{ marginBottom: 6, fontSize: 12, fontWeight: 600, color: t.textPrimary, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{name}</div>
      <Row label="Total images" value={total.toLocaleString()} />
      <Row label="Classes" value={classes} />
      <Row label="Shape (C×H×W)" value={shapeStr} />
    </AugNode>
  )
}

export function ResizeNode({ data, selected }: NodeProps<NodeData & Record<string, unknown>>) {
  return (
    <AugNode label="Resize" color={C.spatial} selected={selected}>
      <Row label="Width"  value={Number(data.width  ?? 224)} />
      <Row label="Height" value={Number(data.height ?? 224)} />
    </AugNode>
  )
}

export function RandomCropNode({ data, selected }: NodeProps<NodeData & Record<string, unknown>>) {
  return (
    <AugNode label="Random Crop" color={C.spatial} selected={selected}>
      <Row label="Size"    value={Number(data.size    ?? 224)} />
      <Row label="Padding" value={Number(data.padding ?? 0)} />
    </AugNode>
  )
}

export function CenterCropNode({ data, selected }: NodeProps<NodeData & Record<string, unknown>>) {
  return (
    <AugNode label="Center Crop" color={C.spatial} selected={selected}>
      <Row label="Size" value={Number(data.size ?? 224)} />
    </AugNode>
  )
}

export function RandomHFlipNode({ data, selected }: NodeProps<NodeData & Record<string, unknown>>) {
  return (
    <AugNode label="Random H-Flip" color={C.spatial} selected={selected}>
      <Row label="Probability" value={Number(data.probability ?? 0.5).toFixed(2)} />
    </AugNode>
  )
}

export function RandomVFlipNode({ data, selected }: NodeProps<NodeData & Record<string, unknown>>) {
  return (
    <AugNode label="Random V-Flip" color={C.spatial} selected={selected}>
      <Row label="Probability" value={Number(data.probability ?? 0.5).toFixed(2)} />
    </AugNode>
  )
}

export function ColorJitterNode({ data, selected }: NodeProps<NodeData & Record<string, unknown>>) {
  return (
    <AugNode label="Color Jitter" color={C.color} selected={selected}>
      <Row label="Brightness" value={Number(data.brightness ?? 0.2).toFixed(2)} />
      <Row label="Contrast"   value={Number(data.contrast   ?? 0.2).toFixed(2)} />
      <Row label="Saturation" value={Number(data.saturation ?? 0.2).toFixed(2)} />
      <Row label="Hue"        value={Number(data.hue        ?? 0.0).toFixed(2)} />
    </AugNode>
  )
}

export function NormalizeNode({ data, selected }: NodeProps<NodeData & Record<string, unknown>>) {
  return (
    <AugNode label="Normalize" color={C.norm} selected={selected}>
      <Row label="Mean R/G/B" value={`${Number(data.meanR ?? 0.485).toFixed(3)} / ${Number(data.meanG ?? 0.456).toFixed(3)} / ${Number(data.meanB ?? 0.406).toFixed(3)}`} />
      <Row label="Std R/G/B"  value={`${Number(data.stdR  ?? 0.229).toFixed(3)} / ${Number(data.stdG  ?? 0.224).toFixed(3)} / ${Number(data.stdB  ?? 0.225).toFixed(3)}`} />
    </AugNode>
  )
}

export function RandomRotationNode({ data, selected }: NodeProps<NodeData & Record<string, unknown>>) {
  return (
    <AugNode label="Random Rotation" color={C.spatial} selected={selected}>
      <Row label="Degrees" value={Number(data.degrees ?? 15)} />
    </AugNode>
  )
}

export function GaussianBlurNode({ data, selected }: NodeProps<NodeData & Record<string, unknown>>) {
  return (
    <AugNode label="Gaussian Blur" color={C.noise} selected={selected}>
      <Row label="Kernel Size" value={Number(data.kernelSize ?? 3)} />
      <Row label="Sigma"       value={Number(data.sigma      ?? 1.0).toFixed(1)} />
    </AugNode>
  )
}

// ── Registration ──────────────────────────────────────────────────────────────

export const augNodeTypes = {
  augSource:      AugSourceNode,
  resize:         ResizeNode,
  randomCrop:     RandomCropNode,
  centerCrop:     CenterCropNode,
  randomHFlip:    RandomHFlipNode,
  randomVFlip:    RandomVFlipNode,
  colorJitter:    ColorJitterNode,
  normalize:      NormalizeNode,
  randomRotation: RandomRotationNode,
  gaussianBlur:   GaussianBlurNode,
}

// Palette definitions (shown in the sidebar)
export const augNodeDefs: {
  type: string
  label: string
  category: string
  description: string
  defaults: Record<string, unknown>
}[] = [
  { type: 'resize',         label: 'Resize',           category: 'spatial', description: 'Rescale all images to a fixed size',               defaults: { width: 224, height: 224 } },
  { type: 'randomCrop',     label: 'Random Crop',      category: 'spatial', description: 'Randomly crop a patch from each image',            defaults: { size: 224, padding: 0 } },
  { type: 'centerCrop',     label: 'Center Crop',      category: 'spatial', description: 'Crop the center region of each image',             defaults: { size: 224 } },
  { type: 'randomHFlip',    label: 'Random H-Flip',    category: 'spatial', description: 'Randomly flip images horizontally',                defaults: { probability: 0.5 } },
  { type: 'randomVFlip',    label: 'Random V-Flip',    category: 'spatial', description: 'Randomly flip images vertically',                  defaults: { probability: 0.5 } },
  { type: 'randomRotation', label: 'Random Rotation',  category: 'spatial', description: 'Randomly rotate images up to ±N degrees',          defaults: { degrees: 15 } },
  { type: 'colorJitter',    label: 'Color Jitter',     category: 'color',   description: 'Randomly alter brightness / contrast / saturation', defaults: { brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.0 } },
  { type: 'gaussianBlur',   label: 'Gaussian Blur',    category: 'color',   description: 'Apply random Gaussian blur',                       defaults: { kernelSize: 3, sigma: 1.0 } },
  { type: 'normalize',      label: 'Normalize',        category: 'norm',    description: 'Subtract mean, divide by std (ImageNet defaults)',   defaults: { meanR: 0.485, meanG: 0.456, meanB: 0.406, stdR: 0.229, stdG: 0.224, stdB: 0.225 } },
]
