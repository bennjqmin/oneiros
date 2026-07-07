import type { AppNode, AppEdge } from '../../types/graph'
import type { ValidationIssue, ValidationResult } from '../../types/validation'
import { topologicalSort } from '../compiler/topoSort'

// ── Cycle detection ───────────────────────────────────────────────────────────

function detectCycles(nodes: AppNode[], edges: AppEdge[]): Set<string> {
  const adj = new Map<string, string[]>()
  for (const node of nodes) adj.set(node.id, [])
  for (const edge of edges) adj.get(edge.source)?.push(edge.target)

  const WHITE = 0, GRAY = 1, BLACK = 2
  const color = new Map<string, number>()
  for (const node of nodes) color.set(node.id, WHITE)
  const cycleNodes = new Set<string>()

  function dfs(id: string): boolean {
    color.set(id, GRAY)
    for (const neighbor of adj.get(id) ?? []) {
      if (color.get(neighbor) === GRAY) { cycleNodes.add(id); cycleNodes.add(neighbor); return true }
      if (color.get(neighbor) === WHITE && dfs(neighbor)) { cycleNodes.add(id); return true }
    }
    color.set(id, BLACK)
    return false
  }
  for (const node of nodes) { if (color.get(node.id) === WHITE) dfs(node.id) }
  return cycleNodes
}

// ── Tensor kind tracking ──────────────────────────────────────────────────────

type TensorKind = 'spatial' | 'flat' | 'unknown'

const SPATIAL_PRODUCERS = new Set(['inputNode', 'conv2dNode', 'conv1dNode', 'maxPool2dNode', 'avgPool2dNode', 'convTranspose2dNode', 'upsampleNode'])
const FLAT_PRODUCERS    = new Set(['denseNode', 'flattenNode', 'adaptiveAvgPool2dNode', 'rnnNode', 'lstmNode', 'gruNode', 'transformerEncoderNode', 'embeddingNode', 'feedForwardNode', 'multiHeadAttentionNode', 'positionalEncodingNode', 'backboneNode', 'hfModelNode', 'reshapeNode'])
const PASSTHROUGH       = new Set(['dropoutNode', 'batchNormNode', 'activationNode', 'gaussianNoiseNode', 'layerNormNode', 'addNode', 'multiplyNode'])
const NEEDS_SPATIAL     = new Set(['conv2dNode', 'maxPool2dNode', 'avgPool2dNode', 'convTranspose2dNode', 'upsampleNode'])
const NEEDS_FLAT        = new Set(['denseNode', 'outputNode', 'rnnNode', 'lstmNode', 'gruNode', 'transformerEncoderNode', 'conv1dNode', 'embeddingNode', 'feedForwardNode', 'multiHeadAttentionNode', 'positionalEncodingNode', 'layerNormNode'])
const MERGE_BINARY      = new Set(['addNode', 'multiplyNode'])

function inferTensorKinds(nodes: AppNode[], edges: AppEdge[]): Map<string, TensorKind> {
  const sorted = topologicalSort(nodes, edges)
  if (!sorted) return new Map()

  const parents = new Map<string, string[]>()
  for (const node of sorted) parents.set(node.id, [])
  for (const edge of edges) parents.get(edge.target)?.push(edge.source)

  const kinds = new Map<string, TensorKind>()
  for (const node of sorted) {
    const type = node.type ?? ''
    const ps = parents.get(node.id) ?? []
    if (SPATIAL_PRODUCERS.has(type)) kinds.set(node.id, 'spatial')
    else if (FLAT_PRODUCERS.has(type)) kinds.set(node.id, 'flat')
    else if (PASSTHROUGH.has(type)) kinds.set(node.id, ps.length > 0 ? (kinds.get(ps[0]) ?? 'unknown') : 'unknown')
    else if (type === 'outputNode') kinds.set(node.id, 'flat')
    else kinds.set(node.id, 'unknown')
  }
  return kinds
}

// ── Shape inference (mirrors backend compiler) ────────────────────────────────

interface FlatShape  { kind: 'flat';    features: number }
interface SpatShape  { kind: 'spatial'; channels: number; height: number; width: number }
type Shape = FlatShape | SpatShape

function flatF(s: Shape) { return s.kind === 'flat' ? s.features : s.channels * s.height * s.width }
function convDim(dim: number, k: number, s: number, p: number) { return Math.max(1, Math.floor((dim + 2 * p - k) / s + 1)) }

function inferShapes(nodes: AppNode[], edges: AppEdge[]): Map<string, Shape> {
  const sorted = topologicalSort(nodes, edges)
  if (!sorted) return new Map()

  const parents = new Map<string, string[]>()
  for (const node of sorted) parents.set(node.id, [])
  for (const edge of edges) parents.get(edge.target)?.push(edge.source)

  const shapes = new Map<string, Shape>()

  for (const node of sorted) {
    const type = node.type ?? ''
    const d = node.data as Record<string, unknown>
    const nid = node.id
    const ps = parents.get(nid) ?? []
    const p0 = ps.length > 0 ? shapes.get(ps[0]) : undefined

    if (type === 'inputNode') {
      shapes.set(nid, { kind: 'spatial', channels: Number(d.channels ?? 1), height: Number(d.height ?? 28), width: Number(d.width ?? 28) })
    } else if (type === 'conv2dNode') {
      const inC = p0?.kind === 'spatial' ? p0.channels : 1
      const ih  = p0?.kind === 'spatial' ? p0.height : 1
      const iw  = p0?.kind === 'spatial' ? p0.width : 1
      const k = Number(d.kernelSize ?? 3), s = Number(d.stride ?? 1), pad = Number(d.padding ?? 0)
      shapes.set(nid, { kind: 'spatial', channels: Number(d.outChannels ?? 32), height: convDim(ih, k, s, pad), width: convDim(iw, k, s, pad) })
      void inC
    } else if (type === 'maxPool2dNode' || type === 'avgPool2dNode') {
      const poolC = p0?.kind === 'spatial' ? p0.channels : 1
      const ih    = p0?.kind === 'spatial' ? p0.height : 1
      const iw    = p0?.kind === 'spatial' ? p0.width : 1
      const k = Number(d.kernelSize ?? 2), s = Number(d.stride ?? k), pad = Number(d.padding ?? 0)
      shapes.set(nid, { kind: 'spatial', channels: poolC, height: convDim(ih, k, s, pad), width: convDim(iw, k, s, pad) })
    } else if (type === 'flattenNode') {
      shapes.set(nid, { kind: 'flat', features: p0 ? flatF(p0) : 0 })
    } else if (type === 'denseNode') {
      const inF = ps.reduce((acc, pid) => acc + (shapes.has(pid) ? flatF(shapes.get(pid)!) : 0), 0)
      shapes.set(nid, { kind: 'flat', features: Number(d.units ?? 128) })
      shapes.set(`${nid}__in`, { kind: 'flat', features: inF })
    } else if (type === 'outputNode') {
      const inF = ps.reduce((acc, pid) => acc + (shapes.has(pid) ? flatF(shapes.get(pid)!) : 0), 0)
      shapes.set(nid, { kind: 'flat', features: Number(d.classes ?? 10) })
      shapes.set(`${nid}__in`, { kind: 'flat', features: inF })
    } else if (PASSTHROUGH.has(type) && p0) {
      shapes.set(nid, p0)
    } else if (type === 'rnnNode' || type === 'gruNode' || type === 'lstmNode') {
      const h = Number(d.hiddenSize ?? 128), bidir = Boolean(d.bidirectional)
      shapes.set(nid, { kind: 'flat', features: h * (bidir ? 2 : 1) })
    } else if (type === 'transformerEncoderNode') {
      shapes.set(nid, { kind: 'flat', features: Number(d.dModel ?? 256) })
    } else if (type === 'adaptiveAvgPool2dNode') {
      const ic = p0?.kind === 'spatial' ? p0.channels : 1
      const sz = Number(d.outputSize ?? 1)
      shapes.set(nid, sz === 1 ? { kind: 'flat', features: ic } : { kind: 'spatial', channels: ic, height: sz, width: sz })
    } else if (type === 'convTranspose2dNode') {
      const ih = p0?.kind === 'spatial' ? p0.height : 1
      const iw = p0?.kind === 'spatial' ? p0.width : 1
      const outC = Number(d.outChannels ?? 32)
      const k = Number(d.kernelSize ?? 2), s = Number(d.stride ?? 2), pad = Number(d.padding ?? 0)
      const outPad = Number(d.outputPadding ?? 0)
      shapes.set(nid, { kind: 'spatial', channels: outC, height: (ih - 1) * s - 2 * pad + k + outPad, width: (iw - 1) * s - 2 * pad + k + outPad })
    } else if (type === 'upsampleNode') {
      const ic = p0?.kind === 'spatial' ? p0.channels : 1
      const ih = p0?.kind === 'spatial' ? p0.height : 1
      const iw = p0?.kind === 'spatial' ? p0.width : 1
      const sf = Number(d.scaleFactor ?? 2)
      shapes.set(nid, { kind: 'spatial', channels: ic, height: Math.floor(ih * sf), width: Math.floor(iw * sf) })
    } else if (type === 'backboneNode') {
      const feats: Record<string, number> = { resnet18: 512, resnet34: 512, resnet50: 2048, mobilenet_v2: 1280, efficientnet_b0: 1280, vgg16: 4096 }
      shapes.set(nid, { kind: 'flat', features: feats[String(d.model ?? 'resnet18')] ?? 512 })
    } else if (type === 'hfModelNode') {
      const outFeats = Number(d.outputFeatures ?? 0)
      shapes.set(nid, { kind: 'flat', features: outFeats > 0 ? outFeats : 768 })
    } else if (type === 'reshapeNode') {
      const target = Number(d.targetFeatures ?? -1)
      shapes.set(nid, target === -1 && p0 ? p0 : { kind: 'flat', features: target })
    } else if (type === 'embeddingNode') {
      const embDim = Number(d.embeddingDim ?? 128)
      const seqLen = p0 ? flatF(p0) : 1
      shapes.set(nid, { kind: 'flat', features: seqLen * embDim })
    } else if (type === 'positionalEncodingNode') {
      shapes.set(nid, { kind: 'flat', features: Number(d.dModel ?? 256) })
    } else if (type === 'feedForwardNode') {
      shapes.set(nid, { kind: 'flat', features: Number(d.dModel ?? 256) })
    } else if (type === 'multiHeadAttentionNode') {
      shapes.set(nid, { kind: 'flat', features: Number(d.embedDim ?? 256) })
    } else if (type === 'concatenateNode') {
      const pShapes = ps.map((pid) => shapes.get(pid)).filter(Boolean) as Shape[]
      if (pShapes.every((s) => s.kind === 'flat')) {
        shapes.set(nid, { kind: 'flat', features: pShapes.reduce((acc, s) => acc + flatF(s), 0) })
      } else if (pShapes.length > 0 && pShapes.every((s) => s.kind === 'spatial')) {
        const ref = pShapes[0] as SpatShape
        shapes.set(nid, { kind: 'spatial', channels: pShapes.reduce((acc, s) => acc + (s as SpatShape).channels, 0), height: ref.height, width: ref.width })
      } else if (p0) {
        shapes.set(nid, p0)
      }
    } else if (type === 'addNode' || type === 'multiplyNode') {
      if (p0) shapes.set(nid, p0)
    }
  }
  return shapes
}

// ── Human-readable node label ─────────────────────────────────────────────────

function nodeLabel(type: string, data: Record<string, unknown>): string {
  const labels: Record<string, string> = {
    inputNode: 'Input', outputNode: 'Output', denseNode: 'Dense', conv2dNode: 'Conv2d',
    conv1dNode: 'Conv1d', maxPool2dNode: 'MaxPool2d', avgPool2dNode: 'AvgPool2d',
    flattenNode: 'Flatten', dropoutNode: 'Dropout', batchNormNode: 'BatchNorm',
    activationNode: 'Activation', rnnNode: 'RNN', lstmNode: 'LSTM', gruNode: 'GRU',
    transformerEncoderNode: 'TransformerEncoder', adaptiveAvgPool2dNode: 'AdaptiveAvgPool2d',
    gaussianNoiseNode: 'Gaussian Noise', reshapeNode: 'Reshape', layerNormNode: 'Layer Norm',
    embeddingNode: 'Embedding', positionalEncodingNode: 'Positional Encoding',
    feedForwardNode: 'Feed Forward', multiHeadAttentionNode: 'Multi-Head Attention',
    concatenateNode: 'Concatenate', addNode: 'Add', multiplyNode: 'Multiply',
    convTranspose2dNode: 'ConvTranspose2d', upsampleNode: 'Upsample', backboneNode: 'Backbone', hfModelNode: 'HF Model',
  }
  return (typeof data.label === 'string' && data.label) ? data.label : (labels[type] ?? type)
}

// ── Main entry point ──────────────────────────────────────────────────────────

export interface GraphValidationOptions {
  /** If provided, used to validate Input node dimensions against dataset */
  customFeatureCount?: number | null
  /** If provided, used to validate Output node class count against dataset */
  customClassCount?: number | null
  /** CV: [C, H, W] shape of images in the dataset */
  cvInputShape?: [number, number, number] | null
  /** CV: number of classes in the image dataset */
  cvClassCount?: number | null
}

export function validateGraph(
  nodes: AppNode[],
  edges: AppEdge[],
  opts: GraphValidationOptions = {},
): ValidationResult {
  const issues: ValidationIssue[] = []

  if (nodes.length === 0) {
    issues.push({ severity: 'error', category: 'structure', message: 'Graph is empty — add at least an Input and Output node.' })
    return { issues, isValid: false }
  }

  const inCount  = new Map<string, number>()
  const outCount = new Map<string, number>()
  for (const n of nodes) { inCount.set(n.id, 0); outCount.set(n.id, 0) }
  for (const e of edges) {
    inCount.set(e.target,  (inCount.get(e.target)  ?? 0) + 1)
    outCount.set(e.source, (outCount.get(e.source) ?? 0) + 1)
  }

  const cycleNodes  = detectCycles(nodes, edges)
  const tensorKinds = inferTensorKinds(nodes, edges)
  const shapes      = inferShapes(nodes, edges)

  const parents = new Map<string, string[]>()
  for (const n of nodes) parents.set(n.id, [])
  for (const e of edges) parents.get(e.target)?.push(e.source)

  const inputNodes  = nodes.filter(n => n.type === 'inputNode')
  const outputNodes = nodes.filter(n => n.type === 'outputNode')

  // ── Graph-level checks ────────────────────────────────────────────────────

  if (inputNodes.length === 0) {
    issues.push({ severity: 'error', category: 'structure',
      message: 'No Input node found.',
      hint: 'Drag an Input node from the sidebar and connect it to the first layer.' })
  }
  if (outputNodes.length === 0) {
    issues.push({ severity: 'error', category: 'structure',
      message: 'No Output node found.',
      hint: 'Drag an Output node from the sidebar and connect it to the last Dense layer.' })
  }
  if (inputNodes.length > 1) {
    issues.push({ severity: 'warning', category: 'structure',
      message: `${inputNodes.length} Input nodes found — only the first is used.` })
  }

  // ── Per-node checks ───────────────────────────────────────────────────────

  for (const node of nodes) {
    const type  = node.type ?? ''
    const data  = (node.data ?? {}) as Record<string, unknown>
    const nid   = node.id
    const label = nodeLabel(type, data)
    const incoming = inCount.get(nid)  ?? 0
    const outgoing = outCount.get(nid) ?? 0
    const ps       = parents.get(nid)  ?? []

    // Cycles
    if (cycleNodes.has(nid)) {
      issues.push({ nodeId: nid, severity: 'error', category: 'structure',
        message: `"${label}" is part of a cycle.`,
        hint: 'Remove the connection that creates the loop.' })
    }

    // Input has incoming edges
    if (type === 'inputNode' && incoming > 0) {
      issues.push({ nodeId: nid, severity: 'error', category: 'structure',
        message: `Input node "${label}" has an incoming connection — it must be the first node.`,
        hint: 'Delete the incoming edge on this Input node.' })
    }

    // Output has outgoing edges
    if (type === 'outputNode' && outgoing > 0) {
      issues.push({ nodeId: nid, severity: 'error', category: 'structure',
        message: `Output node "${label}" has an outgoing connection — it must be the last node.`,
        hint: 'Delete the outgoing edge on this Output node.' })
    }

    // Disconnected non-input node
    if (type !== 'inputNode' && incoming === 0) {
      issues.push({ nodeId: nid, severity: 'warning', category: 'structure',
        message: `"${label}" has no incoming connection.`,
        hint: 'Connect a preceding layer to this node, or remove it if unused.' })
    }

    // Disconnected non-output node
    if (type !== 'outputNode' && outgoing === 0) {
      issues.push({ nodeId: nid, severity: 'warning', category: 'structure',
        message: `"${label}" output is not connected to anything.`,
        hint: 'Connect this node to the next layer, or remove it if unused.' })
    }

    // ── Spatial / flat type checks ─────────────────────────────────────────

    if (NEEDS_SPATIAL.has(type) && ps.length > 0) {
      for (const pid of ps) {
        if (tensorKinds.get(pid) === 'flat') {
          const name = type === 'conv2dNode' ? 'Conv2d' : type === 'maxPool2dNode' ? 'MaxPool2d' : 'AvgPool2d'
          issues.push({ nodeId: nid, severity: 'error', category: 'shape',
            message: `"${label}" (${name}) requires a spatial (image) tensor but receives a flat one.`,
            hint: 'Connect from an Input node or a Conv2d layer, not from a Dense or Flatten node.' })
        }
      }
    }

    if (NEEDS_FLAT.has(type) && ps.length > 0) {
      for (const pid of ps) {
        if (tensorKinds.get(pid) === 'spatial') {
          const name = type === 'denseNode' ? 'Dense' : type === 'outputNode' ? 'Output' : type
          issues.push({ nodeId: nid, severity: 'error', category: 'shape',
            message: `"${label}" (${name}) requires a flat tensor but receives a spatial one.`,
            hint: 'Add a Flatten node between the previous layer and this one.' })
        }
      }
    }

    if (type === 'flattenNode' && ps.length > 0 && tensorKinds.get(ps[0]) === 'flat') {
      issues.push({ nodeId: nid, severity: 'warning', category: 'shape',
        message: `"${label}" (Flatten) is receiving an already-flat tensor — this step is redundant.` })
    }

    // ── Config checks ──────────────────────────────────────────────────────

    if (type === 'conv2dNode') {
      const outC  = Number(data.outChannels ?? 32)
      const grp   = Number(data.groups ?? 1)
      const shape = shapes.get(nid)
      const inC   = ps.length > 0 ? (() => { const s = shapes.get(ps[0]); return s?.kind === 'spatial' ? s.channels : 1 })() : 1
      if (grp > 1 && inC % grp !== 0) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: `"${label}" groups=${grp} doesn't divide in_channels=${inC}.`,
          hint: `Set groups to a value that divides both ${inC} (in_channels) and ${outC} (out_channels).` })
      }
      if (grp > 1 && outC % grp !== 0) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: `"${label}" groups=${grp} doesn't divide out_channels=${outC}.`,
          hint: `Set groups to a value that divides both ${inC} (in_channels) and ${outC} (out_channels).` })
      }
      if (shape?.kind === 'spatial' && (shape.height <= 0 || shape.width <= 0)) {
        issues.push({ nodeId: nid, severity: 'error', category: 'shape',
          message: `"${label}" produces zero-size output (kernel too large for input).`,
          hint: 'Reduce kernel_size or stride, or add padding.' })
      }
    }

    if (type === 'denseNode' || type === 'outputNode') {
      const inShape  = shapes.get(`${nid}__in`)
      const outShape = shapes.get(nid)
      if (inShape && flatF(inShape) === 0) {
        issues.push({ nodeId: nid, severity: 'error', category: 'shape',
          message: `"${label}" has zero input features — nothing is connected.`,
          hint: 'Connect a layer that produces output (Dense, Flatten, etc.) to this node.' })
      }
      if (type === 'outputNode' && outShape && flatF(outShape) === 0) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: `"${label}" has 0 output classes.`,
          hint: 'Set "classes" to the number of classes in your dataset (e.g. 10 for MNIST).' })
      }
    }

    if (type === 'dropoutNode') {
      const p = Number(data.p ?? 0.5)
      if (p < 0 || p >= 1) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: `"${label}" dropout probability ${p} is out of range.`,
          hint: 'Set p to a value in [0, 1).' })
      }
    }

    if (type === 'concatenateNode' && incoming < 2) {
      issues.push({ nodeId: nid, severity: 'error', category: 'structure',
        message: `"${label}" requires at least 2 inputs.`,
        hint: 'Connect two or more layers to this Concatenate node.' })
    }

    if (MERGE_BINARY.has(type)) {
      const handles = new Set(edges.filter((e) => e.target === nid).map((e) => e.targetHandle ?? ''))
      if (!handles.has('in_a') || !handles.has('in_b')) {
        issues.push({ nodeId: nid, severity: 'error', category: 'structure',
          message: `"${label}" requires connections to both inputs (A and B).`,
          hint: 'Connect one layer to the top handle and another to the bottom handle.' })
      }
      if (ps.length >= 2) {
        const s0 = shapes.get(ps[0])
        const s1 = shapes.get(ps[1])
        if (s0 && s1 && flatF(s0) !== flatF(s1)) {
          issues.push({ nodeId: nid, severity: 'error', category: 'shape',
            message: `"${label}" inputs have incompatible shapes.`,
            hint: 'Both inputs must produce tensors with the same shape.' })
        }
      }
    }

    if (type === 'multiHeadAttentionNode') {
      const embedDim = Number(data.embedDim ?? 256)
      const numHeads = Number(data.numHeads ?? 8)
      if (embedDim % numHeads !== 0) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: `"${label}" embed_dim (${embedDim}) must be divisible by num_heads (${numHeads}).`,
          hint: 'Adjust embed_dim or num_heads so embed_dim % num_heads == 0.' })
      }
    }

    if (type === 'feedForwardNode' || type === 'positionalEncodingNode') {
      const dModel = Number(data.dModel ?? 256)
      if (ps.length > 0) {
        const inShape = shapes.get(ps[0])
        if (inShape && inShape.kind === 'flat' && flatF(inShape) !== dModel) {
          issues.push({ nodeId: nid, severity: 'warning', category: 'shape',
            message: `"${label}" expects d_model=${dModel} but input has ${flatF(inShape)} features.`,
            hint: `Add a Dense layer projecting to ${dModel} features before this node.` })
        }
      }
    }

    if ((type === 'rnnNode' || type === 'lstmNode' || type === 'gruNode') && Number(data.numLayers ?? 1) > 1 && Number(data.dropout ?? 0) === 0) {
      issues.push({ nodeId: nid, severity: 'info', category: 'config',
        message: `"${label}" has ${data.numLayers} layers with no dropout between them.`,
        hint: 'Consider setting dropout > 0 to regularise a deep RNN.' })
    }

    // ── Input node ↔ dataset checks ────────────────────────────────────────

    if (type === 'inputNode') {
      const ch = Number(data.channels ?? 1)
      const h  = Number(data.height   ?? 28)
      const w  = Number(data.width    ?? 28)
      const totalFeatures = ch * h * w

      if (opts.customFeatureCount != null) {
        if (totalFeatures !== opts.customFeatureCount) {
          issues.push({ nodeId: nid, severity: 'error', category: 'dataset',
            message: `Input node has ${totalFeatures} features (${ch}×${h}×${w}) but your CSV dataset has ${opts.customFeatureCount} features.`,
            hint: `Set channels=${opts.customFeatureCount}, height=1, width=1 on the Input node to match your data.` })
        }
      }

      if (opts.cvInputShape != null) {
        const [expC, expH, expW] = opts.cvInputShape
        if (ch !== expC) {
          issues.push({ nodeId: nid, severity: 'error', category: 'dataset',
            message: `Input node has ${ch} channel(s) but your image dataset uses ${expC} channel(s).`,
            hint: `Set channels=${expC} on the Input node to match your image dataset.` })
        }
        if (h !== expH || w !== expW) {
          issues.push({ nodeId: nid, severity: 'warning', category: 'dataset',
            message: `Input node is ${h}×${w} but images were detected as ${expH}×${expW}.`,
            hint: `Add a Resize augmentation node (${expW}×${expH}) or update Input H/W to match.` })
        }
      }
    }

    // ── BackboneNode checks ───────────────────────────────────────────────

    if (type === 'backboneNode') {
      const parentIds = parents.get(nid) ?? []
      const parentNode = nodes.find(n => parentIds.includes(n.id))
      const inputChannels = parentNode?.data?.channels as number | undefined

      if (inputChannels != null && inputChannels !== 3) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: `Pretrained backbone expects 3-channel (RGB) input but Input node has ${inputChannels} channels.`,
          hint: 'Set channels=3 on the Input node. Most pretrained models are trained on RGB images.' })
      }
      const h = parentNode?.data?.height as number | undefined
      const w = parentNode?.data?.width  as number | undefined
      if (h != null && w != null && (h < 32 || w < 32)) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: `Pretrained backbone requires at least 32×32 input, but Input is ${h}×${w}.`,
          hint: 'Add a Resize augmentation node to at least 32×32, and update Input H/W accordingly.' })
      }
      const isFrozen    = data.freeze as boolean | undefined
      const isPretrained = data.pretrained as boolean | undefined
      if (isFrozen && !isPretrained) {
        issues.push({ nodeId: nid, severity: 'warning', category: 'config',
          message: 'Backbone is frozen but not using pretrained weights — training will not update any parameters.',
          hint: 'Either enable pretrained weights, or disable freeze so random weights can train.' })
      }
    }

    // ── Hugging Face model checks ─────────────────────────────────────────

    if (type === 'hfModelNode') {
      const modelId = String(data.modelId ?? '').trim()
      const outputFeatures = Number(data.outputFeatures ?? 0)

      if (!modelId) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: 'Hugging Face node has no model ID.',
          hint: 'Open the Hugging Face tab, validate a model, and click “Add to canvas”.' })
      }
      if (outputFeatures <= 0) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: 'Hugging Face node output feature size is unknown.',
          hint: 'Re-import the model from the Hugging Face tab so outputFeatures is set.' })
      }

      const modelKind = String(data.modelKind ?? 'generic')
      const parentIds = parents.get(nid) ?? []
      const parentNode = nodes.find(n => parentIds.includes(n.id))
      const inputChannels = parentNode?.data?.channels as number | undefined

      if (modelKind === 'vision' && inputChannels != null && inputChannels !== 3) {
        issues.push({ nodeId: nid, severity: 'error', category: 'config',
          message: `Vision HF model expects 3-channel RGB input but upstream has ${inputChannels} channels.`,
          hint: 'Set channels=3 on the Input node (e.g. 224×224 for ViT).' })
      }
    }

    if (type === 'outputNode') {
      const classes = Number(data.classes ?? 10)
      if (opts.customClassCount != null && classes !== opts.customClassCount) {
        issues.push({ nodeId: nid, severity: 'warning', category: 'dataset',
          message: `Output node has ${classes} classes but your dataset has ${opts.customClassCount} unique target values.`,
          hint: `Set "classes" on the Output node to ${opts.customClassCount}.` })
      }
      if (opts.cvClassCount != null && classes !== opts.cvClassCount) {
        issues.push({ nodeId: nid, severity: 'warning', category: 'dataset',
          message: `Output node has ${classes} classes but your image dataset has ${opts.cvClassCount} classes.`,
          hint: `Set "classes" on the Output node to ${opts.cvClassCount} to match your image folders.` })
      }
    }
  }

  return {
    issues,
    isValid: issues.every(i => i.severity !== 'error'),
  }
}
