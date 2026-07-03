import type { AppNode, AppEdge } from '../../types/graph'

// ── Shape types ───────────────────────────────────────────────────────────────

export type TensorShape =
  | { kind: 'flat'; features: number }
  | { kind: 'spatial'; channels: number; height: number; width: number }

/** Flat feature count for a shape. */
export function flatFeatures(s: TensorShape): number {
  return s.kind === 'flat' ? s.features : s.channels * s.height * s.width
}

/** Conv / Pool output dimension: floor((in + 2*p - k) / s + 1) */
function convDim(dim: number, kernel: number, stride: number, padding: number): number {
  return Math.max(1, Math.floor((dim + 2 * padding - kernel) / stride + 1))
}

// ── Main inference ────────────────────────────────────────────────────────────

export function inferShapes(
  sortedNodes: AppNode[],
  edges: AppEdge[],
): Map<string, TensorShape> {
  const shapes = new Map<string, TensorShape>()

  const parents = new Map<string, string[]>()
  for (const node of sortedNodes) parents.set(node.id, [])
  for (const edge of edges) parents.get(edge.target)?.push(edge.source)

  for (const node of sortedNodes) {
    const type = node.type ?? ''
    const data = node.data as Record<string, unknown>
    const nodeParents = parents.get(node.id) ?? []
    const firstParentShape = shapes.get(nodeParents[0] ?? '')

    // ── Input ──────────────────────────────────────────────────────────────────
    if (type === 'inputNode') {
      shapes.set(node.id, {
        kind: 'spatial',
        channels: (data.channels as number | undefined) ?? 1,
        height: (data.height as number | undefined) ?? 28,
        width: (data.width as number | undefined) ?? 28,
      })

    // ── Conv2d ─────────────────────────────────────────────────────────────────
    } else if (type === 'conv2dNode') {
      const inH = firstParentShape?.kind === 'spatial' ? firstParentShape.height : 1
      const inW = firstParentShape?.kind === 'spatial' ? firstParentShape.width : 1
      const outC = (data.outChannels as number | undefined) ?? 32
      const k = (data.kernelSize as number | undefined) ?? 3
      const s = (data.stride as number | undefined) ?? 1
      const p = (data.padding as number | undefined) ?? 0
      shapes.set(node.id, { kind: 'spatial', channels: outC, height: convDim(inH, k, s, p), width: convDim(inW, k, s, p) })

    // ── Conv1d (flat → flat: treats features as channels, length=1) ────────────
    } else if (type === 'conv1dNode') {
      const outC = (data.outChannels as number | undefined) ?? 64
      // Conv1d on a (batch, inC, 1) tensor with padding=same gives (batch, outC, 1) → flat
      shapes.set(node.id, { kind: 'flat', features: outC })

    // ── MaxPool2d ──────────────────────────────────────────────────────────────
    } else if (type === 'maxPool2dNode') {
      const inC = firstParentShape?.kind === 'spatial' ? firstParentShape.channels : 1
      const inH = firstParentShape?.kind === 'spatial' ? firstParentShape.height : 1
      const inW = firstParentShape?.kind === 'spatial' ? firstParentShape.width : 1
      const k = (data.kernelSize as number | undefined) ?? 2
      const s = (data.stride as number | undefined) ?? k
      shapes.set(node.id, { kind: 'spatial', channels: inC, height: convDim(inH, k, s, 0), width: convDim(inW, k, s, 0) })

    // ── AvgPool2d ──────────────────────────────────────────────────────────────
    } else if (type === 'avgPool2dNode') {
      const inC = firstParentShape?.kind === 'spatial' ? firstParentShape.channels : 1
      const inH = firstParentShape?.kind === 'spatial' ? firstParentShape.height : 1
      const inW = firstParentShape?.kind === 'spatial' ? firstParentShape.width : 1
      const k = (data.kernelSize as number | undefined) ?? 2
      const s = (data.stride as number | undefined) ?? k
      shapes.set(node.id, { kind: 'spatial', channels: inC, height: convDim(inH, k, s, 0), width: convDim(inW, k, s, 0) })

    // ── AdaptiveAvgPool2d (spatial → flat when outputSize=1, else spatial) ─────
    } else if (type === 'adaptiveAvgPool2dNode') {
      const inC = firstParentShape?.kind === 'spatial' ? firstParentShape.channels : 1
      const outSz = (data.outputSize as number | undefined) ?? 1
      if (outSz === 1) {
        shapes.set(node.id, { kind: 'flat', features: inC })
      } else {
        shapes.set(node.id, { kind: 'spatial', channels: inC, height: outSz, width: outSz })
      }

    // ── Flatten ────────────────────────────────────────────────────────────────
    } else if (type === 'flattenNode') {
      const features = firstParentShape ? flatFeatures(firstParentShape) : 0
      shapes.set(node.id, { kind: 'flat', features })

    // ── ConvTranspose2d (upsampling via transposed convolution) ───────────────
    } else if (type === 'convTranspose2dNode') {
      const inH = firstParentShape?.kind === 'spatial' ? firstParentShape.height : 1
      const inW = firstParentShape?.kind === 'spatial' ? firstParentShape.width : 1
      const outC   = (data.outChannels   as number | undefined) ?? 32
      const k      = (data.kernelSize    as number | undefined) ?? 2
      const s      = (data.stride        as number | undefined) ?? 2
      const p      = (data.padding       as number | undefined) ?? 0
      const outPad = (data.outputPadding as number | undefined) ?? 0
      const outH = (inH - 1) * s - 2 * p + k + outPad
      const outW = (inW - 1) * s - 2 * p + k + outPad
      shapes.set(node.id, { kind: 'spatial', channels: outC, height: outH, width: outW })

    // ── Upsample (scale spatial dims) ─────────────────────────────────────────
    } else if (type === 'upsampleNode') {
      const inC = firstParentShape?.kind === 'spatial' ? firstParentShape.channels : 1
      const inH = firstParentShape?.kind === 'spatial' ? firstParentShape.height : 1
      const inW = firstParentShape?.kind === 'spatial' ? firstParentShape.width : 1
      const sf = (data.scaleFactor as number | undefined) ?? 2
      shapes.set(node.id, { kind: 'spatial', channels: inC, height: Math.floor(inH * sf), width: Math.floor(inW * sf) })

    // ── Backbone (pretrained vision model → flat feature vector) ──────────────
    } else if (type === 'backboneNode') {
      const feats: Record<string, number> = {
        resnet18: 512, resnet34: 512, resnet50: 2048,
        mobilenet_v2: 1280, efficientnet_b0: 1280, vgg16: 4096,
      }
      const model = (data.model as string | undefined) ?? 'resnet18'
      shapes.set(node.id, { kind: 'flat', features: feats[model] ?? 512 })

    // ── Dropout / BatchNorm / Activation / GaussianNoise / LayerNorm (passthrough) ──
    } else if (type === 'dropoutNode' || type === 'batchNormNode' || type === 'activationNode'
      || type === 'gaussianNoiseNode' || type === 'layerNormNode') {
      if (firstParentShape) shapes.set(node.id, { ...firstParentShape })

    // ── Reshape ────────────────────────────────────────────────────────────────
    } else if (type === 'reshapeNode') {
      const target = (data.targetFeatures as number | undefined) ?? -1
      if (target === -1 && firstParentShape) {
        shapes.set(node.id, { ...firstParentShape })
      } else {
        shapes.set(node.id, { kind: 'flat', features: target })
      }

    // ── Embedding ──────────────────────────────────────────────────────────────
    } else if (type === 'embeddingNode') {
      const embDim = (data.embeddingDim as number | undefined) ?? 128
      const seqLen = firstParentShape ? flatFeatures(firstParentShape) : 1
      shapes.set(node.id, { kind: 'flat', features: seqLen * embDim })

    // ── Positional Encoding ────────────────────────────────────────────────────
    } else if (type === 'positionalEncodingNode') {
      const dModel = (data.dModel as number | undefined) ?? 256
      shapes.set(node.id, { kind: 'flat', features: dModel })

    // ── Feed Forward ───────────────────────────────────────────────────────────
    } else if (type === 'feedForwardNode') {
      const dModel = (data.dModel as number | undefined) ?? 256
      shapes.set(node.id, { kind: 'flat', features: dModel })

    // ── Multi-Head Attention ───────────────────────────────────────────────────
    } else if (type === 'multiHeadAttentionNode') {
      const embedDim = (data.embedDim as number | undefined) ?? 256
      shapes.set(node.id, { kind: 'flat', features: embedDim })

    // ── Concatenate ────────────────────────────────────────────────────────────
    } else if (type === 'concatenateNode') {
      const parentShapes = nodeParents.map((pid) => shapes.get(pid)).filter(Boolean) as TensorShape[]
      if (parentShapes.length === 0) {
        shapes.set(node.id, { kind: 'flat', features: 0 })
      } else if (parentShapes.every((s) => s.kind === 'flat')) {
        shapes.set(node.id, { kind: 'flat', features: parentShapes.reduce((sum, s) => sum + flatFeatures(s), 0) })
      } else if (parentShapes.every((s) => s.kind === 'spatial')) {
        const ref = parentShapes[0]
        const sameSize = parentShapes.every((s) => s.kind === 'spatial' && s.height === ref.height && s.width === ref.width)
        if (sameSize && ref.kind === 'spatial') {
          shapes.set(node.id, {
            kind: 'spatial',
            channels: parentShapes.reduce((sum, s) => sum + (s.kind === 'spatial' ? s.channels : 0), 0),
            height: ref.height,
            width: ref.width,
          })
        } else if (firstParentShape) {
          shapes.set(node.id, { ...firstParentShape })
        }
      } else if (firstParentShape) {
        shapes.set(node.id, { ...firstParentShape })
      }

    // ── Add / Multiply (same shape as inputs) ──────────────────────────────────
    } else if (type === 'addNode' || type === 'multiplyNode') {
      if (firstParentShape) shapes.set(node.id, { ...firstParentShape })

    // ── RNN / GRU (flat → flat: last hidden state) ─────────────────────────────
    } else if (type === 'rnnNode' || type === 'gruNode') {
      const hidden = (data.hiddenSize as number | undefined) ?? 128
      const bidir = data.bidirectional === true || data.bidirectional === 'true'
      shapes.set(node.id, { kind: 'flat', features: hidden * (bidir ? 2 : 1) })

    // ── LSTM (flat → flat: last hidden state) ──────────────────────────────────
    } else if (type === 'lstmNode') {
      const hidden = (data.hiddenSize as number | undefined) ?? 128
      const bidir = data.bidirectional === true || data.bidirectional === 'true'
      shapes.set(node.id, { kind: 'flat', features: hidden * (bidir ? 2 : 1) })

    // ── Transformer Encoder (flat → flat: d_model) ─────────────────────────────
    } else if (type === 'transformerEncoderNode') {
      const dModel = (data.dModel as number | undefined) ?? 256
      shapes.set(node.id, { kind: 'flat', features: dModel })

    // ── Dense ──────────────────────────────────────────────────────────────────
    } else if (type === 'denseNode') {
      const units = (data.units as number | undefined) ?? 128
      const parentOutputSum = nodeParents.reduce((sum, pid) => {
        const s = shapes.get(pid)
        return sum + (s ? flatFeatures(s) : 0)
      }, 0)
      shapes.set(node.id, { kind: 'flat', features: units })
      shapes.set(`${node.id}__in`, { kind: 'flat', features: parentOutputSum })

    // ── Output ─────────────────────────────────────────────────────────────────
    } else if (type === 'outputNode') {
      const classes = (data.classes as number | undefined) ?? 10
      const parentOutputSum = nodeParents.reduce((sum, pid) => {
        const s = shapes.get(pid)
        return sum + (s ? flatFeatures(s) : 0)
      }, 0)
      shapes.set(node.id, { kind: 'flat', features: classes })
      shapes.set(`${node.id}__in`, { kind: 'flat', features: parentOutputSum })
    }
  }

  return shapes
}

/** Convenience: get the flat input features a Dense/Output node consumes. */
export function getInputFeatures(shapes: Map<string, TensorShape>, nodeId: string): number {
  const entry = shapes.get(`${nodeId}__in`)
  return entry ? flatFeatures(entry) : 0
}
