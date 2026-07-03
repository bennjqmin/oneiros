import type { AppNode, AppEdge } from '../../types/graph'
import { validateGraph } from '../validation/validateGraph'
import { topologicalSort } from './topoSort'
import { inferShapes, flatFeatures, getInputFeatures } from './shapeInfer'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface CompileResult {
  code: string
  errors: string[]
  warnings: string[]
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function sanitize(id: string): string {
  return id.replace(/[^a-zA-Z0-9]/g, '_')
}

/** Map activation name → Python expression template (use {expr} placeholder) */
const ACTIVATION_MAP: Record<string, string> = {
  relu: 'torch.relu({expr})',
  gelu: 'F.gelu({expr})',
  silu: 'F.silu({expr})',
  elu: 'F.elu({expr})',
  leaky_relu: 'F.leaky_relu({expr})',
  mish: 'F.mish({expr})',
  softplus: 'F.softplus({expr})',
  hardswish: 'F.hardswish({expr})',
  tanh: 'torch.tanh({expr})',
  sigmoid: 'torch.sigmoid({expr})',
  softmax: 'F.softmax({expr}, dim=-1)',
  none: '',
}

/** Which activations need `import torch.nn.functional as F` */
const NEEDS_F = new Set(['gelu', 'silu', 'elu', 'leaky_relu', 'mish', 'softplus', 'hardswish', 'softmax'])

function applyActivation(activation: string, expr: string): string {
  const template = ACTIVATION_MAP[activation] ?? ''
  if (!template) return expr
  return template.replace('{expr}', expr)
}

// ── Main compiler ─────────────────────────────────────────────────────────────

export function compileGraph(
  nodes: AppNode[],
  edges: AppEdge[],
  projectName = 'OneirosModel',
): CompileResult {
  const errors: string[] = []
  const warnings: string[] = []

  if (nodes.length === 0) {
    return { code: '', errors: ['Graph is empty'], warnings }
  }

  const { isValid, issues } = validateGraph(nodes, edges)
  if (!isValid) {
    for (const issue of issues) {
      if (issue.severity === 'error') errors.push(issue.message)
      else warnings.push(issue.message)
    }
    return { code: '', errors, warnings }
  }

  const sorted = topologicalSort(nodes, edges)
  if (!sorted) {
    return { code: '', errors: ['Cycle detected — cannot compile'], warnings }
  }

  const shapes = inferShapes(sorted, edges)

  const parents = new Map<string, string[]>()
  for (const node of sorted) parents.set(node.id, [])
  for (const edge of edges) parents.get(edge.target)?.push(edge.source)

  // Determine which imports are needed
  let needsF = false
  let needsPosEnc = false

  const inputNode = sorted.find((n) => n.type === 'inputNode')
  const batchSize = (inputNode?.data.batchSize as number | undefined) ?? 1
  const channels = (inputNode?.data.channels as number | undefined) ?? 1
  const height = (inputNode?.data.height as number | undefined) ?? 28
  const width = (inputNode?.data.width as number | undefined) ?? 28

  const safeClass = (projectName.replace(/[^a-zA-Z0-9 ]/g, '').split(' ')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join('')) || 'OneirosModel'

  const initLines: string[] = []
  const forwardLines: string[] = []
  const varNames = new Map<string, string>()

  for (const node of sorted) {
    const type = node.type ?? ''
    const sid = sanitize(node.id)
    const nodeParents = parents.get(node.id) ?? []
    const shape = shapes.get(node.id)
    const firstParent = nodeParents[0]

    // ── InputNode ─────────────────────────────────────────────────────────────
    if (type === 'inputNode') {
      varNames.set(node.id, 'x')

    // ── Conv2d ────────────────────────────────────────────────────────────────
    } else if (type === 'conv2dNode') {
      const parentShape = shapes.get(firstParent ?? '')
      const inC = parentShape?.kind === 'spatial' ? parentShape.channels : 1
      const outC = (node.data.outChannels as number | undefined) ?? 32
      const k = (node.data.kernelSize as number | undefined) ?? 3
      const s = (node.data.stride as number | undefined) ?? 1
      const p = (node.data.padding as number | undefined) ?? 0
      const act = (node.data.activation as string | undefined) ?? 'relu'
      initLines.push(`        self.conv_${sid} = nn.Conv2d(${inC}, ${outC}, kernel_size=${k}, stride=${s}, padding=${p})`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = ${applyActivation(act, `self.conv_${sid}(${inp})`)}`)
      varNames.set(node.id, varName)
      if (NEEDS_F.has(act)) needsF = true

    // ── Conv1d (input: flat → treats as (batch, features, 1)) ─────────────────
    } else if (type === 'conv1dNode') {
      const inFeatures = flatFeatures(shapes.get(firstParent ?? '') ?? { kind: 'flat', features: 1 })
      const outC = (node.data.outChannels as number | undefined) ?? 64
      const k = (node.data.kernelSize as number | undefined) ?? 3
      const s = (node.data.stride as number | undefined) ?? 1
      const p = (node.data.padding as number | undefined) ?? 1
      const act = (node.data.activation as string | undefined) ?? 'relu'
      initLines.push(`        self.conv1d_${sid} = nn.Conv1d(${inFeatures}, ${outC}, kernel_size=${k}, stride=${s}, padding=${p})`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      // Reshape flat → (batch, inC, 1), apply conv, squeeze back → flat
      forwardLines.push(`        _c1d_${sid} = ${inp}.unsqueeze(-1)  # (batch, ${inFeatures}, 1)`)
      const convExpr = `self.conv1d_${sid}(_c1d_${sid}).squeeze(-1)`
      forwardLines.push(`        ${varName} = ${applyActivation(act, convExpr)}`)
      varNames.set(node.id, varName)
      if (NEEDS_F.has(act)) needsF = true

    // ── MaxPool2d ─────────────────────────────────────────────────────────────
    } else if (type === 'maxPool2dNode') {
      const k = (node.data.kernelSize as number | undefined) ?? 2
      const s = (node.data.stride as number | undefined) ?? k
      initLines.push(`        self.pool_${sid} = nn.MaxPool2d(kernel_size=${k}, stride=${s})`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = self.pool_${sid}(${inp})`)
      varNames.set(node.id, varName)

    // ── AvgPool2d ─────────────────────────────────────────────────────────────
    } else if (type === 'avgPool2dNode') {
      const k = (node.data.kernelSize as number | undefined) ?? 2
      const s = (node.data.stride as number | undefined) ?? k
      initLines.push(`        self.avgpool_${sid} = nn.AvgPool2d(kernel_size=${k}, stride=${s})`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = self.avgpool_${sid}(${inp})`)
      varNames.set(node.id, varName)

    // ── AdaptiveAvgPool2d ─────────────────────────────────────────────────────
    } else if (type === 'adaptiveAvgPool2dNode') {
      const outSz = (node.data.outputSize as number | undefined) ?? 1
      initLines.push(`        self.adapool_${sid} = nn.AdaptiveAvgPool2d(${outSz})`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      if (outSz === 1) {
        forwardLines.push(`        ${varName} = self.adapool_${sid}(${inp}).view(${inp}.size(0), -1)  # global avg pool → flat`)
      } else {
        forwardLines.push(`        ${varName} = self.adapool_${sid}(${inp})`)
      }
      varNames.set(node.id, varName)

    // ── ConvTranspose2d ───────────────────────────────────────────────────────
    } else if (type === 'convTranspose2dNode') {
      const parentShape = shapes.get(firstParent ?? '')
      const inC = parentShape?.kind === 'spatial' ? parentShape.channels : 1
      const outC   = (node.data.outChannels   as number | undefined) ?? 32
      const k      = (node.data.kernelSize    as number | undefined) ?? 2
      const s      = (node.data.stride        as number | undefined) ?? 2
      const p      = (node.data.padding       as number | undefined) ?? 0
      const outPad = (node.data.outputPadding as number | undefined) ?? 0
      const act    = (node.data.activation    as string | undefined) ?? 'none'
      initLines.push(`        self.convt_${sid} = nn.ConvTranspose2d(${inC}, ${outC}, kernel_size=${k}, stride=${s}, padding=${p}, output_padding=${outPad})`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = ${applyActivation(act, `self.convt_${sid}(${inp})`)}`)
      varNames.set(node.id, varName)
      if (NEEDS_F.has(act)) needsF = true

    // ── Upsample ──────────────────────────────────────────────────────────────
    } else if (type === 'upsampleNode') {
      const sf   = (node.data.scaleFactor as number | undefined) ?? 2
      const mode = (node.data.mode as string | undefined) ?? 'nearest'
      const alignCorners = mode !== 'nearest' ? ', align_corners=False' : ''
      initLines.push(`        self.up_${sid} = nn.Upsample(scale_factor=${sf}, mode='${mode}'${alignCorners})`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = self.up_${sid}(${inp})`)
      varNames.set(node.id, varName)

    // ── Backbone (pretrained torchvision model) ────────────────────────────────
    } else if (type === 'backboneNode') {
      const model    = (node.data.model      as string  | undefined) ?? 'resnet18'
      const frozen   = (node.data.freeze     as boolean | undefined) ?? false
      const outLayer = (node.data.outputLayer as string | undefined) ?? 'avgpool'
      const feats: Record<string, number> = {
        resnet18: 512, resnet34: 512, resnet50: 2048,
        mobilenet_v2: 1280, efficientnet_b0: 1280, vgg16: 4096,
      }
      const numF = feats[model] ?? 512
      // backbone init is handled in compiler.py; here we just emit a forward call
      initLines.push(`        # backbone_${sid}: torchvision ${model} (${outLayer}) → ${numF} features`)
      initLines.push(`        self.backbone_${sid} = _build_backbone_${sid}(self)`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      if (outLayer === 'avgpool') {
        forwardLines.push(`        ${varName} = self.backbone_${sid}(${inp})  # → (batch, ${numF})`)
      } else {
        forwardLines.push(`        ${varName} = self.backbone_${sid}(${inp})`)
      }
      // Inject a helper function at the top of the class
      const pretrained = (node.data.pretrained as boolean | undefined) ?? true
      // The actual backbone building is done via a side-car function
      initLines.unshift(``)
      void frozen  // will be consumed in compiler.py
      void pretrained
      varNames.set(node.id, varName)

    // ── Flatten ───────────────────────────────────────────────────────────────
    } else if (type === 'flattenNode') {
      const inShape = shapes.get(firstParent ?? '')
      const outFeatures = inShape ? flatFeatures(inShape) : '?'
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = ${inp}.view(${inp}.size(0), -1)  # → ${outFeatures} features`)
      varNames.set(node.id, varName)

    // ── Dropout ───────────────────────────────────────────────────────────────
    } else if (type === 'dropoutNode') {
      const p = (node.data.p as number | undefined) ?? 0.5
      initLines.push(`        self.drop_${sid} = nn.Dropout(p=${p})`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = self.drop_${sid}(${inp})`)
      varNames.set(node.id, varName)

    // ── BatchNorm ─────────────────────────────────────────────────────────────
    } else if (type === 'batchNormNode') {
      const eps = (node.data.eps as number | undefined) ?? 1e-5
      const mom = (node.data.momentum as number | undefined) ?? 0.1
      const parentShape = shapes.get(firstParent ?? '')
      const isSpatial = parentShape?.kind === 'spatial'
      const bnArg = isSpatial && parentShape?.kind === 'spatial' ? parentShape.channels
        : (parentShape ? flatFeatures(parentShape) : 0)
      const bnClass = isSpatial ? 'nn.BatchNorm2d' : 'nn.BatchNorm1d'
      initLines.push(`        self.bn_${sid} = ${bnClass}(${bnArg}, eps=${eps}, momentum=${mom})`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = self.bn_${sid}(${inp})`)
      varNames.set(node.id, varName)

    // ── Standalone Activation (passthrough) ───────────────────────────────────
    } else if (type === 'activationNode') {
      const fn = (node.data.fn as string | undefined) ?? 'relu'
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = ${applyActivation(fn, inp)}`)
      varNames.set(node.id, varName)
      if (NEEDS_F.has(fn)) needsF = true

    // ── RNN ───────────────────────────────────────────────────────────────────
    } else if (type === 'rnnNode') {
      const inF = flatFeatures(shapes.get(firstParent ?? '') ?? { kind: 'flat', features: 1 })
      const hidden = (node.data.hiddenSize as number | undefined) ?? 128
      const layers = (node.data.numLayers as number | undefined) ?? 1
      const nonlin = (node.data.nonlinearity as string | undefined) ?? 'tanh'
      const drop = (node.data.dropout as number | undefined) ?? 0
      const bidir = node.data.bidirectional === true || node.data.bidirectional === 'true'
      initLines.push(`        self.rnn_${sid} = nn.RNN(${inF}, ${hidden}, num_layers=${layers}, nonlinearity='${nonlin}', dropout=${drop}, bidirectional=${bidir ? 'True' : 'False'}, batch_first=True)`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        _, _hn_${sid} = self.rnn_${sid}(${inp}.unsqueeze(1))  # add seq_len=1`)
      forwardLines.push(`        ${varName} = _hn_${sid}${bidir ? '.transpose(0,1).reshape(${inp}.size(0), -1)' : '[-1]'}  # last hidden state`)
      varNames.set(node.id, varName)

    // ── GRU ───────────────────────────────────────────────────────────────────
    } else if (type === 'gruNode') {
      const inF = flatFeatures(shapes.get(firstParent ?? '') ?? { kind: 'flat', features: 1 })
      const hidden = (node.data.hiddenSize as number | undefined) ?? 128
      const layers = (node.data.numLayers as number | undefined) ?? 1
      const drop = (node.data.dropout as number | undefined) ?? 0
      const bidir = node.data.bidirectional === true || node.data.bidirectional === 'true'
      initLines.push(`        self.gru_${sid} = nn.GRU(${inF}, ${hidden}, num_layers=${layers}, dropout=${drop}, bidirectional=${bidir ? 'True' : 'False'}, batch_first=True)`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        _, _hn_${sid} = self.gru_${sid}(${inp}.unsqueeze(1))`)
      if (bidir) {
        forwardLines.push(`        ${varName} = _hn_${sid}.transpose(0,1).reshape(${inp}.size(0), -1)`)
      } else {
        forwardLines.push(`        ${varName} = _hn_${sid}[-1]`)
      }
      varNames.set(node.id, varName)

    // ── LSTM ──────────────────────────────────────────────────────────────────
    } else if (type === 'lstmNode') {
      const inF = flatFeatures(shapes.get(firstParent ?? '') ?? { kind: 'flat', features: 1 })
      const hidden = (node.data.hiddenSize as number | undefined) ?? 128
      const layers = (node.data.numLayers as number | undefined) ?? 1
      const drop = (node.data.dropout as number | undefined) ?? 0
      const bidir = node.data.bidirectional === true || node.data.bidirectional === 'true'
      initLines.push(`        self.lstm_${sid} = nn.LSTM(${inF}, ${hidden}, num_layers=${layers}, dropout=${drop}, bidirectional=${bidir ? 'True' : 'False'}, batch_first=True)`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        _, (_hn_${sid}, _) = self.lstm_${sid}(${inp}.unsqueeze(1))`)
      if (bidir) {
        forwardLines.push(`        ${varName} = _hn_${sid}.transpose(0,1).reshape(${inp}.size(0), -1)`)
      } else {
        forwardLines.push(`        ${varName} = _hn_${sid}[-1]`)
      }
      varNames.set(node.id, varName)

    // ── Transformer Encoder ───────────────────────────────────────────────────
    } else if (type === 'transformerEncoderNode') {
      const inF = flatFeatures(shapes.get(firstParent ?? '') ?? { kind: 'flat', features: 1 })
      const dModel = (node.data.dModel as number | undefined) ?? 256
      const nhead = (node.data.nhead as number | undefined) ?? 8
      const ffDim = (node.data.dimFeedforward as number | undefined) ?? 512
      const numLayers = (node.data.numLayers as number | undefined) ?? 2
      const drop = (node.data.dropout as number | undefined) ?? 0.1
      initLines.push(`        self.te_proj_${sid} = nn.Linear(${inF}, ${dModel})  # project to d_model`)
      initLines.push(`        _enc_layer_${sid} = nn.TransformerEncoderLayer(d_model=${dModel}, nhead=${nhead}, dim_feedforward=${ffDim}, dropout=${drop}, batch_first=True)`)
      initLines.push(`        self.te_${sid} = nn.TransformerEncoder(_enc_layer_${sid}, num_layers=${numLayers})`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        _te_in_${sid} = self.te_proj_${sid}(${inp}).unsqueeze(1)  # (batch, 1, d_model)`)
      forwardLines.push(`        ${varName} = self.te_${sid}(_te_in_${sid}).squeeze(1)  # (batch, d_model)`)
      varNames.set(node.id, varName)

    // ── Gaussian Noise ────────────────────────────────────────────────────────
    } else if (type === 'gaussianNoiseNode') {
      const std = (node.data.std as number | undefined) ?? 0.1
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = ${inp} + torch.randn_like(${inp}) * ${std} if self.training else ${inp}`)
      varNames.set(node.id, varName)

    // ── Reshape ───────────────────────────────────────────────────────────────
    } else if (type === 'reshapeNode') {
      const target = (node.data.targetFeatures as number | undefined) ?? -1
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      if (target === -1) {
        forwardLines.push(`        ${varName} = ${inp}.view(${inp}.size(0), -1)`)
      } else {
        forwardLines.push(`        ${varName} = ${inp}.view(${inp}.size(0), ${target})`)
      }
      varNames.set(node.id, varName)

    // ── Layer Norm ────────────────────────────────────────────────────────────
    } else if (type === 'layerNormNode') {
      const inF = flatFeatures(shapes.get(firstParent ?? '') ?? { kind: 'flat', features: 1 })
      const eps = (node.data.eps as number | undefined) ?? 1e-5
      initLines.push(`        self.ln_${sid} = nn.LayerNorm(${inF}, eps=${eps})`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = self.ln_${sid}(${inp})`)
      varNames.set(node.id, varName)

    // ── Embedding ─────────────────────────────────────────────────────────────
    } else if (type === 'embeddingNode') {
      const numEmb = (node.data.numEmbeddings as number | undefined) ?? 1000
      const embDim = (node.data.embeddingDim as number | undefined) ?? 128
      const seqLen = flatFeatures(shapes.get(firstParent ?? '') ?? { kind: 'flat', features: 1 })
      initLines.push(`        self.emb_${sid} = nn.Embedding(${numEmb}, ${embDim})`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = self.emb_${sid}(${inp}.long()).view(${inp}.size(0), -1)  # seq=${seqLen}, dim=${embDim}`)
      varNames.set(node.id, varName)

    // ── Positional Encoding ───────────────────────────────────────────────────
    } else if (type === 'positionalEncodingNode') {
      const dModel = (node.data.dModel as number | undefined) ?? 256
      const maxLen = (node.data.maxLen as number | undefined) ?? 512
      needsPosEnc = true
      initLines.push(`        self.pe_${sid} = PositionalEncoding(${dModel}, max_len=${maxLen})`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = self.pe_${sid}(${inp}.unsqueeze(1)).squeeze(1)`)
      varNames.set(node.id, varName)

    // ── Feed Forward ──────────────────────────────────────────────────────────
    } else if (type === 'feedForwardNode') {
      const dModel = (node.data.dModel as number | undefined) ?? 256
      const ffDim = (node.data.ffDim as number | undefined) ?? 512
      const drop = (node.data.dropout as number | undefined) ?? 0.1
      const act = (node.data.activation as string | undefined) ?? 'relu'
      const actLayer = act === 'gelu' ? 'nn.GELU()' : 'nn.ReLU()'
      if (act === 'gelu') needsF = true
      initLines.push(`        self.ff_${sid} = nn.Sequential(`)
      initLines.push(`            nn.Linear(${dModel}, ${ffDim}),`)
      initLines.push(`            ${actLayer},`)
      initLines.push(`            nn.Dropout(${drop}),`)
      initLines.push(`            nn.Linear(${ffDim}, ${dModel}),`)
      initLines.push(`        )`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        ${varName} = self.ff_${sid}(${inp})`)
      varNames.set(node.id, varName)

    // ── Multi-Head Attention ──────────────────────────────────────────────────
    } else if (type === 'multiHeadAttentionNode') {
      const embedDim = (node.data.embedDim as number | undefined) ?? 256
      const numHeads = (node.data.numHeads as number | undefined) ?? 8
      const drop = (node.data.dropout as number | undefined) ?? 0.1
      initLines.push(`        self.mha_${sid} = nn.MultiheadAttention(${embedDim}, ${numHeads}, dropout=${drop}, batch_first=True)`)
      const inp = resolveInputSingle(nodeParents, varNames)
      const varName = `h_${sid}`
      forwardLines.push(`        _mha_out_${sid}, _ = self.mha_${sid}(${inp}.unsqueeze(1), ${inp}.unsqueeze(1), ${inp}.unsqueeze(1))`)
      forwardLines.push(`        ${varName} = _mha_out_${sid}.squeeze(1)`)
      varNames.set(node.id, varName)

    // ── Concatenate ───────────────────────────────────────────────────────────
    } else if (type === 'concatenateNode') {
      const dim = (node.data.dim as number | undefined) ?? -1
      const varName = `h_${sid}`
      const inputVar = resolveInputMulti(node.id, nodeParents, varNames, forwardLines, dim)
      forwardLines.push(`        ${varName} = ${inputVar}`)
      varNames.set(node.id, varName)

    // ── Add ───────────────────────────────────────────────────────────────────
    } else if (type === 'addNode') {
      const pair = resolveBinaryInputs(node.id, edges, varNames)
      const varName = `h_${sid}`
      if (pair) forwardLines.push(`        ${varName} = ${pair[0]} + ${pair[1]}`)
      varNames.set(node.id, varName)

    // ── Multiply ──────────────────────────────────────────────────────────────
    } else if (type === 'multiplyNode') {
      const pair = resolveBinaryInputs(node.id, edges, varNames)
      const varName = `h_${sid}`
      if (pair) forwardLines.push(`        ${varName} = ${pair[0]} * ${pair[1]}`)
      varNames.set(node.id, varName)

    // ── Dense ─────────────────────────────────────────────────────────────────
    } else if (type === 'denseNode') {
      const inFeatures = getInputFeatures(shapes, node.id)
      const outFeatures = shape?.kind === 'flat' ? shape.features : 128
      initLines.push(`        self.layer_${sid} = nn.Linear(${inFeatures}, ${outFeatures})`)
      const act = (node.data.activation as string | undefined) ?? 'relu'
      const varName = `h_${sid}`
      const inputVar = resolveInputMulti(node.id, nodeParents, varNames, forwardLines)
      forwardLines.push(`        ${varName} = ${applyActivation(act, `self.layer_${sid}(${inputVar})`)}`)
      varNames.set(node.id, varName)
      if (NEEDS_F.has(act)) needsF = true

    // ── Output ────────────────────────────────────────────────────────────────
    } else if (type === 'outputNode') {
      const inFeatures = getInputFeatures(shapes, node.id)
      const outFeatures = shape?.kind === 'flat' ? shape.features : 10
      initLines.push(`        self.layer_${sid} = nn.Linear(${inFeatures}, ${outFeatures})`)
      const inputVar = resolveInputMulti(node.id, nodeParents, varNames, forwardLines)
      forwardLines.push(`        out = self.layer_${sid}(${inputVar})`)
      varNames.set(node.id, 'out')
    }
  }

  const peHelper = needsPosEnc ? [
    ``,
    `class PositionalEncoding(nn.Module):`,
    `    def __init__(self, d_model: int, max_len: int = 512):`,
    `        super().__init__()`,
    `        pe = torch.zeros(max_len, d_model)`,
    `        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)`,
    `        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))`,
    `        pe[:, 0::2] = torch.sin(position * div_term)`,
    `        pe[:, 1::2] = torch.cos(position * div_term)`,
    `        self.register_buffer('pe', pe.unsqueeze(0))`,
    ``,
    `    def forward(self, x: torch.Tensor) -> torch.Tensor:`,
    `        return x + self.pe[:, :x.size(1)]`,
    ``,
  ] : []

  const lines: string[] = [
    `# Generated by Oneiros`,
    `# Project: ${projectName}`,
    ``,
    `import math`,
    `import torch`,
    `import torch.nn as nn`,
    ...(needsF ? [`import torch.nn.functional as F`] : []),
    ...peHelper,
    ``,
    `class ${safeClass}(nn.Module):`,
    `    def __init__(self):`,
    `        super().__init__()`,
    ...initLines,
    ``,
    `    def forward(self, x: torch.Tensor) -> torch.Tensor:`,
    ...forwardLines,
    `        return out`,
    ``,
    ``,
    `if __name__ == '__main__':`,
    `    model = ${safeClass}()`,
    `    print(model)`,
    `    x = torch.randn(${batchSize}, ${channels}, ${height}, ${width})`,
    `    y = model(x)`,
    `    print(f'Output shape: {y.shape}')`,
  ]

  return { code: lines.join('\n'), errors, warnings }
}

// ── Input resolution helpers ──────────────────────────────────────────────────

function resolveInputSingle(nodeParents: string[], varNames: Map<string, string>): string {
  if (nodeParents.length === 0) return 'x'
  return varNames.get(nodeParents[0]) ?? 'x'
}

function resolveInputMulti(
  nodeId: string,
  nodeParents: string[],
  varNames: Map<string, string>,
  forwardLines: string[],
  dim = -1,
): string {
  if (nodeParents.length === 0) return 'x'
  if (nodeParents.length === 1) return varNames.get(nodeParents[0]) ?? 'x'
  const sid = sanitize(nodeId)
  const parts = nodeParents.map((pid) => varNames.get(pid) ?? 'x').join(', ')
  const catVar = `cat_${sid}`
  forwardLines.push(`        ${catVar} = torch.cat([${parts}], dim=${dim})`)
  return catVar
}

function resolveBinaryInputs(
  nodeId: string,
  edges: AppEdge[],
  varNames: Map<string, string>,
): [string, string] | null {
  const aEdge = edges.find((e) => e.target === nodeId && e.targetHandle === 'in_a')
  const bEdge = edges.find((e) => e.target === nodeId && e.targetHandle === 'in_b')
  if (!aEdge || !bEdge) return null
  return [varNames.get(aEdge.source) ?? 'x', varNames.get(bEdge.source) ?? 'x']
}
