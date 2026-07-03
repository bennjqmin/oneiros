import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode, registerPaletteShortcut } from '../registry/nodeRegistry'

const ACTIVATION_LABELS: Record<string, string> = {
  relu: 'ReLU',
  gelu: 'GELU',
  silu: 'SiLU / Swish',
  elu: 'ELU',
  leaky_relu: 'Leaky ReLU',
  mish: 'Mish',
  softplus: 'Softplus',
  hardswish: 'Hard Swish',
  tanh: 'Tanh',
  sigmoid: 'Sigmoid',
  softmax: 'Softmax',
}

interface ActivationData extends Record<string, unknown> {
  label: string
  fn: string
}

type ActivationNodeType = Node<ActivationData, 'activationNode'>

function ActivationNode({ id, data, selected }: NodeProps<ActivationNodeType>) {
  const displayLabel = ACTIVATION_LABELS[data.fn] ?? data.label ?? 'Activation'
  return (
    <BaseNode
      nodeId={id}
      label={displayLabel}
      category="activation"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <path d="M1 12 L5 8 L7 10 L10 4 L15 4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" fill="none" />
        </svg>
      }
    >
      <NodeRow label="fn" value={ACTIVATION_LABELS[data.fn] ?? data.fn} />
    </BaseNode>
  )
}

registerNode({
  type: 'activationNode',
  label: 'Activation',
  description: 'Standalone activation function (passthrough shape)',
  category: 'activation',
  paletteGroup: 'activation',
  defaultData: {
    label: 'Activation',
    fn: 'relu',
  },
  fields: [
    {
      key: 'fn',
      label: 'Function',
      type: 'select',
      options: [
        { value: 'relu', label: 'ReLU' },
        { value: 'gelu', label: 'GELU' },
        { value: 'silu', label: 'SiLU / Swish' },
        { value: 'elu', label: 'ELU' },
        { value: 'leaky_relu', label: 'Leaky ReLU' },
        { value: 'mish', label: 'Mish' },
        { value: 'softplus', label: 'Softplus' },
        { value: 'hardswish', label: 'Hard Swish' },
        { value: 'tanh', label: 'Tanh' },
        { value: 'sigmoid', label: 'Sigmoid' },
        { value: 'softmax', label: 'Softmax (dim=-1)' },
      ],
    },
  ],
  component: ActivationNode,
})

const ACTIVATION_SHORTCUTS: { key: string; fn: string; label: string }[] = [
  { key: 'activationNode:relu', fn: 'relu', label: 'ReLU' },
  { key: 'activationNode:gelu', fn: 'gelu', label: 'GELU' },
  { key: 'activationNode:sigmoid', fn: 'sigmoid', label: 'Sigmoid' },
  { key: 'activationNode:softmax', fn: 'softmax', label: 'Softmax' },
  { key: 'activationNode:tanh', fn: 'tanh', label: 'Tanh' },
  { key: 'activationNode:leaky_relu', fn: 'leaky_relu', label: 'Leaky ReLU' },
]

for (const s of ACTIVATION_SHORTCUTS) {
  registerPaletteShortcut({
    key: s.key,
    type: 'activationNode',
    label: s.label,
    description: `${s.label} activation (passthrough shape)`,
    paletteGroup: 'activation',
    paletteDefaults: { fn: s.fn, label: s.label },
  })
}

export default ActivationNode
