import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface DenseData extends Record<string, unknown> {
  label: string
  units: number
  activation: string
  useBias: boolean
}

type DenseNodeType = Node<DenseData, 'denseNode'>

function DenseNode({ id, data, selected }: NodeProps<DenseNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="Dense"
      category="layers"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <circle cx="3" cy="5" r="1.5" fill="currentColor" />
          <circle cx="3" cy="11" r="1.5" fill="currentColor" />
          <circle cx="13" cy="5" r="1.5" fill="currentColor" />
          <circle cx="13" cy="8" r="1.5" fill="currentColor" />
          <circle cx="13" cy="11" r="1.5" fill="currentColor" />
          <line x1="4.5" y1="5" x2="11.5" y2="5" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
          <line x1="4.5" y1="5" x2="11.5" y2="8" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
          <line x1="4.5" y1="5" x2="11.5" y2="11" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
          <line x1="4.5" y1="11" x2="11.5" y2="5" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
          <line x1="4.5" y1="11" x2="11.5" y2="8" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
          <line x1="4.5" y1="11" x2="11.5" y2="11" stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
        </svg>
      }
    >
      <NodeRow label="Units" value={data.units} />
      <NodeRow label="Activation" value={data.activation} />
      {data.useBias && (
        <NodeRow label="Bias" value="enabled" />
      )}
    </BaseNode>
  )
}

registerNode({
  type: 'denseNode',
  label: 'Dense (Fully Connected)',
  description: 'Fully-connected linear layer',
  category: 'layers',
  paletteGroup: 'basic',
  defaultData: {
    label: 'Dense',
    units: 128,
    activation: 'relu',
    useBias: true,
  },
  fields: [
    { key: 'units', label: 'Units', type: 'number', min: 1, step: 1 },
    {
      key: 'activation',
      label: 'Activation',
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
        { value: 'softmax', label: 'Softmax' },
        { value: 'none', label: 'None' },
      ],
    },
    { key: 'useBias', label: 'Use Bias', type: 'boolean' },
  ],
  component: DenseNode,
})

export default DenseNode
