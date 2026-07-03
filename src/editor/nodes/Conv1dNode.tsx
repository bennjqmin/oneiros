import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface Conv1dData extends Record<string, unknown> {
  label: string
  outChannels: number
  kernelSize: number
  stride: number
  padding: number
  dilation: number
  groups: number
  bias: boolean
  activation: string
}

type Conv1dNodeType = Node<Conv1dData, 'conv1dNode'>

function Conv1dNode({ id, data, selected }: NodeProps<Conv1dNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="Conv1d"
      category="layers"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <rect x="1" y="5" width="14" height="6" rx="1" stroke="currentColor" strokeWidth="1.2" />
          <rect x="4" y="5" width="3" height="6" rx="0.5" fill="currentColor" opacity="0.4" />
          <rect x="9" y="5" width="3" height="6" rx="0.5" fill="currentColor" opacity="0.4" />
        </svg>
      }
    >
      <NodeRow label="Out Channels" value={data.outChannels} />
      <NodeRow label="Kernel" value={data.kernelSize} />
      <NodeRow label="Stride" value={data.stride} />
      <NodeRow label="Activation" value={data.activation} />
    </BaseNode>
  )
}

registerNode({
  type: 'conv1dNode',
  label: 'Conv1d',
  description: '1D convolution — treats flat input as (channels, 1) sequence',
  category: 'layers',
  paletteGroup: 'convolutional',
  defaultData: {
    label: 'Conv1d',
    outChannels: 64,
    kernelSize: 3,
    stride: 1,
    padding: 1,
    dilation: 1,
    groups: 1,
    bias: true,
    activation: 'relu',
  },
  fields: [
    { key: 'outChannels', label: 'Out Channels', type: 'number', min: 1 },
    { key: 'kernelSize', label: 'Kernel Size', type: 'number', min: 1 },
    { key: 'stride', label: 'Stride', type: 'number', min: 1 },
    { key: 'padding', label: 'Padding', type: 'number', min: 0 },
    { key: 'dilation', label: 'Dilation', type: 'number', min: 1 },
    { key: 'groups', label: 'Groups', type: 'number', min: 1 },
    { key: 'bias', label: 'Bias', type: 'boolean' },
    {
      key: 'activation',
      label: 'Activation',
      type: 'select',
      options: [
        { value: 'relu', label: 'ReLU' },
        { value: 'gelu', label: 'GELU' },
        { value: 'silu', label: 'SiLU' },
        { value: 'elu', label: 'ELU' },
        { value: 'leaky_relu', label: 'Leaky ReLU' },
        { value: 'mish', label: 'Mish' },
        { value: 'none', label: 'None' },
      ],
    },
  ],
  component: Conv1dNode,
})

export default Conv1dNode
