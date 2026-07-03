import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface Conv2dData extends Record<string, unknown> {
  label: string
  outChannels: number
  kernelSize: number
  stride: number
  padding: number
  dilation: number
  groups: number
  bias: boolean
  paddingMode: string
  activation: string
}

type Conv2dNodeType = Node<Conv2dData, 'conv2dNode'>

function Conv2dNode({ id, data, selected }: NodeProps<Conv2dNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="Conv2d"
      category="layers"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <rect x="1" y="1" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.2" />
          <rect x="9" y="1" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
          <rect x="1" y="9" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
          <rect x="9" y="9" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.2" opacity="0.3" />
        </svg>
      }
    >
      <NodeRow label="Out Ch" value={data.outChannels} />
      <NodeRow label="Kernel" value={`${data.kernelSize}×${data.kernelSize}`} />
      <NodeRow label="Stride" value={data.stride} />
      <NodeRow label="Pad" value={data.padding} />
      {(data.dilation as number) > 1 && <NodeRow label="Dilation" value={data.dilation} />}
      {(data.groups as number) > 1 && <NodeRow label="Groups" value={data.groups} />}
      <NodeRow label="Act" value={data.activation} />
    </BaseNode>
  )
}

registerNode({
  type: 'conv2dNode',
  label: 'Conv2d',
  description: '2D convolution layer',
  category: 'layers',
  paletteGroup: 'convolutional',
  defaultData: {
    label: 'Conv2d',
    outChannels: 32,
    kernelSize: 3,
    stride: 1,
    padding: 0,
    dilation: 1,
    groups: 1,
    bias: true,
    paddingMode: 'zeros',
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
      key: 'paddingMode',
      label: 'Padding Mode',
      type: 'select',
      options: [
        { value: 'zeros', label: 'Zeros' },
        { value: 'reflect', label: 'Reflect' },
        { value: 'replicate', label: 'Replicate' },
        { value: 'circular', label: 'Circular' },
      ],
    },
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
        { value: 'sigmoid', label: 'Sigmoid' },
        { value: 'tanh', label: 'Tanh' },
        { value: 'none', label: 'None' },
      ],
    },
  ],
  component: Conv2dNode,
})

export default Conv2dNode
