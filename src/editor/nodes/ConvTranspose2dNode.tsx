import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface ConvTranspose2dData extends Record<string, unknown> {
  label: string
  outChannels: number
  kernelSize: number
  stride: number
  padding: number
  outputPadding: number
  groups: number
  bias: boolean
  activation: string
}

type ConvTranspose2dNodeType = Node<ConvTranspose2dData, 'convTranspose2dNode'>

function ConvTranspose2dNode({ id, data, selected }: NodeProps<ConvTranspose2dNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="ConvTranspose2d"
      category="layers"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <rect x="1" y="5" width="4" height="6" rx="1" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
          <rect x="6" y="3" width="4" height="10" rx="1" stroke="currentColor" strokeWidth="1.2" opacity="0.75" />
          <rect x="11" y="1" width="4" height="14" rx="1" stroke="currentColor" strokeWidth="1.2" />
          <line x1="5" y1="8" x2="6" y2="8" stroke="currentColor" strokeWidth="1" opacity="0.6" />
          <line x1="10" y1="8" x2="11" y2="8" stroke="currentColor" strokeWidth="1" opacity="0.6" />
        </svg>
      }
    >
      <NodeRow label="Out Ch" value={data.outChannels} />
      <NodeRow label="Kernel" value={`${data.kernelSize}×${data.kernelSize}`} />
      <NodeRow label="Stride" value={data.stride} />
      <NodeRow label="Pad" value={data.padding} />
      {(data.outputPadding as number) > 0 && <NodeRow label="Out Pad" value={data.outputPadding} />}
      <NodeRow label="Act" value={data.activation} />
    </BaseNode>
  )
}

registerNode({
  type: 'convTranspose2dNode',
  label: 'ConvTranspose2d',
  description: 'Transposed 2D convolution (upsampling / decoder)',
  category: 'layers',
  paletteGroup: 'computerVision',
  defaultData: {
    label: 'ConvTranspose2d',
    outChannels: 32,
    kernelSize: 2,
    stride: 2,
    padding: 0,
    outputPadding: 0,
    groups: 1,
    bias: true,
    activation: 'relu',
  },
  fields: [
    { key: 'outChannels',    label: 'Out Channels',    type: 'number', min: 1 },
    { key: 'kernelSize',     label: 'Kernel Size',     type: 'number', min: 1 },
    { key: 'stride',         label: 'Stride',          type: 'number', min: 1 },
    { key: 'padding',        label: 'Padding',         type: 'number', min: 0 },
    { key: 'outputPadding',  label: 'Output Padding',  type: 'number', min: 0 },
    { key: 'groups',         label: 'Groups',          type: 'number', min: 1 },
    { key: 'bias',           label: 'Bias',            type: 'boolean' },
    {
      key: 'activation',
      label: 'Activation',
      type: 'select',
      options: [
        { value: 'relu',       label: 'ReLU' },
        { value: 'gelu',       label: 'GELU' },
        { value: 'silu',       label: 'SiLU' },
        { value: 'leaky_relu', label: 'Leaky ReLU' },
        { value: 'sigmoid',    label: 'Sigmoid' },
        { value: 'tanh',       label: 'Tanh' },
        { value: 'none',       label: 'None' },
      ],
    },
  ],
  component: ConvTranspose2dNode,
})

export default ConvTranspose2dNode
