import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface MaxPool2dData extends Record<string, unknown> {
  label: string
  kernelSize: number
  stride: number
  padding: number
  dilation: number
  ceilMode: boolean
}

type MaxPool2dNodeType = Node<MaxPool2dData, 'maxPool2dNode'>

function MaxPool2dNode({ id, data, selected }: NodeProps<MaxPool2dNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="MaxPool2d"
      category="layers"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <rect x="1" y="1" width="14" height="14" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
          <rect x="4" y="4" width="8" height="8" rx="1" fill="currentColor" opacity="0.3" />
          <rect x="6" y="6" width="4" height="4" rx="0.5" fill="currentColor" opacity="0.7" />
        </svg>
      }
    >
      <NodeRow label="Kernel" value={`${data.kernelSize}×${data.kernelSize}`} />
      <NodeRow label="Stride" value={data.stride} />
      {(data.padding as number) > 0 && <NodeRow label="Padding" value={data.padding} />}
      {(data.dilation as number) > 1 && <NodeRow label="Dilation" value={data.dilation} />}
    </BaseNode>
  )
}

registerNode({
  type: 'maxPool2dNode',
  label: 'Max Pooling',
  description: '2D max pooling layer',
  category: 'layers',
  paletteGroup: 'convolutional',
  defaultData: {
    label: 'MaxPool2d',
    kernelSize: 2,
    stride: 2,
    padding: 0,
    dilation: 1,
    ceilMode: false,
  },
  fields: [
    { key: 'kernelSize', label: 'Kernel Size', type: 'number', min: 1 },
    { key: 'stride', label: 'Stride', type: 'number', min: 1 },
    { key: 'padding', label: 'Padding', type: 'number', min: 0 },
    { key: 'dilation', label: 'Dilation', type: 'number', min: 1 },
    { key: 'ceilMode', label: 'Ceil Mode', type: 'boolean' },
  ],
  component: MaxPool2dNode,
})

export default MaxPool2dNode
