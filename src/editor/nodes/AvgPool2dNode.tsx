import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface AvgPool2dData extends Record<string, unknown> {
  label: string
  kernelSize: number
  stride: number
  padding: number
  ceilMode: boolean
  countIncludePad: boolean
}

type AvgPool2dNodeType = Node<AvgPool2dData, 'avgPool2dNode'>

function AvgPool2dNode({ id, data, selected }: NodeProps<AvgPool2dNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="AvgPool2d"
      category="layers"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <rect x="1" y="1" width="14" height="14" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
          <rect x="4" y="4" width="8" height="8" rx="1" fill="currentColor" opacity="0.15" />
          <line x1="4" y1="8" x2="12" y2="8" stroke="currentColor" strokeWidth="1" opacity="0.6" />
          <line x1="8" y1="4" x2="8" y2="12" stroke="currentColor" strokeWidth="1" opacity="0.6" />
        </svg>
      }
    >
      <NodeRow label="Kernel" value={`${data.kernelSize}×${data.kernelSize}`} />
      <NodeRow label="Stride" value={data.stride} />
      {(data.padding as number) > 0 && <NodeRow label="Padding" value={data.padding} />}
    </BaseNode>
  )
}

registerNode({
  type: 'avgPool2dNode',
  label: 'Average Pooling',
  description: '2D average pooling',
  category: 'layers',
  paletteGroup: 'convolutional',
  defaultData: {
    label: 'AvgPool2d',
    kernelSize: 2,
    stride: 2,
    padding: 0,
    ceilMode: false,
    countIncludePad: true,
  },
  fields: [
    { key: 'kernelSize', label: 'Kernel Size', type: 'number', min: 1 },
    { key: 'stride', label: 'Stride', type: 'number', min: 1 },
    { key: 'padding', label: 'Padding', type: 'number', min: 0 },
    { key: 'ceilMode', label: 'Ceil Mode', type: 'boolean' },
    { key: 'countIncludePad', label: 'Count Include Pad', type: 'boolean' },
  ],
  component: AvgPool2dNode,
})

export default AvgPool2dNode
