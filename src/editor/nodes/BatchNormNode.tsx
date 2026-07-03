import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface BatchNormData extends Record<string, unknown> {
  label: string
  eps: number
  momentum: number
  affine: boolean
  trackRunningStats: boolean
}

type BatchNormNodeType = Node<BatchNormData, 'batchNormNode'>

function BatchNormNode({ id, data, selected }: NodeProps<BatchNormNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="BatchNorm"
      category="activation"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <line x1="1" y1="8" x2="15" y2="8" stroke="currentColor" strokeWidth="1.2" />
          <line x1="4" y1="3" x2="4" y2="13" stroke="currentColor" strokeWidth="1.2" opacity="0.4" />
          <line x1="8" y1="2" x2="8" y2="14" stroke="currentColor" strokeWidth="1.4" />
          <line x1="12" y1="3" x2="12" y2="13" stroke="currentColor" strokeWidth="1.2" opacity="0.4" />
        </svg>
      }
    >
      <NodeRow label="eps" value={data.eps} />
      <NodeRow label="momentum" value={data.momentum} />
    </BaseNode>
  )
}

registerNode({
  type: 'batchNormNode',
  label: 'BatchNorm',
  description: 'Batch normalisation (1d or 2d auto-detected)',
  category: 'layers',
  paletteGroup: 'convolutional',
  defaultData: {
    label: 'BatchNorm',
    eps: 1e-5,
    momentum: 0.1,
    affine: true,
    trackRunningStats: true,
  },
  fields: [
    { key: 'eps', label: 'Epsilon', type: 'number', min: 0, step: 1e-5 },
    { key: 'momentum', label: 'Momentum', type: 'number', min: 0, max: 1, step: 0.01 },
    { key: 'affine', label: 'Affine (learnable γ/β)', type: 'boolean' },
    { key: 'trackRunningStats', label: 'Track Running Stats', type: 'boolean' },
  ],
  component: BatchNormNode,
})

export default BatchNormNode
