import { t } from '../../theme/tokens'
import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface AdaptiveAvgPool2dData extends Record<string, unknown> {
  label: string
  outputSize: number  // output H=W (usually 1 for global avg pooling)
}

type AdaptiveAvgPool2dNodeType = Node<AdaptiveAvgPool2dData, 'adaptiveAvgPool2dNode'>

function AdaptiveAvgPool2dNode({ id, data, selected }: NodeProps<AdaptiveAvgPool2dNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="AdaptiveAvgPool2d"
      category="layers"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <rect x="1" y="1" width="14" height="14" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
          <circle cx="8" cy="8" r="3" fill="currentColor" opacity="0.3" />
          <circle cx="8" cy="8" r="1.5" fill="currentColor" opacity="0.7" />
          <text x="8" y="8.5" textAnchor="middle" fontSize="3.5" fill="currentColor" opacity="0.9">avg</text>
        </svg>
      }
    >
      <NodeRow label="Output Size" value={`${data.outputSize}×${data.outputSize}`} />
      {data.outputSize === 1 && (
        <div style={{ fontSize: 9, color: t.textFaint, fontStyle: 'italic', padding: '2px 0' }}>global avg pool</div>
      )}
    </BaseNode>
  )
}

registerNode({
  type: 'adaptiveAvgPool2dNode',
  label: 'AdaptiveAvgPool2d',
  description: 'Global average pooling — spatial → flat(channels)',
  category: 'layers',
  paletteGroup: 'convolutional',
  defaultData: {
    label: 'AdaptiveAvgPool2d',
    outputSize: 1,
  },
  fields: [
    { key: 'outputSize', label: 'Output H=W', type: 'number', min: 1 },
  ],
  component: AdaptiveAvgPool2dNode,
})

export default AdaptiveAvgPool2dNode
