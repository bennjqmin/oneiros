import { t } from '../../theme/tokens'
import type { NodeProps, Node } from '@xyflow/react'
import BaseNode from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface FlattenData extends Record<string, unknown> {
  label: string
}

type FlattenNodeType = Node<FlattenData, 'flattenNode'>

function FlattenNode({ id, selected }: NodeProps<FlattenNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="Flatten"
      category="layers"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <rect x="1" y="3" width="6" height="10" rx="1" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
          <line x1="9" y1="8" x2="12" y2="8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          <polyline points="11,6 13,8 11,10" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
          <rect x="13" y="6" width="2" height="4" rx="0.5" fill="currentColor" opacity="0.6" />
        </svg>
      }
    >
      <div style={{ fontSize: 9, color: t.textMuted, fontStyle: 'italic', padding: '2px 0' }}>
        spatial → flat
      </div>
    </BaseNode>
  )
}

registerNode({
  type: 'flattenNode',
  label: 'Flatten',
  description: 'Reshapes spatial tensor to 1D vector',
  category: 'layers',
  paletteGroup: 'basic',
  defaultData: {
    label: 'Flatten',
  },
  fields: [],
  component: FlattenNode,
})

export default FlattenNode
