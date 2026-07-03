import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface DropoutData extends Record<string, unknown> {
  label: string
  p: number
}

type DropoutNodeType = Node<DropoutData, 'dropoutNode'>

function DropoutNode({ id, data, selected }: NodeProps<DropoutNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="Dropout"
      category="regularization"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <circle cx="4" cy="4" r="1.5" fill="currentColor" opacity="0.3" />
          <circle cx="8" cy="4" r="1.5" fill="currentColor" />
          <circle cx="12" cy="4" r="1.5" fill="currentColor" opacity="0.3" />
          <circle cx="4" cy="8" r="1.5" fill="currentColor" />
          <circle cx="8" cy="8" r="1.5" fill="currentColor" opacity="0.3" />
          <circle cx="12" cy="8" r="1.5" fill="currentColor" />
          <circle cx="4" cy="12" r="1.5" fill="currentColor" opacity="0.3" />
          <circle cx="8" cy="12" r="1.5" fill="currentColor" />
          <circle cx="12" cy="12" r="1.5" fill="currentColor" opacity="0.3" />
        </svg>
      }
    >
      <NodeRow label="p (drop rate)" value={data.p} />
    </BaseNode>
  )
}

registerNode({
  type: 'dropoutNode',
  label: 'Dropout',
  description: 'Randomly zeroes elements during training',
  category: 'regularization',
  paletteGroup: 'regularization',
  defaultData: {
    label: 'Dropout',
    p: 0.5,
  },
  fields: [
    { key: 'p', label: 'Drop Probability', type: 'number', min: 0, max: 1, step: 0.05 },
  ],
  component: DropoutNode,
})

export default DropoutNode
