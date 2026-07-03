import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

const MERGE_HANDLES = [
  { id: 'in_a', top: '35%' },
  { id: 'in_b', top: '65%' },
]

interface MultiplyData extends Record<string, unknown> {
  label: string
}

type MultiplyNodeType = Node<MultiplyData, 'multiplyNode'>

function MultiplyNode({ id, selected }: NodeProps<MultiplyNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="Multiply"
      category="utility"
      selected={selected}
      hasTarget={false}
      targetHandles={MERGE_HANDLES}
    >
      <NodeRow label="op" value="a × b" />
    </BaseNode>
  )
}

registerNode({
  type: 'multiplyNode',
  label: 'Multiply',
  description: 'Element-wise multiplication of two tensors (same shape)',
  category: 'utility',
  paletteGroup: 'utility',
  defaultData: { label: 'Multiply' },
  fields: [],
  component: MultiplyNode,
})

export default MultiplyNode
