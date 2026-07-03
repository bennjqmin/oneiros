import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

const MERGE_HANDLES = [
  { id: 'in_a', top: '35%' },
  { id: 'in_b', top: '65%' },
]

interface AddData extends Record<string, unknown> {
  label: string
}

type AddNodeType = Node<AddData, 'addNode'>

function AddNode({ id, selected }: NodeProps<AddNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="Add"
      category="utility"
      selected={selected}
      hasTarget={false}
      targetHandles={MERGE_HANDLES}
    >
      <NodeRow label="op" value="a + b" />
    </BaseNode>
  )
}

registerNode({
  type: 'addNode',
  label: 'Add',
  description: 'Element-wise addition of two tensors (same shape)',
  category: 'utility',
  paletteGroup: 'utility',
  defaultData: { label: 'Add' },
  fields: [],
  component: AddNode,
})

export default AddNode
