import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface ConcatenateData extends Record<string, unknown> {
  label: string
  dim: number
}

type ConcatenateNodeType = Node<ConcatenateData, 'concatenateNode'>

function ConcatenateNode({ id, data, selected }: NodeProps<ConcatenateNodeType>) {
  return (
    <BaseNode nodeId={id} label="Concatenate" category="utility" selected={selected}>
      <NodeRow label="dim" value={data.dim} />
    </BaseNode>
  )
}

registerNode({
  type: 'concatenateNode',
  label: 'Concatenate',
  description: 'Concatenate multiple inputs along a dimension',
  category: 'utility',
  paletteGroup: 'utility',
  defaultData: { label: 'Concatenate', dim: -1 },
  fields: [{ key: 'dim', label: 'Dimension', type: 'number', step: 1 }],
  component: ConcatenateNode,
})

export default ConcatenateNode
