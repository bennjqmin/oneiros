import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface ReshapeData extends Record<string, unknown> {
  label: string
  targetFeatures: number
}

type ReshapeNodeType = Node<ReshapeData, 'reshapeNode'>

function ReshapeNode({ id, data, selected }: NodeProps<ReshapeNodeType>) {
  const target = Number(data.targetFeatures ?? -1)
  return (
    <BaseNode nodeId={id} label="Reshape" category="utility" selected={selected}>
      <NodeRow label="Features" value={target === -1 ? 'auto (-1)' : target} />
    </BaseNode>
  )
}

registerNode({
  type: 'reshapeNode',
  label: 'Reshape',
  description: 'Reshape flat tensor (-1 = auto infer)',
  category: 'utility',
  paletteGroup: 'utility',
  defaultData: { label: 'Reshape', targetFeatures: -1 },
  fields: [{ key: 'targetFeatures', label: 'Target Features (-1=auto)', type: 'number', step: 1 }],
  component: ReshapeNode,
})

export default ReshapeNode
