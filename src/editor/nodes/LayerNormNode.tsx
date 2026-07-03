import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface LayerNormData extends Record<string, unknown> {
  label: string
  eps: number
}

type LayerNormNodeType = Node<LayerNormData, 'layerNormNode'>

function LayerNormNode({ id, data, selected }: NodeProps<LayerNormNodeType>) {
  return (
    <BaseNode nodeId={id} label="Layer Norm" category="transformer" selected={selected}>
      <NodeRow label="eps" value={data.eps} />
    </BaseNode>
  )
}

registerNode({
  type: 'layerNormNode',
  label: 'Layer Norm',
  description: 'Layer normalization (normalized over last dim)',
  category: 'transformer',
  paletteGroup: 'utility',
  defaultData: { label: 'Layer Norm', eps: 1e-5 },
  fields: [{ key: 'eps', label: 'Epsilon', type: 'number', min: 0, step: 1e-6 }],
  component: LayerNormNode,
})

export default LayerNormNode
