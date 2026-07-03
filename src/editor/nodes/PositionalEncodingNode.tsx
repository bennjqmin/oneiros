import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface PositionalEncodingData extends Record<string, unknown> {
  label: string
  dModel: number
  maxLen: number
}

type PositionalEncodingNodeType = Node<PositionalEncodingData, 'positionalEncodingNode'>

function PositionalEncodingNode({ id, data, selected }: NodeProps<PositionalEncodingNodeType>) {
  return (
    <BaseNode nodeId={id} label="Positional Encoding" category="transformer" selected={selected}>
      <NodeRow label="d_model" value={data.dModel} />
      <NodeRow label="max_len" value={data.maxLen} />
    </BaseNode>
  )
}

registerNode({
  type: 'positionalEncodingNode',
  label: 'Positional Encoding',
  description: 'Sinusoidal positional encoding added to sequence',
  category: 'transformer',
  paletteGroup: 'utility',
  defaultData: { label: 'Positional Encoding', dModel: 256, maxLen: 512 },
  fields: [
    { key: 'dModel', label: 'd_model', type: 'number', min: 1 },
    { key: 'maxLen', label: 'Max Length', type: 'number', min: 1 },
  ],
  component: PositionalEncodingNode,
})

export default PositionalEncodingNode
