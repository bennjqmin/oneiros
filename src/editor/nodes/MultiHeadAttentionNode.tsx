import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface MultiHeadAttentionData extends Record<string, unknown> {
  label: string
  embedDim: number
  numHeads: number
  dropout: number
}

type MultiHeadAttentionNodeType = Node<MultiHeadAttentionData, 'multiHeadAttentionNode'>

function MultiHeadAttentionNode({ id, data, selected }: NodeProps<MultiHeadAttentionNodeType>) {
  return (
    <BaseNode nodeId={id} label="Multi-Head Attention" category="transformer" selected={selected}>
      <NodeRow label="embed_dim" value={data.embedDim} />
      <NodeRow label="heads" value={data.numHeads} />
    </BaseNode>
  )
}

registerNode({
  type: 'multiHeadAttentionNode',
  label: 'Multi-Head Attention',
  description: 'Self-attention (batch, 1, embed_dim) pattern',
  category: 'transformer',
  paletteGroup: 'utility',
  defaultData: {
    label: 'Multi-Head Attention',
    embedDim: 256,
    numHeads: 8,
    dropout: 0.1,
  },
  fields: [
    { key: 'embedDim', label: 'Embed Dim', type: 'number', min: 1 },
    { key: 'numHeads', label: 'Num Heads', type: 'number', min: 1 },
    { key: 'dropout', label: 'Dropout', type: 'number', min: 0, max: 1, step: 0.05 },
  ],
  component: MultiHeadAttentionNode,
})

export default MultiHeadAttentionNode
