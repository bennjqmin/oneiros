import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface EmbeddingData extends Record<string, unknown> {
  label: string
  numEmbeddings: number
  embeddingDim: number
}

type EmbeddingNodeType = Node<EmbeddingData, 'embeddingNode'>

function EmbeddingNode({ id, data, selected }: NodeProps<EmbeddingNodeType>) {
  return (
    <BaseNode nodeId={id} label="Embedding" category="transformer" selected={selected}>
      <NodeRow label="Vocab" value={data.numEmbeddings} />
      <NodeRow label="Dim" value={data.embeddingDim} />
    </BaseNode>
  )
}

registerNode({
  type: 'embeddingNode',
  label: 'Embedding',
  description: 'Token embedding lookup (flat seq_len → flat seq×dim)',
  category: 'transformer',
  paletteGroup: 'utility',
  defaultData: { label: 'Embedding', numEmbeddings: 1000, embeddingDim: 128 },
  fields: [
    { key: 'numEmbeddings', label: 'Num Embeddings', type: 'number', min: 1 },
    { key: 'embeddingDim', label: 'Embedding Dim', type: 'number', min: 1 },
  ],
  component: EmbeddingNode,
})

export default EmbeddingNode
