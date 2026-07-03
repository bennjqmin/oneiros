import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface FeedForwardData extends Record<string, unknown> {
  label: string
  dModel: number
  ffDim: number
  dropout: number
  activation: string
}

type FeedForwardNodeType = Node<FeedForwardData, 'feedForwardNode'>

function FeedForwardNode({ id, data, selected }: NodeProps<FeedForwardNodeType>) {
  return (
    <BaseNode nodeId={id} label="Feed Forward" category="transformer" selected={selected}>
      <NodeRow label="d_model" value={data.dModel} />
      <NodeRow label="ff_dim" value={data.ffDim} />
      <NodeRow label="dropout" value={data.dropout} />
    </BaseNode>
  )
}

registerNode({
  type: 'feedForwardNode',
  label: 'Feed Forward',
  description: 'Transformer FFN block (Linear → act → Linear)',
  category: 'transformer',
  paletteGroup: 'utility',
  defaultData: {
    label: 'Feed Forward',
    dModel: 256,
    ffDim: 512,
    dropout: 0.1,
    activation: 'relu',
  },
  fields: [
    { key: 'dModel', label: 'd_model', type: 'number', min: 1 },
    { key: 'ffDim', label: 'FF Dim', type: 'number', min: 1 },
    { key: 'dropout', label: 'Dropout', type: 'number', min: 0, max: 1, step: 0.05 },
    {
      key: 'activation',
      label: 'Activation',
      type: 'select',
      options: [
        { value: 'relu', label: 'ReLU' },
        { value: 'gelu', label: 'GELU' },
      ],
    },
  ],
  component: FeedForwardNode,
})

export default FeedForwardNode
