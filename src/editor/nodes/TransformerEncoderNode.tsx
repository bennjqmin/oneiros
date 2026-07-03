import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface TransformerEncoderData extends Record<string, unknown> {
  label: string
  dModel: number
  nhead: number
  dimFeedforward: number
  numLayers: number
  dropout: number
  activation: string
  normFirst: boolean
}

type TransformerEncoderNodeType = Node<TransformerEncoderData, 'transformerEncoderNode'>

function TransformerEncoderNode({ id, data, selected }: NodeProps<TransformerEncoderNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="Transformer Encoder"
      category="transformer"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <rect x="1" y="2" width="14" height="3" rx="1" stroke="currentColor" strokeWidth="1.1" />
          <rect x="1" y="7" width="14" height="3" rx="1" stroke="currentColor" strokeWidth="1.1" opacity="0.6" />
          <rect x="1" y="12" width="14" height="2" rx="1" stroke="currentColor" strokeWidth="1.1" opacity="0.3" />
          <line x1="4" y1="3.5" x2="12" y2="3.5" stroke="currentColor" strokeWidth="0.8" opacity="0.4" />
          <line x1="4" y1="8.5" x2="12" y2="8.5" stroke="currentColor" strokeWidth="0.8" opacity="0.4" />
        </svg>
      }
    >
      <NodeRow label="d_model" value={data.dModel} />
      <NodeRow label="nhead" value={data.nhead} />
      <NodeRow label="ff_dim" value={data.dimFeedforward} />
      <NodeRow label="Layers" value={data.numLayers} />
    </BaseNode>
  )
}

registerNode({
  type: 'transformerEncoderNode',
  label: 'Transformer Encoder',
  description: 'Multi-layer Transformer encoder (projects input → d_model)',
  category: 'transformer',
  paletteGroup: 'transformer',
  defaultData: {
    label: 'Transformer Encoder',
    dModel: 256,
    nhead: 8,
    dimFeedforward: 512,
    numLayers: 2,
    dropout: 0.1,
    activation: 'relu',
    normFirst: false,
  },
  fields: [
    { key: 'dModel', label: 'd_model', type: 'number', min: 1 },
    { key: 'nhead', label: 'Num Heads', type: 'number', min: 1 },
    { key: 'dimFeedforward', label: 'FF Dim', type: 'number', min: 1 },
    { key: 'numLayers', label: 'Num Layers', type: 'number', min: 1 },
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
    { key: 'normFirst', label: 'Pre-Norm (norm first)', type: 'boolean' },
  ],
  component: TransformerEncoderNode,
})

export default TransformerEncoderNode
