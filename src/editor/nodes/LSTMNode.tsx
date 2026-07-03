import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface LSTMData extends Record<string, unknown> {
  label: string
  hiddenSize: number
  numLayers: number
  dropout: number
  bidirectional: boolean
  projSize: number
}

type LSTMNodeType = Node<LSTMData, 'lstmNode'>

function LSTMNode({ id, data, selected }: NodeProps<LSTMNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="LSTM"
      category="recurrent"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <rect x="1" y="5" width="14" height="6" rx="1" stroke="currentColor" strokeWidth="1.2" />
          <line x1="5" y1="5" x2="5" y2="11" stroke="currentColor" strokeWidth="1" opacity="0.5" />
          <line x1="8" y1="5" x2="8" y2="11" stroke="currentColor" strokeWidth="1" opacity="0.5" />
          <line x1="11" y1="5" x2="11" y2="11" stroke="currentColor" strokeWidth="1" opacity="0.5" />
          <path d="M4 2 L8 4 L12 2" stroke="currentColor" strokeWidth="1" strokeLinecap="round" fill="none" opacity="0.4" />
          <path d="M4 14 L8 12 L12 14" stroke="currentColor" strokeWidth="1" strokeLinecap="round" fill="none" opacity="0.4" />
        </svg>
      }
    >
      <NodeRow label="Hidden Size" value={data.hiddenSize} />
      <NodeRow label="Layers" value={data.numLayers} />
      {data.dropout > 0 && <NodeRow label="Dropout" value={data.dropout} />}
      {data.bidirectional && <NodeRow label="Direction" value="bidir" />}
    </BaseNode>
  )
}

registerNode({
  type: 'lstmNode',
  label: 'LSTM',
  description: 'Long Short-Term Memory cell',
  category: 'recurrent',
  paletteGroup: 'recurrent',
  defaultData: {
    label: 'LSTM',
    hiddenSize: 128,
    numLayers: 1,
    dropout: 0,
    bidirectional: false,
    projSize: 0,
  },
  fields: [
    { key: 'hiddenSize', label: 'Hidden Size', type: 'number', min: 1 },
    { key: 'numLayers', label: 'Num Layers', type: 'number', min: 1 },
    { key: 'projSize', label: 'Proj Size (0 = off)', type: 'number', min: 0 },
    { key: 'dropout', label: 'Dropout', type: 'number', min: 0, max: 1, step: 0.05 },
    {
      key: 'bidirectional',
      label: 'Bidirectional',
      type: 'select',
      options: [
        { value: 'false', label: 'No' },
        { value: 'true', label: 'Yes' },
      ],
    },
  ],
  component: LSTMNode,
})

export default LSTMNode
