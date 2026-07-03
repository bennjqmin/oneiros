import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface GRUData extends Record<string, unknown> {
  label: string
  hiddenSize: number
  numLayers: number
  dropout: number
  bidirectional: boolean
}

type GRUNodeType = Node<GRUData, 'gruNode'>

function GRUNode({ id, data, selected }: NodeProps<GRUNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="GRU"
      category="recurrent"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <circle cx="4" cy="8" r="2" stroke="currentColor" strokeWidth="1.2" />
          <circle cx="12" cy="8" r="2" stroke="currentColor" strokeWidth="1.2" />
          <path d="M6 8 L10 8" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
          <path d="M8 3 C8 3 12 3 12 6" stroke="currentColor" strokeWidth="1" fill="none" opacity="0.5" strokeLinecap="round" />
          <path d="M8 13 C8 13 4 13 4 10" stroke="currentColor" strokeWidth="1" fill="none" opacity="0.5" strokeLinecap="round" />
          <polyline points="11,5.5 12,6 11,6.5" stroke="currentColor" strokeWidth="1" fill="none" opacity="0.5" />
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
  type: 'gruNode',
  label: 'GRU',
  description: 'Gated Recurrent Unit',
  category: 'recurrent',
  paletteGroup: 'recurrent',
  defaultData: {
    label: 'GRU',
    hiddenSize: 128,
    numLayers: 1,
    dropout: 0,
    bidirectional: false,
  },
  fields: [
    { key: 'hiddenSize', label: 'Hidden Size', type: 'number', min: 1 },
    { key: 'numLayers', label: 'Num Layers', type: 'number', min: 1 },
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
  component: GRUNode,
})

export default GRUNode
