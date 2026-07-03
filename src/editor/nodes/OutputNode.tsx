import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface OutputData extends Record<string, unknown> {
  label: string
  classes: number
  name: string
}

type OutputNodeType = Node<OutputData, 'outputNode'>

function OutputNode({ id, data, selected }: NodeProps<OutputNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="Output"
      category="output"
      selected={selected}
      hasSource={false}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
          <rect x="1" y="1" width="14" height="14" rx="2" opacity="0.3" />
          <path d="M5 8h6M9 5l3 3-3 3" strokeWidth="1.5" stroke="currentColor" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      }
    >
      <NodeRow label="Name" value={data.name} />
      <NodeRow label="Classes" value={data.classes} />
    </BaseNode>
  )
}

registerNode({
  type: 'outputNode',
  label: 'Output',
  description: 'Terminal node — defines model output',
  category: 'output',
  paletteGroup: 'basic',
  defaultData: {
    label: 'Output',
    name: 'output',
    classes: 10,
  },
  fields: [
    { key: 'name', label: 'Name', type: 'text', placeholder: 'output' },
    { key: 'classes', label: 'Classes', type: 'number', min: 1, step: 1 },
  ],
  component: OutputNode,
})

export default OutputNode
