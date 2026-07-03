import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface InputData extends Record<string, unknown> {
  label: string
  batchSize: number
  channels: number
  height: number
  width: number
}

type InputNodeType = Node<InputData, 'inputNode'>

function InputNode({ id, data, selected }: NodeProps<InputNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="Input"
      category="input"
      selected={selected}
      hasTarget={false}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
          <rect x="1" y="1" width="14" height="14" rx="2" opacity="0.3" />
          <path d="M8 4v8M4 8h8" strokeWidth="1.5" stroke="currentColor" fill="none" strokeLinecap="round" />
        </svg>
      }
    >
      <NodeRow label="Batch" value={data.batchSize} />
      <NodeRow label="Channels" value={data.channels} />
      <NodeRow
        label="Shape"
        value={`${data.height} × ${data.width}`}
      />
    </BaseNode>
  )
}

registerNode({
  type: 'inputNode',
  label: 'Input',
  description: 'Define the shape of the model input tensor',
  category: 'input',
  paletteGroup: 'basic',
  defaultData: {
    label: 'Input',
    batchSize: 1,
    channels: 1,
    height: 28,
    width: 28,
  },
  fields: [
    { key: 'batchSize', label: 'Batch Size', type: 'number', min: 1, step: 1 },
    { key: 'channels', label: 'Channels', type: 'number', min: 1, step: 1 },
    { key: 'height', label: 'Height', type: 'number', min: 1, step: 1 },
    { key: 'width', label: 'Width', type: 'number', min: 1, step: 1 },
  ],
  component: InputNode,
})

export default InputNode
