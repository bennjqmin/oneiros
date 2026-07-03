import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface UpsampleData extends Record<string, unknown> {
  label: string
  scaleFactor: number
  mode: string
}

type UpsampleNodeType = Node<UpsampleData, 'upsampleNode'>

function UpsampleNode({ id, data, selected }: NodeProps<UpsampleNodeType>) {
  return (
    <BaseNode
      nodeId={id}
      label="Upsample"
      category="layers"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <rect x="4" y="7" width="4" height="4" rx="0.5" stroke="currentColor" strokeWidth="1.2" opacity="0.6" />
          <rect x="1" y="2" width="10" height="10" rx="1" stroke="currentColor" strokeWidth="1.2" />
          <line x1="13" y1="4" x2="13" y2="12" stroke="currentColor" strokeWidth="1.2" strokeDasharray="2 1.5" />
          <line x1="11" y1="4" x2="15" y2="4" stroke="currentColor" strokeWidth="1" opacity="0.6" />
          <line x1="11" y1="12" x2="15" y2="12" stroke="currentColor" strokeWidth="1" opacity="0.6" />
        </svg>
      }
    >
      <NodeRow label="Scale" value={`×${Number(data.scaleFactor).toFixed(1)}`} />
      <NodeRow label="Mode" value={data.mode} />
    </BaseNode>
  )
}

registerNode({
  type: 'upsampleNode',
  label: 'Upsample',
  description: 'Rescale spatial dimensions by scale factor',
  category: 'layers',
  paletteGroup: 'computerVision',
  defaultData: {
    label: 'Upsample',
    scaleFactor: 2,
    mode: 'nearest',
  },
  fields: [
    { key: 'scaleFactor', label: 'Scale Factor', type: 'number', min: 0.1 },
    {
      key: 'mode',
      label: 'Mode',
      type: 'select',
      options: [
        { value: 'nearest',  label: 'Nearest' },
        { value: 'bilinear', label: 'Bilinear' },
        { value: 'bicubic',  label: 'Bicubic' },
      ],
    },
  ],
  component: UpsampleNode,
})

export default UpsampleNode
