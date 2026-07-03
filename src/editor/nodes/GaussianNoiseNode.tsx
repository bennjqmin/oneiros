import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface GaussianNoiseData extends Record<string, unknown> {
  label: string
  std: number
}

type GaussianNoiseNodeType = Node<GaussianNoiseData, 'gaussianNoiseNode'>

function GaussianNoiseNode({ id, data, selected }: NodeProps<GaussianNoiseNodeType>) {
  return (
    <BaseNode nodeId={id} label="Gaussian Noise" category="utility" selected={selected}>
      <NodeRow label="Std" value={data.std} />
    </BaseNode>
  )
}

registerNode({
  type: 'gaussianNoiseNode',
  label: 'Gaussian Noise',
  description: 'Adds train-time Gaussian noise (passthrough shape)',
  category: 'utility',
  paletteGroup: 'utility',
  defaultData: { label: 'Gaussian Noise', std: 0.1 },
  fields: [{ key: 'std', label: 'Std Dev', type: 'number', min: 0, step: 0.01 }],
  component: GaussianNoiseNode,
})

export default GaussianNoiseNode
