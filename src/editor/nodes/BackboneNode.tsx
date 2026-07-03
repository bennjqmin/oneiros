import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

export const BACKBONE_OUTPUT_FEATURES: Record<string, number> = {
  resnet18:        512,
  resnet34:        512,
  resnet50:        2048,
  mobilenet_v2:    1280,
  efficientnet_b0: 1280,
  vgg16:           4096,
}

interface BackboneData extends Record<string, unknown> {
  label: string
  model: string
  pretrained: boolean
  freeze: boolean
  outputLayer: string
}

type BackboneNodeType = Node<BackboneData, 'backboneNode'>

function BackboneNode({ id, data, selected }: NodeProps<BackboneNodeType>) {
  const outFeats = BACKBONE_OUTPUT_FEATURES[data.model as string] ?? '?'
  return (
    <BaseNode
      nodeId={id}
      label="Backbone"
      category="layers"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.2" />
          <circle cx="8" cy="8" r="2.5" stroke="currentColor" strokeWidth="1.2" opacity="0.7" />
          <line x1="8" y1="2" x2="8" y2="5.5" stroke="currentColor" strokeWidth="1" opacity="0.5" />
          <line x1="8" y1="10.5" x2="8" y2="14" stroke="currentColor" strokeWidth="1" opacity="0.5" />
          <line x1="2" y1="8" x2="5.5" y2="8" stroke="currentColor" strokeWidth="1" opacity="0.5" />
          <line x1="10.5" y1="8" x2="14" y2="8" stroke="currentColor" strokeWidth="1" opacity="0.5" />
        </svg>
      }
    >
      <NodeRow label="Model" value={data.model as string} />
      <NodeRow label="Out features" value={outFeats} />
      <NodeRow label="Pretrained" value={(data.pretrained as boolean) ? 'Yes' : 'No'} />
      <NodeRow label="Frozen" value={(data.freeze as boolean) ? 'Yes' : 'No'} />
      {data.pretrained && (
        <div style={{ marginTop: 4, padding: '2px 0', display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{
            fontSize: 9, fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase',
            background: 'rgba(168,85,247,0.15)', border: '1px solid rgba(168,85,247,0.3)',
            color: '#d8b4fe', borderRadius: 3, padding: '1px 5px',
          }}>Pretrained</span>
        </div>
      )}
    </BaseNode>
  )
}

registerNode({
  type: 'backboneNode',
  label: 'Backbone',
  description: 'Pretrained vision backbone (ResNet, MobileNet, VGG, EfficientNet)',
  category: 'layers',
  paletteGroup: 'computerVision',
  defaultData: {
    label: 'Backbone',
    model: 'resnet18',
    pretrained: true,
    freeze: false,
    outputLayer: 'avgpool',
  },
  fields: [
    {
      key: 'model',
      label: 'Architecture',
      type: 'select',
      options: [
        { value: 'resnet18',        label: 'ResNet-18  (512 features)' },
        { value: 'resnet34',        label: 'ResNet-34  (512 features)' },
        { value: 'resnet50',        label: 'ResNet-50  (2048 features)' },
        { value: 'mobilenet_v2',    label: 'MobileNet V2 (1280 features)' },
        { value: 'efficientnet_b0', label: 'EfficientNet-B0 (1280 features)' },
        { value: 'vgg16',           label: 'VGG-16 (4096 features)' },
      ],
    },
    { key: 'pretrained', label: 'Use pretrained weights (ImageNet)', type: 'boolean' },
    { key: 'freeze',     label: 'Freeze backbone weights',           type: 'boolean' },
    {
      key: 'outputLayer',
      label: 'Output tap',
      type: 'select',
      options: [
        { value: 'avgpool', label: 'avgpool — flat feature vector (recommended)' },
        { value: 'layer4',  label: 'layer4 — spatial features (ResNet only)' },
        { value: 'layer3',  label: 'layer3 — spatial features (ResNet only)' },
      ],
    },
  ],
  component: BackboneNode,
})

export default BackboneNode
