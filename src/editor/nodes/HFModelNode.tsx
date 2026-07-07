import type { NodeProps, Node } from '@xyflow/react'
import BaseNode, { NodeRow } from './BaseNode'
import { registerNode } from '../registry/nodeRegistry'

interface HFModelData extends Record<string, unknown> {
  label: string
  modelId: string
  outputFeatures: number
  modelKind: string
  pooling: string
  freeze: boolean
  trustRemoteCode: boolean
}

type HFModelNodeType = Node<HFModelData, 'hfModelNode'>

function HFModelNode({ id, data, selected }: NodeProps<HFModelNodeType>) {
  const modelId = (data.modelId as string) || '—'
  const outFeats = Number(data.outputFeatures) || 0
  const kind = (data.modelKind as string) || 'generic'
  const shortId = modelId.length > 28 ? `${modelId.slice(0, 26)}…` : modelId

  return (
    <BaseNode
      nodeId={id}
      label="HF Model"
      category="layers"
      selected={selected}
      icon={
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <rect x="2" y="3" width="12" height="10" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
          <path d="M5 8h6M8 5v6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" opacity="0.6" />
        </svg>
      }
    >
      <NodeRow label="Model" value={shortId} />
      <NodeRow label="Kind" value={kind} />
      <NodeRow label="Out features" value={outFeats > 0 ? outFeats : '?'} />
      <NodeRow label="Pooling" value={(data.pooling as string) || 'mean'} />
      <NodeRow label="Frozen" value={(data.freeze as boolean) ? 'Yes' : 'No'} />
      {!modelId && (
        <div style={{ marginTop: 4, fontSize: 9, color: 'var(--error)' }}>
          Import a model in the Hugging Face tab
        </div>
      )}
      {modelId && (
        <div style={{ marginTop: 4, padding: '2px 0', display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{
            fontSize: 9, fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase',
            background: 'rgba(255,165,0,0.12)', border: '1px solid rgba(255,165,0,0.35)',
            color: '#fdba74', borderRadius: 3, padding: '1px 5px',
          }}>Hugging Face</span>
        </div>
      )}
    </BaseNode>
  )
}

registerNode({
  type: 'hfModelNode',
  label: 'Hugging Face',
  description: 'Pretrained model from Hugging Face Hub (BERT, ViT, DistilBERT, …)',
  category: 'layers',
  paletteGroup: 'transformer',
  defaultData: {
    label: 'HF Model',
    modelId: '',
    outputFeatures: 0,
    modelKind: 'generic',
    pooling: 'mean',
    freeze: false,
    trustRemoteCode: false,
  },
  fields: [
    { key: 'modelId', label: 'Model ID', type: 'text', placeholder: 'distilbert-base-uncased' },
    { key: 'outputFeatures', label: 'Output features', type: 'number', min: 1, step: 1 },
    {
      key: 'pooling',
      label: 'Pooling',
      type: 'select',
      options: [
        { value: 'mean', label: 'Mean pool (sequence models)' },
        { value: 'cls', label: 'CLS token (BERT-style)' },
        { value: 'pooler', label: 'Pooler output (if available)' },
      ],
    },
    { key: 'freeze', label: 'Freeze weights', type: 'boolean' },
    { key: 'trustRemoteCode', label: 'Trust remote code', type: 'boolean' },
  ],
  component: HFModelNode,
})

export default HFModelNode
