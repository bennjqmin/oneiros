import { t } from '../../theme/tokens'
import { useCallback } from 'react'
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  useReactFlow,
} from '@xyflow/react'
import type { DragEvent } from 'react'
import '@xyflow/react/dist/style.css'

import { useDatasetStore } from '../../store/useDatasetStore'
import { datasetNodeTypes, datasetNodeDefs } from './preprocessingNodes'
import type { DatasetNodeDef } from './preprocessingNodes'

function DatasetFlowInner() {
  const { screenToFlowPosition } = useReactFlow()
  const pipelineNodes = useDatasetStore((s) => s.pipelineNodes)
  const pipelineEdges = useDatasetStore((s) => s.pipelineEdges)
  const onNodesChange = useDatasetStore((s) => s.onPipelineNodesChange)
  const onEdgesChange = useDatasetStore((s) => s.onPipelineEdgesChange)
  const onConnect = useDatasetStore((s) => s.onPipelineConnect)
  const addNode = useDatasetStore((s) => s.addPipelineNode)

  const onDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }, [])

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      const type = e.dataTransfer.getData('application/oneiros-dataset-node')
      if (!type) return
      const def = datasetNodeDefs.find((d) => d.type === type)
      if (!def) return
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY })
      addNode({
        id: `${type}-${Date.now()}`,
        type,
        position,
        data: { label: def.label, ...def.defaultData },
      })
    },
    [screenToFlowPosition, addNode]
  )

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={pipelineNodes}
        edges={pipelineEdges}
        nodeTypes={datasetNodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDragOver={onDragOver}
        onDrop={onDrop}
        defaultEdgeOptions={{ style: { stroke: t.textFaint, strokeWidth: 1.5 }, animated: false }}
        proOptions={{ hideAttribution: true }}
        fitView
        minZoom={0.3}
        maxZoom={2}
      >
        <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#1e2030" />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}

export default function DatasetFlow() {
  return (
    <ReactFlowProvider>
      <DatasetFlowInner />
    </ReactFlowProvider>
  )
}

// ── Palette item used by DatasetPage ─────────────────────────────────────────

export function PipelinePaletteItem({ def }: { def: DatasetNodeDef }) {
  function onDragStart(e: DragEvent<HTMLDivElement>) {
    e.dataTransfer.setData('application/oneiros-dataset-node', def.type)
    e.dataTransfer.effectAllowed = 'copy'
  }

  return (
    <div
      draggable
      onDragStart={onDragStart}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 7,
        padding: '6px 10px',
        borderRadius: 6,
        border: '1px solid transparent',
        cursor: 'grab',
        userSelect: 'none',
        transition: 'background 0.1s, border-color 0.1s',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = t.bgElevated
        e.currentTarget.style.borderColor = t.borderDefault
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent'
        e.currentTarget.style.borderColor = 'transparent'
      }}
    >
      <span style={{ width: 7, height: 7, borderRadius: '50%', background: def.color, flexShrink: 0 }} />
      <div>
        <div style={{ fontSize: 11, fontWeight: 500, color: t.textBody, lineHeight: 1.3 }}>{def.label}</div>
        <div style={{ fontSize: 10, color: t.textFaint, lineHeight: 1.2 }}>{def.description}</div>
      </div>
    </div>
  )
}
