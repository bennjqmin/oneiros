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
import { edfNodeTypes } from './edfNodes'
import type { EDFNodeDef } from './edfNodes'

function EDFFlowInner() {
  const { screenToFlowPosition } = useReactFlow()
  const pipelineNodes   = useDatasetStore((s) => s.edfPipelineNodes)
  const pipelineEdges   = useDatasetStore((s) => s.edfPipelineEdges)
  const onNodesChange   = useDatasetStore((s) => s.onEDFPipelineNodesChange)
  const onEdgesChange   = useDatasetStore((s) => s.onEDFPipelineEdgesChange)
  const onConnect       = useDatasetStore((s) => s.onEDFPipelineConnect)
  const addNode         = useDatasetStore((s) => s.addEDFPipelineNode)

  const onDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }, [])

  const onDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const raw = e.dataTransfer.getData('application/oneiros-edf-node')
    if (!raw) return
    const def: EDFNodeDef = JSON.parse(raw)
    const pos = screenToFlowPosition({ x: e.clientX, y: e.clientY })
    addNode({
      id:       `${def.type}-${Date.now()}`,
      type:     def.type,
      position: pos,
      data:     { label: def.label, ...def.defaultData },
    })
  }, [screenToFlowPosition, addNode])

  return (
    <ReactFlow
      nodes={pipelineNodes}
      edges={pipelineEdges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onDragOver={onDragOver}
      onDrop={onDrop}
      nodeTypes={edfNodeTypes}
      fitView
      fitViewOptions={{ padding: 0.3 }}
      style={{ background: t.bgBase }}
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#27272a" />
      <Controls style={{ bottom: 16, left: 16, top: 'unset' }} />
    </ReactFlow>
  )
}

export default function EDFFlow() {
  return (
    <ReactFlowProvider>
      <EDFFlowInner />
    </ReactFlowProvider>
  )
}

export function EDFPaletteItem({ def }: { def: EDFNodeDef }) {
  function onDragStart(e: DragEvent<HTMLDivElement>) {
    e.dataTransfer.setData('application/oneiros-edf-node', JSON.stringify(def))
    e.dataTransfer.effectAllowed = 'copy'
  }

  return (
    <div
      draggable
      onDragStart={onDragStart}
      style={{
        padding: '6px 12px',
        cursor: 'grab',
        userSelect: 'none',
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        borderBottom: '1px solid #1a1a20',
        transition: 'background 0.1s',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = t.bgElevated }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
        <span style={{ width: 7, height: 7, borderRadius: '50%', background: def.color, flexShrink: 0 }} />
        <span style={{ fontSize: 12, color: t.textBody, fontWeight: 500 }}>{def.label}</span>
      </div>
      <span style={{ fontSize: 10, color: t.textFaint, paddingLeft: 14 }}>{def.description}</span>
    </div>
  )
}
