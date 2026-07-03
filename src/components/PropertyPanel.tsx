import { t } from '../theme/tokens'
import { useRef, useState } from 'react'
import { useGraphStore } from '../store/useGraphStore'
import { getNodeDefinition } from '../editor/registry/nodeRegistry'
import type { NodeField } from '../types/node'
import { useThemeSync } from '../theme/useThemeSync'
import { COLLAPSED_PANEL_WIDTH, CollapseBtn, CollapsedBar, MobileDrawerBackdrop, MobileDrawerHeader, mobileDrawerStyle } from './panelChrome'

const MIN_WIDTH = 200
const MAX_WIDTH = 480

interface PropertyPanelProps {
  mobile?: boolean
  open?: boolean
  onClose?: () => void
}

export default function PropertyPanel({ mobile, open, onClose }: PropertyPanelProps = {}) {
  useThemeSync()
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const nodes = useGraphStore((s) => s.nodes)
  const updateNodeData = useGraphStore((s) => s.updateNodeData)

  const [width, setWidth] = useState(260)
  const [collapsed, setCollapsed] = useState(false)
  const drag = useRef({ active: false, startX: 0, startW: 0 })
  const panelWidth = collapsed ? COLLAPSED_PANEL_WIDTH : width

  const selectedNode = nodes.find((n) => n.id === selectedNodeId)
  const def = selectedNode ? getNodeDefinition(selectedNode.type ?? '') : undefined

  if (mobile && !open) return null

  const inspectorContent = (
    <div style={{ flex: 1, overflowY: 'auto' }}>
      {!selectedNode ? (
        <EmptyState />
      ) : !def ? (
        <div style={{ padding: 14, fontSize: 12, color: t.textFaint }}>
          No definition found for node type "{selectedNode.type}".
        </div>
      ) : (
        <div style={{ padding: '14px 14px' }}>
          <div style={{ marginBottom: 16 }}>
            <NodeIdBadge id={selectedNode.id} />
          </div>
          {def.fields.map((field) => (
            <FieldControl
              key={field.key}
              field={field}
              value={selectedNode.data[field.key]}
              onChange={(val) => updateNodeData(selectedNode.id, { [field.key]: val })}
            />
          ))}
        </div>
      )}
    </div>
  )

  if (mobile) {
    return (
      <>
        <MobileDrawerBackdrop onClose={onClose!} />
        <aside style={{ ...mobileDrawerStyle('right'), display: 'flex', flexDirection: 'column' }}>
          <MobileDrawerHeader title={def?.label ?? 'Inspector'} onClose={onClose!} />
          {inspectorContent}
        </aside>
      </>
    )
  }

  function onResizeMouseDown(e: React.MouseEvent) {
    if (collapsed) return
    e.preventDefault()
    drag.current = { active: true, startX: e.clientX, startW: width }
    const onMove = (ev: MouseEvent) => {
      if (!drag.current.active) return
      // Dragging left edge: moving left increases width
      const next = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, drag.current.startW + drag.current.startX - ev.clientX))
      setWidth(next)
    }
    const onUp = () => {
      drag.current.active = false
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  return (
    <aside
      style={{
        width: panelWidth,
        minWidth: panelWidth,
        background: t.bgPanel,
        borderLeft: `1px solid ${t.borderSubtle}`,
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        overflow: 'hidden',
        position: 'relative',
        transition: 'width 0.18s ease, min-width 0.18s ease',
      }}
    >
      {collapsed ? (
        <CollapsedBar
          side="right"
          label="Inspector"
          onToggle={() => setCollapsed(false)}
        />
      ) : (
        <>
      {/* Resize handle on left edge */}
      <div
        onMouseDown={onResizeMouseDown}
        title="Drag to resize"
        style={{
          position: 'absolute', top: 0, left: 0, width: 4, height: '100%',
          cursor: 'ew-resize', zIndex: 10,
          background: 'transparent', transition: 'background 0.15s',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.background = '#7c3aed66' }}
        onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
      />
      {/* Header */}
      <div
        style={{
          padding: '13px 14px 11px',
          borderBottom: `1px solid ${t.borderSubtle}`,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <CollapseBtn side="right" onClick={() => setCollapsed(true)} title="Collapse inspector" />
        <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: t.textFaint }}>
          Inspector
        </span>
        {selectedNode && def && (
          <span style={{ fontSize: 11, color: t.textMuted, marginLeft: 'auto' }}>{def.label}</span>
        )}
      </div>

      {/* Content */}
      {inspectorContent}
        </>
      )}
    </aside>
  )
}

function EmptyState() {
  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
        gap: 10,
        height: '100%',
        minHeight: 200,
      }}
    >
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#27272a" strokeWidth="1.5">
        <rect x="3" y="3" width="18" height="18" rx="3" />
        <path d="M9 9h6M9 12h6M9 15h4" strokeLinecap="round" />
      </svg>
      <p style={{ fontSize: 12, color: t.textDisabled, textAlign: 'center', margin: 0, lineHeight: 1.5 }}>
        Select a node on the canvas to inspect its properties
      </p>
    </div>
  )
}

function NodeIdBadge({ id }: { id: string }) {
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        background: t.bgElevated,
        border: `1px solid ${t.borderDefault}`,
        borderRadius: 5,
        padding: '3px 8px',
        fontSize: 10,
        color: t.textFaint,
        fontFamily: 'ui-monospace, monospace',
      }}
    >
      {id}
    </div>
  )
}

interface FieldControlProps {
  field: NodeField
  value: unknown
  onChange: (val: unknown) => void
}

function FieldControl({ field, value, onChange }: FieldControlProps) {
  const labelStyle: React.CSSProperties = {
    display: 'block',
    fontSize: 11,
    fontWeight: 500,
    color: t.textMuted,
    marginBottom: 5,
    letterSpacing: '0.02em',
  }

  const inputBaseStyle: React.CSSProperties = {
    width: '100%',
    background: t.bgElevated,
    border: `1px solid ${t.borderDefault}`,
    borderRadius: 6,
    color: t.textPrimary,
    fontSize: 12,
    padding: '6px 9px',
    outline: 'none',
    boxSizing: 'border-box',
    transition: 'border-color 0.1s',
  }

  return (
    <div style={{ marginBottom: 14 }}>
      <label style={labelStyle}>{field.label}</label>

      {field.type === 'number' && (
        <input
          type="number"
          value={typeof value === 'number' ? value : ''}
          min={field.min}
          max={field.max}
          step={field.step ?? 1}
          onChange={(e) => onChange(Number(e.target.value))}
          onFocus={(e) => (e.target.style.borderColor = t.accent)}
          onBlur={(e) => (e.target.style.borderColor = t.borderDefault)}
          style={inputBaseStyle}
        />
      )}

      {field.type === 'text' && (
        <input
          type="text"
          value={typeof value === 'string' ? value : ''}
          placeholder={field.placeholder}
          onChange={(e) => onChange(e.target.value)}
          onFocus={(e) => (e.target.style.borderColor = t.accent)}
          onBlur={(e) => (e.target.style.borderColor = t.borderDefault)}
          style={inputBaseStyle}
        />
      )}

      {field.type === 'select' && (
        <select
          value={typeof value === 'string' ? value : ''}
          onChange={(e) => onChange(e.target.value)}
          onFocus={(e) => (e.target.style.borderColor = t.accent)}
          onBlur={(e) => (e.target.style.borderColor = t.borderDefault)}
          style={{ ...inputBaseStyle, cursor: 'pointer', appearance: 'none' }}
        >
          {field.options?.map((opt) => (
            <option key={opt.value} value={opt.value} style={{ background: t.bgElevated }}>
              {opt.label}
            </option>
          ))}
        </select>
      )}

      {field.type === 'boolean' && (
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 9,
            cursor: 'pointer',
          }}
        >
          <div
            onClick={() => onChange(!value)}
            style={{
              width: 32,
              height: 18,
              borderRadius: 9,
              background: value ? t.accent : t.borderDefault,
              position: 'relative',
              cursor: 'pointer',
              transition: 'background 0.15s',
              flexShrink: 0,
            }}
          >
            <div
              style={{
                position: 'absolute',
                top: 3,
                left: value ? 17 : 3,
                width: 12,
                height: 12,
                borderRadius: '50%',
                background: t.textPrimary,
                transition: 'left 0.15s',
              }}
            />
          </div>
          <span style={{ fontSize: 12, color: value ? t.textBody : t.textFaint }}>
            {value ? 'Enabled' : 'Disabled'}
          </span>
        </label>
      )}
    </div>
  )
}
