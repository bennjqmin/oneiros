import { t } from '../../theme/tokens'
import { Handle, Position } from '@xyflow/react'
import type { ReactNode } from 'react'
import type { NodeCategory } from '../../types/node'
import { useGraphStore } from '../../store/useGraphStore'
import { getNodeDefinition } from '../registry/nodeRegistry'

const categoryConfig: Record<
  NodeCategory,
  { border: string; headerBg: string; textColor: string; handleColor: string; dot: string }
> = {
  input: {
    border: '#3b82f6',
    headerBg: 'rgba(59,130,246,0.08)',
    textColor: '#93c5fd',
    handleColor: '#3b82f6',
    dot: '#3b82f6',
  },
  output: {
    border: '#10b981',
    headerBg: 'rgba(16,185,129,0.08)',
    textColor: '#6ee7b7',
    handleColor: '#10b981',
    dot: '#10b981',
  },
  layers: {
    border: '#8b5cf6',
    headerBg: 'rgba(139,92,246,0.08)',
    textColor: '#c4b5fd',
    handleColor: '#8b5cf6',
    dot: '#8b5cf6',
  },
  recurrent: {
    border: '#ec4899',
    headerBg: 'rgba(236,72,153,0.08)',
    textColor: '#f9a8d4',
    handleColor: '#ec4899',
    dot: '#ec4899',
  },
  attention: {
    border: '#06b6d4',
    headerBg: 'rgba(6,182,212,0.08)',
    textColor: '#67e8f9',
    handleColor: '#06b6d4',
    dot: '#06b6d4',
  },
  activation: {
    border: '#f59e0b',
    headerBg: 'rgba(245,158,11,0.08)',
    textColor: '#fcd34d',
    handleColor: '#f59e0b',
    dot: '#f59e0b',
  },
  utility: {
    border: '#64748b',
    headerBg: 'rgba(100,116,139,0.08)',
    textColor: '#94a3b8',
    handleColor: '#64748b',
    dot: '#64748b',
  },
  regularization: {
    border: '#f97316',
    headerBg: 'rgba(249,115,22,0.08)',
    textColor: '#fdba74',
    handleColor: '#f97316',
    dot: '#f97316',
  },
  transformer: {
    border: '#06b6d4',
    headerBg: 'rgba(6,182,212,0.08)',
    textColor: '#67e8f9',
    handleColor: '#06b6d4',
    dot: '#06b6d4',
  },
}

interface TargetHandleSpec {
  id: string
  top?: string | number
}

interface BaseNodeProps {
  nodeId: string
  label: string
  category: NodeCategory
  selected?: boolean
  children?: ReactNode
  hasSource?: boolean
  hasTarget?: boolean
  targetHandles?: TargetHandleSpec[]
  icon?: ReactNode
}

export default function BaseNode({
  nodeId,
  label,
  category,
  selected = false,
  children,
  hasSource = true,
  hasTarget = true,
  targetHandles,
  icon,
}: BaseNodeProps) {
  const cfg = categoryConfig[category]

  const allIssues = useGraphStore((s) => s.validationIssues)
  const issues = allIssues.filter((i) => i.nodeId === nodeId)

  const nodeType = useGraphStore((s) => s.nodes.find((n) => n.id === nodeId)?.type ?? '')
  const description = getNodeDefinition(nodeType)?.description

  const hasError = issues.some((i) => i.severity === 'error')
  const hasWarning = !hasError && issues.some((i) => i.severity === 'warning')

  const borderColor = hasError
    ? t.error
    : hasWarning
      ? t.warning
      : selected
        ? cfg.border
        : t.borderDefault

  const glow = hasError
    ? `0 0 0 1px ${t.error}40, ${t.shadowPanel}`
    : hasWarning
      ? `0 0 0 1px ${t.warning}40, ${t.shadowPanel}`
      : selected
        ? `0 0 0 1px ${cfg.border}40, ${t.shadowPanel}`
        : t.shadowPanel

  return (
    <div
      style={{
        width: 216,
        background: t.bgElevated,
        borderRadius: 10,
        border: `1px solid ${borderColor}`,
        boxShadow: glow,
        overflow: 'hidden',
        transition: 'border-color 0.15s, box-shadow 0.15s',
      }}
    >
      {targetHandles && targetHandles.length > 0 ? (
        targetHandles.map((h) => (
          <Handle
            key={h.id}
            id={h.id}
            type="target"
            position={Position.Left}
            style={{
              width: 10,
              height: 10,
              background: t.bgElevated,
              border: `2px solid ${cfg.handleColor}`,
              borderRadius: '50%',
              left: -5,
              top: h.top,
            }}
          />
        ))
      ) : hasTarget ? (
        <Handle
          type="target"
          position={Position.Left}
          style={{
            width: 10,
            height: 10,
            background: t.bgElevated,
            border: `2px solid ${cfg.handleColor}`,
            borderRadius: '50%',
            left: -5,
          }}
        />
      ) : null}

      {/* Header */}
      <div
        style={{
          background: cfg.headerBg,
          borderBottom: `1px solid ${t.borderDefault}`,
          padding: '8px 12px',
          display: 'flex',
          alignItems: 'center',
          gap: 7,
        }}
      >
        {icon && (
          <span style={{ color: cfg.textColor, display: 'flex', alignItems: 'center' }}>
            {icon}
          </span>
        )}
        <span
          style={{
            color: cfg.textColor,
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            lineHeight: 1,
          }}
        >
          {label}
        </span>

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 5 }}>
          {/* Validation badge */}
          {(hasError || hasWarning) && (
            <ValidationBadge
              hasError={hasError}
              messages={issues.map((i) => i.message)}
            />
          )}
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: cfg.dot,
              flexShrink: 0,
            }}
          />
        </div>
      </div>

      {/* Body */}
      {children && (
        <div style={{ padding: '10px 12px' }}>
          {children}
        </div>
      )}

      {/* Description footer */}
      {description && (
        <div style={{
          padding: '5px 12px 7px',
          borderTop: children ? `1px solid ${t.borderSubtle}` : undefined,
          fontSize: 10,
          color: t.textDisabled,
          lineHeight: 1.4,
          fontStyle: 'italic',
        }}>
          {description}
        </div>
      )}

      {hasSource && (
        <Handle
          type="source"
          position={Position.Right}
          style={{
            width: 10,
            height: 10,
            background: cfg.handleColor,
            border: `2px solid ${cfg.handleColor}99`,
            borderRadius: '50%',
            right: -5,
          }}
        />
      )}
    </div>
  )
}

// ── Validation badge with tooltip ────────────────────────────────────────────

function ValidationBadge({
  hasError,
  messages,
}: {
  hasError: boolean
  messages: string[]
}) {
  const color = hasError ? t.error : t.warning
  const bg = hasError ? t.errorSubtle : t.warningSubtle

  return (
    <div style={{ position: 'relative' }} className="validation-badge-host">
      <div
        style={{
          width: 16,
          height: 16,
          borderRadius: '50%',
          background: bg,
          border: `1px solid ${color}40`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          cursor: 'default',
        }}
      >
        <span style={{ fontSize: 10, fontWeight: 700, color, lineHeight: 1 }}>
          {hasError ? '!' : '?'}
        </span>
      </div>

      {/* Tooltip */}
      <div
        className="validation-tooltip"
        style={{
          position: 'absolute',
          right: 20,
          top: -4,
          background: t.bgBase,
          border: `1px solid ${color}40`,
          borderRadius: 6,
          padding: '6px 10px',
          minWidth: 180,
          maxWidth: 240,
          zIndex: 999,
          pointerEvents: 'none',
          opacity: 0,
          transition: 'opacity 0.1s',
          boxShadow: t.shadowPanel,
        }}
      >
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              fontSize: 11,
              color: hasError ? t.error : t.warning,
              lineHeight: 1.5,
              marginBottom: i < messages.length - 1 ? 4 : 0,
            }}
          >
            {m}
          </div>
        ))}
      </div>
    </div>
  )
}

// Tooltip hover — injected once into the document
if (typeof document !== 'undefined') {
  const styleId = 'oneiros-validation-tooltip-style'
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style')
    style.id = styleId
    style.textContent = `
      .validation-badge-host:hover .validation-tooltip { opacity: 1 !important; }
    `
    document.head.appendChild(style)
  }
}

// ── Shared helper for node body rows ─────────────────────────────────────────

export function NodeRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 5,
        fontSize: 12,
      }}
    >
      <span style={{ color: '#71717a' }}>{label}</span>
      <span style={{ color: '#e4e4e7', fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </span>
    </div>
  )
}
