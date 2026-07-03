import { t } from '../theme/tokens'
import { useState, useRef } from 'react'
import { useProjectStore } from '../store/useProjectStore'
import { useGraphStore } from '../store/useGraphStore'
import { AppLogo } from './AppLogo'
import ThemeSwitcher from './ThemeSwitcher'
import { useThemeSync } from '../theme/useThemeSync'

function HomeIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="15 18 9 12 15 6" />
    </svg>
  )
}

// ── Icons ─────────────────────────────────────────────────────────────────────

function SaveIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
      <polyline points="17 21 17 13 7 13 7 21" />
      <polyline points="7 3 7 8 15 8" />
    </svg>
  )
}

function FolderOpenIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </svg>
  )
}

function PlayIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5 3 19 12 5 21 5 3" />
    </svg>
  )
}

function SparkleIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17 5.8 21.3l2.4-7.4L2 9.4h7.6z" />
    </svg>
  )
}

function UndoIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 7v6h6" />
      <path d="M3 13a9 9 0 1 0 2.83-6.36L3 9" />
    </svg>
  )
}

function RedoIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 7v6h-6" />
      <path d="M21 13a9 9 0 1 1-2.83-6.36L21 9" />
    </svg>
  )
}

function NodesIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="7" height="7" rx="1" />
      <rect x="14" y="14" width="7" height="7" rx="1" />
      <path d="M10 6h4M6 10v4M14 10v4M10 18h4" />
    </svg>
  )
}

function InspectorIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
    </svg>
  )
}

// ── Topbar ────────────────────────────────────────────────────────────────────

export type AppView = 'model' | 'dataset'

interface TopbarProps {
  view: AppView
  onViewChange: (v: AppView) => void
  trainOpen: boolean
  onToggleTrain: () => void
  aiOpen: boolean
  onToggleAI: () => void
  isMobile?: boolean
  paletteOpen?: boolean
  onTogglePalette?: () => void
  inspectorOpen?: boolean
  onToggleInspector?: () => void
}

export default function Topbar({
  view, onViewChange, trainOpen, onToggleTrain, aiOpen, onToggleAI,
  isMobile, paletteOpen, onTogglePalette, inspectorOpen, onToggleInspector,
}: TopbarProps) {
  useThemeSync()
  const { name, isDirty, setName, saveToFile, loadFromFile, goHome } = useProjectStore()
  const canUndo = useGraphStore((s) => s.canUndo)
  const canRedo = useGraphStore((s) => s.canRedo)
  const undo = useGraphStore((s) => s.undo)
  const redo = useGraphStore((s) => s.redo)
  const validationIssues = useGraphStore((s) => s.validationIssues)

  const errorCount = validationIssues.filter((i) => i.severity === 'error').length
  const warnCount = validationIssues.filter((i) => i.severity === 'warning').length

  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(name)
  const inputRef = useRef<HTMLInputElement>(null)

  function startEdit() {
    setDraft(name)
    setEditing(true)
    setTimeout(() => inputRef.current?.select(), 10)
  }

  function commitEdit() {
    const trimmed = draft.trim()
    if (trimmed) setName(trimmed)
    setEditing(false)
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') commitEdit()
    if (e.key === 'Escape') setEditing(false)
  }

  return (
    <header
      style={{
        height: 52,
        background: t.topbarBg,
        borderBottom: `1px solid ${t.topbarBorder}`,
        display: 'flex',
        alignItems: 'center',
        padding: '0 14px',
        gap: 10,
        flexShrink: 0,
        zIndex: 10,
      }}
    >
      {/* Back to projects */}
      <button
        onClick={goHome}
        title="All Projects"
        style={{
          display: 'flex', alignItems: 'center', gap: 4,
          padding: '4px 7px', borderRadius: 5,
          border: 'none', background: 'transparent',
          color: t.textFaint, fontSize: 11, cursor: 'pointer',
          transition: 'color 0.1s',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.color = t.textSecondary }}
        onMouseLeave={(e) => { e.currentTarget.style.color = t.textFaint }}
      >
        <HomeIcon />
      </button>

      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
        <AppLogo size={isMobile ? 22 : 26} />
        {!isMobile && (
          <span style={{ fontSize: 14, fontWeight: 700, letterSpacing: '-0.02em', color: t.textPrimary }}>
            oneiros
          </span>
        )}
      </div>

      {!isMobile && <Divider />}

      {/* Project name */}
      {!isMobile && (editing ? (
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitEdit}
          onKeyDown={handleKeyDown}
          style={{
            background: t.bgElevated,
            border: `1px solid ${t.accent}`,
            borderRadius: 5,
            color: t.textPrimary,
            fontSize: 13,
            padding: '3px 8px',
            outline: 'none',
            width: 180,
          }}
        />
      ) : (
        <button
          onClick={startEdit}
          style={{
            background: 'transparent',
            border: 'none',
            color: t.textBody,
            fontSize: 13,
            cursor: 'text',
            padding: '3px 6px',
            borderRadius: 5,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
          title="Click to rename project"
        >
          {name}
          {isDirty && (
            <span
              style={{
                width: 5,
                height: 5,
                borderRadius: '50%',
                background: t.accent,
                display: 'inline-block',
                flexShrink: 0,
              }}
            />
          )}
        </button>
      ))}

      {!isMobile && <Divider />}

      {/* View switcher */}
      <div style={{ display: 'flex', gap: 2, background: t.bgElevated, border: `1px solid ${t.borderDefault}`, borderRadius: 6, padding: 2, flexShrink: 0 }}>
        <ViewTab active={view === 'model'} onClick={() => onViewChange('model')} label={isMobile ? 'Model' : 'Model'} compact={isMobile} />
        <ViewTab active={view === 'dataset'} onClick={() => onViewChange('dataset')} label={isMobile ? 'Data' : 'Dataset'} compact={isMobile} />
      </div>

      {isMobile && view === 'model' && (
        <>
          <IconButton onClick={onTogglePalette} title="Node palette" icon={<NodesIcon />} active={paletteOpen} />
          <IconButton onClick={onToggleInspector} title="Inspector" icon={<InspectorIcon />} active={inspectorOpen} />
        </>
      )}

      {!isMobile && (
        <>
      <Divider />

      {/* Undo / Redo */}
      <div style={{ display: 'flex', gap: 2 }}>
        <IconButton
          onClick={undo}
          disabled={!canUndo}
          title="Undo (⌘Z)"
          icon={<UndoIcon />}
        />
        <IconButton
          onClick={redo}
          disabled={!canRedo}
          title="Redo (⌘⇧Z)"
          icon={<RedoIcon />}
        />
      </div>
        </>
      )}

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Validation summary */}
      {!isMobile && (errorCount > 0 || warnCount > 0) && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {errorCount > 0 && (
            <ValidationPill count={errorCount} type="error" />
          )}
          {warnCount > 0 && (
            <ValidationPill count={warnCount} type="warning" />
          )}
        </div>
      )}

      <Divider />

      {/* File + view actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: isMobile ? 4 : 6, flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: isMobile ? 4 : 6, overflowX: 'auto', minWidth: 0 }}>
        {!isMobile && <TopbarButton onClick={loadFromFile} icon={<FolderOpenIcon />} label="Load" />}
        {!isMobile && <TopbarButton onClick={saveToFile} icon={<SaveIcon />} label="Save" accent />}
        {view === 'model' && (
          <>
            {!isMobile && <Divider />}
            <TopbarButton
              onClick={onToggleTrain}
              icon={<PlayIcon />}
              label="Train"
              active={trainOpen}
              accent={trainOpen}
              iconOnly={isMobile}
              title="Train"
            />
          </>
        )}
        </div>
        {!isMobile && <Divider />}
        <ThemeSwitcher />
        {!isMobile && <Divider />}
        <TopbarButton
          onClick={onToggleAI}
          icon={<SparkleIcon />}
          label="AI"
          active={aiOpen}
          accent={aiOpen}
          iconOnly={isMobile}
          title="AI"
        />
      </div>
    </header>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Divider() {
  return <div style={{ width: 1, height: 20, background: t.borderDefault, flexShrink: 0 }} />
}

function ViewTab({ active, onClick, label, compact }: { active: boolean; onClick: () => void; label: string; compact?: boolean }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: compact ? '4px 8px' : '4px 10px',
        borderRadius: 4,
        border: 'none',
        background: active ? t.borderDefault : 'transparent',
        color: active ? t.textPrimary : t.textMuted,
        fontSize: compact ? 11 : 12,
        fontWeight: 500,
        cursor: 'pointer',
        transition: 'background 0.1s, color 0.1s',
        whiteSpace: 'nowrap',
      }}
      onMouseEnter={(e) => { if (!active) e.currentTarget.style.color = t.textSecondary }}
      onMouseLeave={(e) => { if (!active) e.currentTarget.style.color = t.textMuted }}
    >
      {label}
    </button>
  )
}

function ValidationPill({ count, type }: { count: number; type: 'error' | 'warning' }) {
  const isError = type === 'error'
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 4,
        padding: '2px 7px',
        borderRadius: 4,
        background: isError ? 'rgba(239,68,68,0.12)' : 'rgba(245,158,11,0.12)',
        border: `1px solid ${isError ? '#ef444430' : '#f59e0b30'}`,
        fontSize: 11,
        fontWeight: 600,
        color: isError ? '#f87171' : '#fbbf24',
      }}
    >
      <span>{isError ? '✕' : '△'}</span>
      {count}
    </div>
  )
}

function IconButton({
  onClick,
  disabled,
  title,
  icon,
  active,
}: {
  onClick?: () => void
  disabled?: boolean
  title?: string
  icon: React.ReactNode
  active?: boolean
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        width: 32,
        height: 32,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 6,
        border: active ? '1px solid #7c3aed' : '1px solid transparent',
        background: active ? '#7c3aed1a' : 'transparent',
        color: disabled ? t.textDisabled : active ? '#c4b5fd' : t.textMuted,
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'background 0.1s, color 0.1s',
        flexShrink: 0,
      }}
      onMouseEnter={(e) => {
        if (!disabled) {
          e.currentTarget.style.background = t.borderDefault
          e.currentTarget.style.color = t.textPrimary
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent'
        e.currentTarget.style.color = disabled ? t.textDisabled : t.textMuted
      }}
    >
      {icon}
    </button>
  )
}

interface TopbarButtonProps {
  onClick?: () => void
  icon: React.ReactNode
  label: string
  accent?: boolean
  active?: boolean
  disabled?: boolean
  title?: string
  iconOnly?: boolean
}

function TopbarButton({ onClick, icon, label, accent, active, disabled, title, iconOnly }: TopbarButtonProps) {
  const bg = accent ? '#7c3aed1a' : active ? t.borderDefault : 'transparent'
  const color = disabled ? t.textDisabled : accent ? t.accentMuted : active ? t.textPrimary : t.textSecondary
  const border = accent ? '1px solid #7c3aed' : active ? '1px solid #3f3f46' : `1px solid ${t.borderDefault}`

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title ?? (iconOnly ? label || undefined : undefined)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: iconOnly ? 0 : 6,
        padding: iconOnly ? '6px 8px' : '5px 10px',
        borderRadius: 6,
        border,
        background: bg,
        color,
        fontSize: 12,
        fontWeight: 500,
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'background 0.12s, color 0.12s',
        flexShrink: 0,
      }}
      onMouseEnter={(e) => {
        if (!disabled) {
          e.currentTarget.style.background = accent ? '#7c3aed33' : t.borderDefault
          e.currentTarget.style.color = accent ? '#c4b5fd' : t.textPrimary
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = bg
        e.currentTarget.style.color = color
      }}
    >
      {icon}
      {!iconOnly && label}
    </button>
  )
}
