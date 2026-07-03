import { t } from '../theme/tokens'
import { useState } from 'react'
import { useProjectStore } from '../store/useProjectStore'
import type { ProjectEntry } from '../store/useProjectStore'
import { AppLogo } from '../components/AppLogo'
import ThemeSwitcher from '../components/ThemeSwitcher'
import { useThemeSync } from '../theme/useThemeSync'

// ── Helpers ───────────────────────────────────────────────────────────────────

function relativeTime(ms: number): string {
  const diff = Date.now() - ms
  const mins = Math.floor(diff / 60_000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  if (days < 30) return `${days}d ago`
  return new Date(ms).toLocaleDateString()
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function ProjectsPage() {
  useThemeSync()

  const allProjects = useProjectStore((s) => s.allProjects)
  const createProject = useProjectStore((s) => s.createProject)
  const openProject = useProjectStore((s) => s.openProject)

  const sorted = [...allProjects].sort((a, b) => b.updatedAt - a.updatedAt)

  return (
    <div style={{
      minHeight: '100vh',
      background: t.bgBase,
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Header */}
      <header style={{
        height: 56,
        borderBottom: `1px solid ${t.borderSubtle}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        padding: '0 32px',
        gap: 10,
        flexShrink: 0,
      }}>
        <ThemeSwitcher />
        <button
          onClick={() => createProject()}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '7px 16px', borderRadius: 7,
            border: `1px solid ${t.accent}`,
            background: 'rgba(124,58,237,0.12)',
            color: t.accentMuted, fontSize: 13, fontWeight: 600, cursor: 'pointer',
            transition: 'background 0.12s',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(124,58,237,0.24)' }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(124,58,237,0.12)' }}
        >
          <PlusIcon /> New Project
        </button>
      </header>

      {/* Brand hero */}
      <BrandHero />

      {/* Body */}
      <main style={{ flex: 1, padding: '0 32px 48px', maxWidth: 1200, width: '100%', margin: '0 auto' }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: t.textPrimary, margin: '0 0 8px', letterSpacing: '-0.02em' }}>
          Your Projects
        </h1>
        <p style={{ fontSize: 13, color: t.textFaint, margin: '0 0 32px' }}>
          {allProjects.length === 0 ? 'No projects yet — create your first one.' : `${allProjects.length} project${allProjects.length !== 1 ? 's' : ''}`}
        </p>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
          gap: 16,
        }}>
          {/* New project card */}
          <NewProjectCard onCreate={createProject} />

          {/* Existing project cards */}
          {sorted.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onOpen={() => openProject(project.id)}
            />
          ))}
        </div>
      </main>
    </div>
  )
}

// ── Brand hero ────────────────────────────────────────────────────────────────

const FEATURE_PILLS = ['Neural nets', 'XGBoost', 'Datasets', 'Live training']

function BrandHero() {
  return (
    <section style={{
      position: 'relative',
      overflow: 'hidden',
      borderBottom: '1px solid #18181b',
      padding: '44px 32px 36px',
    }}>
      <div className="projects-hero-glow" aria-hidden />
      <div className="projects-hero-grid" aria-hidden />

      <div style={{
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
        maxWidth: 560,
        margin: '0 auto',
      }}>
        <AppLogo size={84} />
        <h1
          className="oneiros-wordmark"
          style={{ fontSize: 'clamp(1.75rem, 4vw, 2.25rem)', margin: '18px 0 10px', lineHeight: 1.1 }}
        >
          oneiros
        </h1>
        <p style={{ fontSize: 14, color: t.textSecondary, margin: 0, letterSpacing: '0.02em' }}>
          Visual machine learning IDE
        </p>
        <p style={{ fontSize: 12, color: t.textFaint, margin: '10px 0 0', lineHeight: 1.6, maxWidth: 400 }}>
          Build models on a node canvas, preprocess datasets, and train — PyTorch, XGBoost, and more, all in the browser.
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 18 }}>
          {FEATURE_PILLS.map((label) => (
            <span
              key={label}
              style={{
                fontSize: 10,
                fontWeight: 600,
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                color: t.textMuted,
                background: 'rgba(24,24,27,0.8)',
                border: `1px solid ${t.borderDefault}`,
                borderRadius: 999,
                padding: '4px 10px',
              }}
            >
              {label}
            </span>
          ))}
        </div>
      </div>
    </section>
  )
}

// ── New project card ──────────────────────────────────────────────────────────

function NewProjectCard({ onCreate }: { onCreate: (name?: string) => void }) {
  const [naming, setNaming] = useState(false)
  const [name, setName] = useState('')

  function handleCreate() {
    onCreate(name.trim() || 'Untitled Project')
    setNaming(false)
    setName('')
  }

  if (naming) {
    return (
      <div style={cardBase}>
        <p style={{ fontSize: 12, color: t.textMuted, margin: '0 0 10px' }}>Project name</p>
        <input
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleCreate()
            if (e.key === 'Escape') { setNaming(false); setName('') }
          }}
          placeholder="Untitled Project"
          style={{
            width: '100%', background: t.borderDefault, border: '1px solid #3f3f46',
            borderRadius: 6, color: t.textPrimary, fontSize: 13,
            padding: '7px 10px', outline: 'none', boxSizing: 'border-box', marginBottom: 10,
          }}
        />
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={handleCreate} style={primaryBtn}>Create</button>
          <button onClick={() => { setNaming(false); setName('') }} style={ghostBtn}>Cancel</button>
        </div>
      </div>
    )
  }

  return (
    <button
      onClick={() => setNaming(true)}
      style={{
        ...cardBase,
        border: '2px dashed #27272a',
        background: 'transparent',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 10,
        minHeight: 160,
        transition: 'border-color 0.15s, background 0.15s',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = t.accent
        e.currentTarget.style.background = 'rgba(124,58,237,0.04)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = t.borderDefault
        e.currentTarget.style.background = 'transparent'
      }}
    >
      <div style={{
        width: 36, height: 36, borderRadius: '50%',
        background: 'rgba(124,58,237,0.1)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: t.accent,
      }}>
        <PlusIcon size={18} />
      </div>
      <span style={{ fontSize: 13, fontWeight: 600, color: t.textFaint }}>New Project</span>
    </button>
  )
}

// ── Project card ──────────────────────────────────────────────────────────────

function ProjectCard({ project, onOpen }: { project: ProjectEntry; onOpen: () => void }) {
  const deleteProject = useProjectStore((s) => s.deleteProject)
  const duplicateProject = useProjectStore((s) => s.duplicateProject)
  const renameProject = useProjectStore((s) => s.renameProject)

  const [confirmDelete, setConfirmDelete] = useState(false)
  const [renaming, setRenaming] = useState(false)
  const [nameInput, setNameInput] = useState(project.name)

  const nodeCount = project.nodes.length
  const edgeCount = project.edges.length

  function handleRename() {
    const trimmed = nameInput.trim()
    if (trimmed) renameProject(project.id, trimmed)
    setRenaming(false)
  }

  return (
    <div style={{
      ...cardBase,
      display: 'flex',
      flexDirection: 'column',
      gap: 0,
    }}>
      {/* Card canvas preview */}
      <div style={{
        height: 80,
        background: t.bgPanel,
        borderRadius: '8px 8px 0 0',
        marginBottom: 14,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        overflow: 'hidden',
        flexShrink: 0,
      }}>
        {nodeCount === 0 ? (
          <span style={{ fontSize: 11, color: t.textDisabled }}>Empty graph</span>
        ) : (
          <MiniGraphPreview nodes={project.nodes} edges={project.edges} />
        )}
      </div>

      {/* Name */}
      {renaming ? (
        <input
          autoFocus
          value={nameInput}
          onChange={(e) => setNameInput(e.target.value)}
          onBlur={handleRename}
          onKeyDown={(e) => { if (e.key === 'Enter') handleRename(); if (e.key === 'Escape') setRenaming(false) }}
          style={{
            background: t.borderDefault, border: '1px solid #3f3f46', borderRadius: 5,
            color: t.textPrimary, fontSize: 14, fontWeight: 600, padding: '3px 7px',
            outline: 'none', width: '100%', boxSizing: 'border-box', marginBottom: 6,
          }}
        />
      ) : (
        <div
          onDoubleClick={() => { setNameInput(project.name); setRenaming(true) }}
          style={{ fontSize: 14, fontWeight: 600, color: t.textPrimary, marginBottom: 5, cursor: 'default' }}
          title="Double-click to rename"
        >
          {project.name}
        </div>
      )}

      {/* Meta */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
        <MetaBadge>{nodeCount} node{nodeCount !== 1 ? 's' : ''}</MetaBadge>
        <MetaBadge>{edgeCount} edge{edgeCount !== 1 ? 's' : ''}</MetaBadge>
        <span style={{ fontSize: 10, color: t.textDisabled, marginLeft: 'auto' }}>
          {relativeTime(project.updatedAt)}
        </span>
      </div>

      {/* Actions */}
      {confirmDelete ? (
        <div style={{ display: 'flex', gap: 6, marginTop: 'auto' }}>
          <span style={{ fontSize: 11, color: '#f87171', flex: 1 }}>Delete?</span>
          <button onClick={() => deleteProject(project.id)} style={{ ...dangerBtn, padding: '3px 10px', fontSize: 11 }}>Yes</button>
          <button onClick={() => setConfirmDelete(false)} style={{ ...ghostBtn, padding: '3px 10px', fontSize: 11 }}>No</button>
        </div>
      ) : (
        <div style={{ display: 'flex', gap: 6, marginTop: 'auto' }}>
          <button onClick={onOpen} style={{ ...primaryBtn, flex: 1 }}>Open</button>
          <IconActionBtn onClick={() => duplicateProject(project.id)} title="Duplicate"><CopyIcon /></IconActionBtn>
          <IconActionBtn onClick={() => setConfirmDelete(true)} title="Delete" danger><TrashIcon /></IconActionBtn>
        </div>
      )}
    </div>
  )
}

// ── Mini graph preview ────────────────────────────────────────────────────────

function MiniGraphPreview({ nodes }: { nodes: { type?: string }[]; edges?: unknown[] }) {
  const typeColors: Record<string, string> = {
    inputNode: '#0ea5e9',
    denseNode: '#8b5cf6',
    outputNode: t.success,
    conv2dNode: t.warning,
    dropoutNode: '#6366f1',
    batchNormNode: '#ec4899',
    maxPool2dNode: '#f97316',
    flattenNode: '#84cc16',
  }
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '0 12px', flexWrap: 'wrap', justifyContent: 'center' }}>
      {nodes.slice(0, 8).map((n, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <div style={{
            width: 10, height: 10, borderRadius: 3,
            background: typeColors[n.type ?? ''] ?? t.textDisabled,
          }} />
          {i < Math.min(nodes.length, 8) - 1 && (
            <div style={{ width: 10, height: 1, background: t.borderDefault }} />
          )}
        </div>
      ))}
      {nodes.length > 8 && <span style={{ fontSize: 9, color: t.textFaint }}>+{nodes.length - 8}</span>}
    </div>
  )
}

// ── Shared styles ─────────────────────────────────────────────────────────────

const cardBase: React.CSSProperties = {
  background: t.bgPanel,
  border: `1px solid ${t.borderSubtle}`,
  borderRadius: 10,
  padding: 14,
  transition: 'border-color 0.15s, box-shadow 0.15s',
}

const primaryBtn: React.CSSProperties = {
  padding: '5px 14px', borderRadius: 6,
  border: `1px solid ${t.accent}`,
  background: 'rgba(124,58,237,0.12)',
  color: t.accentMuted, fontSize: 12, fontWeight: 600, cursor: 'pointer',
}

const ghostBtn: React.CSSProperties = {
  padding: '5px 12px', borderRadius: 6,
  border: `1px solid ${t.borderDefault}`,
  background: 'transparent',
  color: t.textMuted, fontSize: 12, cursor: 'pointer',
}

const dangerBtn: React.CSSProperties = {
  borderRadius: 6, border: '1px solid #7f1d1d',
  background: 'rgba(239,68,68,0.08)',
  color: '#f87171', cursor: 'pointer',
}

function MetaBadge({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      fontSize: 10, color: t.textFaint,
      background: t.bgElevated, border: `1px solid ${t.borderDefault}`,
      borderRadius: 4, padding: '1px 6px',
    }}>
      {children}
    </span>
  )
}

function IconActionBtn({ onClick, title, danger, children }: { onClick: () => void; title: string; danger?: boolean; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      title={title}
      style={{
        width: 28, height: 28, borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center',
        border: `1px solid ${danger ? t.borderDefault : t.borderDefault}`,
        background: 'transparent',
        color: t.textFaint, cursor: 'pointer',
        transition: 'color 0.1s, border-color 0.1s',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.color = danger ? '#f87171' : t.textPrimary
        if (danger) e.currentTarget.style.borderColor = '#7f1d1d'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.color = t.textFaint
        e.currentTarget.style.borderColor = t.borderDefault
      }}
    >
      {children}
    </button>
  )
}

// ── Icons ─────────────────────────────────────────────────────────────────────

function PlusIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  )
}

function CopyIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  )
}

function TrashIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    </svg>
  )
}
