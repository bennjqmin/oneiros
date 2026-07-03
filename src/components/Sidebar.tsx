import { t } from '../theme/tokens'
import { useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { getPaletteItems } from '../editor/registry/nodeRegistry'
import '../editor/nodes/index'
import type { PaletteGroup } from '../types/node'
import { PALETTE_GROUPS, loadCollapsedGroups, saveCollapsedGroups } from '../editor/registry/paletteGroups'
import { useThemeSync } from '../theme/useThemeSync'
import { COLLAPSED_PANEL_WIDTH, CollapseBtn, CollapsedBar, MobileDrawerBackdrop, MobileDrawerHeader, mobileDrawerStyle } from './panelChrome'

interface SidebarProps {
  mobile?: boolean
  open?: boolean
  onClose?: () => void
}

function SearchIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  )
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      width="10"
      height="10"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ transform: open ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.15s' }}
    >
      <polyline points="9 18 15 12 9 6" />
    </svg>
  )
}

const COLLAPSED_WIDTH = COLLAPSED_PANEL_WIDTH
const MIN_WIDTH = 160
const MAX_WIDTH = 420

export default function Sidebar({ mobile, open, onClose }: SidebarProps = {}) {
  useThemeSync()
  const [query, setQuery] = useState('')
  const [width, setWidth] = useState(220)
  const [collapsed, setCollapsed] = useState(false)
  const [collapsedGroups, setCollapsedGroups] = useState<Set<PaletteGroup>>(() => loadCollapsedGroups())
  const drag = useRef({ active: false, startX: 0, startW: 0 })
  const allItems = getPaletteItems()
  const panelWidth = collapsed ? COLLAPSED_WIDTH : width

  if (mobile && !open) return null

  const filtered = query.trim()
    ? allItems.filter(
        (n) =>
          n.label.toLowerCase().includes(query.toLowerCase()) ||
          n.description.toLowerCase().includes(query.toLowerCase())
      )
    : allItems

  const grouped = new Map<PaletteGroup, typeof filtered>()
  for (const item of filtered) {
    const list = grouped.get(item.paletteGroup) ?? []
    list.push(item)
    grouped.set(item.paletteGroup, list)
  }

  function toggleGroup(id: PaletteGroup) {
    setCollapsedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      saveCollapsedGroups(next)
      return next
    })
  }

  function onDragStart(e: DragEvent<HTMLDivElement>, key: string) {
    e.dataTransfer.setData('application/oneiros-palette', key)
    e.dataTransfer.effectAllowed = 'copy'
  }

  function onResizeMouseDown(e: React.MouseEvent) {
    if (collapsed) return
    e.preventDefault()
    drag.current = { active: true, startX: e.clientX, startW: width }
    const onMove = (ev: MouseEvent) => {
      if (!drag.current.active) return
      const next = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, drag.current.startW + ev.clientX - drag.current.startX))
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

  const paletteBody = (
    <>
      <div style={{ padding: '12px 12px 8px', borderBottom: `1px solid ${t.borderSubtle}`, display: 'flex', alignItems: 'center', gap: 6 }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 7, flex: 1, minWidth: 0,
          background: t.bgElevated, border: `1px solid ${t.borderDefault}`,
          borderRadius: 7, padding: '6px 10px',
        }}>
          <span style={{ color: t.textFaint }}><SearchIcon /></span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search nodes…"
            style={{ background: 'transparent', border: 'none', outline: 'none', color: t.textBody, fontSize: 12, flex: 1, minWidth: 0 }}
          />
        </div>
        {!mobile && <CollapseBtn side="left" onClick={() => setCollapsed(true)} title="Collapse palette" />}
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0' }}>
        {PALETTE_GROUPS.map((group) => {
          const items = grouped.get(group.id)
          if (!items || items.length === 0) return null
          const isCollapsed = collapsedGroups.has(group.id)
          return (
            <div key={group.id} style={{ marginBottom: 2 }}>
              <button
                type="button"
                onClick={() => toggleGroup(group.id)}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '6px 14px 4px',
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                <span style={{ color: t.textFaint, display: 'flex' }}><ChevronIcon open={!isCollapsed} /></span>
                <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: t.textFaint }}>
                  {group.label}
                </span>
                <span style={{ marginLeft: 'auto', fontSize: 10, color: t.textDisabled }}>{items.length}</span>
              </button>
              {!isCollapsed && items.map((node) => (
                <PaletteItem
                  key={node.key}
                  label={node.label}
                  description={node.description}
                  color={group.accent}
                  onDragStart={(e) => onDragStart(e, node.key)}
                />
              ))}
            </div>
          )
        })}
        {filtered.length === 0 && (
          <div style={{ padding: '24px 14px', textAlign: 'center', color: t.textDisabled, fontSize: 12 }}>
            No nodes match "{query}"
          </div>
        )}
      </div>

      <div style={{ padding: '8px 14px', borderTop: `1px solid ${t.borderSubtle}`, fontSize: 11, color: t.textDisabled, lineHeight: 1.5 }}>
        {mobile ? 'Drag nodes onto the canvas (or long-press to add)' : 'Drag nodes onto the canvas'}
      </div>
    </>
  )

  if (mobile) {
    return (
      <>
        <MobileDrawerBackdrop onClose={onClose!} />
        <aside style={{ ...mobileDrawerStyle('left'), display: 'flex', flexDirection: 'column' }}>
          <MobileDrawerHeader title="Nodes" onClose={onClose!} />
          {paletteBody}
        </aside>
      </>
    )
  }

  return (
    <aside
      style={{
        width: panelWidth,
        minWidth: panelWidth,
        background: t.bgPanel,
        borderRight: `1px solid ${t.borderSubtle}`,
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
          side="left"
          label="Nodes"
          onToggle={() => setCollapsed(false)}
        />
      ) : (
        <>
      {paletteBody}
      <div
        onMouseDown={onResizeMouseDown}
        title="Drag to resize"
        style={{
          position: 'absolute', top: 0, right: 0, width: 4, height: '100%',
          cursor: 'ew-resize', zIndex: 10,
          background: 'transparent', transition: 'background 0.15s',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.background = '#7c3aed66' }}
        onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
      />
        </>
      )}
    </aside>
  )
}

interface PaletteItemProps {
  label: string
  description: string
  color: string
  onDragStart: (e: DragEvent<HTMLDivElement>) => void
}

function PaletteItem({ label, description, color, onDragStart }: PaletteItemProps) {
  return (
    <div
      draggable
      onDragStart={onDragStart}
      style={{
        margin: '2px 8px', padding: '8px 10px', borderRadius: 7,
        border: '1px solid transparent', cursor: 'grab',
        display: 'flex', alignItems: 'center', gap: 9,
        transition: 'background 0.1s, border-color 0.1s', userSelect: 'none',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = t.bgElevated; e.currentTarget.style.borderColor = t.borderDefault }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = 'transparent' }}
    >
      <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 500, color: t.textBody, lineHeight: 1.3 }}>{label}</div>
        <div style={{ fontSize: 10, color: t.textFaint, lineHeight: 1.3, marginTop: 1 }}>{description}</div>
      </div>
    </div>
  )
}
