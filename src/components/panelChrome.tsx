import { t } from '../theme/tokens'
import type { CSSProperties } from 'react'

export const COLLAPSED_PANEL_WIDTH = 28
export const MOBILE_DRAWER_WIDTH = 'min(320px, 88vw)'

export function MobileDrawerBackdrop({ onClose }: { onClose: () => void }) {
  return (
    <div
      role="presentation"
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: t.overlay,
        zIndex: 40,
      }}
    />
  )
}

export function mobileDrawerStyle(side: 'left' | 'right'): CSSProperties {
  return {
    position: 'fixed',
    top: 52,
    bottom: 0,
    ...(side === 'left' ? { left: 0 } : { right: 0 }),
    width: MOBILE_DRAWER_WIDTH,
    maxWidth: '100vw',
    zIndex: 50,
    background: t.bgPanel,
    borderRight: side === 'left' ? `1px solid ${t.borderSubtle}` : undefined,
    borderLeft: side === 'right' ? `1px solid ${t.borderSubtle}` : undefined,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    boxShadow: side === 'left' ? t.shadowDrawer : '-4px 0 24px rgba(0,0,0,0.4)',
  }
}

export function MobileDrawerHeader({ title, onClose }: { title: string; onClose: () => void }) {
  return (
    <div style={{
      padding: '10px 12px',
      borderBottom: `1px solid ${t.borderSubtle}`,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexShrink: 0,
    }}>
      <span style={{ fontSize: 12, fontWeight: 600, color: t.textPrimary }}>{title}</span>
      <button
        type="button"
        onClick={onClose}
        aria-label="Close panel"
        style={{
          width: 32,
          height: 32,
          borderRadius: 6,
          border: `1px solid ${t.borderDefault}`,
          background: t.bgElevated,
          color: t.textSecondary,
          fontSize: 18,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        ×
      </button>
    </div>
  )
}

export function CollapseBtn({ side, onClick, title }: { side: 'left' | 'right'; onClick: () => void; title: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      style={{
        flexShrink: 0,
        width: 22,
        height: 22,
        borderRadius: 5,
        border: `1px solid ${t.borderDefault}`,
        background: t.bgElevated,
        color: t.textMuted,
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 0,
        transition: 'color 0.12s, border-color 0.12s, background 0.12s',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.color = t.textPrimary
        e.currentTarget.style.borderColor = t.accent
        e.currentTarget.style.background = t.bgHover
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.color = t.textMuted
        e.currentTarget.style.borderColor = t.borderDefault
        e.currentTarget.style.background = t.bgElevated
      }}
    >
      <ChevronIcon direction={side === 'left' ? 'left' : 'right'} />
    </button>
  )
}

export function CollapsedBar({ side, label, onToggle }: { side: 'left' | 'right'; label: string; onToggle: () => void }) {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '10px 0', gap: 10 }}>
      <button
        type="button"
        onClick={onToggle}
        title={`Expand ${label.toLowerCase()}`}
        style={{
          width: 22,
          height: 22,
          borderRadius: 5,
          border: `1px solid ${t.borderDefault}`,
          background: t.bgElevated,
          color: t.textMuted,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 0,
          flexShrink: 0,
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = t.textPrimary
          e.currentTarget.style.borderColor = t.accent
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = t.textMuted
          e.currentTarget.style.borderColor = t.borderDefault
        }}
      >
        <ChevronIcon direction={side === 'left' ? 'right' : 'left'} />
      </button>
      <span
        style={{
          writingMode: 'vertical-rl',
          textOrientation: 'mixed',
          transform: side === 'left' ? 'rotate(180deg)' : undefined,
          fontSize: 9,
          fontWeight: 600,
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          color: t.textDisabled,
          userSelect: 'none',
        }}
      >
        {label}
      </span>
    </div>
  )
}

function ChevronIcon({ direction }: { direction: 'left' | 'right' }) {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      {direction === 'left'
        ? <polyline points="15 18 9 12 15 6" />
        : <polyline points="9 18 15 12 9 6" />}
    </svg>
  )
}

export function LoadingSpinner({ size = 14, color = t.accentMuted }: { size?: number; color?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2.5"
      style={{ animation: 'oneiros-spin 0.8s linear infinite', flexShrink: 0 }}
    >
      <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round" />
    </svg>
  )
}

export function LoadingLabel({ label }: { label: string }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 10, color: t.textMuted }}>
      <LoadingSpinner size={12} />
      {label}
    </span>
  )
}

export function PanelBusyOverlay({ label }: { label: string }) {
  return (
    <div style={{
      position: 'absolute',
      inset: 0,
      background: t.overlay,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 25,
      backdropFilter: 'blur(2px)',
    }}>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 10,
        padding: '16px 20px',
        borderRadius: 10,
        background: t.bgElevated,
        border: `1px solid ${t.borderDefault}`,
      }}>
        <LoadingSpinner size={22} />
        <span style={{ fontSize: 12, color: t.textBody, fontWeight: 500 }}>{label}</span>
      </div>
    </div>
  )
}
