import { useRef, useState, useEffect, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { useThemeStore, type ThemeId } from '../store/useThemeStore'
import { t } from '../theme/tokens'

const THEMES: { id: ThemeId; label: string }[] = [
  { id: 'dark', label: 'Dark' },
  { id: 'light', label: 'Light' },
  { id: 'midnight', label: 'Midnight' },
  { id: 'gruvbox', label: 'Gruvbox' },
  { id: 'nord', label: 'Nord' },
  { id: 'noire', label: 'Black Noire' },
  { id: 'system', label: 'System' },
]

function ThemeIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}

export default function ThemeSwitcher() {
  const themeId = useThemeStore((s) => s.themeId)
  const resolvedTheme = useThemeStore((s) => s.resolvedTheme)
  const setTheme = useThemeStore((s) => s.setTheme)
  const [open, setOpen] = useState(false)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const [menuPos, setMenuPos] = useState({ top: 0, right: 0 })

  const updateMenuPos = useCallback(() => {
    const btn = buttonRef.current
    if (!btn) return
    const rect = btn.getBoundingClientRect()
    setMenuPos({
      top: rect.bottom + 6,
      right: window.innerWidth - rect.right,
    })
  }, [])

  useEffect(() => {
    if (!open) return
    updateMenuPos()
    window.addEventListener('resize', updateMenuPos)
    window.addEventListener('scroll', updateMenuPos, true)
    return () => {
      window.removeEventListener('resize', updateMenuPos)
      window.removeEventListener('scroll', updateMenuPos, true)
    }
  }, [open, updateMenuPos])

  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      const target = e.target as Node
      if (buttonRef.current?.contains(target)) return
      if (menuRef.current?.contains(target)) return
      setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [open])

  void resolvedTheme

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        onClick={() => setOpen((v) => !v)}
        title="Theme"
        aria-label="Change theme"
        aria-expanded={open}
        aria-haspopup="listbox"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 32,
          height: 32,
          borderRadius: 6,
          border: `1px solid ${t.borderDefault}`,
          background: open ? t.bgHover : 'transparent',
          color: t.textSecondary,
          cursor: 'pointer',
          flexShrink: 0,
        }}
      >
        <ThemeIcon />
      </button>

      {open && createPortal(
        <div
          ref={menuRef}
          role="listbox"
          aria-label="Theme options"
          style={{
            position: 'fixed',
            top: menuPos.top,
            right: menuPos.right,
            minWidth: 140,
            maxHeight: 320,
            overflowY: 'auto',
            background: t.bgElevated,
            border: `1px solid ${t.borderDefault}`,
            borderRadius: 8,
            boxShadow: t.shadowPanel,
            zIndex: 9999,
            overflow: 'hidden',
          }}
        >
          {THEMES.map((item) => {
            const active = themeId === item.id
            return (
              <button
                key={item.id}
                type="button"
                role="option"
                aria-selected={active}
                onClick={() => { setTheme(item.id); setOpen(false) }}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 8,
                  padding: '8px 12px',
                  border: 'none',
                  background: active ? t.accentSubtle : 'transparent',
                  color: active ? t.accentMuted : t.textBody,
                  fontSize: 12,
                  fontWeight: active ? 600 : 400,
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                {item.label}
                {active && <CheckIcon />}
              </button>
            )
          })}
        </div>,
        document.body,
      )}
    </>
  )
}
