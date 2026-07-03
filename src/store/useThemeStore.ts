import { create } from 'zustand'

export const THEME_IDS = [
  'dark',
  'light',
  'midnight',
  'gruvbox',
  'nord',
  'noire',
  'system',
] as const

export type ThemeId = (typeof THEME_IDS)[number]
export type ResolvedTheme = Exclude<ThemeId, 'system'>

const STORAGE_KEY = 'oneiros-theme'

function isThemeId(value: string | null): value is ThemeId {
  return value !== null && (THEME_IDS as readonly string[]).includes(value)
}

function getSystemTheme(): 'dark' | 'light' {
  if (typeof window === 'undefined') return 'dark'
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function resolveTheme(themeId: ThemeId): ResolvedTheme {
  if (themeId === 'system') {
    return getSystemTheme()
  }
  return themeId
}

function applyThemeToDom(themeId: ThemeId): ResolvedTheme {
  const resolved = resolveTheme(themeId)
  document.documentElement.dataset.theme = resolved
  document.documentElement.style.colorScheme =
    resolved === 'light' ? 'light' : 'dark'
  return resolved
}

let systemListener: ((e: MediaQueryListEvent) => void) | null = null

function setupSystemListener(onChange: () => void): void {
  if (systemListener) {
    window.matchMedia('(prefers-color-scheme: dark)').removeEventListener('change', systemListener)
  }
  systemListener = () => onChange()
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', systemListener)
}

function removeSystemListener(): void {
  if (systemListener) {
    window.matchMedia('(prefers-color-scheme: dark)').removeEventListener('change', systemListener)
    systemListener = null
  }
}

interface ThemeState {
  themeId: ThemeId
  resolvedTheme: ResolvedTheme
  setTheme: (id: ThemeId) => void
  initTheme: () => void
}

export const useThemeStore = create<ThemeState>((set, get) => ({
  themeId: 'dark',
  resolvedTheme: 'dark',

  setTheme: (id: ThemeId) => {
    try {
      localStorage.setItem(STORAGE_KEY, id)
    } catch {
      // ignore storage errors
    }
    const resolved = applyThemeToDom(id)
    if (id === 'system') {
      setupSystemListener(() => {
        const current = get().themeId
        if (current === 'system') {
          set({ resolvedTheme: applyThemeToDom('system') })
        }
      })
    } else {
      removeSystemListener()
    }
    set({ themeId: id, resolvedTheme: resolved })
  },

  initTheme: () => {
    let saved: ThemeId = 'dark'
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (isThemeId(raw)) {
        saved = raw
      }
    } catch {
      // ignore
    }
    const resolved = applyThemeToDom(saved)
    if (saved === 'system') {
      setupSystemListener(() => {
        if (get().themeId === 'system') {
          set({ resolvedTheme: applyThemeToDom('system') })
        }
      })
    }
    set({ themeId: saved, resolvedTheme: resolved })
  },
}))
