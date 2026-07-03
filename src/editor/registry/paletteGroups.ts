import type { PaletteGroup } from '../../types/node'

export interface PaletteGroupConfig {
  id: PaletteGroup
  label: string
  accent: string
  defaultCollapsed: boolean
}

export const PALETTE_GROUPS: PaletteGroupConfig[] = [
  { id: 'basic', label: 'Basic', accent: '#3b82f6', defaultCollapsed: false },
  { id: 'convolutional', label: 'Convolutional', accent: '#8b5cf6', defaultCollapsed: false },
  { id: 'recurrent', label: 'Recurrent', accent: '#ec4899', defaultCollapsed: false },
  { id: 'transformer', label: 'Transformer', accent: '#06b6d4', defaultCollapsed: false },
  { id: 'regularization', label: 'Regularization', accent: '#f97316', defaultCollapsed: false },
  { id: 'activation', label: 'Activation', accent: '#f59e0b', defaultCollapsed: false },
  { id: 'utility', label: 'Utility', accent: '#64748b', defaultCollapsed: false },
  { id: 'computerVision', label: 'Computer Vision', accent: '#14b8a6', defaultCollapsed: true },
]

const STORAGE_KEY = 'oneiros-palette-collapsed'

export function loadCollapsedGroups(): Set<PaletteGroup> {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return new Set(PALETTE_GROUPS.filter((g) => g.defaultCollapsed).map((g) => g.id))
    }
    return new Set(JSON.parse(raw) as PaletteGroup[])
  } catch {
    return new Set(PALETTE_GROUPS.filter((g) => g.defaultCollapsed).map((g) => g.id))
  }
}

export function saveCollapsedGroups(collapsed: Set<PaletteGroup>): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify([...collapsed]))
  } catch {
    // ignore storage errors
  }
}
