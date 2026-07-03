import type { ComponentType } from 'react'
import type { NodeTypes } from '@xyflow/react'
import type { NodeDefinition, NodeCategory, PaletteItem, PaletteShortcut } from '../../types/node'
import type { NodeData } from '../../types/graph'

const registry = new Map<string, NodeDefinition>()
const paletteDrop = new Map<string, { type: string; defaultData: Partial<NodeData> }>()
const shortcuts: PaletteShortcut[] = []

export function registerNode(definition: NodeDefinition): void {
  registry.set(definition.type, definition)
  paletteDrop.set(definition.type, {
    type: definition.type,
    defaultData: definition.paletteDefaults ?? {},
  })
}

export function registerPaletteShortcut(shortcut: PaletteShortcut): void {
  shortcuts.push(shortcut)
  paletteDrop.set(shortcut.key, {
    type: shortcut.type,
    defaultData: shortcut.paletteDefaults ?? {},
  })
}

export function resolvePaletteDrop(key: string): { type: string; defaultData: Partial<NodeData> } | null {
  const entry = paletteDrop.get(key)
  if (entry) return entry
  if (registry.has(key)) return { type: key, defaultData: {} }
  return null
}

export function getNodeDefinition(type: string): NodeDefinition | undefined {
  return registry.get(type)
}

export function getAllNodes(): NodeDefinition[] {
  return [...registry.values()]
}

export function getNodesByCategory(category: NodeCategory): NodeDefinition[] {
  return [...registry.values()].filter((def) => def.category === category)
}

export function getPaletteItems(): PaletteItem[] {
  const items: PaletteItem[] = [...registry.values()].map((def) => ({
    key: def.type,
    type: def.type,
    label: def.label,
    description: def.description,
    paletteGroup: def.paletteGroup,
  }))
  for (const s of shortcuts) {
    items.push({
      key: s.key,
      type: s.type,
      label: s.label,
      description: s.description,
      paletteGroup: s.paletteGroup,
    })
  }
  return items
}

export function getXYFlowNodeTypes(): NodeTypes {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return Object.fromEntries(
    [...registry.entries()].map(([type, def]) => [type, def.component as ComponentType<any>])
  ) as NodeTypes
}
