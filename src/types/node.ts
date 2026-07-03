import type { ComponentType } from 'react'
import type { NodeProps } from '@xyflow/react'
import type { NodeData } from './graph'

export type NodeCategory =
  | 'input'
  | 'output'
  | 'layers'
  | 'activation'
  | 'recurrent'
  | 'attention'
  | 'utility'
  | 'regularization'
  | 'transformer'

export type PaletteGroup =
  | 'basic'
  | 'convolutional'
  | 'recurrent'
  | 'transformer'
  | 'regularization'
  | 'activation'
  | 'utility'
  | 'computerVision'

export interface NodeFieldOption {
  value: string
  label: string
}

export interface NodeField {
  key: string
  label: string
  type: 'number' | 'text' | 'select' | 'boolean'
  options?: NodeFieldOption[]
  min?: number
  max?: number
  step?: number
  placeholder?: string
}

export interface NodeDefinition {
  type: string
  label: string
  description: string
  category: NodeCategory
  paletteGroup: PaletteGroup
  defaultData: NodeData
  fields: NodeField[]
  /** Optional palette-only shortcut (same type, different defaultData) */
  paletteDefaults?: Partial<NodeData>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  component: ComponentType<NodeProps<any>>
}

export interface PaletteShortcut {
  key: string
  type: string
  label: string
  description: string
  paletteGroup: PaletteGroup
  paletteDefaults?: Partial<NodeData>
}

export interface PaletteItem {
  key: string
  type: string
  label: string
  description: string
  paletteGroup: PaletteGroup
}
