import { create } from 'zustand'
import Papa from 'papaparse'
import { downloadCSV } from '../editor/dataset/mergeCsv'
import {
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from '@xyflow/react'
import type { NodeChange, EdgeChange, Connection } from '@xyflow/react'
import type { AppNode, AppEdge } from '../types/graph'
import type { CustomDatasetPayload, CVDataset } from '../types/training'

const API_BASE = 'http://localhost:8000'

// ── Column metadata ──────────────────────────────────────────────────────────

export type ColumnType = 'number' | 'string' | 'boolean'

export interface ColumnInfo {
  name: string
  type: ColumnType
  nullCount: number
  uniqueCount: number
  min?: number
  max?: number
  mean?: number
  topValues?: string[]
}

// ── Tabular dataset ──────────────────────────────────────────────────────────

export interface LoadedDataset {
  id: string
  name: string
  rows: Record<string, unknown>[]
  columns: ColumnInfo[]
}

// ── EDF / biosignal dataset ──────────────────────────────────────────────────

export interface EDFEvent {
  id: number
  label: string
  count: number
}

export interface EDFDataset {
  sessionId: string
  name: string
  channels: string[]
  sfreq: number
  duration: number        // seconds
  nTimes: number
  events: EDFEvent[]
  previewData: number[][] // (n_channels, preview_samples)
  previewSfreq: number    // sampling rate of previewData
}

// ── EDF pipeline node (visual canvas) ────────────────────────────────────────

function makeEDFSourceNode(name: string): AppNode {
  return {
    id: 'edf-source-1',
    type: 'edfSourceNode',
    position: { x: 60, y: 120 },
    data: { label: 'EDF Source', datasetName: name },
  }
}

// ── Tabular helpers ──────────────────────────────────────────────────────────

function detectType(values: unknown[]): ColumnType {
  const nonNull = values.filter((v) => v !== null && v !== undefined && v !== '')
  if (nonNull.length === 0) return 'string'
  if (nonNull.every((v) => typeof v === 'boolean')) return 'boolean'
  if (nonNull.every((v) => typeof v === 'number' && !isNaN(v as number))) return 'number'
  return 'string'
}

function computeColumn(name: string, values: unknown[]): ColumnInfo {
  const nonNull = values.filter((v) => v !== null && v !== undefined && v !== '')
  const type = detectType(values)
  const nullCount = values.length - nonNull.length
  const uniqueCount = new Set(nonNull.map(String)).size

  if (type === 'number') {
    const nums = nonNull as number[]
    const min = Math.min(...nums)
    const max = Math.max(...nums)
    const mean = nums.reduce((a, b) => a + b, 0) / nums.length
    return { name, type, nullCount, uniqueCount, min, max, mean }
  }

  if (type === 'string') {
    const freq = new Map<string, number>()
    for (const v of nonNull) {
      const s = String(v)
      freq.set(s, (freq.get(s) ?? 0) + 1)
    }
    const topValues = [...freq.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([k]) => k)
    return { name, type, nullCount, uniqueCount, topValues }
  }

  return { name, type, nullCount, uniqueCount }
}

function buildColumns(rows: Record<string, unknown>[]): ColumnInfo[] {
  if (rows.length === 0) return []
  const keys = Object.keys(rows[0])
  return keys.map((key) => computeColumn(key, rows.map((r) => r[key])))
}

function makeSourceNode(datasetName: string): AppNode {
  return {
    id: 'source-1',
    type: 'datasetSource',
    position: { x: 60, y: 120 },
    data: { label: 'Source', datasetName },
  }
}

// ── Store ────────────────────────────────────────────────────────────────────

interface DatasetState {
  // Tabular (CSV / JSON)
  dataset: LoadedDataset | null
  targetColumn: string | null
  pipelineNodes: AppNode[]
  pipelineEdges: AppEdge[]

  // EDF / MNE
  edfDataset: EDFDataset | null
  edfLoading: boolean
  edfError: string | null
  edfPipelineNodes: AppNode[]
  edfPipelineEdges: AppEdge[]
  edfProcessing: boolean
  edfProcessResult: CustomDatasetPayload | null
  edfProcessError: string | null

  // Computer Vision
  cvDataset: CVDataset | null
  cvLoading: boolean
  cvError: string | null
  augPipelineNodes: AppNode[]
  augPipelineEdges: AppEdge[]

  // Tabular actions
  loadFromCSV: (file: File) => Promise<void>
  loadFromJSON: (file: File) => Promise<void>
  loadMergedDataset: (name: string, rows: Record<string, unknown>[]) => void
  exportDatasetAsCSV: () => void
  clearDataset: () => void
  setTargetColumn: (col: string | null) => void
  onPipelineNodesChange: (changes: NodeChange<AppNode>[]) => void
  onPipelineEdgesChange: (changes: EdgeChange<AppEdge>[]) => void
  onPipelineConnect: (connection: Connection) => void
  addPipelineNode: (node: AppNode) => void

  // EDF actions
  loadFromEDF: (file: File) => Promise<void>
  clearEDF: () => void
  processEDFPipeline: (steps: Record<string, unknown>[]) => Promise<CustomDatasetPayload | null>
  onEDFPipelineNodesChange: (changes: NodeChange<AppNode>[]) => void
  onEDFPipelineEdgesChange: (changes: EdgeChange<AppEdge>[]) => void
  onEDFPipelineConnect: (connection: Connection) => void
  addEDFPipelineNode: (node: AppNode) => void

  // CV actions
  loadFromImageZip: (file: File) => Promise<void>
  clearCVDataset: () => void
  onAugPipelineNodesChange: (changes: NodeChange<AppNode>[]) => void
  onAugPipelineEdgesChange: (changes: EdgeChange<AppEdge>[]) => void
  onAugPipelineConnect: (connection: Connection) => void
  addAugPipelineNode: (node: AppNode) => void
}

export const useDatasetStore = create<DatasetState>((set, get) => ({
  dataset: null,
  targetColumn: null,
  pipelineNodes: [],
  pipelineEdges: [],

  edfDataset: null,
  edfLoading: false,
  edfError: null,
  edfPipelineNodes: [],
  edfPipelineEdges: [],
  edfProcessing: false,
  edfProcessResult: null,
  edfProcessError: null,

  cvDataset: null,
  cvLoading: false,
  cvError: null,
  augPipelineNodes: [],
  augPipelineEdges: [],

  // ── Tabular ────────────────────────────────────────────────────────────────

  async loadFromCSV(file) {
    return new Promise((resolve) => {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (result: Papa.ParseResult<Record<string, unknown>>) => {
          const rows = result.data
          const columns = buildColumns(rows)
          const dataset: LoadedDataset = {
            id: `ds-${Date.now()}`,
            name: file.name.replace(/\.csv$/i, ''),
            rows,
            columns,
          }
          const lastCol = columns[columns.length - 1]?.name ?? null
          set({
            dataset,
            targetColumn: lastCol,
            pipelineNodes: [makeSourceNode(dataset.name)],
            pipelineEdges: [],
          })
          resolve()
        },
      })
    })
  },

  async loadFromJSON(file) {
    const text = await file.text()
    try {
      const parsed = JSON.parse(text)
      let rows: Record<string, unknown>[]
      if (Array.isArray(parsed)) {
        rows = parsed as Record<string, unknown>[]
      } else if (typeof parsed === 'object' && parsed !== null) {
        const keys = Object.keys(parsed)
        const len = (parsed[keys[0]] as unknown[]).length
        rows = Array.from({ length: len }, (_, i) =>
          Object.fromEntries(keys.map((k) => [k, (parsed[k] as unknown[])[i]]))
        )
      } else {
        return
      }
      const columns = buildColumns(rows)
      const dataset: LoadedDataset = {
        id: `ds-${Date.now()}`,
        name: file.name.replace(/\.json$/i, ''),
        rows,
        columns,
      }
      const lastCol = columns[columns.length - 1]?.name ?? null
      set({
        dataset,
        targetColumn: lastCol,
        pipelineNodes: [makeSourceNode(dataset.name)],
        pipelineEdges: [],
      })
    } catch {
      console.error('Failed to parse JSON dataset')
    }
  },

  loadMergedDataset(name, rows) {
    const columns = buildColumns(rows)
    const dataset: LoadedDataset = {
      id: `ds-${Date.now()}`,
      name,
      rows,
      columns,
    }
    const lastCol = columns[columns.length - 1]?.name ?? null
    set({
      dataset,
      targetColumn: lastCol,
      pipelineNodes: [makeSourceNode(dataset.name)],
      pipelineEdges: [],
    })
  },

  exportDatasetAsCSV() {
    const { dataset } = get()
    if (!dataset || dataset.rows.length === 0) return
    downloadCSV(dataset.rows, dataset.name)
  },

  clearDataset() {
    set({ dataset: null, targetColumn: null, pipelineNodes: [], pipelineEdges: [] })
  },

  setTargetColumn(col) {
    set({ targetColumn: col })
  },

  onPipelineNodesChange(changes) {
    set({ pipelineNodes: applyNodeChanges(changes, get().pipelineNodes) })
  },

  onPipelineEdgesChange(changes) {
    set({ pipelineEdges: applyEdgeChanges(changes, get().pipelineEdges) })
  },

  onPipelineConnect(connection) {
    set({ pipelineEdges: addEdge({ ...connection, animated: false }, get().pipelineEdges) })
  },

  addPipelineNode(node) {
    set({ pipelineNodes: [...get().pipelineNodes, node] })
  },

  // ── EDF ────────────────────────────────────────────────────────────────────

  async loadFromEDF(file) {
    set({ edfLoading: true, edfError: null, edfDataset: null, edfProcessResult: null })
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_BASE}/api/edf/load`, { method: 'POST', body: formData })
      const json = await res.json() as Record<string, unknown>
      if (!res.ok || json.error) {
        set({ edfLoading: false, edfError: String(json.error ?? 'Upload failed') })
        return
      }
      const edfDataset = json as unknown as EDFDataset
      set({
        edfDataset,
        edfLoading: false,
        edfError: null,
        edfPipelineNodes: [makeEDFSourceNode(edfDataset.name)],
        edfPipelineEdges: [],
      })
    } catch (err) {
      set({ edfLoading: false, edfError: err instanceof Error ? err.message : 'Upload failed' })
    }
  },

  clearEDF() {
    set({ edfDataset: null, edfError: null, edfPipelineNodes: [], edfPipelineEdges: [], edfProcessResult: null })
  },

  async processEDFPipeline(steps) {
    const { edfDataset } = get()
    if (!edfDataset) return null
    set({ edfProcessing: true, edfProcessError: null })
    try {
      const res = await fetch(`${API_BASE}/api/edf/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: edfDataset.sessionId, steps }),
      })
      const json = await res.json() as Record<string, unknown>
      if (!res.ok || json.error) {
        set({ edfProcessing: false, edfProcessError: String(json.error ?? 'Processing failed') })
        return null
      }
      const payload = json as unknown as CustomDatasetPayload
      set({ edfProcessing: false, edfProcessResult: payload })
      return payload
    } catch (err) {
      set({ edfProcessing: false, edfProcessError: err instanceof Error ? err.message : 'Processing failed' })
      return null
    }
  },

  onEDFPipelineNodesChange(changes) {
    set({ edfPipelineNodes: applyNodeChanges(changes, get().edfPipelineNodes) })
  },

  onEDFPipelineEdgesChange(changes) {
    set({ edfPipelineEdges: applyEdgeChanges(changes, get().edfPipelineEdges) })
  },

  onEDFPipelineConnect(connection) {
    set({ edfPipelineEdges: addEdge({ ...connection, animated: false }, get().edfPipelineEdges) })
  },

  addEDFPipelineNode(node) {
    set({ edfPipelineNodes: [...get().edfPipelineNodes, node] })
  },

  // ── CV ─────────────────────────────────────────────────────────────────────

  async loadFromImageZip(file) {
    set({ cvLoading: true, cvError: null, cvDataset: null })
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_BASE}/api/cv/load`, { method: 'POST', body: formData })
      const json = await res.json() as Record<string, unknown>
      if (!res.ok || json.error) {
        set({ cvLoading: false, cvError: String(json.error ?? 'Upload failed') })
        return
      }
      const cvDataset = json as unknown as CVDataset
      // Seed augmentation pipeline with a source node
      const sourceNode: AppNode = {
        id: 'aug-source',
        type: 'augSource',
        position: { x: 80, y: 60 },
        data: {
          label: 'Image Dataset',
          name: cvDataset.name,
          totalImages: cvDataset.totalImages,
          classCount: cvDataset.classNames.length,
          inputShape: cvDataset.inputShape,
        },
      }
      set({
        cvDataset,
        cvLoading: false,
        cvError: null,
        augPipelineNodes: [sourceNode],
        augPipelineEdges: [],
      })
    } catch (err) {
      set({ cvLoading: false, cvError: err instanceof Error ? err.message : 'Upload failed' })
    }
  },

  clearCVDataset() {
    set({ cvDataset: null, cvError: null, augPipelineNodes: [], augPipelineEdges: [] })
  },

  onAugPipelineNodesChange(changes) {
    set({ augPipelineNodes: applyNodeChanges(changes, get().augPipelineNodes) })
  },

  onAugPipelineEdgesChange(changes) {
    set({ augPipelineEdges: applyEdgeChanges(changes, get().augPipelineEdges) })
  },

  onAugPipelineConnect(connection) {
    set({ augPipelineEdges: addEdge({ ...connection, animated: false }, get().augPipelineEdges) })
  },

  addAugPipelineNode(node) {
    set({ augPipelineNodes: [...get().augPipelineNodes, node] })
  },
}))
