import { create } from 'zustand'
import { v4 as uuid } from 'uuid'
import { useProjectStore } from './useProjectStore'
import { useGraphStore } from './useGraphStore'
import type { AppNode } from '../types/graph'

const API_BASE = 'http://localhost:8000'
const TOKEN_KEY = 'oneiros-hf-token'

export interface ImportedHFModel {
  id: string
  modelId: string
  outputFeatures: number
  modelKind: string
  configClass?: string
  pipelineTag?: string | null
  trustRemoteCode: boolean
  pooling: 'mean' | 'cls' | 'pooler'
  freeze: boolean
  importedAt: number
}

export interface HFBackendStatus {
  ready: boolean
  transformers: boolean
  huggingfaceHub: boolean
  hint?: string | null
}

export type HFLoadState = 'idle' | 'checking' | 'searching' | 'validating' | 'importing'

function libraryKey(projectId: string | null): string {
  return projectId ? `oneiros-hf-models-${projectId}` : 'oneiros-hf-models-global'
}

function loadLibrary(projectId: string | null): ImportedHFModel[] {
  try {
    const raw = localStorage.getItem(libraryKey(projectId))
    if (raw) return JSON.parse(raw) as ImportedHFModel[]
  } catch { /* ignore */ }
  return []
}

function persistLibrary(projectId: string | null, models: ImportedHFModel[]): void {
  try {
    localStorage.setItem(libraryKey(projectId), JSON.stringify(models))
  } catch { /* quota */ }
}

function loadToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? ''
  } catch {
    return ''
  }
}

function persistToken(token: string): void {
  try {
    if (token.trim()) localStorage.setItem(TOKEN_KEY, token.trim())
    else localStorage.removeItem(TOKEN_KEY)
  } catch { /* ignore */ }
}

async function parseJsonResponse<T>(res: Response): Promise<T & { error?: string; hint?: string }> {
  let body: T & { error?: string; hint?: string }
  try {
    body = await res.json()
  } catch {
    throw new Error(`Server returned ${res.status} with non-JSON body. Is the backend running?`)
  }
  if (!res.ok) {
    const msg = body.error ?? `Request failed (${res.status})`
    const hint = body.hint ? `\n${body.hint}` : ''
    throw new Error(`${msg}${hint}`)
  }
  return body
}

interface HuggingFaceState {
  backendStatus: HFBackendStatus | null
  backendError: string | null
  hfToken: string
  importedModels: ImportedHFModel[]
  loadState: HFLoadState
  lastError: string | null
  searchQuery: string
  searchResults: Array<{
    id: string
    pipelineTag?: string | null
    downloads?: number | null
    gated?: boolean
  }>

  setHfToken: (token: string) => void
  refreshBackendStatus: () => Promise<void>
  loadLibraryForProject: (projectId: string | null) => void
  searchModels: (query: string) => Promise<void>
  validateAndImport: (modelId: string, opts?: { trustRemoteCode?: boolean; pooling?: ImportedHFModel['pooling']; freeze?: boolean }) => Promise<ImportedHFModel>
  removeImported: (id: string) => void
  addToCanvas: (model: ImportedHFModel, onSwitchToModel?: () => void) => void
  clearError: () => void
}

export const useHuggingFaceStore = create<HuggingFaceState>((set, get) => ({
  backendStatus: null,
  backendError: null,
  hfToken: loadToken(),
  importedModels: [],
  loadState: 'idle',
  lastError: null,
  searchQuery: '',
  searchResults: [],

  setHfToken(token) {
    persistToken(token)
    set({ hfToken: token })
  },

  loadLibraryForProject(projectId) {
    set({ importedModels: loadLibrary(projectId), lastError: null })
  },

  async refreshBackendStatus() {
    set({ loadState: 'checking', backendError: null })
    try {
      const res = await fetch(`${API_BASE}/api/hf/status`)
      const body = await parseJsonResponse<HFBackendStatus>(res)
      set({
        backendStatus: body,
        backendError: body.ready ? null : (body.hint ?? 'Hugging Face dependencies are not installed on the backend.'),
        loadState: 'idle',
      })
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      set({
        backendStatus: null,
        backendError: msg.includes('fetch') || msg.includes('Failed to fetch')
          ? 'Cannot reach the backend at localhost:8000. Start it with: uvicorn main:app --reload --port 8000'
          : msg,
        loadState: 'idle',
      })
    }
  },

  async searchModels(query) {
    const q = query.trim()
    set({ searchQuery: q, loadState: 'searching', lastError: null })
    if (!q) {
      set({ searchResults: [], loadState: 'idle' })
      return
    }
    try {
      const res = await fetch(`${API_BASE}/api/hf/search?q=${encodeURIComponent(q)}&limit=24`)
      const body = await parseJsonResponse<{ models?: HuggingFaceState['searchResults'] }>(res)
      set({ searchResults: body.models ?? [], loadState: 'idle' })
    } catch (err) {
      set({
        lastError: err instanceof Error ? err.message : String(err),
        searchResults: [],
        loadState: 'idle',
      })
    }
  },

  async validateAndImport(modelId, opts = {}) {
    const trimmed = modelId.trim()
    if (!trimmed) throw new Error('Enter a Hugging Face model ID (e.g. distilbert-base-uncased).')

    set({ loadState: 'validating', lastError: null })
    const { hfToken } = get()
    const trustRemoteCode = opts.trustRemoteCode ?? false
    const pooling = opts.pooling ?? 'mean'
    const freeze = opts.freeze ?? false

    let validateBody: {
      valid?: boolean
      modelId?: string
      outputFeatures?: number
      modelKind?: string
      configClass?: string
      needsTrustRemoteCode?: boolean
      pipelineTag?: string | null
    }

    try {
      const res = await fetch(`${API_BASE}/api/hf/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelId: trimmed,
          trustRemoteCode,
          token: hfToken.trim() || undefined,
        }),
      })
      validateBody = await parseJsonResponse(res)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      set({ loadState: 'idle', lastError: msg })
      throw err
    }

    if (validateBody.needsTrustRemoteCode && !trustRemoteCode) {
      const err = new Error('This model requires “Trust remote code”. Enable it and validate again.')
      set({ loadState: 'idle', lastError: err.message })
      throw err
    }

    set({ loadState: 'importing' })

    let info: { pipelineTag?: string | null } = {}
    try {
      const infoRes = await fetch(`${API_BASE}/api/hf/model-info?model_id=${encodeURIComponent(trimmed)}`)
      if (infoRes.ok) info = await infoRes.json()
    } catch { /* optional metadata */ }

    const entry: ImportedHFModel = {
      id: uuid(),
      modelId: validateBody.modelId ?? trimmed,
      outputFeatures: validateBody.outputFeatures ?? 768,
      modelKind: validateBody.modelKind ?? 'generic',
      configClass: validateBody.configClass,
      pipelineTag: info.pipelineTag ?? null,
      trustRemoteCode,
      pooling,
      freeze,
      importedAt: Date.now(),
    }

    const projectId = useProjectStore.getState().currentProjectId
    const existing = get().importedModels
    const withoutDup = existing.filter((m) => m.modelId !== entry.modelId)
    const next = [entry, ...withoutDup]
    persistLibrary(projectId, next)
    set({ importedModels: next, loadState: 'idle', lastError: null })
    return entry
  },

  removeImported(id) {
    const projectId = useProjectStore.getState().currentProjectId
    const next = get().importedModels.filter((m) => m.id !== id)
    persistLibrary(projectId, next)
    set({ importedModels: next })
  },

  addToCanvas(model, onSwitchToModel) {
    const node: AppNode = {
      id: uuid(),
      type: 'hfModelNode',
      position: { x: 280 + Math.random() * 80, y: 180 + Math.random() * 80 },
      data: {
        label: model.modelId.split('/').pop() ?? 'HF Model',
        modelId: model.modelId,
        outputFeatures: model.outputFeatures,
        modelKind: model.modelKind,
        pooling: model.pooling,
        freeze: model.freeze,
        trustRemoteCode: model.trustRemoteCode,
      },
    }
    useGraphStore.getState().addNode(node)
    onSwitchToModel?.()
  },

  clearError() {
    set({ lastError: null })
  },
}))
