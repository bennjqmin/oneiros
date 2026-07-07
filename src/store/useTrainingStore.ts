import { create } from 'zustand'
import type {
  TrainingStatus,
  TrainingConfig,
  EpochMetrics,
  TrainingMessage,
  CustomDatasetPayload,
} from '../types/training'
import type { ValidationIssue } from '../types/validation'
import { DEFAULT_CONFIG } from '../types/training'
import { TrainingSocket } from '../services/trainingSocket'
import { useGraphStore } from './useGraphStore'
import { useDatasetStore } from './useDatasetStore'
import { executePipeline } from '../editor/dataset/pipelineExecutor'
import {
  pingTrainingBackend,
  safeStringifyTrainingPayload,
  validateProcessedDataset,
} from '../editor/dataset/trainingPayloadLimits'
import { validateGraph } from '../editor/validation/validateGraph'
import { validateTabularTraining } from '../editor/validation/validateTabularTraining'
import { deferWork } from '../utils/deferWork'

const API_BASE = 'http://localhost:8000'

interface TrainingState {
  status: TrainingStatus
  statusMessage: string
  config: TrainingConfig
  runId: string | null

  epochMetrics: EpochMetrics[]
  currentEpoch: number
  totalEpochs: number
  currentBatch: number
  totalBatches: number
  currentLoss: number | null
  etaSecs: number | null
  errorMessage: string | null

  // Custom dataset info (set when using CSV data)
  customDatasetInfo: CustomDatasetPayload | null

  preflightIssues: ValidationIssue[]

  setConfig: (patch: Partial<TrainingConfig>) => void
  startTraining: () => Promise<void>
  stopTraining: () => Promise<void>

  // Model export (NN)
  exportWeights: () => void
  exportONNX: () => void
  exportFull: () => void

  // EDF export
  setCustomDatasetFromEDF: (payload: CustomDatasetPayload) => void

  // CV
  cvDatasetRef: { sessionId: string; augmentSteps: unknown[] } | null
  setCVDataset: (ref: { sessionId: string; augmentSteps: unknown[] }) => void
  confusionMatrix: number[][] | null

  // XGBoost
  xgbStatus: 'idle' | 'running' | 'complete' | 'error'
  xgbStatusMessage: string
  xgbResult: XGBResult | null
  xgbError: string | null
  xgbPreflightIssues: ValidationIssue[]
  trainXGBoost: () => Promise<void>
  exportXGB: () => void

  _socket: TrainingSocket | null
  _handleMessage: (msg: TrainingMessage) => void
}

export interface XGBResult {
  task: 'classification' | 'regression'
  // Classification
  trainAccuracy?: number
  valAccuracy?: number
  nClasses?: number
  // Regression
  trainRMSE?: number
  valRMSE?: number
  trainMAE?: number
  valMAE?: number
  trainR2?: number
  valR2?: number
  objective?: string
  valScatter?: { actual: number; predicted: number }[]
  // Common
  bestIteration: number
  nEstimators: number
  featureImportance: { name: string; importance: number }[]
  evals: {
    round: number
    trainLoss: number
    valLoss: number
    trainAccuracy?: number
    valAccuracy?: number
    trainRMSE?: number
    valRMSE?: number
    trainMAE?: number
    valMAE?: number
  }[]
  runId: string
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  status: 'idle',
  statusMessage: '',
  config: { ...DEFAULT_CONFIG },
  runId: null,

  epochMetrics: [],
  currentEpoch: 0,
  totalEpochs: 0,
  currentBatch: 0,
  totalBatches: 0,
  currentLoss: null,
  etaSecs: null,
  errorMessage: null,
  customDatasetInfo: null,
  cvDatasetRef: null,
  confusionMatrix: null,

  preflightIssues: [],

  _socket: null,

  setCVDataset(ref) {
    set({ cvDatasetRef: ref })
  },

  setCustomDatasetFromEDF(payload) {
    set({ customDatasetInfo: payload })
    // Auto-select custom dataset mode so training uses this data
    const { config } = get()
    if (config.dataset !== 'custom') {
      set({ config: { ...config, dataset: 'custom' } })
    }
  },

  xgbStatus: 'idle',
  xgbStatusMessage: '',
  xgbResult: null,
  xgbError: null,
  xgbPreflightIssues: [],

  setConfig(patch) {
    set((s) => ({ config: { ...s.config, ...patch } }))
  },

  async startTraining() {
    const { config, _handleMessage } = get()

    set({
      status: 'connecting',
      statusMessage: 'Preparing dataset…',
      epochMetrics: [],
      currentEpoch: 0,
      totalBatches: 0,
      currentBatch: 0,
      currentLoss: null,
      etaSecs: null,
      errorMessage: null,
      preflightIssues: [],
    })

    await deferWork()

    const { nodes, edges } = useGraphStore.getState().exportGraph()

    // Build custom dataset payload if CSV mode is selected
    let customDataset: CustomDatasetPayload | null = null
    if (config.dataset === 'custom') {
      const dsState = useDatasetStore.getState()
      if (!dsState.dataset || !dsState.targetColumn) {
        set({ status: 'error', errorMessage: 'Load a dataset and select a target column first.' })
        return
      }
      const result = executePipeline(
        dsState.dataset,
        dsState.targetColumn,
        dsState.pipelineNodes,
        dsState.pipelineEdges,
      )
      if (!result.ok) {
        set({ status: 'error', errorMessage: result.error })
        return
      }
      customDataset = result.data
    }

    // For image_folder mode, grab the cvDatasetRef from store
    const cvDataset = config.dataset === 'image_folder' ? get().cvDatasetRef : null
    if (config.dataset === 'image_folder' && !cvDataset) {
      set({ status: 'error', errorMessage: 'No image dataset loaded. Import a zip of class folders in the Dataset tab first.' })
      return
    }

    set({ statusMessage: 'Validating model…' })
    await deferWork()

    // Pre-flight graph validation
    const dsForValidation = useDatasetStore.getState().cvDataset
    const opts = config.dataset === 'custom' && customDataset
      ? { customFeatureCount: customDataset.featureCount, customClassCount: customDataset.classCount }
      : config.dataset === 'image_folder' && dsForValidation
      ? { cvInputShape: dsForValidation.inputShape, cvClassCount: dsForValidation.classNames.length }
      : {}
    const { issues, isValid } = validateGraph(nodes, edges, opts)
    set({ preflightIssues: issues })
    if (!isValid) {
      const errorCount = issues.filter(i => i.severity === 'error').length
      set({ status: 'error', errorMessage: `Fix ${errorCount} error${errorCount > 1 ? 's' : ''} in your graph before training.` })
      return
    }

    set({
      status: 'connecting',
      statusMessage: 'Connecting to backend…',
      customDatasetInfo: customDataset,
    })

    try {
      const res = await fetch(`${API_BASE}/api/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph: { nodes, edges }, config, customDataset, cvDataset }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const { runId } = await res.json() as { runId: string }

      set({ runId })

      const socket = new TrainingSocket()
      set({ _socket: socket })
      socket.connect(`${API_BASE.replace('http', 'ws')}/ws/training/${runId}`, _handleMessage)

    } catch (err) {
      set({
        status: 'error',
        errorMessage: err instanceof Error ? err.message : 'Connection failed',
        statusMessage: '',
      })
    }
  },

  exportWeights() {
    const { runId } = get()
    if (!runId) return
    window.open(`${API_BASE}/api/export/${runId}/weights?filename=model.pt`, '_blank')
  },

  exportONNX() {
    const { runId } = get()
    if (!runId) return
    window.open(`${API_BASE}/api/export/${runId}/onnx?filename=model.onnx`, '_blank')
  },

  exportFull() {
    const { runId } = get()
    if (!runId) return
    window.open(`${API_BASE}/api/export/${runId}/full?filename=model_full.pt`, '_blank')
  },

  async trainXGBoost() {
    const { config } = get()

    const fail = (stage: string, error: string, hint?: string) => {
      console.error(`[Oneiros XGB] Failed at "${stage}":`, error, hint ?? '')
      const msg = hint ? `${error} — ${hint}` : error
      set({
        xgbStatus: 'error',
        xgbStatusMessage: '',
        xgbError: msg,
      })
    }

    set({
      xgbStatus: 'running',
      xgbStatusMessage: 'Checking backend…',
      xgbError: null,
      xgbResult: null,
      xgbPreflightIssues: [],
    })

    await deferWork()

    const backend = await pingTrainingBackend(API_BASE)
    if (!backend.ok) {
      fail('backend', backend.error, backend.hint)
      return
    }

    set({ xgbStatusMessage: 'Validating dataset…' })
    await deferWork()

    const dsState = useDatasetStore.getState()

    const { issues, isValid } = validateTabularTraining(
      dsState.dataset,
      dsState.targetColumn,
      dsState.pipelineNodes,
      dsState.pipelineEdges,
      {
        xgbTask: config.xgbTask,
        xgbNEstimators: config.xgbNEstimators,
        xgbEarlyStoppingRounds: config.xgbEarlyStoppingRounds,
        xgbObjective: config.xgbObjective,
      },
    )
    set({ xgbPreflightIssues: issues })

    if (!isValid) {
      const errorCount = issues.filter((i) => i.severity === 'error').length
      fail(
        'validation',
        `Fix ${errorCount} error${errorCount > 1 ? 's' : ''} before training XGBoost.`,
        issues.find((i) => i.severity === 'error')?.hint,
      )
      return
    }

    set({ xgbStatusMessage: 'Running pipeline on full dataset…' })
    await deferWork()

    let result
    try {
      result = executePipeline(
        dsState.dataset!,
        dsState.targetColumn!,
        dsState.pipelineNodes,
        dsState.pipelineEdges,
        { task: config.xgbTask },
      )
    } catch (err) {
      fail(
        'pipeline',
        err instanceof Error ? err.message : 'Pipeline threw an unexpected error.',
        'Try fewer rows or simpler pipeline steps.',
      )
      return
    }

    if (!result.ok) {
      fail('pipeline', result.error, 'Review Dataset → Pipeline nodes and try again.')
      return
    }

    set({ xgbStatusMessage: 'Checking payload size…' })
    await deferWork()

    const payloadCheck = validateProcessedDataset(result.data)
    if (!payloadCheck.ok) {
      fail('payload', payloadCheck.error, payloadCheck.hint)
      return
    }

    console.info(
      `[Oneiros XGB] Payload OK: ${payloadCheck.rows.toLocaleString()} rows, ` +
      `${payloadCheck.features} features, ~${Math.round(payloadCheck.estimatedJsonBytes / 1024)} KB`,
    )

    set({ xgbStatusMessage: 'Serializing training data…' })
    await deferWork()

    const trainConfig = {
      task:                 config.xgbTask,
      objective:            config.xgbObjective || undefined,
      nEstimators:          config.xgbNEstimators,
      maxDepth:             config.xgbMaxDepth,
      learningRate:         config.xgbLearningRate,
      subsample:            config.xgbSubsample,
      colsampleBytree:      config.xgbColsampleBytree,
      minChildWeight:       config.xgbMinChildWeight,
      gamma:                config.xgbGamma,
      regAlpha:             config.xgbRegAlpha,
      regLambda:            config.xgbRegLambda,
      earlyStoppingRounds:  config.xgbEarlyStoppingRounds,
    }

    const serialized = safeStringifyTrainingPayload(result.data, trainConfig)
    if (!serialized.ok) {
      fail('serialize', serialized.error, serialized.hint)
      return
    }

    set({ xgbStatusMessage: 'Training XGBoost on server…' })
    await deferWork()

    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 15 * 60 * 1000)

      const res = await fetch(`${API_BASE}/api/xgboost/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: serialized.body,
        signal: controller.signal,
      })
      clearTimeout(timeout)

      let json: { error?: string; hint?: string } & Partial<XGBResult>
      try {
        json = await res.json() as typeof json
      } catch {
        fail(
          'response',
          `Server returned non-JSON (HTTP ${res.status}).`,
          'The backend may have crashed. Restart with npm run dev:backend and retry.',
        )
        return
      }

      if (!res.ok) {
        const msg = [json.error, json.hint].filter(Boolean).join(' — ')
        fail('training', msg || `HTTP ${res.status}`, undefined)
        return
      }

      console.info('[Oneiros XGB] Training complete')
      set({ xgbStatus: 'complete', xgbStatusMessage: '', xgbResult: json as XGBResult })
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        fail('training', 'Training timed out after 15 minutes.', 'Lower n_estimators or use fewer rows.')
        return
      }
      fail(
        'network',
        err instanceof Error ? err.message : 'Request failed',
        'Check that the backend is still running on port 8000.',
      )
    }
  },

  exportXGB() {
    const { xgbResult } = get()
    if (!xgbResult?.runId) return
    window.open(`${API_BASE}/api/xgboost/${xgbResult.runId}/export`, '_blank')
  },

  async stopTraining() {
    const { runId, _socket } = get()

    // Always close the socket immediately so the UI transitions out of running
    _socket?.close()
    set({ status: 'stopped', statusMessage: 'Training stopped', _socket: null })

    if (!runId) {
      // Clicked Stop before runId was returned — nothing more to do
      return
    }
    try {
      await fetch(`${API_BASE}/api/train/stop/${runId}`, { method: 'POST' })
    } catch {
      // Network error is fine — the stop event fires on WebSocketDisconnect too
    }
  },

  _handleMessage(msg) {
    const type = msg.type as string

    if (type === 'status') {
      set({ status: 'running', statusMessage: msg.message as string })
    } else if (type === 'epochStart') {
      set({
        status: 'running',
        currentEpoch: msg.epoch as number,
        totalEpochs: msg.totalEpochs as number,
        currentBatch: 0,
        statusMessage: `Epoch ${msg.epoch} / ${msg.totalEpochs}`,
      })
    } else if (type === 'batch') {
      set({
        currentBatch: msg.batch as number,
        totalBatches: msg.totalBatches as number,
        currentLoss: msg.loss as number,
      })
    } else if (type === 'epochEnd') {
      const entry: EpochMetrics = {
        epoch: msg.epoch as number,
        trainLoss: msg.trainLoss as number,
        valLoss: msg.valLoss as number,
        valAccuracy: msg.valAccuracy as number,
        top5Accuracy: msg.top5Accuracy as number | undefined,
        currentLR: msg.currentLR as number | undefined,
      }
      const accStr = `${((msg.valAccuracy as number) * 100).toFixed(1)}%`
      const top5Str = entry.top5Accuracy != null ? ` · top-5 ${(entry.top5Accuracy * 100).toFixed(1)}%` : ''
      set((s) => ({
        epochMetrics: [...s.epochMetrics, entry],
        etaSecs: msg.etaSecs as number,
        statusMessage: `Epoch ${msg.epoch} / ${msg.totalEpochs} — val acc ${accStr}${top5Str}`,
      }))
    } else if (type === 'warning') {
      set({ statusMessage: `⚠ ${msg.message as string}` })
    } else if (type === 'complete') {
      get()._socket?.close()
      const cm = (msg as Record<string, unknown>).confusionMatrix
      set({ status: 'complete', statusMessage: 'Training complete', _socket: null, confusionMatrix: cm as number[][] | null ?? null })
    } else if (type === 'stopped') {
      get()._socket?.close()
      set({ status: 'stopped', statusMessage: 'Training stopped', _socket: null })
    } else if (type === 'error') {
      get()._socket?.close()
      set({ status: 'error', errorMessage: msg.message as string, _socket: null })
    }
  },
}))
