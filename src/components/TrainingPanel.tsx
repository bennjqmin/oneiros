import { t } from '../theme/tokens'
import { useEffect, useRef, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
  Cell,
  ScatterChart,
  Scatter,
  ReferenceLine,
  ZAxis,
} from 'recharts'
import { useTrainingStore } from '../store/useTrainingStore'
import type { XGBResult } from '../store/useTrainingStore'
import { useDatasetStore } from '../store/useDatasetStore'
import { useGraphStore } from '../store/useGraphStore'
import type { TrainingConfig, EpochMetrics, SchedulerType } from '../types/training'
import type { ValidationIssue } from '../types/validation'
import { useDeferredTabularValidation } from '../hooks/useDeferredTabularValidation'
import { LoadingLabel, PanelBusyOverlay } from './panelChrome'
import { useChartTheme } from '../theme/useChartTheme'
import { useThemeSync } from '../theme/useThemeSync'

import type { CodeMode } from './CodePanel'

const API_BASE = 'http://localhost:8000'

interface TrainingPanelProps {
  onClose: () => void
  mobile?: boolean
  onOpenCode?: (mode: CodeMode) => void
}

export default function TrainingPanel({ onClose, mobile, onOpenCode }: TrainingPanelProps) {
  useThemeSync()
  const status = useTrainingStore((s) => s.status)
  const statusMessage = useTrainingStore((s) => s.statusMessage)
  const config = useTrainingStore((s) => s.config)
  const epochMetrics = useTrainingStore((s) => s.epochMetrics)
  const currentEpoch = useTrainingStore((s) => s.currentEpoch)
  const totalEpochs = useTrainingStore((s) => s.totalEpochs)
  const currentBatch = useTrainingStore((s) => s.currentBatch)
  const totalBatches = useTrainingStore((s) => s.totalBatches)
  const currentLoss = useTrainingStore((s) => s.currentLoss)
  const etaSecs = useTrainingStore((s) => s.etaSecs)
  const errorMessage = useTrainingStore((s) => s.errorMessage)
  const customDatasetInfo = useTrainingStore((s) => s.customDatasetInfo)
  const setConfig = useTrainingStore((s) => s.setConfig)
  const startTraining = useTrainingStore((s) => s.startTraining)
  const stopTraining = useTrainingStore((s) => s.stopTraining)
  const exportWeights = useTrainingStore((s) => s.exportWeights)
  const exportONNX = useTrainingStore((s) => s.exportONNX)
  const exportFull = useTrainingStore((s) => s.exportFull)
  const xgbStatus = useTrainingStore((s) => s.xgbStatus)
  const xgbStatusMessage = useTrainingStore((s) => s.xgbStatusMessage)
  const xgbResult = useTrainingStore((s) => s.xgbResult)
  const xgbError = useTrainingStore((s) => s.xgbError)
  const xgbPreflightIssues = useTrainingStore((s) => s.xgbPreflightIssues)
  const trainXGBoost = useTrainingStore((s) => s.trainXGBoost)
  const exportXGB = useTrainingStore((s) => s.exportXGB)
  const preflightIssues = useTrainingStore((s) => s.preflightIssues)
  const csvDataset = useDatasetStore((s) => s.dataset)
  const csvTarget = useDatasetStore((s) => s.targetColumn)
  const pipelineNodes = useDatasetStore((s) => s.pipelineNodes)
  const pipelineEdges = useDatasetStore((s) => s.pipelineEdges)
  const cvDataset = useDatasetStore((s) => s.cvDataset)
  const confusionMatrix = useTrainingStore((s) => s.confusionMatrix)
  // Live validation issues from the graph store (updated on every graph change)
  const liveIssues = useGraphStore((s) => s.validationIssues)

  const [height, setHeight] = useState(300)
  const drag = useRef({ active: false, startY: 0, startH: 0 })

  function onResizeMouseDown(e: React.MouseEvent) {
    e.preventDefault()
    drag.current = { active: true, startY: e.clientY, startH: height }
    const onMove = (ev: MouseEvent) => {
      if (!drag.current.active) return
      // Dragging top edge upward increases height
      const next = Math.max(140, Math.min(window.innerHeight * 0.85, drag.current.startH + drag.current.startY - ev.clientY))
      setHeight(next)
    }
    const onUp = () => {
      drag.current.active = false
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  const isXGB = config.modelType === 'xgboost'
  const isRunning = status === 'running' || status === 'connecting'
  const hasMetrics = epochMetrics.length > 0
  const lastMetrics = epochMetrics[epochMetrics.length - 1]
  const panelHeight = mobile ? 'min(85dvh, 100%)' : height

  const xgbLiveIssues = useDeferredTabularValidation(
    isXGB && xgbStatus !== 'running',
    csvDataset,
    csvTarget,
    pipelineNodes,
    pipelineEdges,
    {
      xgbTask: config.xgbTask,
      xgbNEstimators: config.xgbNEstimators,
      xgbEarlyStoppingRounds: config.xgbEarlyStoppingRounds,
    },
  )

  return (
    <div style={{
      height: panelHeight,
      maxHeight: mobile ? '85dvh' : undefined,
      background: t.topbarBg,
      borderTop: `1px solid ${t.borderSubtle}`,
      display: 'flex',
      flexDirection: 'column',
      flexShrink: 0,
      position: 'relative',
      ...(mobile ? { zIndex: 30 } : {}),
    }}>
      {(isXGB ? xgbStatus === 'running' : isRunning) && (
        <PanelBusyOverlay
          label={
            isXGB
              ? (xgbStatusMessage || 'Training XGBoost…')
              : (statusMessage || 'Preparing…')
          }
        />
      )}
      {/* Resize handle on top edge */}
      {!mobile && (
      <div
        onMouseDown={onResizeMouseDown}
        title="Drag to resize"
        style={{
          position: 'absolute', top: 0, left: 0, right: 0, height: 4,
          cursor: 'ns-resize', zIndex: 10,
          background: 'transparent', transition: 'background 0.15s',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.background = '#7c3aed66' }}
        onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
      />
      )}
      {/* Header */}
      <div style={{
        height: 36,
        display: 'flex',
        alignItems: 'center',
        padding: '0 14px',
        borderBottom: `1px solid ${t.borderSubtle}`,
        gap: 10,
        flexShrink: 0,
      }}>
        {/* Model type toggle */}
        <div style={{ display: 'flex', gap: 2, background: t.bgElevated, border: `1px solid ${t.borderDefault}`, borderRadius: 5, padding: 2 }}>
          {(['nn', 'xgboost'] as const).map((type) => (
            <button key={type} onClick={() => setConfig({ modelType: type })} style={{
              padding: '2px 9px', borderRadius: 4, fontSize: 10, fontWeight: 600,
              border: 'none', cursor: 'pointer',
              background: config.modelType === type ? t.accent : 'transparent',
              color: config.modelType === type ? t.textPrimary : t.textFaint,
              textTransform: 'uppercase', letterSpacing: '0.05em',
            }}>{type === 'nn' ? 'Neural Net' : 'XGBoost'}</button>
          ))}
        </div>

        {!isXGB && <StatusBadge status={status} />}
        {isXGB && <StatusBadge status={xgbStatus === 'running' ? 'running' : xgbStatus === 'complete' ? 'complete' : xgbStatus === 'error' ? 'error' : 'idle'} />}

        {!isXGB && statusMessage && (
          <span style={{ fontSize: 11, color: t.textMuted }}>{statusMessage}</span>
        )}

        {!isXGB && etaSecs !== null && isRunning && (
          <span style={{ fontSize: 11, color: t.textFaint }}>
            ETA {etaSecs < 60 ? `${etaSecs}s` : `${Math.round(etaSecs / 60)}m`}
          </span>
        )}

        <div style={{ flex: 1 }} />

        {!isXGB && isRunning && (
          <PanelBtn onClick={stopTraining} label="Stop" danger />
        )}

        {!isXGB && status === 'complete' && (
          <div style={{ display: 'flex', gap: 5 }}>
            <PanelBtn onClick={exportWeights} label="↓ Weights (.pt)" />
            <PanelBtn onClick={exportFull} label="↓ Full Model" />
            <PanelBtn onClick={exportONNX} label="↓ ONNX" accent />
          </div>
        )}

        {isXGB && xgbStatus === 'complete' && xgbResult && (
          <PanelBtn onClick={exportXGB} label="↓ XGB Model (.json)" accent />
        )}

        <button onClick={onClose} style={{
          background: 'transparent', border: 'none',
          color: t.textFaint, cursor: 'pointer', fontSize: 16, lineHeight: 1,
          borderRadius: 4, padding: '2px 4px',
        }}
          onMouseEnter={(e) => { e.currentTarget.style.color = t.textPrimary }}
          onMouseLeave={(e) => { e.currentTarget.style.color = t.textFaint }}
        >×</button>
      </div>

      {/* Body */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', flexDirection: mobile ? 'column' : 'row' }}>
        {isXGB ? (
          /* ── XGBoost layout ─────────────────────────────────────────────── */
          <>
            <div style={{
              width: mobile ? '100%' : 280,
              borderRight: mobile ? undefined : `1px solid ${t.borderSubtle}`,
              borderBottom: mobile ? `1px solid ${t.borderSubtle}` : undefined,
              padding: '12px 14px',
              overflowY: 'auto',
              flexShrink: 0,
              maxHeight: mobile ? '45%' : undefined,
            }}>
              {xgbStatus === 'error' ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <ErrorBlock message={xgbError ?? 'Unknown error'} />
                  {xgbPreflightIssues.length > 0 && <PreflightPanel issues={xgbPreflightIssues} />}
                  <button
                    onClick={() => useTrainingStore.setState({ xgbStatus: 'idle', xgbError: null, xgbPreflightIssues: [] })}
                    style={{ fontSize: 11, color: t.textMuted, background: 'none', border: 'none', cursor: 'pointer', textAlign: 'left', padding: 0, marginTop: 2 }}
                  >
                    ← Back to config
                  </button>
                </div>
              ) : (
                <>
                  <XGBConfigForm config={config} onChange={setConfig} onTrain={trainXGBoost}
                    running={xgbStatus === 'running'} csvDatasetName={csvDataset?.name ?? null} csvTarget={csvTarget}
                    onOpenCode={onOpenCode ? () => onOpenCode('xgboost') : undefined} />
                  {xgbStatus !== 'running' && xgbLiveIssues.loading && (
                    <div style={{ marginTop: 10 }}>
                      <LoadingLabel label="Checking dataset & pipeline…" />
                    </div>
                  )}
                  {xgbStatus !== 'running' && !xgbLiveIssues.loading && xgbLiveIssues.issues.length > 0 && (
                    <div style={{ marginTop: 10 }}>
                      <PreflightPanel issues={xgbLiveIssues.issues} />
                    </div>
                  )}
                </>
              )}
            </div>
            <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
              {xgbStatus === 'running' ? (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                  <LoadingLabel label={xgbStatusMessage || 'Training XGBoost…'} />
                </div>
              ) : xgbResult ? (
                <XGBResults result={xgbResult} />
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                  <p style={{ fontSize: 12, color: t.textDisabled, margin: 0 }}>Configure and click Train XGBoost</p>
                </div>
              )}
            </div>
          </>
        ) : (
          /* ── Neural network layout ──────────────────────────────────────── */
          <>
            <div style={{ width: mobile ? '100%' : 280, borderRight: mobile ? undefined : `1px solid ${t.borderSubtle}`, borderBottom: mobile ? `1px solid ${t.borderSubtle}` : undefined, padding: '12px 14px', overflowY: 'auto', flexShrink: 0, maxHeight: mobile ? '45%' : undefined }}>
              {status === 'error' ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <ErrorBlock message={errorMessage ?? 'Unknown error'} />
                  {preflightIssues.length > 0 && <PreflightPanel issues={preflightIssues} />}
                  <button
                    onClick={() => useTrainingStore.setState({ status: 'idle', errorMessage: null, preflightIssues: [] })}
                    style={{ fontSize: 11, color: t.textMuted, background: 'none', border: 'none', cursor: 'pointer', textAlign: 'left', padding: 0, marginTop: 2 }}
                  >
                    ← Back to config
                  </button>
                </div>
              ) : (
                <>
                  <ConfigForm config={config} onChange={setConfig} onStart={startTraining} disabled={isRunning}
                    csvDatasetName={csvDataset?.name ?? null} csvTarget={csvTarget} customDatasetInfo={customDatasetInfo}
                    onOpenCode={onOpenCode ? () => onOpenCode('nn') : undefined} />
                  {!isRunning && liveIssues.length > 0 && (
                    <div style={{ marginTop: 10 }}>
                      <PreflightPanel issues={liveIssues as ValidationIssue[]} />
                    </div>
                  )}
                </>
              )}
            </div>

            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '10px 14px', gap: 10 }}>
              {(isRunning || hasMetrics) && totalEpochs > 0 && (
                <div style={{ flexShrink: 0 }}>
                  <ProgressRow label={`Epoch ${currentEpoch} / ${totalEpochs}`} value={currentEpoch / totalEpochs} color="#8b5cf6" />
                  {isRunning && totalBatches > 0 && (
                    <ProgressRow
                      label={`Batch ${currentBatch} / ${totalBatches}${currentLoss !== null ? ` · loss ${currentLoss.toFixed(4)}` : ''}`}
                      value={currentBatch / totalBatches} color="#6366f1"
                    />
                  )}
                  {lastMetrics && (
                    <div style={{ display: 'flex', gap: 14, marginTop: 4, flexWrap: 'wrap' }}>
                      <MetricChip label="Train Loss" value={lastMetrics.trainLoss.toFixed(4)} />
                      <MetricChip label="Val Loss" value={lastMetrics.valLoss.toFixed(4)} />
                      <MetricChip label="Val Acc" value={`${(lastMetrics.valAccuracy * 100).toFixed(1)}%`} accent />
                      {lastMetrics.top5Accuracy != null && (
                        <MetricChip label="Top-5 Acc" value={`${(lastMetrics.top5Accuracy * 100).toFixed(1)}%`} accent />
                      )}
                      {lastMetrics.currentLR !== undefined && config.scheduler !== 'none' && (
                        <MetricChip label="LR" value={lastMetrics.currentLR.toExponential(2)} />
                      )}
                    </div>
                  )}
                </div>
              )}
              {hasMetrics ? (
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10, minHeight: 0, overflow: 'auto' }}>
                  <div style={{ flex: 1, display: 'flex', gap: 12, minHeight: 160, flexDirection: mobile ? 'column' : 'row' }}>
                    <ChartPanel title="Loss" data={epochMetrics}
                      lines={[
                        { key: 'trainLoss' as keyof EpochMetrics, color: '#8b5cf6', label: 'Train' },
                        { key: 'valLoss' as keyof EpochMetrics, color: '#6366f1', label: 'Val' },
                      ]}
                    />
                    <ChartPanel title="Accuracy" data={epochMetrics}
                      lines={[{ key: 'valAccuracy' as keyof EpochMetrics, color: t.success, label: 'Val' }]}
                      percent
                    />
                  </div>
                  {status === 'complete' && confusionMatrix && confusionMatrix.length > 0 && (
                    <ConfusionMatrixPanel
                      matrix={confusionMatrix}
                      classNames={cvDataset?.classNames}
                    />
                  )}
                </div>
              ) : (status === 'idle' || status === 'stopped' || status === 'complete') ? (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <p style={{ fontSize: 12, color: t.textDisabled, textAlign: 'center', margin: 0 }}>
                    {status === 'complete' ? 'Training complete.' : 'Configure and start training.'}
                  </p>
                </div>
              ) : null}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ── Config form ───────────────────────────────────────────────────────────────

function ConfigForm({
  config,
  onChange,
  onStart,
  disabled,
  csvDatasetName,
  csvTarget,
  customDatasetInfo,
  onOpenCode,
}: {
  config: TrainingConfig
  onChange: (patch: Partial<TrainingConfig>) => void
  onStart: () => void
  disabled: boolean
  csvDatasetName: string | null
  csvTarget: string | null
  customDatasetInfo: import('../types/training').CustomDatasetPayload | null
  onOpenCode?: () => void
}) {
  const cvDataset = useDatasetStore((s) => s.cvDataset)
  const cvDatasetRef = useTrainingStore((s) => s.cvDatasetRef)

  const datasetOptions = [
    { value: 'mnist', label: 'MNIST' },
    { value: 'fashion_mnist', label: 'Fashion-MNIST' },
    { value: 'cifar10', label: 'CIFAR-10' },
    ...(csvDatasetName
      ? [{ value: 'custom', label: `CSV: ${csvDatasetName}` }]
      : [{ value: 'custom', label: 'CSV (no dataset loaded)' }]),
    ...(cvDataset
      ? [{ value: 'image_folder', label: `Images: ${cvDataset.name}` }]
      : []),
  ]

  const isCustom = config.dataset === 'custom'
  const isImageFolder = config.dataset === 'image_folder'
  const customReady = isCustom && !!csvDatasetName && !!csvTarget

  return (
    <div>
      <CfgSelect
        label="Dataset"
        value={config.dataset}
        onChange={(v) => onChange({ dataset: v as TrainingConfig['dataset'] })}
        options={datasetOptions}
        disabled={disabled}
      />

      {/* Custom dataset info */}
      {isCustom && !disabled && (
        <div style={{
          marginBottom: 9, padding: '7px 9px',
          background: customReady ? 'rgba(139,92,246,0.06)' : 'rgba(239,68,68,0.06)',
          border: `1px solid ${customReady ? '#7c3aed30' : '#ef444430'}`,
          borderRadius: 6, fontSize: 10, lineHeight: 1.7,
        }}>
          {customReady ? (
            <>
              <div style={{ color: t.accentMuted, fontWeight: 600, marginBottom: 2 }}>{csvDatasetName}</div>
              <div style={{ color: t.textMuted }}>
                Target: <span style={{ color: t.textPrimary }}>{csvTarget}</span>
              </div>
              {customDatasetInfo && (
                <>
                  <div style={{ color: t.textMuted }}>
                    Features: <span style={{ color: t.textPrimary }}>{customDatasetInfo.featureCount}</span>
                    {' · '}Classes: <span style={{ color: t.textPrimary }}>{customDatasetInfo.classCount}</span>
                  </div>
                  <div style={{ color: t.textMuted }}>
                    Train: <span style={{ color: t.textPrimary }}>{customDatasetInfo.trainSamples}</span>
                    {' · '}Val: <span style={{ color: t.textPrimary }}>{customDatasetInfo.valSamples}</span>
                  </div>
                  <div style={{ color: t.textFaint, marginTop: 2 }}>
                    Set Input node: channels={customDatasetInfo.featureCount}, H=1, W=1
                  </div>
                </>
              )}
            </>
          ) : (
            <span style={{ color: '#f87171' }}>
              Load a dataset in the Dataset tab and select a target column.
            </span>
          )}
        </div>
      )}
      {/* Image folder dataset info */}
      {isImageFolder && !disabled && (
        <div style={{
          marginBottom: 9, padding: '7px 9px',
          background: cvDatasetRef ? 'rgba(6,182,212,0.06)' : 'rgba(239,68,68,0.06)',
          border: `1px solid ${cvDatasetRef ? '#06b6d430' : '#ef444430'}`,
          borderRadius: 6, fontSize: 10, lineHeight: 1.7,
        }}>
          {cvDataset && cvDatasetRef ? (
            <>
              <div style={{ color: '#67e8f9', fontWeight: 600, marginBottom: 2 }}>{cvDataset.name}</div>
              <div style={{ color: t.textMuted }}>
                Classes: <span style={{ color: t.textPrimary }}>{cvDataset.classNames.length}</span>
                {' · '}Images: <span style={{ color: t.textPrimary }}>{cvDataset.totalImages.toLocaleString()}</span>
              </div>
              <div style={{ color: t.textMuted }}>
                Shape: <span style={{ color: t.textPrimary }}>{cvDataset.inputShape[0]}×{cvDataset.inputShape[1]}×{cvDataset.inputShape[2]}</span>
              </div>
              {cvDatasetRef.augmentSteps.length === 0 && (
                <div style={{ color: '#fcd34d', marginTop: 3 }}>
                  ⚠ No augmentation — consider Resize + Normalize
                </div>
              )}
            </>
          ) : (
            <span style={{ color: '#f87171' }}>
              Export an image dataset from the Dataset tab first.
            </span>
          )}
        </div>
      )}

      <CfgNumber label="Epochs" value={config.epochs} onChange={(v) => onChange({ epochs: v })} min={1} disabled={disabled} />
      <CfgNumber label="Batch Size" value={config.batchSize} onChange={(v) => onChange({ batchSize: v })} min={1} disabled={disabled} />
      <CfgNumber label="Learning Rate" value={config.learningRate} onChange={(v) => onChange({ learningRate: v })} step={0.0001} disabled={disabled} />
      <CfgSelect
        label="Optimizer"
        value={config.optimizer}
        onChange={(v) => onChange({ optimizer: v as TrainingConfig['optimizer'] })}
        options={[
          { value: 'adam', label: 'Adam' },
          { value: 'sgd', label: 'SGD' },
          { value: 'rmsprop', label: 'RMSprop' },
        ]}
        disabled={disabled}
      />

      {/* LR Scheduler */}
      <CfgSelect
        label="LR Scheduler"
        value={config.scheduler}
        onChange={(v) => onChange({ scheduler: v as SchedulerType })}
        options={[
          { value: 'none', label: 'None' },
          { value: 'step', label: 'StepLR' },
          { value: 'cosine', label: 'CosineAnnealing' },
          { value: 'plateau', label: 'ReduceOnPlateau' },
          { value: 'exponential', label: 'ExponentialLR' },
        ]}
        disabled={disabled}
      />

      {/* Contextual scheduler params */}
      {config.scheduler === 'step' && (
        <>
          <CfgNumber label="Step Size (epochs)" value={config.schedulerStepSize} onChange={(v) => onChange({ schedulerStepSize: v })} min={1} disabled={disabled} />
          <CfgNumber label="Gamma" value={config.schedulerGamma} onChange={(v) => onChange({ schedulerGamma: v })} step={0.05} min={0} disabled={disabled} />
        </>
      )}
      {config.scheduler === 'cosine' && (
        <CfgNumber label="T_max (epochs)" value={config.schedulerTMax} onChange={(v) => onChange({ schedulerTMax: v })} min={1} disabled={disabled} />
      )}
      {config.scheduler === 'exponential' && (
        <CfgNumber label="Gamma" value={config.schedulerGamma} onChange={(v) => onChange({ schedulerGamma: v })} step={0.05} min={0} disabled={disabled} />
      )}

      <button
        onClick={onStart}
        disabled={disabled || (isCustom && !customReady)}
        style={{
          width: '100%',
          marginTop: 12,
          padding: '7px 0',
          borderRadius: 6,
          border: `1px solid ${t.accent}`,
          background: (disabled || (isCustom && !customReady)) ? '#1a1a2e' : '#7c3aed1a',
          color: (disabled || (isCustom && !customReady)) ? t.textDisabled : t.accentMuted,
          fontSize: 12,
          fontWeight: 600,
          cursor: (disabled || (isCustom && !customReady)) ? 'not-allowed' : 'pointer',
          transition: 'background 0.12s',
        }}
        onMouseEnter={(e) => { if (!disabled && !isCustom || customReady) e.currentTarget.style.background = '#7c3aed33' }}
        onMouseLeave={(e) => { e.currentTarget.style.background = (disabled || (isCustom && !customReady)) ? '#1a1a2e' : '#7c3aed1a' }}
      >
        Start Training
      </button>

      {onOpenCode && (
        <button
          type="button"
          onClick={onOpenCode}
          disabled={disabled}
          style={{
            width: '100%',
            marginTop: 8,
            padding: '6px 0',
            borderRadius: 6,
            border: `1px solid ${t.borderDefault}`,
            background: disabled ? 'transparent' : t.accentSubtle,
            color: disabled ? t.textDisabled : t.accentMuted,
            fontSize: 11,
            fontWeight: 600,
            cursor: disabled ? 'not-allowed' : 'pointer',
          }}
        >
          View Code
        </button>
      )}
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  const cfg: Record<string, { color: string; bg: string; label: string }> = {
    idle: { color: t.textFaint, bg: 'transparent', label: 'Idle' },
    connecting: { color: '#fbbf24', bg: 'rgba(245,158,11,0.1)', label: 'Connecting…' },
    running: { color: '#34d399', bg: 'rgba(16,185,129,0.1)', label: 'Training' },
    stopped: { color: t.textSecondary, bg: 'rgba(161,161,170,0.1)', label: 'Stopped' },
    complete: { color: '#818cf8', bg: 'rgba(129,140,248,0.1)', label: 'Complete' },
    error: { color: '#f87171', bg: 'rgba(239,68,68,0.1)', label: 'Error' },
  }
  const c = cfg[status] ?? cfg.idle

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 5,
      padding: '2px 8px', borderRadius: 4, background: c.bg,
    }}>
      {status === 'running' && (
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#34d399', animation: 'pulse 1.5s ease-in-out infinite', flexShrink: 0 }} />
      )}
      <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: c.color }}>
        {c.label}
      </span>
    </div>
  )
}

function ProgressRow({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontSize: 10, color: t.textMuted, marginBottom: 3 }}>{label}</div>
      <div style={{ height: 4, background: t.borderDefault, borderRadius: 2, overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${Math.min(value * 100, 100)}%`,
          background: color,
          borderRadius: 2,
          transition: 'width 0.3s ease',
        }} />
      </div>
    </div>
  )
}

function MetricChip({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <span style={{ fontSize: 9, color: t.textFaint, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</span>
      <span style={{ fontSize: 13, fontWeight: 600, color: accent ? t.success : t.textPrimary, fontVariantNumeric: 'tabular-nums' }}>{value}</span>
    </div>
  )
}

function ChartPanel({
  title,
  data,
  lines,
  percent,
}: {
  title: string
  data: EpochMetrics[]
  lines: { key: keyof EpochMetrics; color: string; label: string }[]
  percent?: boolean
}) {
  const chart = useChartTheme()
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: t.textFaint, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>
        {title}
      </div>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart data={data} margin={{ top: 2, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chart.gridStroke} />
          <XAxis dataKey="epoch" stroke={chart.axisStroke} tick={{ fontSize: 9, fill: t.textFaint }} />
          <YAxis
            stroke={chart.axisStroke}
            tick={{ fontSize: 9, fill: t.textFaint }}
            tickFormatter={percent ? (v: number) => `${(v * 100).toFixed(0)}%` : undefined}
          />
          <Tooltip
            contentStyle={{ background: t.bgElevated, border: `1px solid ${t.borderDefault}`, fontSize: 11, borderRadius: 6 }}
            labelStyle={{ color: t.textMuted }}
            formatter={percent ? (v: unknown) => [`${((v as number) * 100).toFixed(1)}%`] : undefined}
          />
          {lines.length > 1 && <Legend wrapperStyle={{ fontSize: 9, color: t.textMuted }} />}
          {lines.map((l) => (
            <Line
              key={l.key}
              type="monotone"
              dataKey={l.key}
              stroke={l.color}
              strokeWidth={1.5}
              dot={false}
              name={l.label}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Rich error + preflight panel ─────────────────────────────────────────────

const CATEGORY_META: Record<string, { label: string; color: string; icon: string }> = {
  structure: { label: 'Structure', color: '#f87171', icon: '⛓' },
  shape:     { label: 'Shape',     color: '#fb923c', icon: '⬡' },
  config:    { label: 'Config',    color: '#facc15', icon: '⚙' },
  dataset:   { label: 'Dataset',   color: '#60a5fa', icon: '📊' },
  training:  { label: 'Runtime',   color: '#f472b6', icon: '⚡' },
  other:     { label: 'Other',     color: t.accentMuted, icon: '●' },
}

const SEV_COLOR: Record<string, string> = { error: '#f87171', warning: '#fbbf24', info: '#60a5fa' }
const SEV_BG:    Record<string, string> = { error: 'rgba(239,68,68,0.07)', warning: 'rgba(245,158,11,0.07)', info: 'rgba(96,165,250,0.07)' }

/** Parse common backend error strings into a user-friendly object. */
function parseTrainingError(raw: string): { title: string; detail: string; hint: string } {
  if (raw.includes('Shape mismatch') || raw.includes('cannot be multiplied') || raw.includes('size mismatch')) {
    const match = raw.match(/\((\d+x\d+) and (\d+x\d+)\)/)
    const isCV = raw.includes('image dataset') || raw.includes('Backbone') || raw.includes('channel')
    return {
      title: 'Shape Mismatch',
      detail: match ? `Got tensor ${match[1]} but layer expects ${match[2]}.` : raw,
      hint: isCV
        ? 'Check your Input node C/H/W matches the image dataset shape. Add a Resize augmentation if sizes differ. Backbone requires channels=3.'
        : 'Check that your Input node dimensions match your data. For a CSV with N features set channels=N, height=1, width=1.',
    }
  }
  if (raw.includes('Backbone expects') || raw.includes('backbone') || raw.includes('expected input') && raw.includes('channel')) {
    return { title: 'Backbone Channel Error', detail: raw, hint: 'Pretrained backbones require 3-channel (RGB) input. Set channels=3 on the Input node.' }
  }
  if (raw.includes('kernel is larger') || raw.includes('output size is too small') || raw.includes('Convolutional kernel')) {
    return { title: 'Kernel Too Large', detail: raw, hint: 'Your convolutional kernel is bigger than the feature map. Add padding, reduce kernel size, reduce pooling, or use a larger input size.' }
  }
  if (raw.includes('Loss became NaN')) {
    return { title: 'Loss is NaN', detail: raw, hint: 'Lower your learning rate (try 0.0001), add BatchNorm layers, or check your labels are in the correct range.' }
  }
  if (raw.includes('Loss exploded')) {
    return { title: 'Loss Exploded', detail: raw, hint: 'Lower your learning rate or add gradient clipping.' }
  }
  if (raw.includes('out of memory') || raw.includes('OutOfMemory')) {
    return { title: 'Out of Memory', detail: raw, hint: 'Reduce batch size, add Dropout/pooling, or use a smaller model.' }
  }
  if (raw.includes('torchvision is required') || raw.includes('torchvision is not installed')) {
    return { title: 'Missing Dependency', detail: raw, hint: 'Run: pip install torchvision in your backend directory.' }
  }
  if (raw.includes('CV session') || raw.includes('session') && raw.includes('not found')) {
    return { title: 'Session Expired', detail: raw, hint: 'Re-upload your image zip in the Dataset tab — sessions reset when the backend restarts.' }
  }
  if (raw.includes('Image dataset session missing')) {
    return { title: 'No Image Dataset', detail: raw, hint: 'Go to Dataset → Images tab, export your dataset to training first.' }
  }
  if (raw.includes('Invalid or expired run')) {
    return { title: 'Connection Error', detail: raw, hint: 'Click Start Training again — the run ID expired before the WebSocket could connect.' }
  }
  if (raw.includes('WebSocket connection error') || raw.includes('Connection failed')) {
    return { title: 'Backend Unreachable', detail: raw, hint: 'Make sure the backend is running: cd backend && uvicorn main:app --reload --port 8000' }
  }
  if (raw.includes('Graph is empty') || raw.includes('No Input') || raw.includes('No Output')) {
    return { title: 'Invalid Graph', detail: raw, hint: 'Build a complete graph: Input → layers → Output, all connected.' }
  }
  if (raw.includes('xgboost') && raw.includes('not installed')) {
    return { title: 'Missing Dependency', detail: raw, hint: 'Run: pip install xgboost scikit-learn in the backend directory.' }
  }
  if (raw.includes('No training data') || raw.includes('Load a CSV') || raw.includes('No dataset loaded')) {
    return { title: 'No Dataset', detail: raw, hint: 'Import a CSV in Dataset tab, set a target column, and connect Source → Split in the pipeline.' }
  }
  if (raw.includes('Validation set is empty') || raw.includes('train ratio')) {
    return { title: 'Split Error', detail: raw, hint: 'Lower the train ratio on the Split node or add more rows to your dataset.' }
  }
  if (raw.includes('No numeric feature') || raw.includes('zero columns') || raw.includes('zero features')) {
    return { title: 'Pipeline Error', detail: raw, hint: 'One-hot encode categoricals or ensure numeric features remain after the pipeline.' }
  }
  if (raw.includes('at least 2 classes') || raw.includes('Classification requires')) {
    return { title: 'Classification Error', detail: raw, hint: 'Pick a target with multiple classes or switch Task to Regression.' }
  }
  if (raw.includes('Pipeline crashed') || raw.includes('Pipeline execution crashed')) {
    return { title: 'Pipeline Crash', detail: raw, hint: 'Reduce dataset size or remove one-hot / balance-classes steps.' }
  }
  if (raw.includes('serialize') || raw.includes('Serialized payload') || raw.includes('Failed to serialize')) {
    return { title: 'Payload Too Large', detail: raw, hint: 'Filter rows/features — large payloads crash the browser silently.' }
  }
  if (raw.includes('Cannot reach backend') || raw.includes('Backend returned')) {
    return { title: 'Backend Offline', detail: raw, hint: 'Run npm run dev:backend in a terminal.' }
  }
  if (raw.includes('Matrix too large') || raw.includes('Too many features') || raw.includes('Too many rows')) {
    return { title: 'Dataset Too Large', detail: raw, hint: 'Filter rows or reduce features in Dataset → Pipeline.' }
  }
  if (raw.includes('greater than -1') || raw.includes('squaredlogerror') || raw.includes('rmsle')) {
    return { title: 'Objective Error', detail: raw, hint: 'Use reg:squarederror for targets that can be negative, or shift targets above -1.' }
  }
  if (raw.includes('crashed unexpectedly') || raw.includes('native error')) {
    return { title: 'Backend Crash', detail: raw, hint: 'Restart the backend (npm run dev:backend). Try fewer rows/trees or remove Balance Classes from the pipeline.' }
  }
  if (raw.includes('Dataset too large') || raw.includes('Too many features')) {
    return { title: 'Dataset Too Large', detail: raw, hint: 'Filter rows or reduce features in the Dataset pipeline before training.' }
  }
  if (raw.includes('Regression targets') || raw.includes('non-numeric target')) {
    return { title: 'Regression Error', detail: raw, hint: 'Use a numeric target column and set Task to Regression in the XGBoost panel.' }
  }
  if (raw.includes('early_stopping') || raw.includes('n_estimators')) {
    return { title: 'Hyperparameter Error', detail: raw, hint: 'Set early_stopping_rounds lower than n_estimators.' }
  }
  if (raw.includes('Task is set to') || raw.includes('looks like')) {
    return { title: 'Task Mismatch', detail: raw, hint: 'Match the Task toggle (Classification / Regression) to your target column type.' }
  }
  if (raw.includes('NaN values') || raw.includes('Fill NaN')) {
    return { title: 'Invalid Features', detail: raw, hint: 'Add a Fill NaN transform in Dataset → Pipeline.' }
  }
  if (raw.includes('Model not found') || raw.includes('train first')) {
    return { title: 'No Trained Model', detail: raw, hint: 'Train an XGBoost model before exporting.' }
  }
  if (raw.includes('XGBoost training failed') || raw.includes('XGBoost classification failed') || raw.includes('XGBoost regression failed')) {
    return { title: 'XGBoost Error', detail: raw, hint: 'Check dataset, target column, task type, and pipeline configuration.' }
  }
  return { title: 'Training Error', detail: raw, hint: '' }
}

function ErrorBlock({ message }: { message: string }) {
  const { title, detail, hint } = parseTrainingError(message)
  return (
    <div style={{ padding: '10px 12px', background: 'rgba(239,68,68,0.08)', border: '1px solid #ef444430', borderRadius: 7, lineHeight: 1.6 }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: '#f87171', marginBottom: 3 }}>{title}</div>
      <div style={{ fontSize: 11, color: '#fca5a5', fontFamily: 'ui-monospace,monospace', wordBreak: 'break-word', marginBottom: hint ? 6 : 0 }}>{detail}</div>
      {hint && (
        <div style={{ fontSize: 11, color: '#fbbf24', background: 'rgba(245,158,11,0.08)', borderRadius: 5, padding: '4px 8px', display: 'flex', gap: 5 }}>
          <span style={{ opacity: 0.7 }}>💡</span><span>{hint}</span>
        </div>
      )}
    </div>
  )
}

function PreflightPanel({ issues }: { issues: ValidationIssue[] }) {
  if (issues.length === 0) return null
  const errors   = issues.filter(i => i.severity === 'error')
  const warnings = issues.filter(i => i.severity === 'warning')
  const infos    = issues.filter(i => i.severity === 'info')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {errors.length > 0 && (
        <div style={{ fontSize: 10, fontWeight: 700, color: '#f87171', textTransform: 'uppercase', letterSpacing: '0.07em', display: 'flex', alignItems: 'center', gap: 5 }}>
          <span>✗</span> {errors.length} Error{errors.length > 1 ? 's' : ''} — fix before training
        </div>
      )}
      {[...errors, ...warnings, ...infos].map((issue, i) => {
        const cat = CATEGORY_META[issue.category] ?? CATEGORY_META.other
        const bg  = SEV_BG[issue.severity]  ?? SEV_BG.info
        const col = SEV_COLOR[issue.severity] ?? SEV_COLOR.info
        return (
          <div key={i} style={{ padding: '7px 10px', background: bg, border: `1px solid ${col}20`, borderLeft: `3px solid ${col}`, borderRadius: '0 5px 5px 0', lineHeight: 1.55 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: issue.hint ? 3 : 0 }}>
              <span style={{ fontSize: 9, fontWeight: 700, color: cat.color, background: `${cat.color}18`, borderRadius: 3, padding: '1px 5px', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                {cat.icon} {cat.label}
              </span>
              {issue.nodeId && (
                <span style={{ fontSize: 9, color: t.textFaint, fontFamily: 'ui-monospace,monospace' }}>node:{issue.nodeId.slice(0, 6)}</span>
              )}
            </div>
            <div style={{ fontSize: 11, color: t.textBody }}>{issue.message}</div>
            {issue.hint && (
              <div style={{ fontSize: 10, color: t.textSecondary, marginTop: 2, display: 'flex', gap: 4 }}>
                <span>↳</span><span>{issue.hint}</span>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function PanelBtn({ onClick, label, danger, accent }: { onClick: () => void; label: string; danger?: boolean; accent?: boolean }) {
  return (
    <button onClick={onClick} style={{
      padding: '3px 10px', borderRadius: 5,
      border: `1px solid ${danger ? '#7f1d1d' : accent ? t.accent : t.borderDefault}`,
      background: danger ? 'rgba(239,68,68,0.08)' : accent ? 'rgba(124,58,237,0.12)' : 'transparent',
      color: danger ? '#f87171' : accent ? t.accentMuted : t.textMuted,
      fontSize: 11, fontWeight: 500, cursor: 'pointer',
    }}>
      {label}
    </button>
  )
}

function CfgSelect({
  label, value, onChange, options, disabled,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string }[]
  disabled?: boolean
}) {
  return (
    <div style={{ marginBottom: 9 }}>
      <div style={{ fontSize: 10, color: t.textMuted, marginBottom: 3, fontWeight: 500 }}>{label}</div>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: '100%', background: t.bgElevated, border: `1px solid ${t.borderDefault}`,
          borderRadius: 5, color: disabled ? t.textFaint : t.textPrimary,
          fontSize: 11, padding: '4px 7px', outline: 'none', cursor: disabled ? 'not-allowed' : 'pointer',
        }}
      >
        {options.map((o) => <option key={o.value} value={o.value} style={{ background: t.bgElevated }}>{o.label}</option>)}
      </select>
    </div>
  )
}

function CfgNumber({
  label, value, onChange, min, step, disabled,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  step?: number
  disabled?: boolean
}) {
  return (
    <div style={{ marginBottom: 9 }}>
      <div style={{ fontSize: 10, color: t.textMuted, marginBottom: 3, fontWeight: 500 }}>{label}</div>
      <input
        type="number"
        value={value}
        disabled={disabled}
        min={min}
        step={step ?? 1}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{
          width: '100%', background: t.bgElevated, border: `1px solid ${t.borderDefault}`,
          borderRadius: 5, color: disabled ? t.textFaint : t.textPrimary,
          fontSize: 11, padding: '4px 7px', outline: 'none',
          cursor: disabled ? 'not-allowed' : 'text',
          boxSizing: 'border-box',
        }}
      />
    </div>
  )
}

// ── XGBoost config form ───────────────────────────────────────────────────────

function XGBConfigForm({ config, onChange, onTrain, running, csvDatasetName, csvTarget, onOpenCode }: {
  config: TrainingConfig
  onChange: (patch: Partial<TrainingConfig>) => void
  onTrain: () => void
  running: boolean
  csvDatasetName: string | null
  csvTarget: string | null
  onOpenCode?: () => void
}) {
  const ready = !!csvDatasetName && !!csvTarget
  const isRegression = config.xgbTask === 'regression'

  const [regObjectives, setRegObjectives] = useState<{ value: string; label: string }[]>([])
  useEffect(() => {
    fetch(`${API_BASE}/api/xgboost/meta`)
      .then((r) => r.json())
      .then((d: { regressionObjectives: { value: string; label: string }[] }) => setRegObjectives(d.regressionObjectives))
      .catch(() => setRegObjectives([
        { value: 'reg:squarederror', label: 'Squared Error (RMSE)' },
        { value: 'reg:absoluteerror', label: 'Absolute Error (MAE)' },
        { value: 'reg:squaredlogerror', label: 'Squared Log Error' },
        { value: 'reg:tweedie', label: 'Tweedie' },
      ]))
  }, [])

  return (
    <div>
      {/* Dataset status */}
      <div style={{ marginBottom: 9, padding: '7px 9px', background: ready ? 'rgba(139,92,246,0.06)' : 'rgba(239,68,68,0.06)',
        border: `1px solid ${ready ? '#7c3aed30' : '#ef444430'}`, borderRadius: 6, fontSize: 10, lineHeight: 1.7 }}>
        {ready ? (
          <>
            <div style={{ color: t.accentMuted, fontWeight: 600 }}>{csvDatasetName}</div>
            <div style={{ color: t.textMuted }}>Target: <span style={{ color: t.textPrimary }}>{csvTarget}</span></div>
          </>
        ) : (
          <span style={{ color: '#f87171' }}>Load a CSV dataset in the Dataset tab and set a target column.</span>
        )}
      </div>

      {/* Task toggle */}
      <div style={{ marginBottom: 8 }}>
        <div style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em', color: t.textFaint, marginBottom: 5 }}>Task</div>
        <div style={{ display: 'flex', gap: 3, background: t.bgElevated, border: `1px solid ${t.borderDefault}`, borderRadius: 5, padding: 2 }}>
          {(['classification', 'regression'] as const).map((task) => (
            <button key={task} disabled={running} onClick={() => onChange({ xgbTask: task, xgbObjective: '' })} style={{
              flex: 1, padding: '3px 0', borderRadius: 4, fontSize: 10, fontWeight: 600, border: 'none',
              background: config.xgbTask === task ? (task === 'regression' ? '#b45309' : '#4c1d95') : 'transparent',
              color: config.xgbTask === task ? t.textPrimary : t.textFaint,
              cursor: running ? 'not-allowed' : 'pointer', textTransform: 'capitalize',
            }}>{task}</button>
          ))}
        </div>
      </div>

      {/* Regression objective */}
      {isRegression && regObjectives.length > 0 && (
        <CfgSelect
          label="Objective"
          value={config.xgbObjective || 'reg:squarederror'}
          onChange={(v) => onChange({ xgbObjective: v })}
          options={regObjectives}
          disabled={running}
        />
      )}

      <div style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em', color: t.textFaint, marginBottom: 6 }}>
        Hyperparameters
      </div>
      <CfgNumber label="n_estimators" value={config.xgbNEstimators} onChange={(v) => onChange({ xgbNEstimators: v })} min={1} disabled={running} />
      <CfgNumber label="max_depth" value={config.xgbMaxDepth} onChange={(v) => onChange({ xgbMaxDepth: v })} min={1} disabled={running} />
      <CfgNumber label="learning_rate (eta)" value={config.xgbLearningRate} onChange={(v) => onChange({ xgbLearningRate: v })} step={0.01} min={0.001} disabled={running} />
      <CfgNumber label="subsample" value={config.xgbSubsample} onChange={(v) => onChange({ xgbSubsample: v })} step={0.05} min={0.1} disabled={running} />
      <CfgNumber label="colsample_bytree" value={config.xgbColsampleBytree} onChange={(v) => onChange({ xgbColsampleBytree: v })} step={0.05} min={0.1} disabled={running} />
      <CfgNumber label="min_child_weight" value={config.xgbMinChildWeight} onChange={(v) => onChange({ xgbMinChildWeight: v })} min={0} disabled={running} />
      <CfgNumber label="gamma (min split loss)" value={config.xgbGamma} onChange={(v) => onChange({ xgbGamma: v })} step={0.1} min={0} disabled={running} />
      <CfgNumber label="reg_alpha (L1)" value={config.xgbRegAlpha} onChange={(v) => onChange({ xgbRegAlpha: v })} step={0.1} min={0} disabled={running} />
      <CfgNumber label="reg_lambda (L2)" value={config.xgbRegLambda} onChange={(v) => onChange({ xgbRegLambda: v })} step={0.1} min={0} disabled={running} />
      <CfgNumber label="early_stopping_rounds (0=off)" value={config.xgbEarlyStoppingRounds} onChange={(v) => onChange({ xgbEarlyStoppingRounds: v })} min={0} disabled={running} />

      <button onClick={onTrain} disabled={running || !ready} style={{
        width: '100%', marginTop: 12, padding: '7px 0', borderRadius: 6,
        border: `1px solid ${isRegression ? '#d97706' : t.warning}`,
        background: (running || !ready) ? '#1a1a2e' : (isRegression ? 'rgba(180,83,9,0.15)' : 'rgba(245,158,11,0.12)'),
        color: (running || !ready) ? t.textDisabled : '#fbbf24',
        fontSize: 12, fontWeight: 600, cursor: (running || !ready) ? 'not-allowed' : 'pointer',
      }}>
        {running ? 'Training…' : `⚡ Train (${isRegression ? 'Regression' : 'Classification'})`}
      </button>

      {onOpenCode && (
        <button
          type="button"
          onClick={onOpenCode}
          disabled={!ready}
          style={{
            width: '100%',
            marginTop: 8,
            padding: '6px 0',
            borderRadius: 6,
            border: `1px solid ${t.borderDefault}`,
            background: ready ? t.accentSubtle : 'transparent',
            color: ready ? t.accentMuted : t.textDisabled,
            fontSize: 11,
            fontWeight: 600,
            cursor: ready ? 'pointer' : 'not-allowed',
          }}
        >
          View Code
        </button>
      )}
    </div>
  )
}

// ── XGBoost results ───────────────────────────────────────────────────────────

function XGBResults({ result }: { result: XGBResult }) {
  const chart = useChartTheme()
  const top10 = result.featureImportance.slice(0, 10)
  const isRegression = result.task === 'regression'
  const showAccuracyChart =
    !isRegression &&
    result.evals.some((e) => e.valAccuracy != null)
  const showRmseChart =
    isRegression &&
    result.evals.some((e) => e.valRMSE != null)
  const showMaeChart =
    isRegression &&
    result.evals.some((e) => e.valMAE != null)
  const scatter = result.valScatter ?? []
  const scatterDomain = scatter.length > 0
    ? (() => {
        const vals = scatter.flatMap((p) => [p.actual, p.predicted])
        const lo = Math.min(...vals)
        const hi = Math.max(...vals)
        const pad = (hi - lo) * 0.05 || 1
        return { lo: lo - pad, hi: hi + pad }
      })()
    : null
  const residuals = scatter.map((p) => ({
    predicted: p.predicted,
    residual: p.actual - p.predicted,
  }))
  const lossTitle = isRegression
    ? `Objective (${result.objective?.replace('reg:', '') ?? 'loss'}) per Round`
    : 'Loss per Round'

  const chartBox = (title: string, children: React.ReactNode) => (
    <div style={{ flex: 1, minWidth: 220 }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: t.textFaint, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>
        {title}
      </div>
      <ResponsiveContainer width="100%" height={220}>
        {children}
      </ResponsiveContainer>
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', padding: '10px 14px', gap: 10 }}>
      {/* Summary chips */}
      <div style={{ display: 'flex', gap: 16, flexShrink: 0, flexWrap: 'wrap' }}>
        {isRegression ? (
          <>
            <MetricChip label="Val RMSE" value={result.valRMSE?.toFixed(4) ?? '–'} accent />
            <MetricChip label="Val MAE" value={result.valMAE?.toFixed(4) ?? '–'} />
            <MetricChip label="Val R²" value={result.valR2?.toFixed(4) ?? '–'} accent />
            <MetricChip label="Train RMSE" value={result.trainRMSE?.toFixed(4) ?? '–'} />
            <MetricChip label="Train MAE" value={result.trainMAE?.toFixed(4) ?? '–'} />
            <MetricChip label="Train R²" value={result.trainR2?.toFixed(4) ?? '–'} />
          </>
        ) : (
          <>
            <MetricChip label="Val Accuracy" value={`${((result.valAccuracy ?? 0) * 100).toFixed(2)}%`} accent />
            <MetricChip label="Train Accuracy" value={`${((result.trainAccuracy ?? 0) * 100).toFixed(2)}%`} />
            <MetricChip label="Classes" value={String(result.nClasses ?? '–')} />
          </>
        )}
        <MetricChip label="Best Round" value={`${result.bestIteration} / ${result.nEstimators}`} />
      </div>

      {/* Training curves */}
      {result.evals.length > 0 && (
        <div style={{ display: 'flex', gap: 12, flexShrink: 0, flexWrap: 'wrap' }}>
          {chartBox(lossTitle,
            <LineChart data={result.evals} margin={{ top: 2, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chart.gridStroke} />
              <XAxis dataKey="round" stroke={chart.axisStroke} tick={{ fontSize: 9, fill: t.textFaint }} />
              <YAxis stroke={chart.axisStroke} tick={{ fontSize: 9, fill: t.textFaint }} />
              <Tooltip contentStyle={{ background: t.bgElevated, border: `1px solid ${t.borderDefault}`, fontSize: 11, borderRadius: 6 }} labelStyle={{ color: t.textMuted }} />
              <Legend wrapperStyle={{ fontSize: 9, color: t.textMuted }} />
              <Line type="monotone" dataKey="trainLoss" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Train" isAnimationActive={false} />
              <Line type="monotone" dataKey="valLoss" stroke="#fbbf24" strokeWidth={1.5} dot={false} name="Val" isAnimationActive={false} />
            </LineChart>,
          )}

          {showRmseChart && chartBox('RMSE per Round',
            <LineChart data={result.evals} margin={{ top: 2, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chart.gridStroke} />
              <XAxis dataKey="round" stroke={chart.axisStroke} tick={{ fontSize: 9, fill: t.textFaint }} />
              <YAxis stroke={chart.axisStroke} tick={{ fontSize: 9, fill: t.textFaint }} />
              <Tooltip contentStyle={{ background: t.bgElevated, border: `1px solid ${t.borderDefault}`, fontSize: 11, borderRadius: 6 }} labelStyle={{ color: t.textMuted }} />
              <Legend wrapperStyle={{ fontSize: 9, color: t.textMuted }} />
              <Line type="monotone" dataKey="trainRMSE" stroke="#38bdf8" strokeWidth={1.5} dot={false} name="Train RMSE" isAnimationActive={false} />
              <Line type="monotone" dataKey="valRMSE" stroke="#0ea5e9" strokeWidth={1.5} dot={false} name="Val RMSE" isAnimationActive={false} />
            </LineChart>,
          )}

          {showMaeChart && chartBox('MAE per Round',
            <LineChart data={result.evals} margin={{ top: 2, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chart.gridStroke} />
              <XAxis dataKey="round" stroke={chart.axisStroke} tick={{ fontSize: 9, fill: t.textFaint }} />
              <YAxis stroke={chart.axisStroke} tick={{ fontSize: 9, fill: t.textFaint }} />
              <Tooltip contentStyle={{ background: t.bgElevated, border: `1px solid ${t.borderDefault}`, fontSize: 11, borderRadius: 6 }} labelStyle={{ color: t.textMuted }} />
              <Legend wrapperStyle={{ fontSize: 9, color: t.textMuted }} />
              <Line type="monotone" dataKey="trainMAE" stroke="#a78bfa" strokeWidth={1.5} dot={false} name="Train MAE" isAnimationActive={false} />
              <Line type="monotone" dataKey="valMAE" stroke="#8b5cf6" strokeWidth={1.5} dot={false} name="Val MAE" isAnimationActive={false} />
            </LineChart>,
          )}

          {showAccuracyChart && chartBox('Accuracy per Round',
            <LineChart data={result.evals} margin={{ top: 2, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chart.gridStroke} />
              <XAxis dataKey="round" stroke={chart.axisStroke} tick={{ fontSize: 9, fill: t.textFaint }} />
              <YAxis
                stroke={chart.axisStroke}
                tick={{ fontSize: 9, fill: t.textFaint }}
                domain={[0, 1]}
                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{ background: t.bgElevated, border: `1px solid ${t.borderDefault}`, fontSize: 11, borderRadius: 6 }}
                labelStyle={{ color: t.textMuted }}
                formatter={(v: unknown) => [`${((v as number) * 100).toFixed(1)}%`]}
              />
              <Legend wrapperStyle={{ fontSize: 9, color: t.textMuted }} />
              <Line type="monotone" dataKey="trainAccuracy" stroke="#10b981" strokeWidth={1.5} dot={false} name="Train" isAnimationActive={false} />
              <Line type="monotone" dataKey="valAccuracy" stroke="#34d399" strokeWidth={1.5} dot={false} name="Val" isAnimationActive={false} />
            </LineChart>,
          )}
        </div>
      )}

      {/* Regression diagnostics */}
      {isRegression && scatter.length > 0 && scatterDomain && (
        <div style={{ display: 'flex', gap: 12, flexShrink: 0, flexWrap: 'wrap' }}>
          {chartBox('Predicted vs Actual (val)',
            <ScatterChart margin={{ top: 2, right: 8, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chart.gridStroke} />
              <XAxis
                type="number"
                dataKey="actual"
                name="Actual"
                domain={[scatterDomain.lo, scatterDomain.hi]}
                stroke={chart.axisStroke}
                tick={{ fontSize: 9, fill: t.textFaint }}
              />
              <YAxis
                type="number"
                dataKey="predicted"
                name="Predicted"
                domain={[scatterDomain.lo, scatterDomain.hi]}
                stroke={chart.axisStroke}
                tick={{ fontSize: 9, fill: t.textFaint }}
              />
              <ZAxis range={[20, 20]} />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                contentStyle={{ background: t.bgElevated, border: `1px solid ${t.borderDefault}`, fontSize: 11, borderRadius: 6 }}
                labelStyle={{ color: t.textMuted }}
                formatter={(v: unknown) => [(v as number).toFixed(4)]}
              />
              <ReferenceLine
                segment={[
                  { x: scatterDomain.lo, y: scatterDomain.lo },
                  { x: scatterDomain.hi, y: scatterDomain.hi },
                ]}
                stroke={t.textFaint}
                strokeDasharray="4 4"
              />
              <Scatter data={scatter} fill="#38bdf8" fillOpacity={0.65} isAnimationActive={false} />
            </ScatterChart>,
          )}

          {chartBox('Residuals (val)',
            <ScatterChart margin={{ top: 2, right: 8, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chart.gridStroke} />
              <XAxis
                type="number"
                dataKey="predicted"
                name="Predicted"
                stroke={chart.axisStroke}
                tick={{ fontSize: 9, fill: t.textFaint }}
              />
              <YAxis
                type="number"
                dataKey="residual"
                name="Residual"
                stroke={chart.axisStroke}
                tick={{ fontSize: 9, fill: t.textFaint }}
              />
              <ZAxis range={[20, 20]} />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                contentStyle={{ background: t.bgElevated, border: `1px solid ${t.borderDefault}`, fontSize: 11, borderRadius: 6 }}
                labelStyle={{ color: t.textMuted }}
                formatter={(v: unknown) => [(v as number).toFixed(4)]}
              />
              <ReferenceLine y={0} stroke={t.textFaint} strokeDasharray="4 4" />
              <Scatter data={residuals} fill="#a78bfa" fillOpacity={0.65} isAnimationActive={false} />
            </ScatterChart>,
          )}
        </div>
      )}

      <div style={{ flexShrink: 0, display: 'flex', gap: 12, minHeight: 220 }}>
        {/* Feature importance */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: t.textFaint, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>
            Feature Importance (top 10)
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={top10} layout="vertical" margin={{ top: 2, right: 8, left: 8, bottom: 2 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chart.gridStroke} horizontal={false} />
              <XAxis type="number" stroke={chart.axisStroke} tick={{ fontSize: 9, fill: t.textFaint }} />
              <YAxis dataKey="name" type="category" width={90} tick={{ fontSize: 9, fill: t.textSecondary }} />
              <Tooltip
                contentStyle={{ background: t.bgElevated, border: `1px solid ${t.borderDefault}`, fontSize: 11, borderRadius: 6 }}
                labelStyle={{ color: t.textMuted }}
                formatter={(v: unknown) => [(v as number).toFixed(4)]}
              />
              <Bar dataKey="importance" radius={[0, 3, 3, 0]}>
                {top10.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? t.warning : i < 3 ? '#d97706' : '#92400e'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

// ── Confusion matrix heatmap ──────────────────────────────────────────────────

function ConfusionMatrixPanel({ matrix, classNames }: { matrix: number[][]; classNames?: string[] }) {
  const n = matrix.length
  if (n === 0) return null

  const maxVal = Math.max(...matrix.flat())
  const labels = classNames?.slice(0, n) ?? matrix.map((_, i) => String(i))

  function cellColor(v: number) {
    const t = maxVal > 0 ? v / maxVal : 0
    const r = Math.round(99 + t * (16 - 99))
    const g = Math.round(102 + t * (185 - 102))
    const b = Math.round(241 + t * (129 - 241))
    return `rgb(${r},${g},${b})`
  }

  const cellSize = Math.min(48, Math.max(20, Math.floor(280 / (n + 1))))
  const fontSize = Math.max(8, Math.min(11, cellSize - 6))

  return (
    <div style={{ flexShrink: 0, marginTop: 8 }}>
      <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: t.textFaint, marginBottom: 6 }}>
        Confusion Matrix (validation set)
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', fontSize }}>
          <thead>
            <tr>
              <th style={{ padding: 2, color: t.textDisabled, fontSize: fontSize - 1 }}>P↓ T→</th>
              {labels.map((lbl) => (
                <th key={lbl} style={{ padding: 2, color: t.textMuted, minWidth: cellSize, textAlign: 'center', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: cellSize }}>
                  {lbl.length > 6 ? lbl.slice(0, 5) + '…' : lbl}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, ri) => (
              <tr key={ri}>
                <td style={{ padding: 2, color: t.textMuted, fontSize: fontSize - 1, whiteSpace: 'nowrap', maxWidth: cellSize * 1.5, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {(labels[ri] ?? '').length > 6 ? (labels[ri] ?? '').slice(0, 5) + '…' : (labels[ri] ?? '')}
                </td>
                {row.map((v, ci) => (
                  <td key={ci} style={{
                    width: cellSize, height: cellSize, textAlign: 'center',
                    background: cellColor(v),
                    color: v / maxVal > 0.5 ? t.bgBase : t.textPrimary,
                    borderRadius: 2,
                    fontWeight: ri === ci ? 700 : 400,
                    border: ri === ci ? '1px solid rgba(16,185,129,0.5)' : '1px solid transparent',
                  }}>
                    {v}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
