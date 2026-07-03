import { t } from '../theme/tokens'
import { useEffect, useRef, useState } from 'react'
import { useDatasetStore } from '../store/useDatasetStore'
import type { ColumnInfo, EDFDataset } from '../store/useDatasetStore'
import type { CVDataset } from '../types/training'
import { useTrainingStore } from '../store/useTrainingStore'
import DatasetFlow, { PipelinePaletteItem } from '../editor/dataset/DatasetFlow'
import EDFFlow, { EDFPaletteItem } from '../editor/dataset/EDFFlow'
import AugFlow, { AugPaletteItem } from '../editor/dataset/AugFlow'
import MergeCSVView from '../editor/dataset/MergeCSVView'
import { datasetNodeDefs } from '../editor/dataset/preprocessingNodes'
import { edfNodeDefs } from '../editor/dataset/edfNodes'
import { augNodeDefs } from '../editor/dataset/augmentNodes'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { useTabularModelInfo } from '../hooks/useTabularModelInfo'
import { LoadingLabel } from '../components/panelChrome'
import { useThemeSync } from '../theme/useThemeSync'
import type { LoadedDataset } from '../store/useDatasetStore'

const PREVIEW_ROWS = 200
const API_BASE = 'http://localhost:8000'
const MAX_PLOT_ROWS = 4000

// ── Icons ─────────────────────────────────────────────────────────────────────

function UploadIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )
}

function TableIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3h18v18H3z" />
      <path d="M3 9h18M3 15h18M9 3v18M15 3v18" />
    </svg>
  )
}

function FlowIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="7" width="6" height="6" rx="1" />
      <rect x="16" y="7" width="6" height="6" rx="1" />
      <line x1="8" y1="10" x2="16" y2="10" />
    </svg>
  )
}

function ChartIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6" y1="20" x2="6" y2="14" /><line x1="2" y1="20" x2="22" y2="20" />
    </svg>
  )
}

function TrashIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
    </svg>
  )
}

function BrainIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9.5 2a2.5 2.5 0 0 1 0 5" />
      <path d="M14.5 2a2.5 2.5 0 0 0 0 5" />
      <path d="M5 7a5 5 0 0 0 5 5 5 5 0 0 0 5-5" />
      <path d="M5 7a4 4 0 0 0 0 8h1" />
      <path d="M19 7a4 4 0 0 1 0 8h-1" />
      <path d="M9 17v-2" /><path d="M15 17v-2" />
      <path d="M9 20a3 3 0 0 0 6 0" />
    </svg>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

function ImageIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <polyline points="21 15 16 10 5 21" />
    </svg>
  )
}

function MergeIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 3H5a2 2 0 0 0-2 2v3" />
      <path d="M21 8V5a2 2 0 0 0-2-2h-3" />
      <path d="M3 16v3a2 2 0 0 0 2 2h3" />
      <path d="M16 21h3a2 2 0 0 0 2-2v-3" />
      <line x1="12" y1="8" x2="12" y2="16" />
      <polyline points="8 12 12 16 16 12" />
    </svg>
  )
}

function DownloadIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  )
}

type ActiveTab = 'table' | 'pipeline' | 'visualize' | 'edf' | 'cv' | 'merge'

export default function DatasetPage({ mobile, active = true, onOpenCode }: {
  mobile?: boolean
  active?: boolean
  onOpenCode?: () => void
}) {
  useThemeSync()
  const {
    dataset, loadFromCSV, loadFromJSON, exportDatasetAsCSV,
    clearDataset, targetColumn, setTargetColumn,
    edfDataset, loadFromEDF, clearEDF, edfLoading, edfError,
    cvDataset, loadFromImageZip, clearCVDataset, cvLoading, cvError,
  } = useDatasetStore()

  const [activeTab, setActiveTab] = useState<ActiveTab>('table')
  const csvInputRef  = useRef<HTMLInputElement>(null)
  const jsonInputRef = useRef<HTMLInputElement>(null)
  const edfInputRef  = useRef<HTMLInputElement>(null)
  const cvInputRef   = useRef<HTMLInputElement>(null)

  // Auto-switch to tab when dataset loads
  useEffect(() => {
    if (edfDataset) setActiveTab('edf')
  }, [edfDataset])

  useEffect(() => {
    if (cvDataset) setActiveTab('cv')
  }, [cvDataset])

  async function handleCSV(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) { await loadFromCSV(file); setActiveTab('table') }
    e.target.value = ''
  }

  async function handleJSON(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) { await loadFromJSON(file); setActiveTab('table') }
    e.target.value = ''
  }

  async function handleEDF(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) await loadFromEDF(file)
    e.target.value = ''
  }

  async function handleCV(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) await loadFromImageZip(file)
    e.target.value = ''
  }

  const hasTabular = !!dataset
  const hasEDF     = !!edfDataset
  const hasCV      = !!cvDataset

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: t.bgBase }}>

      {/* Toolbar */}
      <div style={{
        height: mobile ? 'auto' : 48,
        minHeight: 48,
        background: t.bgPanel,
        borderBottom: `1px solid ${t.borderSubtle}`,
        display: 'flex',
        alignItems: 'center',
        padding: mobile ? '8px 12px' : '0 16px',
        gap: 10,
        flexShrink: 0,
        minWidth: 0,
        flexWrap: mobile ? 'wrap' : 'nowrap',
      }}>
        {!mobile && (
        <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: t.textFaint }}>
          Dataset Studio
        </span>
        )}

        {/* Active dataset label */}
        {hasTabular && activeTab !== 'edf' && activeTab !== 'cv' && activeTab !== 'merge' && (
          <>
            <span style={{ fontSize: 11, color: t.textDisabled, margin: '0 2px' }}>›</span>
            <span style={{ fontSize: 12, color: t.textSecondary, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 160 }}>{dataset.name}</span>
          </>
        )}
        {hasEDF && activeTab === 'edf' && (
          <>
            <span style={{ fontSize: 11, color: t.textDisabled, margin: '0 2px' }}>›</span>
            <span style={{ fontSize: 12, color: '#c4b5fd', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 160 }}>{edfDataset.name}</span>
            <span style={{ fontSize: 10, color: t.textFaint, background: t.bgElevated, border: `1px solid ${t.borderDefault}`, borderRadius: 4, padding: '1px 7px', flexShrink: 0 }}>
              {edfDataset.channels.length} ch · {edfDataset.sfreq} Hz · {edfDataset.duration.toFixed(1)} s
            </span>
          </>
        )}
        {hasCV && activeTab === 'cv' && (
          <>
            <span style={{ fontSize: 11, color: t.textDisabled, margin: '0 2px' }}>›</span>
            <span style={{ fontSize: 12, color: '#67e8f9', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 160 }}>{cvDataset!.name}</span>
            <span style={{ fontSize: 10, color: t.textFaint, background: t.bgElevated, border: `1px solid ${t.borderDefault}`, borderRadius: 4, padding: '1px 7px', flexShrink: 0 }}>
              {cvDataset!.classNames.length} classes · {cvDataset!.totalImages.toLocaleString()} images · {cvDataset!.inputShape[0]}×{cvDataset!.inputShape[1]}×{cvDataset!.inputShape[2]}
            </span>
          </>
        )}

        <div style={{ flex: 1 }} />

        {/* Target column selector (tabular only) */}
        {hasTabular && activeTab !== 'edf' && activeTab !== 'cv' && activeTab !== 'merge' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 7, flexShrink: 0 }}>
            <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: t.textFaint, whiteSpace: 'nowrap' }}>
              Target
            </span>
            <select
              value={targetColumn ?? ''}
              onChange={(e) => setTargetColumn(e.target.value || null)}
              style={{
                background: t.bgElevated, border: `1px solid ${t.borderDefault}`,
                borderRadius: 5, color: targetColumn ? t.accentMuted : t.textFaint,
                fontSize: 11, padding: '3px 7px', outline: 'none', cursor: 'pointer',
                maxWidth: 140,
              }}
            >
              <option value="" style={{ background: t.bgElevated }}>— none —</option>
              {dataset!.columns.map((col) => (
                <option key={col.name} value={col.name} style={{ background: t.bgElevated }}>{col.name}</option>
              ))}
            </select>
          </div>
        )}

        {/* Tab toggle */}
        <div style={{ display: 'flex', gap: 2, background: t.bgElevated, border: `1px solid ${t.borderDefault}`, borderRadius: 6, padding: 2, flexShrink: 0 }}>
          <TabBtn active={activeTab === 'merge'} onClick={() => setActiveTab('merge')} icon={<MergeIcon />} label="Merge CSV" merge />
          {hasTabular && (
            <>
              <TabBtn active={activeTab === 'table'}     onClick={() => setActiveTab('table')}     icon={<TableIcon />} label="Table" />
              <TabBtn active={activeTab === 'pipeline'}  onClick={() => setActiveTab('pipeline')}  icon={<FlowIcon />}  label="Pipeline" />
              <TabBtn active={activeTab === 'visualize'} onClick={() => setActiveTab('visualize')} icon={<ChartIcon />} label="Visualize" />
            </>
          )}
          {hasEDF && (
            <TabBtn active={activeTab === 'edf'} onClick={() => setActiveTab('edf')} icon={<BrainIcon />} label="EDF" accent />
          )}
          {hasCV && (
            <TabBtn active={activeTab === 'cv'} onClick={() => setActiveTab('cv')} icon={<ImageIcon />} label="Images" cv />
          )}
        </div>

        {hasTabular && activeTab !== 'edf' && activeTab !== 'cv' && activeTab !== 'merge' && (
          <button
            onClick={() => exportDatasetAsCSV()}
            style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '4px 8px', borderRadius: 5, border: `1px solid ${t.borderDefault}`, background: 'transparent', color: t.textMuted, fontSize: 11, cursor: 'pointer', flexShrink: 0 }}
            onMouseEnter={(e) => { e.currentTarget.style.color = t.textPrimary; e.currentTarget.style.borderColor = t.borderStrong }}
            onMouseLeave={(e) => { e.currentTarget.style.color = t.textMuted; e.currentTarget.style.borderColor = t.borderDefault }}
          >
            <DownloadIcon /> Export CSV
          </button>
        )}

        {/* Clear buttons */}
        {hasTabular && activeTab !== 'edf' && activeTab !== 'cv' && activeTab !== 'merge' && (
          <button
            onClick={() => { clearDataset(); if (!hasEDF) setActiveTab('table') }}
            style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '4px 8px', borderRadius: 5, border: `1px solid ${t.borderDefault}`, background: 'transparent', color: t.textMuted, fontSize: 11, cursor: 'pointer', flexShrink: 0 }}
            onMouseEnter={(e) => { e.currentTarget.style.color = '#f87171'; e.currentTarget.style.borderColor = '#7f1d1d' }}
            onMouseLeave={(e) => { e.currentTarget.style.color = t.textMuted; e.currentTarget.style.borderColor = t.borderDefault }}
          >
            <TrashIcon /> Clear
          </button>
        )}
        {hasEDF && activeTab === 'edf' && (
          <button
            onClick={() => { clearEDF(); if (hasTabular) setActiveTab('table') }}
            style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '4px 8px', borderRadius: 5, border: `1px solid ${t.borderDefault}`, background: 'transparent', color: t.textMuted, fontSize: 11, cursor: 'pointer', flexShrink: 0 }}
            onMouseEnter={(e) => { e.currentTarget.style.color = '#f87171'; e.currentTarget.style.borderColor = '#7f1d1d' }}
            onMouseLeave={(e) => { e.currentTarget.style.color = t.textMuted; e.currentTarget.style.borderColor = t.borderDefault }}
          >
            <TrashIcon /> Clear EDF
          </button>
        )}
        {hasCV && activeTab === 'cv' && (
          <button
            onClick={() => { clearCVDataset(); if (hasTabular) setActiveTab('table') }}
            style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '4px 8px', borderRadius: 5, border: `1px solid ${t.borderDefault}`, background: 'transparent', color: t.textMuted, fontSize: 11, cursor: 'pointer', flexShrink: 0 }}
            onMouseEnter={(e) => { e.currentTarget.style.color = '#f87171'; e.currentTarget.style.borderColor = '#7f1d1d' }}
            onMouseLeave={(e) => { e.currentTarget.style.color = t.textMuted; e.currentTarget.style.borderColor = t.borderDefault }}
          >
            <TrashIcon /> Clear Images
          </button>
        )}

        <input ref={csvInputRef}  type="file" accept=".csv"  onChange={handleCSV}  style={{ display: 'none' }} />
        <input ref={jsonInputRef} type="file" accept=".json" onChange={handleJSON} style={{ display: 'none' }} />
        <input ref={edfInputRef}  type="file" accept=".edf"  onChange={handleEDF}  style={{ display: 'none' }} />
        <input ref={cvInputRef}   type="file" accept=".zip"  onChange={handleCV}   style={{ display: 'none' }} />

        <UploadButton label="Import CSV"    onClick={() => csvInputRef.current?.click()} />
        <UploadButton label="Import JSON"   onClick={() => jsonInputRef.current?.click()} accent />
        <UploadButton label="Import EDF"    onClick={() => edfInputRef.current?.click()} edf />
        <UploadButton label="Import Images" onClick={() => cvInputRef.current?.click()} cv />
      </div>

      {/* Loading indicators */}
      {edfLoading && (
        <div style={{ background: 'rgba(139,92,246,0.08)', borderBottom: '1px solid #8b5cf620', padding: '6px 16px', display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#c4b5fd', flexShrink: 0 }}>
          <Spinner color="#8b5cf6" /> Parsing EDF file with MNE…
        </div>
      )}
      {edfError && (
        <div style={{ background: 'rgba(239,68,68,0.07)', borderBottom: '1px solid #ef444420', padding: '6px 16px', fontSize: 12, color: '#fca5a5', flexShrink: 0 }}>
          EDF error: {edfError}
        </div>
      )}
      {cvLoading && (
        <div style={{ background: 'rgba(6,182,212,0.08)', borderBottom: '1px solid #06b6d420', padding: '6px 16px', display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#67e8f9', flexShrink: 0 }}>
          <Spinner color="#06b6d4" /> Extracting and scanning image dataset…
        </div>
      )}
      {cvError && (
        <div style={{ background: 'rgba(239,68,68,0.07)', borderBottom: '1px solid #ef444420', padding: '6px 16px', fontSize: 12, color: '#fca5a5', flexShrink: 0 }}>
          Image dataset error: {cvError}
        </div>
      )}

      {/* Model-relevant dataset summary (tabular) */}
      {hasTabular && activeTab !== 'edf' && activeTab !== 'cv' && dataset && (
        <TabularModelInfoBar
          dataset={dataset}
          targetColumn={targetColumn}
          usePipeline={activeTab === 'pipeline'}
        />
      )}

      {/* Body */}
      {activeTab === 'merge' ? (
        <MergeCSVView onMerged={() => setActiveTab('table')} />
      ) : !hasTabular && !hasEDF && !hasCV ? (
        <EmptyState
          onImportCSV={() => csvInputRef.current?.click()}
          onImportJSON={() => jsonInputRef.current?.click()}
          onImportEDF={() => edfInputRef.current?.click()}
          onImportImages={() => cvInputRef.current?.click()}
          onMergeCSV={() => setActiveTab('merge')}
        />
      ) : activeTab === 'edf' && hasEDF ? (
        <EDFView edf={edfDataset!} active={active} />
      ) : activeTab === 'cv' && hasCV ? (
        <CVView cv={cvDataset!} active={active} />
      ) : activeTab === 'table' && hasTabular ? (
        <TableView dataset={dataset!} />
      ) : activeTab === 'visualize' && hasTabular ? (
        <VisualizeView dataset={dataset!} />
      ) : activeTab === 'pipeline' && hasTabular ? (
        <PipelineView mobile={mobile} active={active} onOpenCode={onOpenCode} />
      ) : hasTabular ? (
        <TableView dataset={dataset!} />
      ) : null}
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────

function EmptyState({ onImportCSV, onImportJSON, onImportEDF, onImportImages, onMergeCSV }: {
  onImportCSV: () => void
  onImportJSON: () => void
  onImportEDF: () => void
  onImportImages: () => void
  onMergeCSV: () => void
}) {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 20 }}>
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#27272a" strokeWidth="1.2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
      <div style={{ textAlign: 'center' }}>
        <p style={{ fontSize: 14, color: t.textFaint, margin: '0 0 6px' }}>No dataset loaded</p>
        <p style={{ fontSize: 12, color: t.textDisabled, margin: 0 }}>Import tabular data, a biosignal recording, or an image dataset</p>
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
        <UploadButton label="Import CSV" onClick={onImportCSV} />
        <UploadButton label="Import JSON" onClick={onImportJSON} accent />
        <UploadButton label="Merge CSVs" onClick={onMergeCSV} merge />
        <UploadButton label="Import EDF" onClick={onImportEDF} edf />
        <UploadButton label="Import Images" onClick={onImportImages} cv />
      </div>
      <div style={{ display: 'flex', gap: 24, marginTop: 8 }}>
        {[
          { icon: '⊞', title: 'CSV / JSON', desc: 'Tabular data for classification\nor regression with NN / XGBoost' },
          { icon: '⊕', title: 'Merge CSV', desc: 'Combine multiple CSVs\nwith identical columns' },
          { icon: '〜', title: 'EDF (MNE)', desc: 'EEG / ECG / EMG recordings\nFiltered, epoched, then trained' },
          { icon: '⊡', title: 'Images (ZIP)', desc: 'Class folders → image classifier\nWith visual augmentation pipeline' },
        ].map(({ icon, title, desc }) => (
          <div key={title} style={{ textAlign: 'center', maxWidth: 150 }}>
            <div style={{ fontSize: 22, marginBottom: 6, opacity: 0.3 }}>{icon}</div>
            <div style={{ fontSize: 12, color: t.textFaint, fontWeight: 600, marginBottom: 3 }}>{title}</div>
            <div style={{ fontSize: 11, color: t.textDisabled, whiteSpace: 'pre-line', lineHeight: 1.5 }}>{desc}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Tabular model summary bar ─────────────────────────────────────────────────

function TabularModelInfoBar({
  dataset,
  targetColumn,
  usePipeline,
}: {
  dataset: LoadedDataset
  targetColumn: string | null
  usePipeline?: boolean
}) {
  const { info, loading } = useTabularModelInfo(dataset, targetColumn, !!usePipeline)

  return (
    <div style={{
      padding: '7px 16px',
      background: t.bgPanel,
      borderBottom: `1px solid ${t.borderSubtle}`,
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      flexWrap: 'wrap',
      flexShrink: 0,
      opacity: loading ? 0.75 : 1,
      transition: 'opacity 0.15s ease',
    }}>
      <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: t.textFaint, marginRight: 2 }}>
        For model
      </span>

      {loading && usePipeline && (
        <LoadingLabel label="Updating pipeline preview…" />
      )}

      <InfoChip label="Rows" value={info.totalRows.toLocaleString()} accent />
      {info.previewRows != null && (
        <InfoChip label="Preview" value={`${info.previewRows.toLocaleString()} shown`} />
      )}
      <InfoChip
        label="Features"
        value={
          info.source === 'pipeline'
            ? `${info.featureCount} (after pipeline)`
            : `${info.featureCount} numeric${info.categoricalFeatures > 0 ? ` + ${info.categoricalFeatures} cat.` : ''}`
        }
      />
      <InfoChip
        label="Target"
        value={info.targetColumn ?? 'not set'}
        accent={!!info.targetColumn}
        warn={!info.targetColumn}
      />

      {info.targetColumn && info.taskType === 'classification' && info.classCount != null && (
        <InfoChip
          label="Classes"
          value={info.classPreview ? `${info.classCount} (${info.classPreview})` : String(info.classCount)}
          accent
        />
      )}
      {info.targetColumn && info.taskType === 'regression' && (
        <InfoChip label="Task" value="Regression" accent />
      )}

      {info.inputShape && (
        <InfoChip label="Input shape" value={`${info.inputShape} (C×H×W)`} hint="Set on Input node" />
      )}
      {info.outputClasses != null && (
        <InfoChip label="Output classes" value={String(info.outputClasses)} hint="Set on Output node" />
      )}

      {info.trainSamples != null && info.valSamples != null && (
        <InfoChip label="Split" value={`${info.trainSamples} train · ${info.valSamples} val`} />
      )}

      {info.missingTargets > 0 && (
        <InfoChip label="Missing targets" value={String(info.missingTargets)} warn />
      )}

      {info.pipelineError && (
        <span style={{ fontSize: 10, color: '#fca5a5' }}>Pipeline: {info.pipelineError}</span>
      )}

      {!info.targetColumn && (
        <span style={{ fontSize: 10, color: t.textMuted, marginLeft: 'auto' }}>
          Pick a target column to see class count and Output node size
        </span>
      )}
      {info.targetColumn && info.featureCount === 0 && (
        <span style={{ fontSize: 10, color: '#fcd34d', marginLeft: info.targetColumn ? undefined : 'auto' }}>
          No numeric features yet — encode categoricals in the Pipeline tab
        </span>
      )}
    </div>
  )
}

function InfoChip({
  label,
  value,
  accent,
  warn,
  hint,
}: {
  label: string
  value: string
  accent?: boolean
  warn?: boolean
  hint?: string
}) {
  const border = warn ? '#f59e0b40' : accent ? '#7c3aed40' : t.borderDefault
  const bg = warn ? 'rgba(245,158,11,0.08)' : accent ? t.accentSubtle : t.bgElevated
  const color = warn ? '#fcd34d' : accent ? '#c4b5fd' : t.textSecondary
  return (
    <span
      title={hint}
      style={{
        fontSize: 10,
        color,
        background: bg,
        border: `1px solid ${border}`,
        borderRadius: 4,
        padding: '2px 8px',
        whiteSpace: 'nowrap',
        maxWidth: 280,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}
    >
      <span style={{ color: t.textFaint }}>{label}: </span>
      {value}
    </span>
  )
}

// ── Table view ────────────────────────────────────────────────────────────────

function TableView({ dataset }: { dataset: NonNullable<ReturnType<typeof useDatasetStore.getState>['dataset']> }) {
  const preview = dataset.rows.slice(0, PREVIEW_ROWS)
  const cols = dataset.columns

  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

      {/* Column stats sidebar */}
      <div style={{
        width: 220, background: t.bgPanel,
        borderRight: `1px solid ${t.borderSubtle}`,
        overflowY: 'auto', flexShrink: 0,
      }}>
        <div style={{ padding: '10px 12px 6px', fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: t.textFaint }}>
          Columns
        </div>
        {cols.map((col) => <ColumnCard key={col.name} col={col} />)}
      </div>

      {/* Data table */}
      <div style={{ flex: 1, overflow: 'auto', position: 'relative' }}>
        <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
          <thead>
            <tr style={{ background: t.bgPanel, position: 'sticky', top: 0, zIndex: 1 }}>
              <th style={thStyle}>#</th>
              {cols.map((col) => (
                <th key={col.name} style={thStyle}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <TypeDot type={col.type} />
                    {col.name}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.map((row, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)' }}>
                <td style={indexCellStyle}>{i + 1}</td>
                {cols.map((col) => {
                  const val = row[col.name]
                  const isNull = val === null || val === undefined || val === ''
                  return (
                    <td key={col.name} style={{ ...tdStyle, color: isNull ? t.textDisabled : col.type === 'number' ? '#a5b4fc' : t.textBody }}>
                      {isNull ? 'null' : String(val)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── EDF view ──────────────────────────────────────────────────────────────────

type EDFSubTab = 'info' | 'pipeline' | 'visualize'

function EDFView({ edf, active = true }: { edf: EDFDataset; active?: boolean }) {
  const [subTab, setSubTab] = useState<EDFSubTab>('info')
  const [vizType, setVizType] = useState<'waveform' | 'psd'>('waveform')
  const [selChannels, setSelChannels] = useState<string[]>(edf.channels.slice(0, 8))
  const [duration, setDuration] = useState(5)
  const [fmin, setFmin] = useState(0)
  const [fmax, setFmax] = useState(80)
  const [plotImg, setPlotImg] = useState<string | null>(null)
  const [plotLoading, setPlotLoading] = useState(false)
  const [plotError, setPlotError] = useState<string | null>(null)
  const { edfProcessing, edfProcessResult, edfProcessError, edfPipelineNodes } = useDatasetStore()
  const setCustomDataset = useTrainingStore((s) => s.setCustomDatasetFromEDF)

  async function handlePlot() {
    setPlotLoading(true)
    setPlotError(null)
    try {
      const res = await fetch(`${API_BASE}/api/edf/plot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: edf.sessionId,
          config: { plot_type: vizType, channels: selChannels, duration, fmin, fmax },
        }),
      })
      if (!res.ok) {
        const j = await res.json().catch(() => null)
        setPlotError((j?.error as string) ?? `HTTP ${res.status}`)
        return
      }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      setPlotImg((prev) => { if (prev) URL.revokeObjectURL(prev); return url })
    } catch (e) {
      setPlotError(e instanceof Error ? e.message : 'Failed to reach backend')
    } finally {
      setPlotLoading(false)
    }
  }

  async function handleExport() {
    // Derive pipeline steps from the EDF pipeline nodes
    const { edfPipelineNodes: nodes, processEDFPipeline } = useDatasetStore.getState()
    const steps = nodes
      .filter((n) => n.type !== 'edfSourceNode')
      .map((n) => {
        const { label: _l, ...params } = n.data as Record<string, unknown>
        return { type: n.type?.replace('Node', '').replace('edfResample', 'resample').replace('notchFilter', 'notch').replace('pickChannels', 'pick_channels').replace('edfEpochOutput', 'epoch'), ...params }
      })
      .filter((s) => s.type !== 'edfEpochOutput')
    const payload = await processEDFPipeline(steps as Record<string, unknown>[])
    if (payload && setCustomDataset) {
      setCustomDataset(payload)
    }
  }

  const COLORS = ['#6366f1','#0ea5e9',t.success,t.warning,t.error,'#a855f7','#ec4899','#14b8a6','#f97316','#84cc16']

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Sub-tab bar */}
      <div style={{ height: 36, background: t.bgPanel, borderBottom: `1px solid ${t.borderSubtle}`, display: 'flex', alignItems: 'center', padding: '0 14px', gap: 4, flexShrink: 0 }}>
        {(['info', 'pipeline', 'visualize'] as EDFSubTab[]).map((tab) => (
          <button key={tab} onClick={() => setSubTab(tab)} style={{
            padding: '3px 10px', borderRadius: 5, fontSize: 11, border: 'none', cursor: 'pointer',
            background: subTab === tab ? 'rgba(139,92,246,0.18)' : 'transparent',
            color: subTab === tab ? '#c4b5fd' : t.textMuted, fontWeight: subTab === tab ? 600 : 400,
          }}>
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Content */}
      {subTab === 'info' && (
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {/* Left: stats */}
          <div style={{ width: 240, background: t.bgPanel, borderRight: `1px solid ${t.borderSubtle}`, overflowY: 'auto', flexShrink: 0, padding: '12px 0' }}>
            <SectionLabel>Recording Info</SectionLabel>
            {[
              { k: 'Channels', v: edf.channels.length },
              { k: 'Sampling rate', v: `${edf.sfreq} Hz` },
              { k: 'Duration', v: `${edf.duration.toFixed(2)} s` },
              { k: 'Total samples', v: edf.nTimes.toLocaleString() },
            ].map(({ k, v }) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 14px', fontSize: 11 }}>
                <span style={{ color: t.textMuted }}>{k}</span>
                <span style={{ color: t.textBody, fontWeight: 500 }}>{v}</span>
              </div>
            ))}
            {edf.events.length > 0 && (
              <>
                <SectionLabel style={{ marginTop: 10 }}>Events</SectionLabel>
                {edf.events.map((ev) => (
                  <div key={ev.id} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 14px', fontSize: 11 }}>
                    <span style={{ color: t.textMuted }}>{ev.label}</span>
                    <span style={{ color: '#a5b4fc', fontWeight: 500 }}>{ev.count}×</span>
                  </div>
                ))}
              </>
            )}
          </div>
          {/* Right: channel list + mini waveform preview */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px' }}>
            <SectionLabel>Channels ({edf.channels.length})</SectionLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 20 }}>
              {edf.channels.map((ch, i) => (
                <span key={ch} style={{ fontSize: 11, padding: '2px 8px', background: t.bgElevated, border: `1px solid ${COLORS[i % COLORS.length]}30`, borderRadius: 4, color: COLORS[i % COLORS.length] }}>
                  {ch}
                </span>
              ))}
            </div>
            {edf.previewData.length > 0 && (
              <>
                <SectionLabel>Signal Preview (first 10 s, {edf.channels.slice(0, 8).length} channels)</SectionLabel>
                <EEGMiniChart data={edf.previewData.slice(0, 8)} sfreq={edf.previewSfreq} channels={edf.channels.slice(0, 8)} colors={COLORS} />
              </>
            )}
          </div>
        </div>
      )}

      {subTab === 'pipeline' && (
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {/* Palette */}
          <div style={{ width: 190, background: t.bgPanel, borderRight: `1px solid ${t.borderSubtle}`, padding: '10px 0', flexShrink: 0, overflowY: 'auto' }}>
            <div style={{ padding: '2px 12px 6px', fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: t.textFaint }}>EDF Transforms</div>
            {edfNodeDefs.map((def) => <EDFPaletteItem key={def.type} def={def} />)}
            <div style={{ padding: '10px 12px 4px', marginTop: 8, borderTop: `1px solid ${t.borderSubtle}`, fontSize: 10, color: t.textDisabled, lineHeight: 1.5 }}>
              Drag transforms onto the canvas. Ends with an Epoch Output node.
            </div>
          </div>
          {/* Canvas */}
          <div style={{ flex: 1, position: 'relative', display: 'flex', flexDirection: 'column' }}>
            <div style={{ flex: 1 }}>
              {active && <EDFFlow />}
            </div>
            {/* Export bar */}
            <div style={{ height: 46, background: t.bgPanel, borderTop: `1px solid ${t.borderSubtle}`, display: 'flex', alignItems: 'center', padding: '0 14px', gap: 10, flexShrink: 0 }}>
              {edfProcessResult ? (
                <span style={{ fontSize: 11, color: '#6ee7b7' }}>
                  ✓ Exported {edfProcessResult.trainSamples + edfProcessResult.valSamples} epochs · {edfProcessResult.featureCount} features · {edfProcessResult.classCount} classes
                </span>
              ) : edfProcessError ? (
                <span style={{ fontSize: 11, color: '#fca5a5' }}>Error: {edfProcessError}</span>
              ) : (
                <span style={{ fontSize: 11, color: t.textFaint }}>
                  Build your pipeline then export epochs to the Model for training.
                </span>
              )}
              <div style={{ flex: 1 }} />
              <button
                onClick={handleExport}
                disabled={edfProcessing || edfPipelineNodes.length < 2}
                style={{
                  padding: '5px 14px', borderRadius: 6, fontSize: 12, fontWeight: 600,
                  border: '1px solid #8b5cf6',
                  background: edfProcessing ? '#1a1a2e' : 'rgba(139,92,246,0.15)',
                  color: edfProcessing ? t.textFaint : '#c4b5fd',
                  cursor: edfProcessing || edfPipelineNodes.length < 2 ? 'not-allowed' : 'pointer',
                }}
              >
                {edfProcessing ? 'Processing…' : '⟡ Export to Training'}
              </button>
            </div>
          </div>
        </div>
      )}

      {subTab === 'visualize' && (
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {/* Controls */}
          <div style={{ width: 240, background: t.bgPanel, borderRight: `1px solid ${t.borderSubtle}`, overflowY: 'auto', flexShrink: 0, padding: '12px 0', display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '0 12px', display: 'flex', flexDirection: 'column', gap: 10 }}>
              <VizLabel>Plot Type</VizLabel>
              <div style={{ display: 'flex', gap: 6 }}>
                {(['waveform', 'psd'] as const).map((plotType) => (
                  <button key={plotType} onClick={() => setVizType(plotType)} style={{
                    flex: 1, padding: '5px 0', borderRadius: 6, fontSize: 11, cursor: 'pointer',
                    border: vizType === plotType ? '1px solid #8b5cf6' : `1px solid ${t.borderDefault}`,
                    background: vizType === plotType ? 'rgba(139,92,246,0.15)' : 'transparent',
                    color: vizType === plotType ? '#c4b5fd' : t.textMuted,
                    fontWeight: vizType === plotType ? 600 : 400,
                  }}>
                    {plotType === 'waveform' ? '〜 Waveform' : '⊞ PSD'}
                  </button>
                ))}
              </div>

              <VizLabel>Channels</VizLabel>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, maxHeight: 120, overflowY: 'auto' }}>
                {edf.channels.map((ch) => {
                  const on = selChannels.includes(ch)
                  return (
                    <button key={ch} onClick={() => setSelChannels(on ? selChannels.filter(c => c !== ch) : [...selChannels, ch].slice(0, 16))}
                      style={{ padding: '2px 7px', borderRadius: 4, fontSize: 10, cursor: 'pointer',
                        border: on ? '1px solid #8b5cf6' : `1px solid ${t.borderDefault}`,
                        background: on ? 'rgba(139,92,246,0.15)' : 'transparent',
                        color: on ? '#c4b5fd' : t.textFaint }}>
                      {ch}
                    </button>
                  )
                })}
              </div>

              {vizType === 'waveform' && (
                <>
                  <VizLabel>Duration (s)</VizLabel>
                  <input type="range" min={1} max={Math.min(30, edf.duration)} value={duration} onChange={(e) => setDuration(+e.target.value)}
                    style={{ accentColor: '#8b5cf6', width: '100%' }} />
                  <span style={{ fontSize: 10, color: t.textFaint }}>{duration} s</span>
                </>
              )}
              {vizType === 'psd' && (
                <>
                  <VizLabel>Freq Range (Hz)</VizLabel>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <input type="number" value={fmin} onChange={(e) => setFmin(+e.target.value)} min={0} max={edf.sfreq / 2 - 1}
                      style={{ flex: 1, background: t.bgElevated, border: `1px solid ${t.borderDefault}`, borderRadius: 4, color: t.textPrimary, fontSize: 11, padding: '3px 6px', outline: 'none' }} />
                    <span style={{ fontSize: 11, color: t.textFaint, alignSelf: 'center' }}>–</span>
                    <input type="number" value={fmax} onChange={(e) => setFmax(+e.target.value)} min={1} max={edf.sfreq / 2}
                      style={{ flex: 1, background: t.bgElevated, border: `1px solid ${t.borderDefault}`, borderRadius: 4, color: t.textPrimary, fontSize: 11, padding: '3px 6px', outline: 'none' }} />
                  </div>
                </>
              )}
            </div>

            <div style={{ flex: 1 }} />
            <div style={{ padding: '12px 12px 0' }}>
              <button onClick={handlePlot} disabled={plotLoading || selChannels.length === 0}
                style={{ width: '100%', padding: '8px 0', borderRadius: 7, border: '1px solid #8b5cf6',
                  background: plotLoading ? '#1a1a2e' : 'rgba(139,92,246,0.15)',
                  color: plotLoading ? t.textFaint : '#c4b5fd',
                  fontSize: 13, fontWeight: 600, cursor: plotLoading ? 'not-allowed' : 'pointer' }}>
                {plotLoading ? 'Generating…' : '⟡ Plot'}
              </button>
            </div>
          </div>

          {/* Chart area */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: t.bgBase }}>
            <div style={{ flex: 1, overflow: 'auto', display: 'flex', alignItems: 'flex-start', justifyContent: 'center', padding: 24 }}>
              {plotLoading ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, marginTop: 80 }}>
                  <Spinner color="#8b5cf6" /><span style={{ fontSize: 12, color: t.textFaint }}>Generating plot…</span>
                </div>
              ) : plotError ? (
                <div style={{ maxWidth: 420, padding: '14px 18px', background: 'rgba(239,68,68,0.08)', border: '1px solid #ef444430', borderRadius: 8, fontSize: 13, color: '#fca5a5', lineHeight: 1.5 }}>
                  <strong style={{ display: 'block', marginBottom: 4 }}>Plot error</strong>{plotError}
                </div>
              ) : plotImg ? (
                <img src={plotImg} alt="EDF plot" style={{ maxWidth: '100%', borderRadius: 10, border: `1px solid ${t.borderSubtle}`, boxShadow: '0 8px 32px rgba(0,0,0,0.5)' }} />
              ) : (
                <div style={{ textAlign: 'center', marginTop: 80 }}>
                  <div style={{ fontSize: 40, marginBottom: 14, opacity: 0.2 }}>〜</div>
                  <p style={{ fontSize: 13, color: t.textFaint, margin: '0 0 6px' }}>Select channels and click <strong style={{ color: t.textMuted }}>Plot</strong></p>
                  <p style={{ fontSize: 11, color: t.textDisabled }}>{edf.channels.length} channels · {edf.sfreq} Hz · {edf.duration.toFixed(1)} s</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ── CV view ───────────────────────────────────────────────────────────────────

type CVSubTab = 'info' | 'augmentation' | 'export'

const CV_COLORS = ['#0ea5e9','#06b6d4',t.success,t.warning,'#a855f7','#ec4899','#f97316','#84cc16','#6366f1',t.error]

function CVView({ cv, active = true }: { cv: CVDataset; active?: boolean }) {
  const [subTab, setSubTab] = useState<CVSubTab>('info')
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewImg, setPreviewImg] = useState<string | null>(null)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [previewIndex, setPreviewIndex] = useState(0)
  const augPipelineNodes = useDatasetStore((s) => s.augPipelineNodes)
  const setTrainingConfig = useTrainingStore((s) => s.setConfig)
  const setCVDatasetTraining = useTrainingStore((s) => s.setCVDataset)

  async function handlePreview() {
    setPreviewLoading(true)
    setPreviewError(null)
    const steps = augPipelineNodes
      .filter((n) => n.type !== 'augSource')
      .map((n) => {
        const { label: _l, ...params } = n.data as Record<string, unknown>
        return { type: n.type, ...params }
      })
    try {
      const res = await fetch(`${API_BASE}/api/cv/augment_preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: cv.sessionId, steps, imageIndex: previewIndex }),
      })
      if (!res.ok) {
        const j = await res.json().catch(() => null)
        setPreviewError((j?.error as string) ?? `HTTP ${res.status}`)
        return
      }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      setPreviewImg((prev) => { if (prev) URL.revokeObjectURL(prev); return url })
    } catch (e) {
      setPreviewError(e instanceof Error ? e.message : 'Preview failed')
    } finally {
      setPreviewLoading(false)
    }
  }

  function handleExportToTraining() {
    const steps = augPipelineNodes
      .filter((n) => n.type !== 'augSource')
      .map((n) => {
        const { label: _l, ...params } = n.data as Record<string, unknown>
        return { type: n.type, ...params }
      })
    // Store CV dataset info and augment steps in training store
    if (setCVDatasetTraining) {
      setCVDatasetTraining({ sessionId: cv.sessionId, augmentSteps: steps })
    }
    setTrainingConfig({ dataset: 'image_folder' })
  }

  const chartData = cv.classNames.map((cls, i) => ({
    name: cls,
    count: cv.classCounts[cls] ?? 0,
    fill: CV_COLORS[i % CV_COLORS.length],
  }))

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Sub-tab bar */}
      <div style={{ height: 36, background: t.bgPanel, borderBottom: `1px solid ${t.borderSubtle}`, display: 'flex', alignItems: 'center', padding: '0 14px', gap: 4, flexShrink: 0 }}>
        {(['info', 'augmentation', 'export'] as CVSubTab[]).map((tab) => (
          <button key={tab} onClick={() => setSubTab(tab)} style={{
            padding: '3px 10px', borderRadius: 5, fontSize: 11, border: 'none', cursor: 'pointer',
            background: subTab === tab ? 'rgba(6,182,212,0.18)' : 'transparent',
            color: subTab === tab ? '#67e8f9' : t.textMuted, fontWeight: subTab === tab ? 600 : 400,
          }}>
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Warnings */}
      {cv.warnings.length > 0 && (
        <div style={{ background: 'rgba(245,158,11,0.07)', borderBottom: '1px solid #f59e0b20', padding: '6px 14px', flexShrink: 0 }}>
          {cv.warnings.map((w, i) => (
            <div key={i} style={{ fontSize: 11, color: '#fcd34d', display: 'flex', alignItems: 'flex-start', gap: 6, marginBottom: i < cv.warnings.length - 1 ? 3 : 0 }}>
              <span style={{ flexShrink: 0, marginTop: 1 }}>⚠</span>{w}
            </div>
          ))}
        </div>
      )}

      {/* Info sub-tab */}
      {subTab === 'info' && (
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {/* Left: stats */}
          <div style={{ width: 240, background: t.bgPanel, borderRight: `1px solid ${t.borderSubtle}`, overflowY: 'auto', flexShrink: 0, padding: '12px 0' }}>
            <SectionLabel>Dataset Info</SectionLabel>
            {[
              { k: 'Name',        v: cv.name },
              { k: 'Classes',     v: cv.classNames.length },
              { k: 'Total images',v: cv.totalImages.toLocaleString() },
              { k: 'Input shape', v: `${cv.inputShape[0]}×${cv.inputShape[1]}×${cv.inputShape[2]}` },
            ].map(({ k, v }) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 14px', fontSize: 11 }}>
                <span style={{ color: t.textMuted }}>{k}</span>
                <span style={{ color: t.textBody, fontWeight: 500 }}>{v}</span>
              </div>
            ))}
            <SectionLabel style={{ marginTop: 10 }}>Class Distribution</SectionLabel>
            {cv.classNames.map((cls, i) => (
              <div key={cls} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 14px', fontSize: 11 }}>
                <span style={{ color: CV_COLORS[i % CV_COLORS.length] }}>{cls}</span>
                <span style={{ color: t.textBody }}>{(cv.classCounts[cls] ?? 0).toLocaleString()}</span>
              </div>
            ))}
          </div>

          {/* Right: chart + thumbnails */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>
            <SectionLabel>Class Balance</SectionLabel>
            <div style={{ height: 220, marginBottom: 24 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 28, left: 8 }}>
                  <XAxis dataKey="name" tick={{ fill: t.textMuted, fontSize: 10 }} axisLine={{ stroke: t.borderDefault }} tickLine={false} />
                  <YAxis tick={{ fill: t.textFaint, fontSize: 10 }} axisLine={{ stroke: t.borderDefault }} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: t.bgElevated, border: `1px solid ${t.borderDefault}`, borderRadius: 6, fontSize: 11 }}
                    labelStyle={{ color: t.textPrimary }}
                    itemStyle={{ color: t.textSecondary }}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <SectionLabel>Thumbnails</SectionLabel>
            {cv.classNames.map((cls, ci) => {
              const thumbs = cv.thumbnails[cls] ?? []
              if (!thumbs.length) return null
              return (
                <div key={cls} style={{ marginBottom: 16 }}>
                  <div style={{ fontSize: 11, color: CV_COLORS[ci % CV_COLORS.length], fontWeight: 600, marginBottom: 8 }}>
                    {cls} <span style={{ color: t.textFaint, fontWeight: 400 }}>({cv.classCounts[cls]} images)</span>
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    {thumbs.map((b64, ti) => (
                      <img
                        key={ti}
                        src={`data:image/jpeg;base64,${b64}`}
                        alt={`${cls} thumb ${ti}`}
                        style={{ width: 64, height: 64, borderRadius: 6, objectFit: 'cover', border: `1px solid ${CV_COLORS[ci % CV_COLORS.length]}40` }}
                      />
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Augmentation sub-tab */}
      {subTab === 'augmentation' && (
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {/* Palette */}
          <div style={{ width: 195, background: t.bgPanel, borderRight: `1px solid ${t.borderSubtle}`, padding: '10px 0', flexShrink: 0, overflowY: 'auto' }}>
            <div style={{ padding: '2px 12px 6px', fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: t.textFaint }}>
              Augmentations
            </div>
            {augNodeDefs.map((def) => <AugPaletteItem key={def.type} def={def} />)}
            <div style={{ padding: '10px 12px 4px', marginTop: 8, borderTop: `1px solid ${t.borderSubtle}`, fontSize: 10, color: t.textDisabled, lineHeight: 1.5 }}>
              Drag transforms onto the canvas. Applied during training only.
            </div>
          </div>

          {/* Canvas + preview panel */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ flex: 1 }}>
              {active && <AugFlow />}
            </div>
            {/* Preview bar */}
            <div style={{ borderTop: `1px solid ${t.borderSubtle}`, background: t.bgPanel, padding: '8px 14px', display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
              <span style={{ fontSize: 11, color: t.textFaint }}>Preview image #</span>
              <input
                type="number" min={0} value={previewIndex}
                onChange={(e) => setPreviewIndex(Math.max(0, +e.target.value))}
                style={{ width: 60, background: t.bgElevated, border: `1px solid ${t.borderDefault}`, borderRadius: 4, color: t.textPrimary, fontSize: 11, padding: '3px 6px', outline: 'none' }}
              />
              <button
                onClick={handlePreview}
                disabled={previewLoading}
                style={{ padding: '4px 12px', borderRadius: 6, fontSize: 11, fontWeight: 600, border: '1px solid #06b6d4', background: previewLoading ? '#1a1a2e' : 'rgba(6,182,212,0.12)', color: previewLoading ? t.textFaint : '#67e8f9', cursor: previewLoading ? 'not-allowed' : 'pointer' }}
              >
                {previewLoading ? 'Generating…' : '⟡ Preview Augmentation'}
              </button>
              {previewError && <span style={{ fontSize: 11, color: '#fca5a5' }}>Error: {previewError}</span>}
              <div style={{ flex: 1 }} />
              {previewImg && (
                <img src={previewImg} alt="Aug preview" style={{ height: 80, borderRadius: 6, border: `1px solid ${t.borderDefault}` }} />
              )}
            </div>
          </div>
        </div>
      )}

      {/* Export sub-tab */}
      {subTab === 'export' && (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
          <div style={{ maxWidth: 500, padding: '32px 40px', background: t.bgPanel, border: `1px solid ${t.borderSubtle}`, borderRadius: 12, display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div>
              <div style={{ fontSize: 15, fontWeight: 600, color: t.textPrimary, marginBottom: 6 }}>Export to Training</div>
              <div style={{ fontSize: 12, color: t.textMuted, lineHeight: 1.6 }}>
                This sends your image dataset session (+ any augmentation steps you've built) to the Model page for training.
                Switch to the Model page and choose the neural network that processes your images.
              </div>
            </div>

            {/* Dataset summary */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {[
                { k: 'Dataset', v: cv.name },
                { k: 'Classes', v: cv.classNames.join(', ') },
                { k: 'Total images', v: cv.totalImages.toLocaleString() },
                { k: 'Input shape', v: `${cv.inputShape[0]}×${cv.inputShape[1]}×${cv.inputShape[2]}` },
                { k: 'Aug. nodes', v: augPipelineNodes.filter((n) => n.type !== 'augSource').length },
              ].map(({ k, v }) => (
                <div key={k}>
                  <div style={{ fontSize: 10, color: t.textFaint, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 2 }}>{k}</div>
                  <div style={{ fontSize: 12, color: t.textBody, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{v}</div>
                </div>
              ))}
            </div>

            {augPipelineNodes.filter((n) => n.type !== 'augSource').length === 0 && (
              <div style={{ padding: '8px 12px', background: 'rgba(245,158,11,0.08)', border: '1px solid #f59e0b30', borderRadius: 6, fontSize: 11, color: '#fcd34d', display: 'flex', gap: 6 }}>
                <span>⚠</span>
                <span>No augmentation steps configured — consider adding at least a Resize and Normalize node in the Augmentation tab.</span>
              </div>
            )}

            <button
              onClick={handleExportToTraining}
              style={{ padding: '10px 0', borderRadius: 8, fontSize: 13, fontWeight: 700, border: '1px solid #06b6d4', background: 'rgba(6,182,212,0.15)', color: '#67e8f9', cursor: 'pointer' }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(6,182,212,0.25)' }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(6,182,212,0.15)' }}
            >
              ⟡ Export to Model Training
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ── EEG mini chart (in-browser waveform preview) ──────────────────────────────

function EEGMiniChart({ data, sfreq, channels, colors }: { data: number[][], sfreq: number, channels: string[], colors: string[] }) {
  const maxPts = 512
  const step = Math.max(1, Math.floor(data[0]?.length / maxPts))
  const downsampled = data.map(ch => ch.filter((_, i) => i % step === 0))
  const times = downsampled[0]?.map((_, i) => (i * step) / sfreq) ?? []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {downsampled.map((ch, ci) => {
        const vals = ch
        const mn = Math.min(...vals), mx = Math.max(...vals), rng = mx - mn || 1
        const pts = vals.map((v, i) => `${(i / (vals.length - 1)) * 100},${100 - ((v - mn) / rng) * 100}`).join(' ')
        return (
          <div key={ci} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 9, color: colors[ci % colors.length], width: 40, textAlign: 'right', flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{channels[ci]}</span>
            <svg viewBox={`0 0 100 100`} preserveAspectRatio="none" style={{ flex: 1, height: 28 }}>
              <polyline points={pts} fill="none" stroke={colors[ci % colors.length]} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
            </svg>
          </div>
        )
      })}
      <div style={{ paddingLeft: 48, fontSize: 9, color: t.textDisabled, display: 'flex', justifyContent: 'space-between' }}>
        <span>0 s</span><span>{times[times.length - 1]?.toFixed(1)} s</span>
      </div>
    </div>
  )
}

function SectionLabel({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{ padding: '4px 14px 6px', fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: t.textFaint, ...style }}>
      {children}
    </div>
  )
}

// ── Pipeline view ─────────────────────────────────────────────────────────────

function PipelineView({ mobile, active = true, onOpenCode }: {
  mobile?: boolean
  active?: boolean
  onOpenCode?: () => void
}) {
  const [paletteOpen, setPaletteOpen] = useState(false)

  const palette = (
    <>
      <div style={{ padding: '2px 12px 6px', fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: t.textFaint }}>
        Transforms
      </div>
      {datasetNodeDefs.map((def) => (
        <PipelinePaletteItem key={def.type} def={def} />
      ))}
      <div style={{ padding: '10px 12px 4px', marginTop: 8, borderTop: `1px solid ${t.borderSubtle}`, fontSize: 10, color: t.textDisabled, lineHeight: 1.5 }}>
        Drag transforms onto the canvas
      </div>
    </>
  )

  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden', position: 'relative' }}>
      {mobile ? (
        <>
          <button
            type="button"
            onClick={() => setPaletteOpen(true)}
            style={{
              position: 'absolute', top: 10, left: 10, zIndex: 5,
              padding: '6px 10px', borderRadius: 6,
              border: `1px solid ${t.borderDefault}`, background: t.bgElevated,
              color: t.textBody, fontSize: 11, fontWeight: 600, cursor: 'pointer',
            }}
          >
            Transforms
          </button>
          {paletteOpen && (
            <>
              <div role="presentation" onClick={() => setPaletteOpen(false)} style={{ position: 'fixed', inset: 0, background: t.overlay, zIndex: 40 }} />
              <div style={{
                position: 'fixed', top: 52, left: 0, bottom: 0, width: 'min(280px, 88vw)',
                zIndex: 50, background: t.bgPanel, borderRight: `1px solid ${t.borderSubtle}`,
                overflowY: 'auto', padding: '10px 0',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0 12px 8px', alignItems: 'center' }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: t.textPrimary }}>Transforms</span>
                  <button type="button" onClick={() => setPaletteOpen(false)} style={{ background: 'none', border: 'none', color: t.textMuted, fontSize: 18, cursor: 'pointer' }}>×</button>
                </div>
                {palette}
              </div>
            </>
          )}
        </>
      ) : (
        <div style={{
          width: 190, background: t.bgPanel,
          borderRight: `1px solid ${t.borderSubtle}`,
          padding: '10px 0',
          flexShrink: 0,
          overflowY: 'auto',
        }}>
          {palette}
        </div>
      )}

      {/* Pipeline canvas — unmount React Flow when leaving pipeline tab or app view */}
      <div style={{ flex: 1, position: 'relative' }}>
        {onOpenCode && (
          <button
            type="button"
            onClick={onOpenCode}
            title="View generated Python pipeline code"
            style={{
              position: 'absolute',
              top: 10,
              right: 10,
              zIndex: 5,
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '6px 11px',
              borderRadius: 7,
              border: `1px solid ${t.accent}`,
              background: 'rgba(124,58,237,0.14)',
              color: t.accentMuted,
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              boxShadow: '0 4px 14px rgba(0,0,0,0.35)',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(124,58,237,0.28)' }}
            onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(124,58,237,0.14)' }}
          >
            View Python
          </button>
        )}
        {active && <DatasetFlow />}
      </div>
    </div>
  )
}

// ── Visualize view ────────────────────────────────────────────────────────────

interface ChartDef {
  id: string
  label: string
  needsY: boolean
  allowHue: boolean
  numX: boolean
  numY: boolean
}

const CHART_GROUPS = [
  { label: 'Distribution', ids: ['hist', 'kde', 'count'] },
  { label: 'Relational', ids: ['scatter', 'line', 'regression', 'joint'] },
  { label: 'Categorical', ids: ['bar', 'box', 'violin', 'strip', 'swarm', 'point'] },
  { label: 'Multi-variate', ids: ['heatmap', 'pairplot'] },
]

function VisualizeView({ dataset }: { dataset: NonNullable<ReturnType<typeof useDatasetStore.getState>['dataset']> }) {
  const cols = dataset.columns
  const numCols = cols.filter((c) => c.type === 'number')

  const [chartType, setChartType] = useState('scatter')
  const [xCol, setXCol] = useState(numCols[0]?.name ?? cols[0]?.name ?? '')
  const [yCol, setYCol] = useState(numCols[1]?.name ?? numCols[0]?.name ?? '')
  const [hueCol, setHueCol] = useState('')
  const [palette, setPalette] = useState('mako')
  const [chartDefs, setChartDefs] = useState<ChartDef[]>([])
  const [palettes, setPalettes] = useState<string[]>([])

  const [imgSrc, setImgSrc] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const activeDef = chartDefs.find((d) => d.id === chartType) ?? null

  // Fetch chart metadata from backend once on mount
  useEffect(() => {
    const STATIC_DEFS: ChartDef[] = [
      { id: 'scatter', label: 'Scatter', needsY: true, allowHue: true, numX: true, numY: true },
      { id: 'line', label: 'Line', needsY: true, allowHue: true, numX: true, numY: true },
      { id: 'regression', label: 'Regression', needsY: true, allowHue: false, numX: true, numY: true },
      { id: 'bar', label: 'Bar', needsY: true, allowHue: true, numX: false, numY: true },
      { id: 'box', label: 'Box', needsY: true, allowHue: true, numX: false, numY: true },
      { id: 'violin', label: 'Violin', needsY: true, allowHue: true, numX: false, numY: true },
      { id: 'strip', label: 'Strip', needsY: true, allowHue: true, numX: false, numY: true },
      { id: 'swarm', label: 'Swarm', needsY: true, allowHue: true, numX: false, numY: true },
      { id: 'point', label: 'Point', needsY: true, allowHue: true, numX: false, numY: true },
      { id: 'count', label: 'Count', needsY: false, allowHue: true, numX: false, numY: false },
      { id: 'hist', label: 'Histogram', needsY: false, allowHue: true, numX: true, numY: false },
      { id: 'kde', label: 'KDE', needsY: false, allowHue: true, numX: true, numY: false },
      { id: 'joint', label: 'Joint', needsY: true, allowHue: false, numX: true, numY: true },
      { id: 'heatmap', label: 'Correlation Map', needsY: false, allowHue: false, numX: false, numY: false },
      { id: 'pairplot', label: 'Pair Plot', needsY: false, allowHue: true, numX: false, numY: false },
    ]
    setChartDefs(STATIC_DEFS)
    setPalettes(['mako', 'rocket', 'viridis', 'plasma', 'magma', 'Set2', 'tab10', 'cividis', 'flare', 'crest', 'icefire', 'husl'])

    fetch(`${API_BASE}/api/plot/meta`)
      .then((r) => r.json())
      .then((data: { chartDefs: ChartDef[]; palettes: string[] }) => {
        setChartDefs(data.chartDefs)
        setPalettes(data.palettes)
      })
      .catch(() => { /* use static fallback already set above */ })
  }, [])

  async function handlePlot() {
    setLoading(true)
    setError(null)
    try {
      const rows = dataset.rows.slice(0, MAX_PLOT_ROWS)
      const res = await fetch(`${API_BASE}/api/plot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rows,
          config: {
            chartType,
            xCol: xCol || null,
            yCol: (activeDef?.needsY ? yCol : null) || null,
            hueCol: hueCol || null,
            palette,
            maxRows: MAX_PLOT_ROWS,
          },
        }),
      })
      if (!res.ok) {
        const json = await res.json().catch(() => null)
        if (!json) {
          setError(`HTTP ${res.status}`)
        } else if (json.error) {
          setError(json.error)
        } else if (json.detail) {
          // FastAPI Pydantic validation errors come as detail[]
          const detail = Array.isArray(json.detail)
            ? json.detail.map((d: { msg?: string; loc?: string[] }) => `${d.loc?.join('.')} — ${d.msg}`).join('\n')
            : String(json.detail)
          setError(detail)
        } else {
          setError(JSON.stringify(json))
        }
        setLoading(false)
        return
      }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      setImgSrc((prev) => { if (prev) URL.revokeObjectURL(prev); return url })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to reach backend')
    } finally {
      setLoading(false)
    }
  }

  function handleDownload() {
    if (!imgSrc) return
    const a = document.createElement('a')
    a.href = imgSrc
    a.download = `${chartType}_${xCol}_${yCol || 'plot'}.png`
    a.click()
  }

  const xOptions = activeDef?.numX ? numCols : cols
  const yOptions = activeDef?.numY ? numCols : cols

  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

      {/* ── Left panel: controls ─────────────────────────────────────────── */}
      <div style={{
        width: 240, background: t.bgPanel,
        borderRight: `1px solid ${t.borderSubtle}`,
        overflowY: 'auto', flexShrink: 0,
        display: 'flex', flexDirection: 'column',
      }}>
        {/* Chart type picker */}
        <div style={{ padding: '12px 12px 6px' }}>
          <VizLabel>Chart Type</VizLabel>
          {CHART_GROUPS.map((group) => {
            const defs = chartDefs.filter((d) => group.ids.includes(d.id))
            if (defs.length === 0) return null
            return (
              <div key={group.label} style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: t.textDisabled, marginBottom: 4 }}>
                  {group.label}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {defs.map((d) => (
                    <button
                      key={d.id}
                      onClick={() => setChartType(d.id)}
                      style={{
                        padding: '3px 9px', borderRadius: 5, fontSize: 11,
                        border: chartType === d.id ? '1px solid #7c3aed' : `1px solid ${t.borderDefault}`,
                        background: chartType === d.id ? 'rgba(124,58,237,0.15)' : 'transparent',
                        color: chartType === d.id ? t.accentMuted : t.textMuted,
                        cursor: 'pointer', fontWeight: chartType === d.id ? 600 : 400,
                        transition: 'all 0.1s',
                      }}
                    >
                      {d.label}
                    </button>
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        <div style={{ borderTop: `1px solid ${t.borderSubtle}`, padding: '12px 12px', display: 'flex', flexDirection: 'column', gap: 10 }}>

          {/* X Axis */}
          {activeDef?.id !== 'heatmap' && activeDef?.id !== 'pairplot' && (
            <VizSelect
              label={`X Axis${activeDef?.numX ? ' (numeric)' : ''}`}
              value={xCol}
              onChange={setXCol}
              options={xOptions.map((c) => c.name)}
            />
          )}

          {/* Y Axis */}
          {activeDef?.needsY && (
            <VizSelect
              label={`Y Axis${activeDef?.numY ? ' (numeric)' : ''}`}
              value={yCol}
              onChange={setYCol}
              options={yOptions.map((c) => c.name)}
            />
          )}

          {/* Hue */}
          {activeDef?.allowHue && (
            <VizSelect
              label="Colour by (hue)"
              value={hueCol}
              onChange={setHueCol}
              options={cols.map((c) => c.name)}
              optional
            />
          )}

          {/* Palette */}
          <div>
            <VizLabel>Palette</VizLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {palettes.slice(0, 12).map((p) => (
                <button
                  key={p}
                  onClick={() => setPalette(p)}
                  title={p}
                  style={{
                    padding: '2px 7px', borderRadius: 4, fontSize: 10,
                    border: palette === p ? '1px solid #7c3aed' : `1px solid ${t.borderDefault}`,
                    background: palette === p ? 'rgba(124,58,237,0.15)' : t.bgElevated,
                    color: palette === p ? t.accentMuted : t.textFaint,
                    cursor: 'pointer', transition: 'all 0.1s',
                  }}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div style={{ flex: 1 }} />

        {/* Plot button */}
        <div style={{ padding: 12, borderTop: `1px solid ${t.borderSubtle}` }}>
          {dataset.rows.length > MAX_PLOT_ROWS && (
            <p style={{ fontSize: 10, color: t.textFaint, marginBottom: 8 }}>
              Sampling {MAX_PLOT_ROWS.toLocaleString()} of {dataset.rows.length.toLocaleString()} rows
            </p>
          )}
          <button
            onClick={handlePlot}
            disabled={loading}
            style={{
              width: '100%', padding: '8px 0', borderRadius: 7,
              border: `1px solid ${t.accent}`,
              background: loading ? '#1a1a2e' : 'rgba(124,58,237,0.15)',
              color: loading ? t.textDisabled : t.accentMuted,
              fontSize: 13, fontWeight: 600, cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'background 0.12s',
            }}
            onMouseEnter={(e) => { if (!loading) e.currentTarget.style.background = 'rgba(124,58,237,0.28)' }}
            onMouseLeave={(e) => { if (!loading) e.currentTarget.style.background = 'rgba(124,58,237,0.15)' }}
          >
            {loading ? 'Generating…' : '⟡ Plot'}
          </button>
        </div>
      </div>

      {/* ── Right panel: chart display ───────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: t.bgBase }}>

        {/* Toolbar above chart */}
        {imgSrc && !loading && (
          <div style={{
            height: 36, display: 'flex', alignItems: 'center',
            padding: '0 14px', gap: 8, borderBottom: `1px solid ${t.borderSubtle}`,
            background: t.bgPanel, flexShrink: 0,
          }}>
            <span style={{ fontSize: 11, color: t.textFaint }}>
              {chartType} · {xCol}{yCol ? ` × ${yCol}` : ''}{hueCol ? ` · hue: ${hueCol}` : ''}
            </span>
            <div style={{ flex: 1 }} />
            <button
              onClick={handleDownload}
              style={{
                padding: '3px 10px', borderRadius: 5, fontSize: 11,
                border: `1px solid ${t.borderDefault}`, background: 'transparent',
                color: t.textMuted, cursor: 'pointer',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.color = t.accentMuted; e.currentTarget.style.borderColor = t.accent }}
              onMouseLeave={(e) => { e.currentTarget.style.color = t.textMuted; e.currentTarget.style.borderColor = t.borderDefault }}
            >
              ↓ Save PNG
            </button>
          </div>
        )}

        <div style={{ flex: 1, overflow: 'auto', display: 'flex', alignItems: 'flex-start', justifyContent: 'center', padding: 24 }}>
          {loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, marginTop: 80 }}>
              <Spinner />
              <span style={{ fontSize: 12, color: t.textFaint }}>Generating plot…</span>
            </div>
          ) : error ? (
            <div style={{
              maxWidth: 420, padding: '14px 18px',
              background: 'rgba(239,68,68,0.08)', border: '1px solid #ef444430',
              borderRadius: 8, fontSize: 13, color: '#fca5a5', lineHeight: 1.5,
            }}>
              <strong style={{ display: 'block', marginBottom: 4 }}>Plot error</strong>
              {error}
            </div>
          ) : imgSrc ? (
            <img
              src={imgSrc}
              alt="Generated chart"
              style={{
                maxWidth: '100%', borderRadius: 10,
                border: `1px solid ${t.borderSubtle}`,
                boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
              }}
            />
          ) : (
            <div style={{ textAlign: 'center', marginTop: 80 }}>
              <div style={{ fontSize: 40, marginBottom: 14, opacity: 0.2 }}>⟡</div>
              <p style={{ fontSize: 13, color: t.textFaint, margin: '0 0 6px' }}>
                Configure a chart and click <strong style={{ color: t.textMuted }}>Plot</strong>
              </p>
              <p style={{ fontSize: 11, color: t.textDisabled, margin: 0 }}>
                {dataset.rows.length.toLocaleString()} rows · {cols.length} columns available
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Visualize sub-components ──────────────────────────────────────────────────

function VizLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: t.textFaint, marginBottom: 5 }}>
      {children}
    </div>
  )
}

function VizSelect({ label, value, onChange, options, optional }: {
  label: string
  value: string
  onChange: (v: string) => void
  options: string[]
  optional?: boolean
}) {
  return (
    <div>
      <VizLabel>{label}</VizLabel>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: '100%', background: t.bgElevated, border: `1px solid ${t.borderDefault}`,
          borderRadius: 5, color: t.textPrimary, fontSize: 11,
          padding: '4px 7px', outline: 'none', cursor: 'pointer',
        }}
      >
        {optional && <option value="" style={{ background: t.bgElevated }}>— none —</option>}
        {options.map((o) => <option key={o} value={o} style={{ background: t.bgElevated }}>{o}</option>)}
      </select>
    </div>
  )
}

function Spinner({ color = t.accent }: { color?: string }) {
  return (
    <div style={{
      width: 18, height: 18, borderRadius: '50%',
      border: '2px solid #27272a',
      borderTopColor: color,
      animation: 'spin 0.7s linear infinite',
      flexShrink: 0,
    }} />
  )
}

// Inject spinner keyframe
if (typeof document !== 'undefined') {
  const id = 'oneiros-spinner-style'
  if (!document.getElementById(id)) {
    const s = document.createElement('style')
    s.id = id
    s.textContent = '@keyframes spin { to { transform: rotate(360deg); } }'
    document.head.appendChild(s)
  }
}

// ── Column card ───────────────────────────────────────────────────────────────

function ColumnCard({ col }: { col: ColumnInfo }) {
  return (
    <div style={{ padding: '8px 12px', borderBottom: '1px solid #1a1a20' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 4 }}>
        <TypeDot type={col.type} />
        <span style={{ fontSize: 11, fontWeight: 500, color: t.textBody, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {col.name}
        </span>
      </div>
      <div style={{ fontSize: 10, color: t.textFaint }}>
        {col.type === 'number' && (
          <>
            <span>{col.min?.toFixed(2)} – {col.max?.toFixed(2)}</span>
            <span style={{ margin: '0 4px', color: t.borderDefault }}>·</span>
            <span>μ {col.mean?.toFixed(2)}</span>
          </>
        )}
        {col.type === 'string' && (
          <span>{col.uniqueCount} unique</span>
        )}
        {col.nullCount > 0 && (
          <span style={{ color: '#f87171', marginLeft: 4 }}>{col.nullCount} null</span>
        )}
      </div>
      {col.type === 'number' && (
        <NullBar nullFrac={col.nullCount / (col.uniqueCount + col.nullCount)} />
      )}
    </div>
  )
}

function NullBar({ nullFrac }: { nullFrac: number }) {
  if (nullFrac === 0) return null
  return (
    <div style={{ marginTop: 4, height: 3, background: t.borderDefault, borderRadius: 2, overflow: 'hidden' }}>
      <div style={{ height: '100%', width: `${nullFrac * 100}%`, background: '#f87171', borderRadius: 2 }} />
    </div>
  )
}

function TypeDot({ type }: { type: string }) {
  const color = type === 'number' ? '#a5b4fc' : type === 'boolean' ? '#86efac' : '#fcd34d'
  return <span style={{ width: 6, height: 6, borderRadius: '50%', background: color, flexShrink: 0, display: 'inline-block' }} />
}

// ── Table styles ──────────────────────────────────────────────────────────────

const thStyle: React.CSSProperties = {
  padding: '8px 12px',
  textAlign: 'left',
  fontWeight: 500,
  fontSize: 11,
  color: t.textMuted,
  borderBottom: `1px solid ${t.borderSubtle}`,
  whiteSpace: 'nowrap',
  letterSpacing: '0.02em',
}

const tdStyle: React.CSSProperties = {
  padding: '5px 12px',
  borderBottom: '1px solid rgba(255,255,255,0.03)',
  whiteSpace: 'nowrap',
  maxWidth: 200,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
}

const indexCellStyle: React.CSSProperties = {
  ...tdStyle,
  color: t.textDisabled,
  fontSize: 10,
  textAlign: 'right',
  paddingRight: 8,
  minWidth: 36,
}

// ── Shared components ─────────────────────────────────────────────────────────

function TabBtn({ active, onClick, icon, label, accent, cv, merge }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string; accent?: boolean; cv?: boolean; merge?: boolean }) {
  const bg = active ? (merge ? 'rgba(34,197,94,0.18)' : cv ? 'rgba(6,182,212,0.18)' : accent ? 'rgba(245,158,11,0.18)' : t.borderDefault) : 'transparent'
  const color = active ? (merge ? '#86efac' : cv ? '#67e8f9' : accent ? '#fbbf24' : t.textPrimary) : t.textMuted
  return (
    <button
      onClick={onClick}
      style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '4px 9px', borderRadius: 4, border: 'none', background: bg, color, fontSize: 11, fontWeight: 500, cursor: 'pointer' }}
    >
      {icon}{label}
    </button>
  )
}

function UploadButton({ label, onClick, accent, edf, cv, merge }: { label: string; onClick: () => void; accent?: boolean; edf?: boolean; cv?: boolean; merge?: boolean }) {
  const borderCol = merge ? '#22c55e' : cv ? '#06b6d4' : edf ? '#8b5cf6' : accent ? '#0ea5e9' : t.borderDefault
  const bgCol     = merge ? 'rgba(34,197,94,0.08)' : cv ? 'rgba(6,182,212,0.08)' : edf ? 'rgba(139,92,246,0.08)' : accent ? 'rgba(14,165,233,0.08)' : 'transparent'
  const textCol   = merge ? '#86efac' : cv ? '#67e8f9' : edf ? '#c4b5fd' : accent ? '#7dd3fc' : t.textSecondary
  const bgHov     = merge ? 'rgba(34,197,94,0.15)' : cv ? 'rgba(6,182,212,0.15)' : edf ? 'rgba(139,92,246,0.15)' : accent ? 'rgba(14,165,233,0.15)' : t.borderDefault
  const textHov   = merge ? '#bbf7d0' : cv ? '#a5f3fc' : edf ? '#ddd6fe' : accent ? '#bae6fd' : t.textPrimary
  return (
    <button
      onClick={onClick}
      style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '5px 10px', borderRadius: 6, border: `1px solid ${borderCol}`, background: bgCol, color: textCol, fontSize: 12, fontWeight: 500, cursor: 'pointer', flexShrink: 0 }}
      onMouseEnter={(e) => { e.currentTarget.style.background = bgHov; e.currentTarget.style.color = textHov }}
      onMouseLeave={(e) => { e.currentTarget.style.background = bgCol; e.currentTarget.style.color = textCol }}
    >
      <UploadIcon />{label}
    </button>
  )
}
