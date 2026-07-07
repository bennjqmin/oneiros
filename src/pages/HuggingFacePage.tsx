import { t } from '../theme/tokens'
import { useCallback, useEffect, useState } from 'react'
import { useHuggingFaceStore } from '../store/useHuggingFaceStore'
import { useProjectStore } from '../store/useProjectStore'
import { LoadingLabel } from '../components/panelChrome'
import { useThemeSync } from '../theme/useThemeSync'

const POPULAR_MODELS = [
  'distilbert-base-uncased',
  'bert-base-uncased',
  'google/vit-base-patch16-224',
  'sentence-transformers/all-MiniLM-L6-v2',
  'facebook/deit-base-distilled-patch16-224',
  'microsoft/resnet-50',
]

function SearchIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  )
}

function PlusIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
      <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  )
}

function TrashIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    </svg>
  )
}

function StatusBanner({
  ready,
  error,
  onRetry,
}: {
  ready: boolean | null
  error: string | null
  onRetry: () => void
}) {
  if (ready === null && !error) return null

  const bg = ready ? t.successSubtle : t.errorSubtle
  const border = ready ? 'var(--success)' : t.error
  const color = ready ? t.success : t.error

  return (
    <div style={{
      margin: '0 0 16px',
      padding: '10px 14px',
      borderRadius: 8,
      border: `1px solid ${border}`,
      background: bg,
      display: 'flex',
      alignItems: 'flex-start',
      justifyContent: 'space-between',
      gap: 12,
    }}>
      <div>
        <div style={{ fontSize: 12, fontWeight: 600, color }}>
          {ready ? 'Backend ready' : 'Backend issue'}
        </div>
        <div style={{ fontSize: 11, color: t.textSecondary, marginTop: 4, lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>
          {ready
            ? 'transformers and huggingface_hub are installed. You can search and import models.'
            : error ?? 'Hugging Face dependencies are missing on the backend.'}
        </div>
      </div>
      {!ready && (
        <button
          type="button"
          onClick={onRetry}
          style={{
            flexShrink: 0, padding: '5px 10px', borderRadius: 6,
            border: `1px solid ${t.borderDefault}`, background: t.bgElevated,
            color: t.textBody, fontSize: 11, cursor: 'pointer',
          }}
        >
          Retry
        </button>
      )}
    </div>
  )
}

function ErrorBox({ message, onDismiss }: { message: string; onDismiss?: () => void }) {
  return (
    <div style={{
      marginTop: 12, padding: '10px 12px', borderRadius: 8,
      background: t.errorSubtle, border: `1px solid ${t.error}`,
    }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: t.error, marginBottom: 4 }}>Error</div>
      <div style={{ fontSize: 11, color: t.textBody, lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>{message}</div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          style={{
            marginTop: 8, padding: '4px 8px', borderRadius: 4, border: 'none',
            background: 'transparent', color: t.textMuted, fontSize: 10, cursor: 'pointer',
          }}
        >
          Dismiss
        </button>
      )}
    </div>
  )
}

interface HuggingFacePageProps {
  mobile?: boolean
  active?: boolean
  onSwitchToModel?: () => void
}

export default function HuggingFacePage({ mobile, active = true, onSwitchToModel }: HuggingFacePageProps) {
  useThemeSync()

  const projectId = useProjectStore((s) => s.currentProjectId)
  const backendStatus = useHuggingFaceStore((s) => s.backendStatus)
  const backendError = useHuggingFaceStore((s) => s.backendError)
  const hfToken = useHuggingFaceStore((s) => s.hfToken)
  const importedModels = useHuggingFaceStore((s) => s.importedModels)
  const loadState = useHuggingFaceStore((s) => s.loadState)
  const lastError = useHuggingFaceStore((s) => s.lastError)
  const searchResults = useHuggingFaceStore((s) => s.searchResults)
  const refreshBackendStatus = useHuggingFaceStore((s) => s.refreshBackendStatus)
  const loadLibraryForProject = useHuggingFaceStore((s) => s.loadLibraryForProject)
  const searchModels = useHuggingFaceStore((s) => s.searchModels)
  const validateAndImport = useHuggingFaceStore((s) => s.validateAndImport)
  const removeImported = useHuggingFaceStore((s) => s.removeImported)
  const addToCanvas = useHuggingFaceStore((s) => s.addToCanvas)
  const setHfToken = useHuggingFaceStore((s) => s.setHfToken)
  const clearError = useHuggingFaceStore((s) => s.clearError)

  const [modelIdInput, setModelIdInput] = useState('')
  const [searchInput, setSearchInput] = useState('')
  const [trustRemoteCode, setTrustRemoteCode] = useState(false)
  const [freeze, setFreeze] = useState(false)
  const [pooling, setPooling] = useState<'mean' | 'cls' | 'pooler'>('mean')
  const [importSuccess, setImportSuccess] = useState<string | null>(null)

  useEffect(() => {
    if (!active) return
    loadLibraryForProject(projectId)
    refreshBackendStatus()
  }, [active, projectId, loadLibraryForProject, refreshBackendStatus])

  const busy = loadState === 'checking' || loadState === 'searching' || loadState === 'validating' || loadState === 'importing'

  const handleSearch = useCallback((e?: React.FormEvent) => {
    e?.preventDefault()
    searchModels(searchInput)
  }, [searchInput, searchModels])

  const handleImport = useCallback(async (modelId?: string) => {
    setImportSuccess(null)
    clearError()
    const id = (modelId ?? modelIdInput).trim()
    if (!id) return
    try {
      const entry = await validateAndImport(id, { trustRemoteCode, pooling, freeze })
      setImportSuccess(`Imported ${entry.modelId} (${entry.outputFeatures} features)`)
      setModelIdInput('')
    } catch {
      /* error stored in store */
    }
  }, [modelIdInput, trustRemoteCode, pooling, freeze, validateAndImport, clearError])

  if (!active) return null

  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      background: t.bgBase,
      minHeight: 0,
    }}>
      <div style={{
        padding: mobile ? '12px 14px 0' : '16px 20px 0',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
          <h1 style={{ margin: 0, fontSize: mobile ? 16 : 18, fontWeight: 700, color: t.textPrimary }}>
            Hugging Face
          </h1>
          <span style={{
            fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em',
            padding: '2px 7px', borderRadius: 4,
            background: 'rgba(255,165,0,0.12)', border: '1px solid rgba(255,165,0,0.35)', color: '#fdba74',
          }}>
            Hub
          </span>
        </div>
        <p style={{ margin: '0 0 12px', fontSize: 12, color: t.textSecondary, lineHeight: 1.5, maxWidth: 720 }}>
          Search Hugging Face Hub, validate a model, import it into your project, then add it to the model canvas for training.
        </p>
        <StatusBanner
          ready={backendStatus?.ready ?? null}
          error={backendError}
          onRetry={() => refreshBackendStatus()}
        />
      </div>

      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: mobile ? 'column' : 'row',
        gap: mobile ? 0 : 1,
        overflow: 'hidden',
        minHeight: 0,
        borderTop: `1px solid ${t.borderSubtle}`,
      }}>
        {/* Search panel */}
        <section style={{
          width: mobile ? '100%' : 300,
          flexShrink: 0,
          background: t.bgPanel,
          borderRight: mobile ? 'none' : `1px solid ${t.borderSubtle}`,
          borderBottom: mobile ? `1px solid ${t.borderSubtle}` : 'none',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          maxHeight: mobile ? 240 : undefined,
        }}>
          <div style={{ padding: '12px 14px', borderBottom: `1px solid ${t.borderSubtle}` }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: t.textMuted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Search Hub
            </div>
            <form onSubmit={handleSearch} style={{ display: 'flex', gap: 6 }}>
              <input
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder="bert, vit, distilbert…"
                style={{
                  flex: 1, padding: '7px 10px', borderRadius: 6,
                  border: `1px solid ${t.borderDefault}`, background: t.bgInput,
                  color: t.textPrimary, fontSize: 12,
                }}
              />
              <button
                type="submit"
                disabled={busy}
                style={{
                  padding: '7px 10px', borderRadius: 6, border: 'none',
                  background: t.accent, color: '#fff', cursor: busy ? 'wait' : 'pointer',
                  display: 'flex', alignItems: 'center',
                }}
              >
                <SearchIcon />
              </button>
            </form>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '8px 10px' }}>
            {loadState === 'searching' && <LoadingLabel label="Searching…" />}
            {loadState !== 'searching' && searchResults.length === 0 && searchInput.trim() && (
              <div style={{ fontSize: 11, color: t.textMuted, padding: 8 }}>No models found.</div>
            )}
            {searchResults.map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => { setModelIdInput(m.id); clearError() }}
                style={{
                  width: '100%', textAlign: 'left', padding: '8px 10px', marginBottom: 4,
                  borderRadius: 6, border: `1px solid ${t.borderSubtle}`, background: t.bgElevated,
                  cursor: 'pointer', color: t.textBody,
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 600, color: t.textPrimary, wordBreak: 'break-all' }}>{m.id}</div>
                <div style={{ fontSize: 10, color: t.textMuted, marginTop: 2 }}>
                  {m.pipelineTag ?? 'unknown task'}
                  {m.gated ? ' · gated' : ''}
                  {m.downloads != null ? ` · ${(m.downloads / 1000).toFixed(0)}k dl` : ''}
                </div>
              </button>
            ))}
            <div style={{ marginTop: 12, fontSize: 10, color: t.textFaint, padding: '0 4px' }}>Popular</div>
            {POPULAR_MODELS.map((id) => (
              <button
                key={id}
                type="button"
                onClick={() => { setModelIdInput(id); clearError() }}
                style={{
                  width: '100%', textAlign: 'left', padding: '6px 8px', marginTop: 4,
                  borderRadius: 4, border: 'none', background: 'transparent',
                  color: t.textSecondary, fontSize: 11, cursor: 'pointer',
                }}
              >
                {id}
              </button>
            ))}
          </div>
        </section>

        {/* Import panel */}
        <section style={{
          flex: 1,
          overflowY: 'auto',
          padding: mobile ? 14 : '16px 20px',
          minWidth: 0,
        }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: t.textMuted, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Import model
          </div>

          <label style={{ display: 'block', fontSize: 11, color: t.textSecondary, marginBottom: 4 }}>
            Model ID
          </label>
          <input
            value={modelIdInput}
            onChange={(e) => { setModelIdInput(e.target.value); clearError(); setImportSuccess(null) }}
            placeholder="distilbert-base-uncased or google/vit-base-patch16-224"
            style={{
              width: '100%', boxSizing: 'border-box', padding: '9px 11px', borderRadius: 7,
              border: `1px solid ${t.borderDefault}`, background: t.bgInput,
              color: t.textPrimary, fontSize: 13, marginBottom: 12,
            }}
          />

          <label style={{ display: 'block', fontSize: 11, color: t.textSecondary, marginBottom: 4 }}>
            HF access token (optional — for gated/private models)
          </label>
          <input
            type="password"
            value={hfToken}
            onChange={(e) => setHfToken(e.target.value)}
            placeholder="hf_…"
            autoComplete="off"
            style={{
              width: '100%', boxSizing: 'border-box', padding: '8px 11px', borderRadius: 7,
              border: `1px solid ${t.borderDefault}`, background: t.bgInput,
              color: t.textPrimary, fontSize: 12, marginBottom: 12,
            }}
          />

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 14 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: t.textBody, cursor: 'pointer' }}>
              <input type="checkbox" checked={trustRemoteCode} onChange={(e) => setTrustRemoteCode(e.target.checked)} />
              Trust remote code
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: t.textBody, cursor: 'pointer' }}>
              <input type="checkbox" checked={freeze} onChange={(e) => setFreeze(e.target.checked)} />
              Freeze weights
            </label>
          </div>

          <label style={{ display: 'block', fontSize: 11, color: t.textSecondary, marginBottom: 4 }}>
            Pooling
          </label>
          <select
            value={pooling}
            onChange={(e) => setPooling(e.target.value as typeof pooling)}
            style={{
              width: '100%', maxWidth: 320, padding: '8px 10px', borderRadius: 7,
              border: `1px solid ${t.borderDefault}`, background: t.bgInput,
              color: t.textPrimary, fontSize: 12, marginBottom: 16,
            }}
          >
            <option value="mean">Mean pool (sequence models)</option>
            <option value="cls">CLS token (BERT-style)</option>
            <option value="pooler">Pooler output</option>
          </select>

          <button
            type="button"
            disabled={busy || !modelIdInput.trim()}
            onClick={() => handleImport()}
            style={{
              padding: '9px 16px', borderRadius: 7, border: 'none',
              background: busy || !modelIdInput.trim() ? t.bgElevated : t.accent,
              color: busy || !modelIdInput.trim() ? t.textDisabled : '#fff',
              fontSize: 12, fontWeight: 600, cursor: busy ? 'wait' : 'pointer',
            }}
          >
            {loadState === 'validating' ? 'Validating…' : loadState === 'importing' ? 'Importing…' : 'Validate & import'}
          </button>

          {lastError && <ErrorBox message={lastError} onDismiss={clearError} />}
          {importSuccess && (
            <div style={{
              marginTop: 12, padding: '10px 12px', borderRadius: 8,
              background: t.successSubtle, border: `1px solid ${t.success}`,
              fontSize: 11, color: t.success,
            }}>
              {importSuccess}
            </div>
          )}

          <div style={{
            marginTop: 24, padding: 14, borderRadius: 8,
            background: t.bgElevated, border: `1px solid ${t.borderSubtle}`,
            fontSize: 11, color: t.textSecondary, lineHeight: 1.6,
          }}>
            <strong style={{ color: t.textBody }}>Input wiring tips</strong>
            <ul style={{ margin: '8px 0 0', paddingLeft: 18 }}>
              <li>Text models: flat input as token IDs — e.g. channels=128, height=1, width=1 for seq length 128.</li>
              <li>Vision models (ViT): spatial RGB input — channels=3 with matching H×W (often 224×224).</li>
              <li>First run downloads weights from Hugging Face; gated models need a token.</li>
            </ul>
          </div>
        </section>

        {/* Library panel */}
        <section style={{
          width: mobile ? '100%' : 280,
          flexShrink: 0,
          background: t.bgPanel,
          borderLeft: mobile ? 'none' : `1px solid ${t.borderSubtle}`,
          borderTop: mobile ? `1px solid ${t.borderSubtle}` : 'none',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          minHeight: mobile ? 200 : undefined,
        }}>
          <div style={{
            padding: '12px 14px', borderBottom: `1px solid ${t.borderSubtle}`,
            fontSize: 11, fontWeight: 600, color: t.textMuted, textTransform: 'uppercase', letterSpacing: '0.06em',
          }}>
            Imported ({importedModels.length})
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 12px' }}>
            {importedModels.length === 0 && (
              <div style={{ fontSize: 11, color: t.textMuted, lineHeight: 1.5, padding: '8px 4px' }}>
                No models yet. Validate a model ID to add it here, then place it on the canvas.
              </div>
            )}
            {importedModels.map((m) => (
              <div
                key={m.id}
                style={{
                  padding: '10px 12px', marginBottom: 8, borderRadius: 8,
                  border: `1px solid ${t.borderDefault}`, background: t.bgElevated,
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 600, color: t.textPrimary, wordBreak: 'break-all' }}>
                  {m.modelId}
                </div>
                <div style={{ fontSize: 10, color: t.textMuted, marginTop: 4 }}>
                  {m.outputFeatures} features · {m.modelKind} · {m.pooling} pool
                </div>
                <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
                  <button
                    type="button"
                    onClick={() => addToCanvas(m, onSwitchToModel)}
                    style={{
                      flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
                      padding: '6px 8px', borderRadius: 6, border: 'none',
                      background: t.accent, color: '#fff', fontSize: 11, fontWeight: 600, cursor: 'pointer',
                    }}
                  >
                    <PlusIcon /> Canvas
                  </button>
                  <button
                    type="button"
                    onClick={() => removeImported(m.id)}
                    title="Remove from library"
                    style={{
                      padding: '6px 8px', borderRadius: 6,
                      border: `1px solid ${t.borderDefault}`, background: t.bgInput,
                      color: t.textMuted, cursor: 'pointer',
                    }}
                  >
                    <TrashIcon />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
