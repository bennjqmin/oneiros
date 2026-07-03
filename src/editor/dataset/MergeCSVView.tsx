import { useMemo, useRef, useState } from 'react'
import { t } from '../../theme/tokens'
import { useDatasetStore } from '../../store/useDatasetStore'
import {
  parseCSVFile,
  validateCSVColumns,
  concatRows,
  downloadCSV,
  mergedDatasetName,
  type ParsedCSVFile,
} from './mergeCsv'

function TrashIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
    </svg>
  )
}

function UploadIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )
}

export default function MergeCSVView({ onMerged }: { onMerged: () => void }) {
  const loadMergedDataset = useDatasetStore((s) => s.loadMergedDataset)
  const inputRef = useRef<HTMLInputElement>(null)
  const [files, setFiles] = useState<ParsedCSVFile[]>([])
  const [parsing, setParsing] = useState(false)
  const [parseError, setParseError] = useState<string | null>(null)
  const [merging, setMerging] = useState(false)
  const [mergedRows, setMergedRows] = useState<Record<string, unknown>[] | null>(null)

  const validation = useMemo(() => validateCSVColumns(files), [files])
  const canMerge = files.length >= 2 && validation.ok && !parsing && !merging
  const totalRows = files.reduce((sum, f) => sum + f.rowCount, 0)
  const previewRows = mergedRows ?? (validation.ok && files.length >= 2 ? concatRows(files) : null)

  async function handleFilesSelected(selected: FileList | null) {
    if (!selected || selected.length === 0) return
    setParseError(null)
    setParsing(true)
    setMergedRows(null)
    try {
      const parsed = await Promise.all(Array.from(selected).map(parseCSVFile))
      setFiles((prev) => [...prev, ...parsed])
    } catch (err) {
      setParseError(err instanceof Error ? err.message : 'Failed to parse CSV file.')
    } finally {
      setParsing(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index))
    setMergedRows(null)
  }

  function clearAll() {
    setFiles([])
    setMergedRows(null)
    setParseError(null)
  }

  async function handleMerge() {
    if (!canMerge) return
    setMerging(true)
    const rows = concatRows(files)
    const name = mergedDatasetName(files)
    loadMergedDataset(name, rows)
    setMergedRows(rows)
    setMerging(false)
    onMerged()
  }

  function handleExport() {
    if (!previewRows || previewRows.length === 0) return
    const name = files.length > 0 ? mergedDatasetName(files) : 'merged'
    downloadCSV(previewRows, name)
  }

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{
        padding: '12px 16px',
        background: t.bgPanel,
        borderBottom: `1px solid ${t.borderSubtle}`,
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        flexWrap: 'wrap',
        flexShrink: 0,
      }}>
        <span style={{ fontSize: 12, color: t.textSecondary, flex: 1, minWidth: 200 }}>
          Upload CSV files with identical columns to merge them into one dataset.
        </span>
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          multiple
          onChange={(e) => handleFilesSelected(e.target.files)}
          style={{ display: 'none' }}
        />
        <button
          onClick={() => inputRef.current?.click()}
          disabled={parsing}
          style={{
            display: 'flex', alignItems: 'center', gap: 5,
            padding: '5px 10px', borderRadius: 6,
            border: `1px solid ${t.borderDefault}`,
            background: 'rgba(34,197,94,0.08)',
            color: '#86efac',
            fontSize: 12, fontWeight: 500, cursor: parsing ? 'wait' : 'pointer',
            opacity: parsing ? 0.7 : 1,
          }}
        >
          <UploadIcon /> Add CSV files
        </button>
        {files.length > 0 && (
          <button
            onClick={clearAll}
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              padding: '5px 10px', borderRadius: 6,
              border: `1px solid ${t.borderDefault}`,
              background: 'transparent',
              color: t.textMuted,
              fontSize: 12, cursor: 'pointer',
            }}
          >
            <TrashIcon /> Clear all
          </button>
        )}
      </div>

      {parsing && (
        <div style={{ background: 'rgba(34,197,94,0.08)', borderBottom: `1px solid rgba(34,197,94,0.15)`, padding: '6px 16px', fontSize: 12, color: '#86efac', flexShrink: 0 }}>
          Parsing CSV files…
        </div>
      )}
      {parseError && (
        <div style={{ background: 'rgba(239,68,68,0.07)', borderBottom: '1px solid #ef444420', padding: '6px 16px', fontSize: 12, color: '#fca5a5', flexShrink: 0 }}>
          {parseError}
        </div>
      )}
      {!validation.ok && files.length > 1 && (
        <div style={{ background: 'rgba(239,68,68,0.07)', borderBottom: '1px solid #ef444420', padding: '6px 16px', fontSize: 12, color: '#fca5a5', flexShrink: 0 }}>
          {validation.message}
        </div>
      )}

      <div style={{ flex: 1, overflow: 'auto', padding: 16 }}>
        {files.length === 0 ? (
          <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            gap: 16, minHeight: 280, color: t.textFaint,
          }}>
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={t.textDisabled} strokeWidth="1.2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="12" y1="18" x2="12" y2="12" />
              <line x1="9" y1="15" x2="15" y2="15" />
            </svg>
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: 14, margin: '0 0 6px' }}>No CSV files added</p>
              <p style={{ fontSize: 12, color: t.textDisabled, margin: 0 }}>
                Add at least two files with matching column headers
              </p>
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16, maxWidth: 720 }}>
            <div>
              <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: t.textFaint, marginBottom: 8 }}>
                Files ({files.length})
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {files.map((f, i) => (
                  <div
                    key={`${f.name}-${i}`}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 10,
                      padding: '8px 12px',
                      background: t.bgElevated,
                      border: `1px solid ${t.borderDefault}`,
                      borderRadius: 6,
                    }}
                  >
                    <span style={{ fontSize: 12, color: t.textPrimary, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {f.name}
                    </span>
                    <span style={{ fontSize: 11, color: t.textMuted, flexShrink: 0 }}>
                      {f.rowCount.toLocaleString()} rows · {f.columns.length} cols
                    </span>
                    {i === 0 && (
                      <span style={{ fontSize: 10, color: '#86efac', background: 'rgba(34,197,94,0.12)', borderRadius: 4, padding: '1px 6px', flexShrink: 0 }}>
                        reference
                      </span>
                    )}
                    <button
                      onClick={() => removeFile(i)}
                      style={{
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        width: 24, height: 24, borderRadius: 4,
                        border: `1px solid ${t.borderDefault}`,
                        background: 'transparent', color: t.textMuted, cursor: 'pointer', flexShrink: 0,
                      }}
                      title="Remove file"
                    >
                      <TrashIcon />
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {files.length > 0 && files[0].columns.length > 0 && (
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: t.textFaint, marginBottom: 8 }}>
                  Columns (from {files[0].name})
                </div>
                <div style={{
                  display: 'flex', flexWrap: 'wrap', gap: 6,
                  padding: '10px 12px',
                  background: t.bgElevated,
                  border: `1px solid ${t.borderDefault}`,
                  borderRadius: 6,
                }}>
                  {files[0].columns.map((col) => (
                    <span
                      key={col}
                      style={{
                        fontSize: 11, color: t.textSecondary,
                        background: t.bgPanel,
                        border: `1px solid ${t.borderSubtle}`,
                        borderRadius: 4,
                        padding: '2px 8px',
                      }}
                    >
                      {col}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {files.length >= 2 && validation.ok && (
              <div style={{
                padding: '10px 12px',
                background: 'rgba(34,197,94,0.06)',
                border: `1px solid rgba(34,197,94,0.2)`,
                borderRadius: 6,
                fontSize: 12,
                color: '#86efac',
              }}>
                Ready to merge: {files.length} files · {totalRows.toLocaleString()} total rows
              </div>
            )}

            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                onClick={handleMerge}
                disabled={!canMerge}
                style={{
                  padding: '8px 14px', borderRadius: 6, border: 'none',
                  background: canMerge ? 'rgba(34,197,94,0.25)' : t.bgElevated,
                  color: canMerge ? '#86efac' : t.textDisabled,
                  fontSize: 12, fontWeight: 600, cursor: canMerge ? 'pointer' : 'not-allowed',
                }}
              >
                {merging ? 'Merging…' : 'Merge & open in studio'}
              </button>
              <button
                onClick={handleExport}
                disabled={!previewRows || previewRows.length === 0}
                style={{
                  padding: '8px 14px', borderRadius: 6,
                  border: `1px solid ${t.borderDefault}`,
                  background: 'transparent',
                  color: previewRows && previewRows.length > 0 ? t.textSecondary : t.textDisabled,
                  fontSize: 12, fontWeight: 500,
                  cursor: previewRows && previewRows.length > 0 ? 'pointer' : 'not-allowed',
                }}
              >
                Export CSV
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
