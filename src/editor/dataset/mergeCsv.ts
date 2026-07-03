import Papa from 'papaparse'

export interface ParsedCSVFile {
  file: File
  name: string
  rows: Record<string, unknown>[]
  columns: string[]
  rowCount: number
}

export function columnsEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  return a.every((col, i) => col === b[i])
}

export function parseCSVFile(file: File): Promise<ParsedCSVFile> {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (result: Papa.ParseResult<Record<string, unknown>>) => {
        const rows = result.data
        const columns =
          result.meta.fields ??
          (rows.length > 0 ? Object.keys(rows[0]) : [])
        resolve({
          file,
          name: file.name,
          rows,
          columns,
          rowCount: rows.length,
        })
      },
      error: (err: Error) => reject(err),
    })
  })
}

export function validateCSVColumns(
  files: ParsedCSVFile[],
): { ok: true } | { ok: false; message: string; fileName?: string } {
  if (files.length === 0) {
    return { ok: false, message: 'Add at least one CSV file.' }
  }
  const reference = files[0].columns
  for (let i = 1; i < files.length; i++) {
    const f = files[i]
    if (!columnsEqual(reference, f.columns)) {
      return {
        ok: false,
        message: `Columns in "${f.name}" do not match "${files[0].name}". Expected [${reference.join(', ')}], got [${f.columns.join(', ')}].`,
        fileName: f.name,
      }
    }
  }
  return { ok: true }
}

export function concatRows(files: ParsedCSVFile[]): Record<string, unknown>[] {
  return files.flatMap((f) => f.rows)
}

export function downloadCSV(
  rows: Record<string, unknown>[],
  filename: string,
): void {
  const csv = Papa.unparse(rows)
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename.endsWith('.csv') ? filename : `${filename}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export function mergedDatasetName(files: ParsedCSVFile[]): string {
  if (files.length === 0) return 'merged'
  const base = files[0].name.replace(/\.csv$/i, '')
  return `${base}_merged`
}
