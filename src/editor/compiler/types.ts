export type CodeMode = 'nn' | 'pipeline' | 'xgboost'

export interface CompileResult {
  code: string
  errors: string[]
  warnings: string[]
  filename: string
  label: string
}
