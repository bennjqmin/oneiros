import type { AppNode, AppEdge } from '../../types/graph'

export interface FilterStep {
  column: string
  operator: string
  value: number
}

export interface PipelineConfig {
  dropDuplicates: boolean
  dropColumns: string[]
  shuffle: boolean
  shuffleSeed: number
  ordinalEncode: boolean
  ordinalEncodeColumns: string[]
  oneHotEncode: boolean
  oneHotColumns: string[]
  fillNaN: boolean
  fillNaNStrategy: string
  fillNaNConstant: number
  logTransform: boolean
  logTransformColumns: string[]
  clipOutliers: boolean
  clipStdFactor: number
  binColumnEnabled: boolean
  binColumn: string
  binCount: number
  binStrategy: string
  normalize: boolean
  normalizeMethod: 'min-max' | 'zscore'
  normalizeColumns: string[]
  standardScaler: boolean
  standardScalerColumns: string[]
  selectKBest: boolean
  selectK: number
  balanceClasses: boolean
  trainRatio: number
}

export function defaultPipelineConfig(): PipelineConfig {
  return {
    dropDuplicates: false,
    dropColumns: [],
    shuffle: false,
    shuffleSeed: 42,
    ordinalEncode: false,
    ordinalEncodeColumns: [],
    oneHotEncode: false,
    oneHotColumns: [],
    fillNaN: false,
    fillNaNStrategy: 'mean',
    fillNaNConstant: 0,
    logTransform: false,
    logTransformColumns: [],
    clipOutliers: false,
    clipStdFactor: 3,
    binColumnEnabled: false,
    binColumn: '',
    binCount: 5,
    binStrategy: 'equal-width',
    normalize: false,
    normalizeMethod: 'min-max',
    normalizeColumns: [],
    standardScaler: false,
    standardScalerColumns: [],
    selectKBest: false,
    selectK: 10,
    balanceClasses: false,
    trainRatio: 0.8,
  }
}

function parseColumns(raw: unknown): string[] {
  if (!raw) return []
  return String(raw).split(',').map((s) => s.trim()).filter(Boolean)
}

export function topoSort(nodes: AppNode[], edges: AppEdge[]): AppNode[] {
  const inDegree = new Map<string, number>(nodes.map((n) => [n.id, 0]))
  const adj = new Map<string, string[]>(nodes.map((n) => [n.id, []]))
  for (const e of edges) {
    adj.get(e.source)?.push(e.target)
    inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1)
  }
  const queue = nodes.filter((n) => inDegree.get(n.id) === 0)
  const result: AppNode[] = []
  while (queue.length) {
    const node = queue.shift()!
    result.push(node)
    for (const nid of adj.get(node.id) ?? []) {
      const d = (inDegree.get(nid) ?? 0) - 1
      inDegree.set(nid, d)
      if (d === 0) queue.push(nodes.find((n) => n.id === nid)!)
    }
  }
  return result.filter(Boolean)
}

export function reachableFromSource(nodes: AppNode[], edges: AppEdge[]): Set<string> {
  const sourceIds = nodes.filter((n) => n.type === 'datasetSource').map((n) => n.id)
  if (sourceIds.length === 0) return new Set<string>()

  const adj = new Map<string, string[]>()
  for (const e of edges) {
    if (!adj.has(e.source)) adj.set(e.source, [])
    adj.get(e.source)!.push(e.target)
  }

  const seen = new Set<string>()
  const queue = [...sourceIds]
  while (queue.length) {
    const id = queue.shift()!
    if (seen.has(id)) continue
    seen.add(id)
    for (const next of adj.get(id) ?? []) queue.push(next)
  }
  return seen
}

export function getReachableSortedNodes(nodes: AppNode[], edges: AppEdge[]): AppNode[] {
  const reachable = reachableFromSource(nodes, edges)
  return topoSort(nodes, edges).filter((n) => reachable.has(n.id))
}

export interface ParsedPipeline {
  sorted: AppNode[]
  filters: FilterStep[]
  config: PipelineConfig
}

export function parsePipelineNodes(nodes: AppNode[], edges: AppEdge[]): ParsedPipeline {
  const sorted = getReachableSortedNodes(nodes, edges)
  const config = defaultPipelineConfig()
  const filters: FilterStep[] = []

  for (const node of sorted) {
    const data = node.data as Record<string, unknown>
    switch (node.type) {
      case 'datasetSource':
        break
      case 'normalizeNode':
        config.normalize = true
        config.normalizeMethod = (data.method as string) === 'zscore' ? 'zscore' : 'min-max'
        config.normalizeColumns = parseColumns(data.columns)
        break
      case 'standardScalerNode':
        config.standardScaler = true
        config.standardScalerColumns = parseColumns(data.columns)
        break
      case 'logTransformNode':
        config.logTransform = true
        config.logTransformColumns = parseColumns(data.columns)
        break
      case 'clipOutliersNode':
        config.clipOutliers = true
        config.clipStdFactor = typeof data.stdFactor === 'number' ? data.stdFactor : 3
        break
      case 'binColumnNode':
        config.binColumnEnabled = true
        config.binColumn = String(data.column ?? '')
        config.binCount = typeof data.bins === 'number' ? data.bins : 5
        config.binStrategy = String(data.strategy ?? 'equal-width')
        break
      case 'fillNaNNode':
        config.fillNaN = true
        config.fillNaNStrategy = String(data.strategy ?? 'mean')
        config.fillNaNConstant = typeof data.constant === 'number' ? data.constant : 0
        break
      case 'dropColumnsNode':
        config.dropColumns = parseColumns(data.columns)
        break
      case 'dropDuplicatesNode':
        config.dropDuplicates = true
        break
      case 'selectKBestNode':
        config.selectKBest = true
        config.selectK = typeof data.k === 'number' ? data.k : 10
        break
      case 'balanceClassesNode':
        config.balanceClasses = true
        break
      case 'oneHotEncodeNode':
        config.oneHotEncode = true
        config.oneHotColumns = parseColumns(data.columns)
        break
      case 'ordinalEncodeNode':
        config.ordinalEncode = true
        config.ordinalEncodeColumns = parseColumns(data.columns)
        break
      case 'shuffleNode':
        config.shuffle = true
        config.shuffleSeed = typeof data.seed === 'number' ? data.seed : 42
        break
      case 'splitNode':
        config.trainRatio = typeof data.trainRatio === 'number' ? data.trainRatio : 0.8
        break
      case 'filterNode': {
        const col = String(data.column ?? '')
        if (col) {
          filters.push({
            column: col,
            operator: String(data.operator ?? '>'),
            value: parseFloat(String(data.value ?? '0')),
          })
        }
        break
      }
    }
  }

  return { sorted, filters, config }
}
