import { useEffect, useMemo, useRef, useState } from 'react'
import type { TrainingConfig } from '../types/training'
import type { ValidationIssue } from '../types/validation'
import type { LoadedDataset } from '../store/useDatasetStore'
import type { AppNode, AppEdge } from '../types/graph'
import { pipelineStructureKey } from '../editor/dataset/datasetModelInfo'
import { validateTabularTraining } from '../editor/validation/validateTabularTraining'
import { deferWork } from '../utils/deferWork'

const PIPELINE_PREVIEW_ROWS = 200
const DEBOUNCE_MS = 200

export function useDeferredTabularValidation(
  enabled: boolean,
  dataset: LoadedDataset | null,
  targetColumn: string | null,
  pipelineNodes: AppNode[],
  pipelineEdges: AppEdge[],
  config: Pick<TrainingConfig, 'xgbTask' | 'xgbNEstimators' | 'xgbEarlyStoppingRounds' | 'xgbObjective'>,
) {
  const pipelineKey = useMemo(
    () => pipelineStructureKey(pipelineNodes, pipelineEdges),
    [pipelineNodes, pipelineEdges],
  )

  const [issues, setIssues] = useState<ValidationIssue[]>([])
  const [loading, setLoading] = useState(false)
  const requestId = useRef(0)

  useEffect(() => {
    if (!enabled) {
      setIssues([])
      setLoading(false)
      return
    }

    const id = ++requestId.current
    setLoading(true)

    const timer = setTimeout(() => {
      void deferWork().then(() => {
        if (requestId.current !== id) return
        const result = validateTabularTraining(
          dataset,
          targetColumn,
          pipelineNodes,
          pipelineEdges,
          config,
          { pipelineMaxRows: PIPELINE_PREVIEW_ROWS },
        )
        if (requestId.current !== id) return
        setIssues(result.issues.filter((i) => i.severity !== 'info'))
        setLoading(false)
      })
    }, DEBOUNCE_MS)

    return () => clearTimeout(timer)
  }, [
    enabled,
    dataset,
    targetColumn,
    pipelineKey,
    pipelineNodes,
    pipelineEdges,
    config.xgbTask,
    config.xgbNEstimators,
    config.xgbEarlyStoppingRounds,
    config.xgbObjective,
  ])

  return { issues, loading }
}
