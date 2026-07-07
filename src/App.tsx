import { t } from './theme/tokens'
import { useEffect, useRef, useState } from 'react'
import FlowEditor from './editor/FlowEditor'
import Sidebar from './components/Sidebar'
import PropertyPanel from './components/PropertyPanel'
import Topbar from './components/Topbar'
import type { AppView } from './components/Topbar'
import CodePanel, { type CodeMode } from './components/CodePanel'
import TrainingPanel from './components/TrainingPanel'
import AIPanel from './components/AIPanel'
import DatasetPage from './pages/DatasetPage'
import HuggingFacePage from './pages/HuggingFacePage'
import ProjectsPage from './pages/ProjectsPage'
import { useGraphStore } from './store/useGraphStore'
import { useProjectStore } from './store/useProjectStore'
import { useTrainingStore } from './store/useTrainingStore'
import { useIsMobile } from './hooks/useBreakpoint'
import { LoadingLabel } from './components/panelChrome'
import { useThemeSync } from './theme/useThemeSync'

const AUTOSAVE_DELAY = 1500

export default function App() {
  useThemeSync()

  const currentProjectId = useProjectStore((s) => s.currentProjectId)
  const saveCurrentProject = useProjectStore((s) => s.saveCurrentProject)
  const markDirty = useProjectStore((s) => s.markDirty)
  const trainingStatus = useTrainingStore((s) => s.status)
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const isMobile = useIsMobile()

  const [view, setView] = useState<AppView>('model')
  const [pendingView, setPendingView] = useState<AppView | null>(null)
  const [codePanel, setCodePanel] = useState<{ mode: CodeMode } | null>(null)
  const [trainOpen, setTrainOpen] = useState(false)
  const [aiOpen, setAiOpen] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const saveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  function openCode(mode: CodeMode) {
    setCodePanel({ mode })
    if (mode === 'xgboost') setTrainOpen(true)
  }

  function closeCode() {
    setCodePanel(null)
  }

  function handleViewChange(next: AppView) {
    if (next === view) return
    setPendingView(next)
    requestAnimationFrame(() => {
      setView(next)
      setPendingView(null)
    })
  }

  // Close code panel when switching away from its context
  useEffect(() => {
    if (!codePanel) return
    if (codePanel.mode === 'nn' && view !== 'model') closeCode()
    if (codePanel.mode === 'pipeline' && view !== 'dataset') closeCode()
  }, [view, codePanel])

  // Auto-open training panel when training starts
  useEffect(() => {
    if (trainingStatus === 'running' || trainingStatus === 'connecting') {
      setTrainOpen(true)
    }
  }, [trainingStatus])

  // On mobile, open inspector when a node is selected
  useEffect(() => {
    if (isMobile && view === 'model' && selectedNodeId) {
      setInspectorOpen(true)
    }
  }, [isMobile, view, selectedNodeId])

  // Close drawers when leaving model view
  useEffect(() => {
    if (view !== 'model') {
      setPaletteOpen(false)
      setInspectorOpen(false)
    }
  }, [view])

  function switchToModelView() {
    handleViewChange('model')
  }

  // Auto-save to project store on graph change
  useEffect(() => {
    if (!currentProjectId) return
    const unsubscribe = useGraphStore.subscribe(() => {
      markDirty()
      clearTimeout(saveTimer.current)
      saveTimer.current = setTimeout(() => {
        saveCurrentProject()
      }, AUTOSAVE_DELAY)
    })
    return () => {
      unsubscribe()
      clearTimeout(saveTimer.current)
    }
  }, [currentProjectId, markDirty, saveCurrentProject])

  // Keyboard shortcuts
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const mod = e.metaKey || e.ctrlKey
      if (!mod) return
      if (e.key === 'z' && !e.shiftKey) {
        e.preventDefault()
        useGraphStore.getState().undo()
      }
      if ((e.key === 'z' && e.shiftKey) || e.key === 'y') {
        e.preventDefault()
        useGraphStore.getState().redo()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  // Show projects home when no project is open
  if (!currentProjectId) {
    return <ProjectsPage />
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100dvh',
      width: '100vw',
      background: t.bgBase,
      overflow: 'hidden',
    }}>
      <Topbar
        view={view}
        onViewChange={handleViewChange}
        trainOpen={trainOpen}
        onToggleTrain={() => setTrainOpen((o) => !o)}
        aiOpen={aiOpen}
        onToggleAI={() => setAiOpen((o) => !o)}
        isMobile={isMobile}
        paletteOpen={paletteOpen}
        onTogglePalette={() => setPaletteOpen((o) => !o)}
        inspectorOpen={inspectorOpen}
        onToggleInspector={() => setInspectorOpen((o) => !o)}
      />

      <div style={{ flex: 1, overflow: 'hidden', minHeight: 0, position: 'relative' }}>
        {/* Both views stay mounted for fast switching; display:none hides React Flow nodes (visibility:hidden does not). */}
        <div style={{
          position: 'absolute',
          inset: 0,
          display: view === 'model' ? 'flex' : 'none',
          overflow: 'hidden',
          zIndex: view === 'model' ? 2 : 0,
        }}>
          {!isMobile && <Sidebar />}
          {isMobile && (
            <Sidebar mobile open={paletteOpen} onClose={() => setPaletteOpen(false)} />
          )}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
            <main style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
              {view === 'model' && <FlowEditor />}
              {aiOpen && <AIPanel onClose={() => setAiOpen(false)} mobile={isMobile} />}
            </main>
            {trainOpen && (
              <TrainingPanel
                onClose={() => { closeCode(); setTrainOpen(false) }}
                mobile={isMobile}
                onOpenCode={openCode}
              />
            )}
            {trainOpen && codePanel?.mode === 'xgboost' && (
              <CodePanel mode="xgboost" onClose={closeCode} mobile={isMobile} />
            )}
            {codePanel?.mode === 'nn' && (
              <CodePanel mode="nn" onClose={closeCode} mobile={isMobile} />
            )}
          </div>
          {!isMobile && <PropertyPanel />}
          {isMobile && (
            <PropertyPanel mobile open={inspectorOpen} onClose={() => setInspectorOpen(false)} />
          )}
        </div>

        <div style={{
          position: 'absolute',
          inset: 0,
          display: view === 'dataset' ? 'flex' : 'none',
          flexDirection: 'column',
          overflow: 'hidden',
          zIndex: view === 'dataset' ? 2 : 0,
          minHeight: 0,
        }}>
          <DatasetPage
            mobile={isMobile}
            active={view === 'dataset'}
            onOpenCode={() => openCode('pipeline')}
          />
          {codePanel?.mode === 'pipeline' && (
            <CodePanel mode="pipeline" onClose={closeCode} mobile={isMobile} />
          )}
        </div>

        <div style={{
          position: 'absolute',
          inset: 0,
          display: view === 'huggingface' ? 'flex' : 'none',
          flexDirection: 'column',
          overflow: 'hidden',
          zIndex: view === 'huggingface' ? 2 : 0,
          minHeight: 0,
        }}>
          <HuggingFacePage
            mobile={isMobile}
            active={view === 'huggingface'}
            onSwitchToModel={switchToModelView}
          />
        </div>

        {pendingView && (
          <div style={{
            position: 'absolute',
            inset: 0,
            background: t.overlay,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 60,
            pointerEvents: 'none',
          }}>
            <LoadingLabel label="Switching view…" />
          </div>
        )}
      </div>
    </div>
  )
}
