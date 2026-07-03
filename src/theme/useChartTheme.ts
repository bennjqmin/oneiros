import { useMemo } from 'react'
import { useThemeStore } from '../store/useThemeStore'

export interface ChartTheme {
  gridStroke: string
  axisStroke: string
  axisTick: string
  tooltipBg: string
  tooltipBorder: string
  textColor: string
}

function readCssVar(name: string, fallback: string): string {
  if (typeof document === 'undefined') return fallback
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
}

export function useChartTheme(): ChartTheme {
  const resolvedTheme = useThemeStore((s) => s.resolvedTheme)

  return useMemo(() => ({
    gridStroke: readCssVar('--chart-grid', '#27272a'),
    axisStroke: readCssVar('--chart-axis', '#52525b'),
    axisTick: readCssVar('--text-faint', '#52525b'),
    tooltipBg: readCssVar('--chart-tooltip-bg', '#18181b'),
    tooltipBorder: readCssVar('--chart-tooltip-border', '#27272a'),
    textColor: readCssVar('--text-secondary', '#a1a1aa'),
  }), [resolvedTheme])
}
