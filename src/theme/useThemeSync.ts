import { useThemeStore } from '../store/useThemeStore'

/** Subscribe to theme changes so parent layouts re-render when the palette switches. */
export function useThemeSync(): void {
  useThemeStore((s) => s.resolvedTheme)
}
