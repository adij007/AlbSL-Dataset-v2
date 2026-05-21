import { DEFAULT_PATHS } from './constants'

const KEY = 'albsl_webui_paths_v1'

export type PathSettings = typeof DEFAULT_PATHS

export function loadPathSettings(): PathSettings {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return { ...DEFAULT_PATHS }
    return { ...DEFAULT_PATHS, ...JSON.parse(raw) }
  } catch {
    return { ...DEFAULT_PATHS }
  }
}

export function savePathSettings(p: Partial<PathSettings>) {
  const next = { ...loadPathSettings(), ...p }
  localStorage.setItem(KEY, JSON.stringify(next))
  return next
}
