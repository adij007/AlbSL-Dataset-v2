/** Albanian finger-spelling letters (matches `albsl_app_v2.ALBANIAN_LETTERS`). */
export const ALBANIAN_LETTERS = [
  'A', 'B', 'C', 'Ç', 'D', 'Dh', 'E', 'Ë', 'F', 'G', 'Gj', 'H', 'I', 'J', 'K',
  'L', 'Ll', 'M', 'N', 'Nj', 'O', 'P', 'Q', 'R', 'Rr', 'S', 'Sh', 'T', 'Th',
  'U', 'V', 'X', 'Xh', 'Y', 'Z', 'Zh',
] as const

export const DEFAULT_PATHS = {
  weights: 'outputs/albsl_mlp.pt',
  fused_weights: 'outputs/fused_phase3.pt',
  albsl_model: 'models/trained/albsl_model_final/model_full.pt',
  landmarks_json: 'datasets/processed/assets/albsl_landmarks.json',
  dynamic_templates_json: 'datasets/processed/assets/albsl_dynamic_templates.json',
  words_dict_json: 'datasets/processed/assets/albsl_words_dictionary.json',
  unified_coords_json: 'datasets/json_dataset/coordinates.json',
  recordings_h5: 'keypoints.h5',
  keypoints_dir: 'datasets/processed/core_data/data/keypoints',
  alfabeti_h5: 'datasets/processed/core_data/data/alfabeti_keypoints.h5',
  legacy_h5: 'keypoints.h5',
}

export function apiUrl(path: string): string {
  const base = import.meta.env.VITE_API_BASE || ''
  return `${base}${path.startsWith('/') ? path : `/${path}`}`
}

export function wsUrl(path: string): string {
  const env = import.meta.env.VITE_WS_BASE as string | undefined
  if (env) return `${env.replace(/\/$/, '')}${path}`
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}${path.startsWith('/') ? path : `/${path}`}`
}
