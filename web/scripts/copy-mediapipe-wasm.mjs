/**
 * Copy MediaPipe Tasks Vision WASM bundle into public/ so the runtime matches
 * the same @mediapipe/tasks-vision version installed in node_modules (Vite bundles
 * the JS from npm; loading WASM from a different CDN version breaks init).
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const webRoot = path.join(__dirname, '..')
const from = path.join(webRoot, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm')
const to = path.join(webRoot, 'public', 'mediapipe-wasm')

if (!fs.existsSync(from)) {
  console.warn('[copy-mediapipe-wasm] skip: node_modules/@mediapipe/tasks-vision/wasm not found (run npm install in web/)')
  process.exit(0)
}

fs.mkdirSync(to, { recursive: true })
for (const name of fs.readdirSync(from)) {
  fs.copyFileSync(path.join(from, name), path.join(to, name))
}
console.log('[copy-mediapipe-wasm] copied', fs.readdirSync(from).length, 'files to', to)
