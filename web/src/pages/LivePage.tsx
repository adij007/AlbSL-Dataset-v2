import AddIcon from '@mui/icons-material/Add'
import BackspaceIcon from '@mui/icons-material/Backspace'
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord'
import KeyboardReturnIcon from '@mui/icons-material/KeyboardReturn'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  Chip,
  FormControlLabel,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material'
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision'
import { useCallback, useEffect, useRef, useState } from 'react'
import { ALBANIAN_LETTERS, apiUrl, wsUrl } from '../constants'
import { loadPathSettings } from '../settingsStorage'

/** Avoid showing `[object Event]` for DOM / WebSocket failures. */
function formatUnknownErr(e: unknown): string {
  if (e instanceof Error) return e.message
  if (typeof e === 'string') return e
  if (typeof DOMException !== 'undefined' && e instanceof DOMException) return e.message
  if (typeof ErrorEvent !== 'undefined' && e instanceof ErrorEvent && e.message) return e.message
  if (typeof Event !== 'undefined' && e instanceof Event) {
    return e.type === 'error' ? 'Connection or media error (check the browser console).' : `Event: ${e.type}`
  }
  if (e && typeof e === 'object' && 'message' in e && typeof (e as { message: unknown }).message === 'string') {
    return (e as { message: string }).message
  }
  try {
    return JSON.stringify(e)
  } catch {
    return 'Unknown error'
  }
}

type WsResult = {
  top3: [string, number][]
  idle_detected: boolean
  idle_prob: number
  detected: boolean
  actionable: boolean
  shown_letter: string
  auto_append_letter: string | null
  fusion_used: boolean
}

const HAND_TASK =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

/** WASM served from `public/mediapipe-wasm` (same build as npm package; see `scripts/copy-mediapipe-wasm.mjs`). */
function mediapipeWasmDirectory(): string {
  return new URL('mediapipe-wasm/', new URL(import.meta.env.BASE_URL, window.location.origin)).href
}

function handednessToSide(displayName: string): { side: string; is_left: boolean } {
  const raw = displayName.toLowerCase()
  const hand_side = raw === 'right' ? 'left' : 'right'
  return { side: hand_side, is_left: hand_side === 'left' }
}

type HandPayload = {
  /** Raw MediaPipe landmarks (unmirrored image); use for overlay + fusion crop on the video element. */
  raw: NormalizedLandmark[]
  /** Mirrored x to match ``cv2.flip(..., 1)`` + desktop training / inference. */
  serverXyz: number[][]
  is_left: boolean
  score: number
  side: string
}

function handsFromResult(result: HandLandmarkerResult): HandPayload[] {
  const out: HandPayload[] = []
  const handLms = result.landmarks ?? []
  for (let i = 0; i < handLms.length; i++) {
    const pts = handLms[i]
    let score = 0
    let side = 'right'
    let is_left = false
    const h0 = result.handedness?.[i]?.[0]
    if (h0) {
      score = h0.score
      const s = handednessToSide(h0.displayName)
      side = s.side
      is_left = s.is_left
    }
    const serverXyz = pts.map((p) => [1 - p.x, p.y, p.z])
    out.push({ raw: pts, serverXyz, is_left, score, side })
  }
  return out
}

function pickPrimaryHand(hands: HandPayload[]): HandPayload | undefined {
  if (!hands.length) return undefined
  return hands.reduce((a, b) => (b.score > a.score ? b : a))
}

function bboxFromRawLandmarks(pts: NormalizedLandmark[], vw: number, vh: number, pad = 0.25) {
  const xs = pts.map((p) => p.x * vw)
  const ys = pts.map((p) => p.y * vh)
  const minX = Math.max(0, Math.min(...xs) - pad * (Math.max(...xs) - Math.min(...xs)))
  const maxX = Math.min(vw, Math.max(...xs) + pad * (Math.max(...xs) - Math.min(...xs)))
  const minY = Math.max(0, Math.min(...ys) - pad * (Math.max(...ys) - Math.min(...ys)))
  const maxY = Math.min(vh, Math.max(...ys) + pad * (Math.max(...ys) - Math.min(...ys)))
  return { minX, minY, maxX, maxY }
}

/** Only draw overlay when MediaPipe is this confident (reduces “ghost” hands). */
const OVERLAY_MIN_HAND_SCORE = 0.62

function drawHandSkeleton(
  ctx: CanvasRenderingContext2D,
  vw: number,
  vh: number,
  pts: NormalizedLandmark[],
  isLeft: boolean,
) {
  const bone = isLeft ? 'rgba(0,200,255,0.92)' : 'rgba(0,255,0,0.92)'
  const joint = isLeft ? '#00d4ff' : '#66ff66'
  const lw = Math.max(2, vw / 220)
  const jr = Math.max(2.5, vw / 110)
  const conns = HandLandmarker.HAND_CONNECTIONS
  ctx.strokeStyle = bone
  ctx.lineWidth = lw
  ctx.lineCap = 'round'
  for (const { start, end } of conns) {
    const a = pts[start]
    const b = pts[end]
    ctx.beginPath()
    ctx.moveTo(a.x * vw, a.y * vh)
    ctx.lineTo(b.x * vw, b.y * vh)
    ctx.stroke()
  }
  ctx.fillStyle = joint
  for (const p of pts) {
    ctx.beginPath()
    ctx.arc(p.x * vw, p.y * vh, jr, 0, 2 * Math.PI)
    ctx.fill()
  }
}

export default function LivePage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const cropRef = useRef<HTMLCanvasElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const handRef = useRef<HandLandmarker | null>(null)
  const rafRef = useRef<number>(0)
  const lastTs = useRef(0)

  const [status, setStatus] = useState<string>('Idle')
  const [err, setErr] = useState<string | null>(null)
  const [res, setRes] = useState<WsResult | null>(null)
  const [word, setWord] = useState<string[]>([])
  const [labelIdx, setLabelIdx] = useState(0)
  const [fusionJpeg, setFusion] = useState(true)
  const [running, setRunning] = useState(false)
  const [recMode, setRecMode] = useState<'idle' | 'countdown' | 'record'>('idle')
  const [recUntil, setRecUntil] = useState<number | undefined>()
  const [recCount, setRecCount] = useState(0)
  const recFramesRef = useRef<{ xyz: number[][]; is_left: boolean }[]>([])
  const submitGuard = useRef(false)
  const expectWsClose = useRef(false)
  /** When false, the RAF loop exits and must not schedule another frame (prevents frozen overlay after Stop). */
  const liveLoopActiveRef = useRef(false)

  const clearHandOverlay = useCallback(() => {
    const ovr = overlayRef.current
    const video = videoRef.current
    if (!ovr) return
    const w = video && video.videoWidth > 0 ? video.videoWidth : ovr.width
    const h = video && video.videoHeight > 0 ? video.videoHeight : ovr.height
    if (w <= 0 || h <= 0) {
      const ctx = ovr.getContext('2d')
      if (ctx && ovr.width > 0 && ovr.height > 0) ctx.clearRect(0, 0, ovr.width, ovr.height)
      return
    }
    ovr.width = w
    ovr.height = h
    ovr.getContext('2d')?.clearRect(0, 0, w, h)
  }, [])

  const sendInit = useCallback((ws: WebSocket) => {
    const p = loadPathSettings()
    ws.send(
      JSON.stringify({
        type: 'init',
        config: {
          weights: p.weights,
          fused_weights: p.fused_weights,
          albsl_model: p.albsl_model,
          landmarks_json: p.landmarks_json,
          dynamic_templates_json: p.dynamic_templates_json,
          words_dict_json: p.words_dict_json,
          unified_coords_json: p.unified_coords_json,
        },
      }),
    )
  }, [])

  const stop = useCallback(() => {
    liveLoopActiveRef.current = false
    expectWsClose.current = true
    cancelAnimationFrame(rafRef.current)
    rafRef.current = 0
    wsRef.current?.close()
    wsRef.current = null
    handRef.current?.close()
    handRef.current = null
    const v = videoRef.current?.srcObject as MediaStream | undefined
    v?.getTracks().forEach((t) => t.stop())
    if (videoRef.current) videoRef.current.srcObject = null
    clearHandOverlay()
    setRunning(false)
    setStatus('Stopped')
  }, [clearHandOverlay])

  useEffect(() => () => stop(), [stop])

  const loop = useCallback(() => {
    if (!liveLoopActiveRef.current) {
      clearHandOverlay()
      return
    }

    const video = videoRef.current
    const ws = wsRef.current
    const hand = handRef.current
    if (!video || !ws || ws.readyState !== WebSocket.OPEN || !hand || video.readyState < 2) {
      clearHandOverlay()
      if (liveLoopActiveRef.current) rafRef.current = requestAnimationFrame(loop)
      return
    }
    const ts = performance.now()
    if (ts - lastTs.current < 33) {
      if (liveLoopActiveRef.current) rafRef.current = requestAnimationFrame(loop)
      return
    }
    lastTs.current = ts

    try {
      const result = hand.detectForVideo(video, ts)
      const hands = handsFromResult(result)
      const vw = video.videoWidth
      const vh = video.videoHeight

      const ovr = overlayRef.current
      if (ovr && vw > 0 && vh > 0) {
        ovr.width = vw
        ovr.height = vh
        const ox = ovr.getContext('2d')
        if (ox) {
          ox.clearRect(0, 0, vw, vh)
          const lms = result.landmarks ?? []
          const handed = result.handedness ?? []
          for (let i = 0; i < lms.length; i++) {
            const pts = lms[i]
            const h0 = handed[i]?.[0]
            if (!h0 || h0.score < OVERLAY_MIN_HAND_SCORE) continue
            const isLeft = handednessToSide(h0.displayName).is_left
            drawHandSkeleton(ox, vw, vh, pts, isLeft)
          }
        }
      }

      const primary = pickPrimaryHand(hands)

      let fusion_b64 = ''
      if (fusionJpeg && primary && vw > 0) {
        const { minX, minY, maxX, maxY } = bboxFromRawLandmarks(primary.raw, vw, vh)
        const cw = cropRef.current
        if (cw) {
          cw.width = 224
          cw.height = 224
          const ctx = cw.getContext('2d')
          if (ctx) {
            ctx.drawImage(video, minX, minY, maxX - minX, maxY - minY, 0, 0, 224, 224)
            fusion_b64 = cw.toDataURL('image/jpeg', 0.85).replace(/^data:image\/jpeg;base64,/, '')
          }
        }
      }

      ws.send(
        JSON.stringify({
          type: 'frame',
          ts_ms: Math.round(ts),
          frame_h: vh || 480,
          frame_w: vw || 640,
          hands: hands.map((h) => ({
            xyz: h.serverXyz,
            is_left: h.is_left,
            score: h.score,
            side: h.side,
          })),
          fusion_jpeg: fusion_b64 || undefined,
        }),
      )

      if (recMode === 'record' && primary && recFramesRef.current.length < 30) {
        recFramesRef.current.push({ xyz: primary.serverXyz, is_left: primary.is_left })
        setRecCount(recFramesRef.current.length)
      }
    } catch (e) {
      console.error('[Live] frame loop', e)
      clearHandOverlay()
    }

    if (liveLoopActiveRef.current) rafRef.current = requestAnimationFrame(loop)
  }, [fusionJpeg, recMode, clearHandOverlay])

  useEffect(() => {
    if (running && liveLoopActiveRef.current) {
      rafRef.current = requestAnimationFrame(loop)
    }
    return () => cancelAnimationFrame(rafRef.current)
  }, [running, loop])

  async function start() {
    setErr(null)
    expectWsClose.current = false
    liveLoopActiveRef.current = false
    setStatus('Starting…')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
      const video = videoRef.current
      if (!video) return
      video.srcObject = stream
      await video.play()

      const fileset = await FilesetResolver.forVisionTasks(mediapipeWasmDirectory())
      let hand: HandLandmarker
      try {
        hand = await HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: HAND_TASK, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
        })
      } catch (gpuErr) {
        try {
          hand = await HandLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: HAND_TASK, delegate: 'CPU' },
            runningMode: 'VIDEO',
            numHands: 2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
          })
        } catch (cpuErr) {
          throw new Error(
            `MediaPipe HandLandmarker failed (GPU: ${formatUnknownErr(gpuErr)}; CPU: ${formatUnknownErr(cpuErr)}).`,
          )
        }
      }
      handRef.current = hand

      const ws = new WebSocket(wsUrl('/api/ws/live'))
      wsRef.current = ws
      await new Promise<void>((resolve, reject) => {
        const t = window.setTimeout(() => {
          reject(
            new Error(
              'WebSocket open timed out. From the repo root run: python -m uvicorn web.server.main:app --host 127.0.0.1 --port 8765 (or npm run api from the web folder), then npm run dev so /api is proxied.',
            ),
          )
        }, 12000)
        ws.onopen = () => {
          window.clearTimeout(t)
          resolve()
        }
        ws.onerror = () => {
          window.clearTimeout(t)
          reject(
            new Error(
              'Could not open WebSocket to the API. Start the server on port 8765, or set VITE_WS_BASE in web/.env to your API URL.',
            ),
          )
        }
      })
      sendInit(ws)
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data as string)
          if (msg.type === 'ready') setStatus('Live')
          if (msg.type === 'result') setRes(msg as WsResult)
          if (msg.type === 'error') setErr(typeof msg.message === 'string' ? msg.message : formatUnknownErr(msg))
        } catch {
          setErr('Invalid message from inference server (JSON parse failed).')
        }
      }
      ws.onerror = () => {
        liveLoopActiveRef.current = false
        clearHandOverlay()
        setErr('WebSocket error while live. Check the API process and network.')
        setRunning(false)
        setStatus('Stopped')
      }
      ws.onclose = (ev) => {
        liveLoopActiveRef.current = false
        clearHandOverlay()
        if (expectWsClose.current) {
          setRunning(false)
          return
        }
        setErr(`WebSocket closed (${ev.code}${ev.reason ? `: ${ev.reason}` : ''}).`)
        setRunning(false)
        setStatus('Stopped')
      }

      liveLoopActiveRef.current = true
      setRunning(true)
    } catch (e) {
      setErr(formatUnknownErr(e))
      setStatus('Error')
      stop()
    }
  }

  useEffect(() => {
    if (recMode !== 'record' || recCount !== 30) {
      submitGuard.current = false
      return
    }
    if (submitGuard.current) return
    submitGuard.current = true
    const frames = recFramesRef.current
    const p = loadPathSettings()
    const label = ALBANIAN_LETTERS[labelIdx]
    fetch(apiUrl('/api/record/landmarks'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        recordings_h5: p.recordings_h5,
        label,
        source: 'webui-live',
        frames: frames.map((f) => ({ xyz: f.xyz.flat(), is_left: f.is_left })),
      }),
    })
      .then((r) => r.json())
      .then((j) => {
        setStatus(`Recorded → ${j.path ?? 'ok'}`)
        setRecMode('idle')
        setRecCount(0)
        recFramesRef.current = []
      })
      .catch((e) => {
        setErr(formatUnknownErr(e))
        submitGuard.current = false
      })
  }, [recMode, recCount, labelIdx])

  useEffect(() => {
    if (recMode !== 'countdown' || !recUntil) return
    const t = setInterval(() => {
      if (Date.now() >= recUntil) {
        recFramesRef.current = []
        setRecCount(0)
        setRecMode('record')
        clearInterval(t)
      }
    }, 150)
    return () => clearInterval(t)
  }, [recMode, recUntil])

  function appendFromPred() {
    if (!res?.top3?.[0]) return
    if (!res.actionable) return
    setWord((w) => [...w, res.top3![0][0]])
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === ' ' || e.code === 'Space') {
      e.preventDefault()
      appendFromPred()
    }
    if (e.key === 'Backspace') setWord((w) => w.slice(0, -1))
    if (e.key === 'Enter') setWord([])
    if (e.key.toLowerCase() === 'c') setWord([])
    if (e.key.toLowerCase() === 'l') setLabelIdx((i) => (i + 1) % ALBANIAN_LETTERS.length)
    if (e.key.toLowerCase() === 'k') setLabelIdx((i) => (i + ALBANIAN_LETTERS.length - 1) % ALBANIAN_LETTERS.length)
    if (e.key.toLowerCase() === 'r') {
      setRecMode('countdown')
      setRecUntil(Date.now() + 3000)
    }
  }

  return (
    <Stack spacing={2} onKeyDown={onKeyDown} tabIndex={0}>
      <Typography variant="h4" sx={{ fontWeight: 700 }}>
        Live recognition
      </Typography>
      <Typography color="text.secondary" sx={{ maxWidth: 800 }}>
        Preview is mirrored like a selfie. MediaPipe tracks up to two hands (green ≈ right, cyan ≈ left); the API
        picks the primary hand for the letter model. Landmark x is flipped for the server to match the desktop
        OpenCV pipeline. Optional JPEG crops feed the fusion model when weights are present.
      </Typography>
      {err && <Alert severity="error">{err}</Alert>}
      <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 1, alignItems: 'center' }}>
        <Button variant="contained" onClick={start} disabled={running}>
          Start camera
        </Button>
        <Button variant="outlined" onClick={stop} disabled={!running}>
          Stop
        </Button>
        <FormControlLabel
          control={<Checkbox checked={fusionJpeg} onChange={(e) => setFusion(e.target.checked)} />}
          label="Send fusion crop (JPEG)"
        />
        <Chip label={status} color={status === 'Live' ? 'success' : 'default'} />
      </Stack>

      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'flex-start' }}>
        <Box
          sx={{
            position: 'relative',
            display: 'inline-block',
            maxWidth: '100%',
            borderRadius: 2,
            overflow: 'hidden',
            transform: 'scaleX(-1)',
            lineHeight: 0,
          }}
        >
          <video
            ref={videoRef}
            muted
            playsInline
            style={{ width: 480, maxWidth: '100%', height: 'auto', display: 'block' }}
          />
          <canvas
            ref={overlayRef}
            style={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              pointerEvents: 'none',
            }}
          />
        </Box>
        <canvas ref={cropRef} style={{ display: 'none' }} />
        <Card variant="outlined" sx={{ minWidth: 260, flex: 1 }}>
          <CardContent>
            <Typography variant="overline">Prediction</Typography>
            <Typography variant="h3" sx={{ fontWeight: 700 }}>
              {res?.shown_letter ?? '—'}
            </Typography>
            <Stack spacing={0.5} sx={{ mt: 1 }}>
              {(res?.top3 ?? []).map(([l, p]) => (
                <Typography key={l} variant="body2">
                  {l} — {(p * 100).toFixed(1)}%
                </Typography>
              ))}
            </Stack>
            {res?.idle_detected && (
              <Typography variant="caption" color="text.secondary">
                Idle / No-Sign {(res.idle_prob * 100).toFixed(0)}%
              </Typography>
            )}
            {res?.fusion_used && (
              <Typography variant="caption" color="success.main">
                Fusion used
              </Typography>
            )}
            <Typography variant="h6" sx={{ mt: 2 }}>
              Word
            </Typography>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              {word.join('') || '—'}
            </Typography>
            {res?.auto_append_letter && (
              <Typography variant="caption">Auto: {res.auto_append_letter}</Typography>
            )}
          </CardContent>
        </Card>
      </Box>

      <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 1, alignItems: 'center' }}>
        <Typography>Label</Typography>
        <Select
          size="small"
          value={labelIdx}
          onChange={(e) => setLabelIdx(Number(e.target.value))}
          sx={{ minWidth: 100 }}
        >
          {ALBANIAN_LETTERS.map((l, i) => (
            <MenuItem key={l} value={i}>
              {l}
            </MenuItem>
          ))}
        </Select>
        <Button startIcon={<AddIcon />} onClick={appendFromPred} disabled={!res?.actionable}>
          Append (Space)
        </Button>
        <Button startIcon={<BackspaceIcon />} onClick={() => setWord((w) => w.slice(0, -1))}>
          Backspace
        </Button>
        <Button startIcon={<KeyboardReturnIcon />} onClick={() => setWord([])}>
          Clear word (Enter)
        </Button>
        <Button
          startIcon={<FiberManualRecordIcon />}
          onClick={() => {
            setRecMode('countdown')
            setRecUntil(Date.now() + 3000)
          }}
          color="error"
          variant="outlined"
        >
          Record 3s → 30 fr (R)
        </Button>
      </Stack>
      <Typography variant="caption" color="text.secondary">
        Hotkeys: L/K label · R countdown+record · Space append · Backspace · Enter clear word · C clear
      </Typography>
      {recMode === 'countdown' && recUntil && (
        <Alert severity="warning">
          Recording starts in {Math.max(0, Math.ceil((recUntil - Date.now()) / 1000))}s…
        </Alert>
      )}
      {recMode === 'record' && <Alert severity="info">Recording {recCount}/30</Alert>}
    </Stack>
  )
}
