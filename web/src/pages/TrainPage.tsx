import {
  Alert,
  Box,
  Button,
  Checkbox,
  FormControlLabel,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { useRef, useState } from 'react'
import { apiUrl } from '../constants'

export default function TrainPage() {
  const [log, setLog] = useState('')
  const [running, setRunning] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const [form, setForm] = useState({
    keypoints_dir: 'datasets/processed/core_data/data/keypoints',
    alfabeti_h5: 'datasets/processed/core_data/data/alfabeti_keypoints.h5',
    legacy_h5: 'keypoints.h5',
    out: 'outputs/albsl_mlp.pt',
    epochs: 50,
    batch_size: 128,
    lr: 0.001,
    device: 'cuda',
    sequence_len: 8,
    sequence_stride: 2,
    min_valid_frames: 4,
    idle_ratio: 0.35,
    hidden_dim: 192,
    layers: 2,
    dropout: 0.25,
    workers: 0,
    no_augment: false,
  })

  async function start() {
    setRunning(true)
    setLog('')
    const ac = new AbortController()
    abortRef.current = ac
    try {
      const res = await fetch(apiUrl('/api/train/stream'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          lr: Number(form.lr),
          legacy_h5: form.legacy_h5 || null,
        }),
        signal: ac.signal,
      })
      if (!res.ok || !res.body) {
        setLog(await res.text())
        return
      }
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const parts = buf.split('\n\n')
        buf = parts.pop() || ''
        for (const block of parts) {
          for (const line of block.split('\n')) {
            if (line.startsWith('data: ')) {
              const payload = line.slice(6)
              setLog((prev) => prev + payload + '\n')
            }
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setLog(String(e))
    } finally {
      setRunning(false)
      abortRef.current = null
    }
  }

  function cancel() {
    abortRef.current?.abort()
  }

  return (
    <Stack spacing={2}>
      <Typography variant="h4" sx={{ fontWeight: 700 }}>
        Train temporal model
      </Typography>
      <Typography color="text.secondary">
        Streams stdout from <code>python Script/albsl_app_v2.py train …</code>. Use CPU if CUDA is unavailable.
      </Typography>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ flexWrap: 'wrap', gap: 2 }}>
        <TextField label="Out checkpoint" value={form.out} onChange={(e) => setForm({ ...form, out: e.target.value })} sx={{ minWidth: 220 }} />
        <TextField type="number" label="Epochs" value={form.epochs} onChange={(e) => setForm({ ...form, epochs: +e.target.value })} />
        <TextField type="number" label="Batch size" value={form.batch_size} onChange={(e) => setForm({ ...form, batch_size: +e.target.value })} />
        <TextField label="Device" value={form.device} onChange={(e) => setForm({ ...form, device: e.target.value })} sx={{ width: 120 }} />
      </Stack>
      <FormControlLabel
        control={
          <Checkbox
            checked={form.no_augment}
            onChange={(e) => setForm({ ...form, no_augment: e.target.checked })}
          />
        }
        label="No augment (--no-augment)"
      />
      <Stack direction="row" sx={{ gap: 1 }}>
        <Button variant="contained" onClick={start} disabled={running}>
          Start training
        </Button>
        <Button variant="outlined" onClick={cancel} disabled={!running}>
          Cancel
        </Button>
      </Stack>
      <Alert severity="info">Ensure the API server is running from the repository root.</Alert>
      <Box
        component="pre"
        sx={{
          p: 2,
          borderRadius: 2,
          bgcolor: 'action.hover',
          maxHeight: 480,
          overflow: 'auto',
          fontSize: 12,
          whiteSpace: 'pre-wrap',
        }}
      >
        {log || '—'}
      </Box>
    </Stack>
  )
}
