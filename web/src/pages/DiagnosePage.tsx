import {
  Alert,
  Box,
  Button,
  LinearProgress,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { useState } from 'react'
import { apiUrl } from '../constants'

export default function DiagnosePage() {
  const [keypointsDir, setKd] = useState('datasets/processed/core_data/data/keypoints')
  const [alfabeti, setAlf] = useState('datasets/processed/core_data/data/alfabeti_keypoints.h5')
  const [legacy, setLeg] = useState('keypoints.h5')
  const [data, setData] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  async function run() {
    setLoading(true)
    setErr(null)
    try {
      const r = await fetch(apiUrl('/api/diagnose'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keypoints_dir: keypointsDir, alfabeti_h5: alfabeti, legacy_h5: legacy }),
      })
      if (!r.ok) throw new Error(await r.text())
      setData(await r.json())
    } catch (e) {
      setErr(String(e))
    } finally {
      setLoading(false)
    }
  }

  const clips = data && typeof data === 'object' && data !== null && 'clips' in data
    ? (data as { clips?: { per_letter?: { letter: string; frames: number }[] } }).clips
    : undefined

  return (
    <Stack spacing={3}>
      <Typography variant="h4" sx={{ fontWeight: 700 }}>
        Dataset diagnostics
      </Typography>
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ gap: 2 }}>
        <TextField label="Keypoints dir" fullWidth value={keypointsDir} onChange={(e) => setKd(e.target.value)} />
        <TextField label="Alfabeti H5" fullWidth value={alfabeti} onChange={(e) => setAlf(e.target.value)} />
        <TextField label="Legacy H5" fullWidth value={legacy} onChange={(e) => setLeg(e.target.value)} />
      </Stack>
      <Button variant="contained" onClick={run} disabled={loading}>
        {loading ? 'Running…' : 'Run diagnose'}
      </Button>
      {err && <Alert severity="error">{err}</Alert>}
      {clips?.per_letter && (
        <Box>
          <Typography variant="subtitle1" gutterBottom>
            Frames per letter (clips)
          </Typography>
          <Stack spacing={0.5}>
            {clips.per_letter.map((row) => (
              <Box key={row.letter} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography sx={{ width: 40 }}>{row.letter}</Typography>
                <LinearProgress
                  variant="determinate"
                  sx={{ flex: 1, height: 8, borderRadius: 1 }}
                  value={Math.min(100, (row.frames / Math.max(1, clips.per_letter!.reduce((m, x) => Math.max(m, x.frames), 0))) * 100)}
                />
                <Typography variant="caption" sx={{ width: 48, textAlign: 'right' }}>
                  {row.frames}
                </Typography>
              </Box>
            ))}
          </Stack>
        </Box>
      )}
      {data != null && (
        <Box component="pre" sx={{ p: 2, borderRadius: 2, bgcolor: 'action.hover', overflow: 'auto', fontSize: 12 }}>
          {JSON.stringify(data, null, 2)}
        </Box>
      )}
    </Stack>
  )
}
