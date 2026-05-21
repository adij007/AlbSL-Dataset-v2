import { Box, Button, Card, CardContent, Stack, Typography } from '@mui/material'
import { Link } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { apiUrl } from '../constants'

type Health = {
  ok: boolean
  paths: Record<string, boolean | string>
}

export default function DashboardPage() {
  const [h, setH] = useState<Health | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    fetch(apiUrl('/api/health'))
      .then((r) => r.json())
      .then(setH)
      .catch(() => setErr('API unreachable. Run `npm run api` from the web folder (repo root).'))
  }, [])

  return (
    <Stack spacing={3}>
      <Box>
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 700 }}>
          Dashboard
        </Typography>
        <Typography color="text.secondary" sx={{ maxWidth: 720 }}>
          AlbSL v2 recognizes Albanian finger-spelling from hand landmarks. Use Diagnose to inspect your dataset,
          Train to fit the temporal model, and Live for browser-based recognition with the same backend logic as
          the OpenCV desktop app.
        </Typography>
      </Box>

      {err && (
        <Typography color="error" variant="body2">
          {err}
        </Typography>
      )}

      <Box
        sx={{
          display: 'grid',
          gap: 2,
          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' },
        }}
      >
        {h &&
          Object.entries(h.paths).map(([k, v]) => (
            <Card key={k} variant="outlined">
              <CardContent>
                <Typography variant="overline" color="text.secondary">
                  {k}
                </Typography>
                <Typography variant="body1" sx={{ fontWeight: 600 }}>
                  {typeof v === 'boolean' ? (v ? 'Found' : 'Missing') : String(v)}
                </Typography>
              </CardContent>
            </Card>
          ))}
      </Box>

      <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 1 }}>
        <Button component={Link} to="/diagnose" variant="contained">
          Run diagnose
        </Button>
        <Button component={Link} to="/train" variant="outlined">
          Train model
        </Button>
        <Button component={Link} to="/live" variant="outlined">
          Open live
        </Button>
        <Button component={Link} to="/settings" variant="text">
          Paths
        </Button>
      </Stack>
    </Stack>
  )
}
