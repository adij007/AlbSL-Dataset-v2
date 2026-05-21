import { Stack, Typography } from '@mui/material'

export default function AboutPage() {
  return (
    <Stack spacing={2} sx={{ maxWidth: 720 }}>
      <Typography variant="h4" sx={{ fontWeight: 700 }} gutterBottom>
        About
      </Typography>
      <Typography>
        AlbSL v2 WebUI wraps the Python pipeline in{' '}
        <code>Script/albsl_app_v2.py</code>: dataset diagnostics, training, and live inference over WebSocket with
        optional fusion crops (JPEG) for parity with the desktop fusion path.
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Desktop live controls: L/K label, R record countdown, Space append, Backspace, Enter commit word, C clear, Y
        confirm to CSV, Q quit — see the script docstring. The web Live page maps these to buttons and hotkeys where
        applicable.
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Run the API from the repository root: <code>npm run api</code> inside the <code>web</code> folder, or{' '}
        <code>python -m uvicorn web.server.main:app --host 127.0.0.1 --port 8765</code>.
      </Typography>
    </Stack>
  )
}
