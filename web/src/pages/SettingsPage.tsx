import { Alert, Button, Stack, TextField, Typography } from '@mui/material'
import { useState } from 'react'
import { DEFAULT_PATHS } from '../constants'
import { loadPathSettings, savePathSettings } from '../settingsStorage'

export default function SettingsPage() {
  const [s, setS] = useState(loadPathSettings)
  const [saved, setSaved] = useState(false)

  function save() {
    savePathSettings(s)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  function reset() {
    setS({ ...DEFAULT_PATHS })
  }

  return (
    <Stack spacing={2} sx={{ maxWidth: 560 }}>
      <Typography variant="h4" sx={{ fontWeight: 700 }}>
        Paths
      </Typography>
      <Typography color="text.secondary">
        Stored in <code>localStorage</code>. Live and other pages read these when connecting to the API.
      </Typography>
      {(Object.keys(DEFAULT_PATHS) as (keyof typeof DEFAULT_PATHS)[]).map((key) => (
        <TextField
          key={key}
          label={key}
          fullWidth
          value={s[key]}
          onChange={(e) => setS((prev) => ({ ...prev, [key]: e.target.value }))}
        />
      ))}
      <Stack direction="row" sx={{ gap: 1 }}>
        <Button variant="contained" onClick={save}>
          Save
        </Button>
        <Button variant="outlined" onClick={reset}>
          Reset defaults
        </Button>
      </Stack>
      {saved && <Alert severity="success">Saved.</Alert>}
    </Stack>
  )
}
