import type { ReactNode } from 'react'
import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import { CssBaseline, GlobalStyles, ThemeProvider, createTheme } from '@mui/material'

export type ThemePreset = 'md3' | 'oneui'

type Ctx = {
  mode: 'light' | 'dark'
  setMode: (m: 'light' | 'dark') => void
  preset: ThemePreset
  setPreset: (p: ThemePreset) => void
}

const AppThemeCtx = createContext<Ctx | null>(null)

function md3Theme(mode: 'light' | 'dark') {
  return createTheme({
    palette: {
      mode,
      primary: { main: mode === 'light' ? '#006A6B' : '#4FD8D8' },
      secondary: { main: mode === 'light' ? '#4A4458' : '#CCC2DC' },
      background: {
        default: mode === 'light' ? '#F6FAFA' : '#0F1414',
        paper: mode === 'light' ? '#FFFFFF' : '#1A2121',
      },
    },
    shape: { borderRadius: 14 },
    typography: {
      fontFamily: '"DM Sans", "Source Sans 3", "Segoe UI", system-ui, sans-serif',
      h1: { fontWeight: 600, letterSpacing: '-0.02em' },
      h2: { fontWeight: 600 },
      button: { textTransform: 'none', fontWeight: 600 },
    },
    components: {
      MuiButton: { defaultProps: { disableElevation: true } },
      MuiCard: { styleOverrides: { root: { borderRadius: 18 } } },
    },
  })
}

function oneUiTheme(mode: 'light' | 'dark') {
  return createTheme({
    palette: {
      mode,
      primary: { main: mode === 'light' ? '#0077C8' : '#57A8FF' },
      secondary: { main: mode === 'light' ? '#5C5F62' : '#A0A4A8' },
      background: {
        default: mode === 'light' ? '#F7F7F7' : '#101010',
        paper: mode === 'light' ? '#FFFFFF' : '#1C1C1C',
      },
    },
    shape: { borderRadius: 22 },
    typography: {
      fontFamily: '"Source Sans 3", "Segoe UI", system-ui, sans-serif',
      h1: { fontWeight: 700 },
      button: { textTransform: 'none', fontWeight: 600 },
    },
    components: {
      MuiButton: { defaultProps: { disableElevation: true } },
      MuiCard: { styleOverrides: { root: { borderRadius: 26 } } },
    },
  })
}

export function AppThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<'light' | 'dark'>('dark')
  const [preset, setPreset] = useState<ThemePreset>('md3')

  const theme = useMemo(
    () => (preset === 'oneui' ? oneUiTheme(mode) : md3Theme(mode)),
    [mode, preset],
  )

  const value = useMemo(
    () => ({
      mode,
      setMode,
      preset,
      setPreset,
    }),
    [mode, preset],
  )

  return (
    <AppThemeCtx.Provider value={value}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <GlobalStyles
          styles={{
            '@import':
              'url(https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400..700;1,9..40,400..700&family=Source+Sans+3:ital,wght@0,400..700;1,400..700&display=swap)',
            body: { background: theme.palette.background.default },
          }}
        />
        {children}
      </ThemeProvider>
    </AppThemeCtx.Provider>
  )
}

export function useAppTheme() {
  const v = useContext(AppThemeCtx)
  if (!v) throw new Error('useAppTheme outside provider')
  return v
}

export function useToggleTheme() {
  const { mode, setMode } = useAppTheme()
  return useCallback(() => setMode(mode === 'dark' ? 'light' : 'dark'), [mode, setMode])
}
