import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppThemeProvider } from './theme'
import AppLayout from './layout/AppLayout'
import DashboardPage from './pages/DashboardPage'
import DiagnosePage from './pages/DiagnosePage'
import TrainPage from './pages/TrainPage'
import LivePage from './pages/LivePage'
import SettingsPage from './pages/SettingsPage'
import AboutPage from './pages/AboutPage'

export default function App() {
  return (
    <AppThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/diagnose" element={<DiagnosePage />} />
            <Route path="/train" element={<TrainPage />} />
            <Route path="/live" element={<LivePage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppThemeProvider>
  )
}
