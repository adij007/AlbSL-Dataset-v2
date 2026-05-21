import MenuIcon from '@mui/icons-material/Menu'
import Brightness4Icon from '@mui/icons-material/Brightness4'
import Brightness7Icon from '@mui/icons-material/Brightness7'
import {
  AppBar,
  Box,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  useMediaQuery,
} from '@mui/material'
import { useTheme } from '@mui/material/styles'
import { Link, Outlet, useLocation } from 'react-router-dom'
import { useState } from 'react'
import HomeIcon from '@mui/icons-material/Home'
import BiotechIcon from '@mui/icons-material/Biotech'
import ModelTrainingIcon from '@mui/icons-material/ModelTraining'
import VideocamIcon from '@mui/icons-material/Videocam'
import SettingsIcon from '@mui/icons-material/Settings'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import PaletteIcon from '@mui/icons-material/Palette'
import { useAppTheme, useToggleTheme } from '../theme'

const drawerWidth = 260

const nav = [
  { to: '/', label: 'Dashboard', icon: <HomeIcon /> },
  { to: '/diagnose', label: 'Diagnose', icon: <BiotechIcon /> },
  { to: '/train', label: 'Train', icon: <ModelTrainingIcon /> },
  { to: '/live', label: 'Live', icon: <VideocamIcon /> },
  { to: '/settings', label: 'Settings', icon: <SettingsIcon /> },
  { to: '/about', label: 'About', icon: <InfoOutlinedIcon /> },
]

export default function AppLayout() {
  const theme = useTheme()
  const narrow = useMediaQuery(theme.breakpoints.down('md'))
  const [open, setOpen] = useState(false)
  const loc = useLocation()
  const toggleDark = useToggleTheme()
  const { mode, preset, setPreset } = useAppTheme()

  const drawer = (
    <Box sx={{ pt: 1 }}>
      <Typography variant="h6" sx={{ px: 2, pb: 1, fontWeight: 700 }}>
        AlbSL v2
      </Typography>
      <Divider />
      <List>
        {nav.map((item) => (
          <ListItemButton
            key={item.to}
            component={Link}
            to={item.to}
            selected={loc.pathname === item.to}
            onClick={() => narrow && setOpen(false)}
          >
            <ListItemIcon sx={{ color: 'inherit' }}>{item.icon}</ListItemIcon>
            <ListItemText primary={item.label} />
          </ListItemButton>
        ))}
      </List>
    </Box>
  )

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar
        position="fixed"
        elevation={0}
        color="transparent"
        sx={{
          borderBottom: 1,
          borderColor: 'divider',
          backdropFilter: 'blur(12px)',
          bgcolor: (t) => (t.palette.mode === 'dark' ? 'rgba(26,33,33,0.85)' : 'rgba(255,255,255,0.85)'),
        }}
      >
        <Toolbar>
          {narrow && (
            <IconButton edge="start" onClick={() => setOpen(true)} aria-label="menu">
              <MenuIcon />
            </IconButton>
          )}
          <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 700 }}>
            Albanian Sign — WebUI
          </Typography>
          <IconButton
            onClick={() => setPreset(preset === 'md3' ? 'oneui' : 'md3')}
            title="Toggle MD3 / One UI style"
            color="inherit"
          >
            <PaletteIcon />
          </IconButton>
          <IconButton onClick={toggleDark} color="inherit" aria-label="theme">
            {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
          </IconButton>
        </Toolbar>
      </AppBar>

      {!narrow && (
        <Drawer
          variant="permanent"
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
          }}
        >
          <Toolbar />
          {drawer}
        </Drawer>
      )}

      {narrow && (
        <Drawer open={open} onClose={() => setOpen(false)}>
          <Toolbar />
          {drawer}
        </Drawer>
      )}

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: { xs: 2, md: 3 },
          width: { md: `calc(100% - ${drawerWidth}px)` },
        }}
      >
        <Toolbar />
        <Outlet />
      </Box>
    </Box>
  )
}
