import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Overview from './pages/Overview'
import Teams from './pages/Teams'
import TeamDetail from './pages/TeamDetail'
import Schedule from './pages/Schedule'
import Players from './pages/Players'
import Ratings from './pages/Ratings'
import PlayerGrades from './pages/PlayerGrades'
import SeasonProjections from './pages/SeasonProjections'
import Coaches from './pages/Coaches'
import Chat from './pages/Chat'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/teams" element={<Teams />} />
        <Route path="/teams/:abbr" element={<TeamDetail />} />
        <Route path="/schedule" element={<Schedule />} />
        <Route path="/players" element={<Players />} />
        <Route path="/ratings" element={<Ratings />} />
        <Route path="/player-grades" element={<PlayerGrades />} />
        <Route path="/projections" element={<SeasonProjections />} />
        <Route path="/coaches" element={<Coaches />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}
