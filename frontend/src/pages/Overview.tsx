import { useQuery } from '@tanstack/react-query'
import { CheckCircle, XCircle, Users, Calendar, UserRound, Briefcase } from 'lucide-react'
import { apiClient } from '../api/client'
import type { HealthResponse } from '../api/types'
import { useWeeklySchedule } from '../api/schedules'
import PageHeader from '../components/ui/PageHeader'
import StatCard from '../components/ui/StatCard'
import SkeletonCard from '../components/ui/SkeletonCard'

const CURRENT_SEASON = 2024
const CURRENT_WEEK = 18

function HealthBanner() {
  const { data, isLoading } = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const { data } = await apiClient.get<HealthResponse>('/health')
      return data
    },
    refetchInterval: 60_000,
  })

  if (isLoading) return <div className="h-10 bg-slate-800 rounded-lg animate-pulse mb-6" />

  const healthy = data?.status === 'healthy'
  return (
    <div
      className={`flex items-center gap-3 px-4 py-3 rounded-lg mb-6 text-sm font-medium ${
        healthy ? 'bg-green-900/40 text-green-300 border border-green-800' : 'bg-red-900/40 text-red-300 border border-red-800'
      }`}
    >
      {healthy ? <CheckCircle size={18} /> : <XCircle size={18} />}
      <span>API {healthy ? 'healthy' : 'unavailable'}</span>
      {data?.systems && (
        <span className="ml-auto text-xs opacity-70">
          Player grading: {data.systems.player_grading ? 'on' : 'off'} &nbsp;·&nbsp;
          Coaching analytics: {data.systems.coaching_analytics ? 'on' : 'off'}
        </span>
      )}
    </div>
  )
}

function CurrentWeekGames() {
  const { data, isLoading } = useWeeklySchedule(CURRENT_SEASON, CURRENT_WEEK)

  if (isLoading) return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {[1, 2, 3, 4].map((i) => <SkeletonCard key={i} rows={3} />)}
    </div>
  )

  const games = data?.games ?? []

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {games.slice(0, 8).map((g) => (
        <div key={g.game_id ?? `${g.away_team}-${g.home_team}`} className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex justify-between items-center text-xs text-slate-400 mb-2">
            <span>{g.gameday ?? ''} {g.gametime ?? ''}</span>
            <span className="text-slate-500">{g.stadium ?? ''}</span>
          </div>
          <div className="flex justify-between items-center">
            <div className="text-center">
              <p className="font-bold text-lg text-white">{g.away_team}</p>
              {g.away_score !== null && (
                <p className="text-2xl font-bold text-brand-gold">{g.away_score}</p>
              )}
            </div>
            <span className="text-slate-500 text-sm font-medium">@</span>
            <div className="text-center">
              <p className="font-bold text-lg text-white">{g.home_team}</p>
              {g.home_score !== null && (
                <p className="text-2xl font-bold text-brand-gold">{g.home_score}</p>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

export default function Overview() {
  return (
    <div className="p-6">
      <PageHeader title="Overview" subtitle={`NFL ${CURRENT_SEASON} Season Dashboard`} />
      <HealthBanner />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard label="Teams" value="32" icon={<Users size={22} />} />
        <StatCard label="Season" value={CURRENT_SEASON} icon={<Calendar size={22} />} />
        <StatCard label="Current Week" value={`Week ${CURRENT_WEEK}`} icon={<UserRound size={22} />} />
        <StatCard label="Data Source" value="nfl_data_py" icon={<Briefcase size={22} />} />
      </div>

      <h2 className="text-lg font-semibold text-slate-200 mb-4">
        Week {CURRENT_WEEK} Games
      </h2>
      <CurrentWeekGames />
    </div>
  )
}
