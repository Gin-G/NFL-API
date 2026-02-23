import { useState } from 'react'
import { useRosters, usePlayerStats } from '../api/players'
import { getAvailableSeasons, getCurrentNFLSeason } from '../utils/nflDate'
import type { RosterPlayer, PlayerStat } from '../api/types'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'
import StatBarChart from '../components/charts/StatBarChart'

const SEASONS = getAvailableSeasons()
const POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K', 'DT', 'DE', 'LB', 'CB', 'S']

type Tab = 'roster' | 'stats'

function RosterTable({ players }: { players: RosterPlayer[] }) {
  if (!players.length) return <p className="text-slate-400 text-sm">No players found.</p>

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700 text-slate-400 text-left">
            <th className="py-2 pr-4 font-medium">#</th>
            <th className="py-2 pr-4 font-medium">Name</th>
            <th className="py-2 pr-4 font-medium">Pos</th>
            <th className="py-2 pr-4 font-medium">Team</th>
            <th className="py-2 pr-4 font-medium">Ht</th>
            <th className="py-2 pr-4 font-medium">Wt</th>
            <th className="py-2 font-medium">College</th>
          </tr>
        </thead>
        <tbody>
          {players.slice(0, 100).map((p, i) => (
            <tr key={`${p.player_id}-${i}`} className="border-b border-slate-800 hover:bg-slate-800/50">
              <td className="py-2 pr-4 text-slate-400">{p.jersey_number ?? '—'}</td>
              <td className="py-2 pr-4 text-white font-medium">{p.player_name}</td>
              <td className="py-2 pr-4 text-slate-300">{p.position}</td>
              <td className="py-2 pr-4 text-slate-300">{p.team}</td>
              <td className="py-2 pr-4 text-slate-400">{p.height ?? '—'}</td>
              <td className="py-2 pr-4 text-slate-400">{p.weight ?? '—'}</td>
              <td className="py-2 text-slate-400 truncate max-w-[120px]">{p.college ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function statLeaders(stats: PlayerStat[], field: keyof PlayerStat, n = 10) {
  return stats
    .filter((s) => typeof s[field] === 'number' && (s[field] as number) > 0)
    .sort((a, b) => ((b[field] as number) ?? 0) - ((a[field] as number) ?? 0))
    .slice(0, n)
    .map((s) => ({ name: s.player_name, value: s[field] as number }))
}

function StatsPanel({ season, team, position, week }: { season: number; team?: string; position?: string; week?: number }) {
  const { data, isLoading, error, refetch } = usePlayerStats(season, position, team, week)

  if (isLoading) return <div className="space-y-3">{Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} rows={3} />)}</div>
  if (error) return <ErrorCard message="Failed to load stats" onRetry={() => refetch()} />
  if (!data?.data.length) return <p className="text-slate-400 text-sm">No stats found for these filters.</p>

  const stats = data.data
  const showPassing = !position || position === 'QB'
  const showRushing = !position || ['QB', 'RB'].includes(position)
  const showReceiving = !position || ['WR', 'TE', 'RB'].includes(position)

  return (
    <div className="space-y-8">
      <p className="text-slate-400 text-sm">{data.total_records} records</p>

      {showPassing && (
        <div>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Passing Yards Leaders</h3>
          <StatBarChart data={statLeaders(stats, 'passing_yards')} label="Pass Yds" />
        </div>
      )}
      {showRushing && (
        <div>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Rushing Yards Leaders</h3>
          <StatBarChart data={statLeaders(stats, 'rushing_yards')} label="Rush Yds" color="#d4b94e" />
        </div>
      )}
      {showReceiving && (
        <div>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Receiving Yards Leaders</h3>
          <StatBarChart data={statLeaders(stats, 'receiving_yards')} label="Rec Yds" color="#3b82f6" />
        </div>
      )}
    </div>
  )
}

export default function Players() {
  const [tab, setTab] = useState<Tab>('roster')
  const [season, setSeason] = useState(getCurrentNFLSeason())
  const [team, setTeam] = useState('')
  const [position, setPosition] = useState('')
  const [week, setWeek] = useState<number | undefined>(undefined)

  const {
    data: rosterData,
    isLoading: rosterLoading,
    error: rosterError,
    refetch: rosterRefetch,
  } = useRosters(season, team || undefined, position || undefined, week)

  return (
    <div className="p-6">
      <PageHeader title="Players" subtitle="Rosters and statistics" />

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-6">
        <select
          value={season}
          onChange={(e) => setSeason(Number(e.target.value))}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
        >
          {SEASONS.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <input
          placeholder="Team (e.g. KC)"
          value={team}
          onChange={(e) => setTeam(e.target.value.toUpperCase().slice(0, 3))}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm w-28 placeholder-slate-500"
        />
        <select
          value={position}
          onChange={(e) => setPosition(e.target.value)}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
        >
          <option value="">All positions</option>
          {POSITIONS.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <input
          type="number"
          placeholder="Week"
          value={week ?? ''}
          onChange={(e) => setWeek(e.target.value ? Number(e.target.value) : undefined)}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm w-20 placeholder-slate-500"
          min={1}
          max={18}
        />
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-6 bg-slate-800 rounded-lg p-1 w-fit">
        {(['roster', 'stats'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-1.5 rounded-md text-sm font-medium capitalize transition-colors ${
              tab === t ? 'bg-brand-green text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'roster' && (
        <>
          {rosterLoading && <div className="space-y-2">{Array.from({ length: 5 }).map((_, i) => <SkeletonCard key={i} />)}</div>}
          {rosterError && <ErrorCard message="Failed to load roster" onRetry={() => rosterRefetch()} />}
          {rosterData && (
            <>
              <p className="text-slate-400 text-sm mb-3">{rosterData.total_players} players (showing up to 100)</p>
              <RosterTable players={rosterData.data} />
            </>
          )}
        </>
      )}

      {tab === 'stats' && (
        <StatsPanel
          season={season}
          team={team || undefined}
          position={position || undefined}
          week={week}
        />
      )}
    </div>
  )
}
