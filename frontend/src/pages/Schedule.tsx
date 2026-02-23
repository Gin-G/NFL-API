import { useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useSchedules } from '../api/schedules'
import { getAvailableSeasons, getCurrentNFLSeason, getCurrentNFLWeek } from '../utils/nflDate'
import type { Game } from '../api/types'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'

const SEASONS = getAvailableSeasons()
const MAX_WEEKS = 18

function GameCard({ game }: { game: Game }) {
  const finished = game.home_score !== null && game.away_score !== null
  const awayWon = finished && (game.away_score ?? 0) > (game.home_score ?? 0)
  const homeWon = finished && (game.home_score ?? 0) > (game.away_score ?? 0)

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <div className="flex justify-between text-xs text-slate-400 mb-3">
        <span>{game.gameday ?? ''} {game.gametime ?? ''}</span>
        <span>{game.game_type ?? 'REG'}{game.div_game ? ' · DIV' : ''}</span>
      </div>

      <div className="grid grid-cols-3 items-center gap-2">
        <div className={`text-center ${awayWon ? 'text-white' : 'text-slate-400'}`}>
          <p className={`text-lg font-bold ${awayWon ? 'text-white' : ''}`}>{game.away_team}</p>
          {finished && (
            <p className={`text-2xl font-bold ${awayWon ? 'text-brand-gold' : ''}`}>
              {game.away_score}
            </p>
          )}
        </div>
        <div className="text-center text-slate-500 text-sm font-medium">
          {finished ? 'FINAL' : 'vs'}
        </div>
        <div className={`text-center ${homeWon ? 'text-white' : 'text-slate-400'}`}>
          <p className={`text-lg font-bold ${homeWon ? 'text-white' : ''}`}>{game.home_team}</p>
          {finished && (
            <p className={`text-2xl font-bold ${homeWon ? 'text-brand-gold' : ''}`}>
              {game.home_score}
            </p>
          )}
        </div>
      </div>

      {game.stadium && (
        <p className="text-xs text-slate-500 text-center mt-2 truncate">{game.stadium}</p>
      )}
    </div>
  )
}

export default function Schedule() {
  const defaultSeason = getCurrentNFLSeason()
  const [season, setSeason] = useState(defaultSeason)
  const [week, setWeek] = useState(getCurrentNFLWeek(defaultSeason))

  const { data, isLoading, error, refetch } = useSchedules(season, week)

  return (
    <div className="p-6">
      <PageHeader title="Schedule" subtitle="NFL game results and upcoming games" />

      {/* Controls */}
      <div className="flex flex-wrap gap-4 mb-6">
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-400">Season</label>
          <select
            value={season}
            onChange={(e) => setSeason(Number(e.target.value))}
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          >
            {SEASONS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-400">Week</label>
          <button
            onClick={() => setWeek((w) => Math.max(1, w - 1))}
            disabled={week === 1}
            className="p-1 rounded text-slate-400 hover:text-white disabled:opacity-30"
          >
            <ChevronLeft size={18} />
          </button>
          <span className="text-white font-semibold w-8 text-center">{week}</span>
          <button
            onClick={() => setWeek((w) => Math.min(MAX_WEEKS, w + 1))}
            disabled={week === MAX_WEEKS}
            className="p-1 rounded text-slate-400 hover:text-white disabled:opacity-30"
          >
            <ChevronRight size={18} />
          </button>
        </div>
      </div>

      {isLoading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} rows={4} />)}
        </div>
      )}

      {error && <ErrorCard message="Failed to load schedule" onRetry={() => refetch()} />}

      {data && (
        <>
          <p className="text-sm text-slate-400 mb-4">{data.total_games} games</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {data.data.map((g) => (
              <GameCard key={g.game_id ?? `${g.away_team}-${g.home_team}-${g.week}`} game={g} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
