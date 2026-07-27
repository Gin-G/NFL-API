import { useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useSchedules } from '../api/schedules'
import { getAvailableSeasons, getCurrentNFLSeason, getCurrentNFLWeek } from '../utils/nflDate'
import type { Game } from '../api/types'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'
import GamePBP from '../components/GamePBP'

const SEASONS = getAvailableSeasons()
const MAX_WEEKS = 22

const PLAYOFF_ROUND: Record<number, string> = {
  19: 'Wild Card',
  20: 'Divisional',
  21: 'Conference',
  22: 'Super Bowl',
}

function weekLabel(week: number): string {
  return PLAYOFF_ROUND[week] ?? `Week ${week}`
}

function GameCard({ game, onClick }: { game: Game; onClick?: () => void }) {
  const finished = game.home_score !== null && game.away_score !== null
  const awayWon = finished && (game.away_score ?? 0) > (game.home_score ?? 0)
  const homeWon = finished && (game.home_score ?? 0) > (game.away_score ?? 0)
  const isOT = finished && game.overtime === 1

  return (
    <div
      className={`bg-slate-800 rounded-xl p-4 border border-slate-700 ${onClick ? 'cursor-pointer hover:ring-1 ring-slate-500 transition-shadow' : ''}`}
      onClick={onClick}
    >
      <div className="flex justify-between text-xs text-slate-400 mb-3">
        <span>{game.gameday ?? ''}{game.gametime ? ` · ${game.gametime}` : ''}</span>
        <span className="flex items-center gap-1">
          {game.div_game ? <span className="text-slate-500">DIV ·</span> : null}
          {game.game_type && game.game_type !== 'REG' ? (
            <span className="text-amber-400 font-semibold">{game.game_type}</span>
          ) : null}
        </span>
      </div>

      <div className="grid grid-cols-3 items-center gap-2">
        <div className={`text-center ${awayWon ? 'text-white' : 'text-slate-400'}`}>
          <p className={`text-lg font-bold ${awayWon ? 'text-white' : ''}`}>{game.away_team}</p>
          {game.away_qb_name && (
            <p className="text-xs text-slate-500 truncate">{game.away_qb_name}</p>
          )}
          {finished && (
            <p className={`text-2xl font-bold mt-1 ${awayWon ? 'text-brand-gold' : ''}`}>
              {game.away_score}
            </p>
          )}
        </div>
        <div className="text-center text-slate-500 text-sm font-medium">
          {finished ? (isOT ? 'FINAL/OT' : 'FINAL') : 'vs'}
        </div>
        <div className={`text-center ${homeWon ? 'text-white' : 'text-slate-400'}`}>
          <p className={`text-lg font-bold ${homeWon ? 'text-white' : ''}`}>{game.home_team}</p>
          {game.home_qb_name && (
            <p className="text-xs text-slate-500 truncate">{game.home_qb_name}</p>
          )}
          {finished && (
            <p className={`text-2xl font-bold mt-1 ${homeWon ? 'text-brand-gold' : ''}`}>
              {game.home_score}
            </p>
          )}
        </div>
      </div>

      {(game.away_coach || game.home_coach) && (
        <div className="flex justify-between mt-2 text-xs text-slate-600">
          <span className="truncate max-w-[45%]">{game.away_coach ?? ''}</span>
          <span className="truncate max-w-[45%] text-right">{game.home_coach ?? ''}</span>
        </div>
      )}

      {game.stadium && (
        <p className="text-xs text-slate-500 text-center mt-1 truncate">{game.stadium}</p>
      )}
    </div>
  )
}

export default function Schedule() {
  const defaultSeason = getCurrentNFLSeason()
  const [season, setSeason] = useState(defaultSeason)
  const [week, setWeek] = useState(getCurrentNFLWeek(defaultSeason))
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null)

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
          <span className="text-white font-semibold min-w-[80px] text-center">{weekLabel(week)}</span>
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
            {data.data.map((g) => {
              const finished = g.home_score !== null && g.away_score !== null
              return (
                <GameCard
                  key={g.game_id ?? `${g.away_team}-${g.home_team}-${g.week}`}
                  game={g}
                  onClick={finished && g.game_id ? () => setSelectedGameId(g.game_id!) : undefined}
                />
              )
            })}
          </div>
        </>
      )}

      {selectedGameId && (
        <GamePBP gameId={selectedGameId} onClose={() => setSelectedGameId(null)} />
      )}
    </div>
  )
}
