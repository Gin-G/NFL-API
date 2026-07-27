import { useState } from 'react'
import { X } from 'lucide-react'
import { useGamePBP } from '../api/pbp'
import type { PBPPlay } from '../api/types'
import SkeletonCard from './ui/SkeletonCard'

const PLAY_TYPES = ['All', 'run', 'pass', 'punt', 'field_goal', 'kickoff', 'no_play']
const QUARTERS = ['All', '1', '2', '3', '4', 'OT']

function epaColor(epa: number | null | undefined): string {
  if (epa == null) return 'text-slate-400'
  if (epa > 0) return 'text-green-400'
  if (epa < 0) return 'text-red-400'
  return 'text-slate-400'
}

function downStr(play: PBPPlay): string {
  if (play.down == null) return '—'
  const suffix = ['st', 'nd', 'rd', 'th'][Math.min((play.down ?? 1) - 1, 3)]
  return `${play.down}${suffix} & ${play.ydstogo ?? '?'}`
}

export default function GamePBP({ gameId, onClose }: { gameId: string; onClose: () => void }) {
  const { data, isLoading } = useGamePBP(gameId)
  const [playTypeFilter, setPlayTypeFilter] = useState('All')
  const [quarterFilter, setQuarterFilter] = useState('All')

  // Derive header info from first play
  const firstPlay = data?.data?.[0]
  const awayTeam = firstPlay?.away_team ?? firstPlay?.posteam ?? '?'
  const homeTeam = firstPlay?.home_team ?? firstPlay?.defteam ?? '?'

  // Try to get team names from the game_id format: YYYY_WW_AWAY_HOME
  let awayDisplay = awayTeam
  let homeDisplay = homeTeam
  if (gameId) {
    const parts = gameId.split('_')
    if (parts.length === 4) {
      awayDisplay = parts[2]
      homeDisplay = parts[3]
    }
  }

  const plays: PBPPlay[] = data?.data ?? []

  const filtered = plays.filter((p) => {
    const matchType = playTypeFilter === 'All' || p.play_type === playTypeFilter
    const matchQtr =
      quarterFilter === 'All' ||
      (quarterFilter === 'OT' ? (p.qtr ?? 0) > 4 : String(p.qtr) === quarterFilter)
    return matchType && matchQtr
  })

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Backdrop */}
      <div className="flex-1 bg-black/50" onClick={onClose} />

      {/* Panel */}
      <div className="w-full max-w-2xl bg-slate-900 border-l border-slate-700 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-700 shrink-0">
          <div>
            <h2 className="text-lg font-bold text-white">
              {awayDisplay} @ {homeDisplay}
            </h2>
            <p className="text-xs text-slate-400">{gameId}</p>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white p-1">
            <X size={20} />
          </button>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-3 p-3 border-b border-slate-700 shrink-0">
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-400">Play Type</label>
            <select
              value={playTypeFilter}
              onChange={(e) => setPlayTypeFilter(e.target.value)}
              className="bg-slate-700 border border-slate-600 text-white rounded px-2 py-1 text-xs"
            >
              {PLAY_TYPES.map((t) => (
                <option key={t} value={t}>{t === 'All' ? 'All Types' : t}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-400">Quarter</label>
            <select
              value={quarterFilter}
              onChange={(e) => setQuarterFilter(e.target.value)}
              className="bg-slate-700 border border-slate-600 text-white rounded px-2 py-1 text-xs"
            >
              {QUARTERS.map((q) => (
                <option key={q} value={q}>{q === 'All' ? 'All Qtrs' : `Q${q}`}</option>
              ))}
            </select>
          </div>
          {data && (
            <span className="text-xs text-slate-500 self-center ml-auto">
              {filtered.length} / {data.total_plays} plays
            </span>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {isLoading && (
            <div className="p-4 space-y-3">
              {Array.from({ length: 8 }).map((_, i) => <SkeletonCard key={i} rows={2} />)}
            </div>
          )}

          {!isLoading && data?.status === 'no_data' && (
            <div className="p-6 text-center text-slate-400 text-sm">
              {data.message ?? 'No play-by-play data available for this game.'}
            </div>
          )}

          {!isLoading && filtered.length > 0 && (
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-slate-900 border-b border-slate-700">
                <tr className="text-slate-400 text-left">
                  <th className="py-2 px-3 font-medium w-10">Qtr</th>
                  <th className="py-2 px-2 font-medium w-14">Time</th>
                  <th className="py-2 px-2 font-medium w-10">Team</th>
                  <th className="py-2 px-2 font-medium w-20">Down</th>
                  <th className="py-2 px-2 font-medium">Description</th>
                  <th className="py-2 px-2 font-medium w-10 text-right">Yds</th>
                  <th className="py-2 px-3 font-medium w-14 text-right">EPA</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((play, i) => (
                  <tr
                    key={`${play.play_id ?? i}`}
                    className="border-b border-slate-800 hover:bg-slate-800/50"
                  >
                    <td className="py-1.5 px-3 text-slate-400">{play.qtr ?? '—'}</td>
                    <td className="py-1.5 px-2 text-slate-400">{play.time ?? '—'}</td>
                    <td className="py-1.5 px-2 text-slate-300 font-medium">{play.posteam ?? '—'}</td>
                    <td className="py-1.5 px-2 text-slate-400">{downStr(play)}</td>
                    <td className="py-1.5 px-2 text-slate-300 max-w-[280px]">
                      <span className="line-clamp-2">{play.desc ?? '—'}</span>
                    </td>
                    <td className="py-1.5 px-2 text-right text-slate-300">
                      {play.yards_gained != null ? play.yards_gained : '—'}
                    </td>
                    <td className={`py-1.5 px-3 text-right font-medium ${epaColor(play.epa)}`}>
                      {play.epa != null ? play.epa.toFixed(2) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {!isLoading && !data?.status && filtered.length === 0 && plays.length > 0 && (
            <div className="p-6 text-center text-slate-400 text-sm">
              No plays match the current filters.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
