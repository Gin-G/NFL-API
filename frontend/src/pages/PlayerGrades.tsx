import { useState } from 'react'
import { Info } from 'lucide-react'
import { usePlayerGradeLeaderboard } from '../api/grades'
import { getAvailableSeasons, getCurrentNFLSeason } from '../utils/nflDate'
import type { PlayerGradeEntry } from '../api/types'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'
import GradeScaleLegend from '../components/ui/GradeScaleLegend'
import PlayerGradesTable from '../components/PlayerGradesTable'
import PlayerPanel from '../components/PlayerPanel'
import GradeScore from '../components/ui/GradeScore'
import GradeMeter from '../components/ui/GradeMeter'
import { gradeLabel } from '../utils/gradeScale'

const SEASONS = getAvailableSeasons()
const POSITIONS = ['QB', 'RB', 'WR', 'TE']
const LIMITS = [25, 50, 100, 250]

/** Grade summary shown at the top of a player's slide-over panel. */
function GradeSummary({ player }: { player: PlayerGradeEntry }) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
      <div className="flex items-center gap-4 mb-4">
        <GradeScore value={player.grade} size="lg" />
        <div>
          <p className="text-white text-sm font-medium">{gradeLabel(player.grade)}</p>
          <p className="text-slate-400 text-xs mt-0.5">
            vs other {player.position}s · {player.games} game{player.games === 1 ? '' : 's'} ·{' '}
            {player.fppg.toFixed(1)} FPPG
          </p>
        </div>
      </div>
      <div className="space-y-3">
        <GradeMeter value={player.opportunity_grade} label="Opportunity (volume / role)" />
        <GradeMeter
          value={player.efficiency_grade}
          label="Efficiency (per play)"
          lowConfidence
          title="Per-play efficiency is noisier than opportunity."
        />
      </div>
    </div>
  )
}

export default function PlayerGrades() {
  // Player grades need played games, so default to the most recent season that
  // has them rather than the upcoming one.
  const [season, setSeason] = useState(getCurrentNFLSeason())
  const [position, setPosition] = useState('')
  const [throughWeek, setThroughWeek] = useState<number | undefined>(undefined)
  const [limit, setLimit] = useState(100)
  const [selected, setSelected] = useState<PlayerGradeEntry | null>(null)

  const { data, isLoading, error, refetch } = usePlayerGradeLeaderboard(
    season,
    position || undefined,
    throughWeek,
    limit,
  )

  return (
    <div className="p-6">
      <PageHeader
        title="Player Grades"
        subtitle="Opportunity-weighted player grades — 0–100, where 50 is average for the position"
      />

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <select
          value={season}
          onChange={(e) => setSeason(Number(e.target.value))}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          aria-label="Season"
        >
          {SEASONS.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <select
          value={position}
          onChange={(e) => setPosition(e.target.value)}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          aria-label="Position"
        >
          <option value="">All positions</option>
          {POSITIONS.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
        <select
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          aria-label="How many players"
        >
          {LIMITS.map((n) => (
            <option key={n} value={n}>Top {n} by grade</option>
          ))}
        </select>
        <GradeScaleLegend average="positional average" />
      </div>

      {/* Through-week slider */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <label className="text-xs text-slate-400 uppercase tracking-wide" htmlFor="through-week">
          Through week
        </label>
        <input
          id="through-week"
          type="range"
          min={1}
          max={18}
          step={1}
          value={throughWeek ?? 18}
          onChange={(e) => setThroughWeek(Number(e.target.value))}
          className="w-48 accent-[#1a7a4a]"
        />
        <span className="text-sm text-white tabular-nums w-24">
          {throughWeek ? `Week ${throughWeek}` : 'Full season'}
        </span>
        {throughWeek !== undefined && (
          <button
            onClick={() => setThroughWeek(undefined)}
            className="text-xs text-slate-400 hover:text-white underline transition-colors"
          >
            Reset to full season
          </button>
        )}
      </div>

      {/* How to read these grades */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 mb-6 flex gap-3">
        <Info size={16} className="text-slate-400 shrink-0 mt-0.5" />
        <div className="text-xs text-slate-400 space-y-1.5">
          <p>
            Grades are <span className="text-slate-200 font-medium">position-relative</span>: 50 is
            average <em>for that position</em>, so compare a WR to other WRs, not to QBs.
          </p>
          <p>
            They are <span className="text-slate-200 font-medium">opportunity-weighted</span> (WOPR,
            target share, carries) — they reward role, not just past points. A high-opportunity
            player who underperformed still grades well, which is the point: opportunity is the
            stable, predictive part. Per-play{' '}
            <span className="text-slate-300">efficiency is noisier</span> and is shown dimmed.
          </p>
          <p>
            Each grade blends a preseason prior with in-season results and updates as games are
            played. Drag <span className="text-slate-300">Through week</span> for a point-in-time
            view of how a grade moved over the season.
          </p>
        </div>
      </div>

      {isLoading && (
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <SkeletonCard key={i} rows={2} />
          ))}
        </div>
      )}

      {error && <ErrorCard message="Failed to load player grades" onRetry={() => refetch()} />}

      {data?.status === 'no_data' && (
        <p className="text-slate-400 text-sm">
          {data.message} Player grades need played games — pick a completed season.
        </p>
      )}

      {data && data.status !== 'no_data' && (
        <>
          <p className="text-slate-400 text-sm mb-3">
            {data.total} graded {position ? `${position}s` : 'players'} ·{' '}
            {throughWeek ? `through week ${throughWeek}` : 'full season'} · showing top{' '}
            {Math.min(limit, data.data.length)} by grade — click a player for their breakdown
          </p>
          <PlayerGradesTable players={data.data} onSelect={setSelected} />
        </>
      )}

      {selected && (
        <PlayerPanel
          playerId={selected.player_id}
          playerName={selected.player_name}
          subtitle={`${selected.position} · ${season}${throughWeek ? ` · through week ${throughWeek}` : ''}`}
          season={season}
          onClose={() => setSelected(null)}
        >
          <GradeSummary player={selected} />
        </PlayerPanel>
      )}
    </div>
  )
}
