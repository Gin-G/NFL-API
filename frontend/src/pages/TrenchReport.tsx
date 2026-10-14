import { useState } from 'react'
import { Info } from 'lucide-react'
import { useLineUnits, useLineMatchups, useLineBacktest } from '../api/trenches'
import { getAvailableSeasons, getDefaultSeason, getCurrentNFLWeek } from '../utils/nflDate'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'
import GradeScore from '../components/ui/GradeScore'
import LineMatchupTable from '../components/LineMatchupTable'
import LineBandTable from '../components/LineBandTable'

const SEASONS = getAvailableSeasons()
const WEEKS = Array.from({ length: 18 }, (_, i) => i + 1)
const UNITS = [
  { key: 'run_block', label: 'Run block', side: 'Offense' },
  { key: 'pass_block', label: 'Pass protect', side: 'Offense' },
  { key: 'run_stop', label: 'Run stop', side: 'Defense' },
  { key: 'pass_rush', label: 'Pass rush', side: 'Defense' },
] as const

export default function TrenchReport() {
  const defaultSeason = getDefaultSeason()
  const [season, setSeason] = useState(defaultSeason)
  const [week, setWeek] = useState(() => getCurrentNFLWeek(defaultSeason))
  const [tab, setTab] = useState<'matchups' | 'units'>('matchups')

  const matchups = useLineMatchups(season, week)
  const units = useLineUnits(season, week)
  const backtest = useLineBacktest(3)

  const unitRows = units.data?.units ?? []
  const matchupRows = matchups.data?.matchups ?? []
  const loading = matchups.isLoading || units.isLoading
  const error = matchups.error || units.error

  return (
    <div className="p-6">
      <PageHeader
        title="Trench Report"
        subtitle="Line-of-scrimmage ratings and weekly line matchups — 0–100, where 50 is league average"
      />

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
          value={week}
          onChange={(e) => setWeek(Number(e.target.value))}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          aria-label="Week"
        >
          {WEEKS.map((w) => (
            <option key={w} value={w}>Week {w}</option>
          ))}
        </select>
        <div className="flex rounded-lg overflow-hidden border border-slate-600">
          {(['matchups', 'units'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1.5 text-sm ${
                tab === t ? 'bg-slate-600 text-white' : 'bg-slate-800 text-slate-400'
              }`}
            >
              {t === 'matchups' ? 'This week' : 'Unit grades'}
            </button>
          ))}
        </div>
      </div>

      {/* How to read the number — the part that keeps it honest. */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 mb-6 flex gap-3">
        <Info size={16} className="text-slate-400 shrink-0 mt-0.5" />
        <div className="text-xs text-slate-400 space-y-1.5">
          <p>
            <span className="text-slate-200 font-medium">Matchup rating</span> — one unit against
            the one it faces. 50 is an <em>even</em> matchup, not a mediocre one, so a rating in the
            eighties means the edge is large rather than that the offense is good.
          </p>
          <p>
            <span className="text-slate-200 font-medium">Read the opponent column first.</span> A
            team's own line is already reflected in its players' recent form, so the half of this
            rating that is new information is the front it is running at. In four seasons of
            backtesting, the opponent's run-stop grade moved yards per carry by about{' '}
            <span className="text-slate-300">0.4 across its range</span>; the combined rating looks
            more dramatic mostly because good offenses rush well regardless.
          </p>
          <p>
            Ratings for a week use only games played before it, so what is shown here before
            kickoff is the same number the backtest below grades afterwards. Pass protection is the
            most stable of the four; a defense's sack rate is nearly noise, which is why pass rush
            is built from pressures instead.
          </p>
        </div>
      </div>

      {loading && (
        <div className="space-y-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <SkeletonCard key={i} rows={2} />
          ))}
        </div>
      )}

      {!loading && error && (
        <ErrorCard
          message="Could not load line ratings."
          onRetry={() => {
            matchups.refetch()
            units.refetch()
          }}
        />
      )}

      {!loading && !error && tab === 'matchups' && (
        matchupRows.length === 0 ? (
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 text-sm text-slate-400">
            No line matchups for {season} week {week} yet. Ratings need a few weeks of play-by-play
            in the season, and the week needs a schedule.
          </div>
        ) : (
          <LineMatchupTable rows={matchupRows} basis={matchups.data?.basis} />
        )
      )}

      {!loading && !error && tab === 'units' && (
        unitRows.length === 0 ? (
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 text-sm text-slate-400">
            No unit grades for {season} through week {week}.
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {UNITS.map((unit) => {
              const ranked = [...unitRows]
                .filter((r) => r[unit.key] !== null)
                .sort((a, b) => (b[unit.key] ?? 0) - (a[unit.key] ?? 0))
              return (
                <div key={unit.key} className="bg-slate-800 border border-slate-700 rounded-xl p-4">
                  <div className="flex items-baseline justify-between mb-3">
                    <h3 className="text-sm font-semibold text-slate-200">{unit.label}</h3>
                    <span className="text-xs text-slate-500">{unit.side}</span>
                  </div>
                  <ol className="space-y-1">
                    {ranked.map((row, i) => (
                      <li key={row.team} className="flex items-center gap-2 text-sm">
                        <span className="w-6 text-right text-xs text-slate-500 tabular-nums">
                          {i + 1}
                        </span>
                        <span className="w-12 font-medium text-slate-200">{row.team}</span>
                        <GradeScore value={row[unit.key]} size="sm" />
                      </li>
                    ))}
                  </ol>
                </div>
              )
            })}
          </div>
        )
      )}

      {backtest.data && (
        <div className="mt-8">
          <h2 className="text-sm font-semibold text-slate-200 mb-1">
            What these ratings actually delivered
          </h2>
          <p className="text-xs text-slate-500 mb-4">
            {backtest.data.team_games.toLocaleString()} team-games from week{' '}
            {backtest.data.first_week} on, {backtest.data.seasons.join('–')}. Each rating was built
            only from games played before the one it is graded against.
          </p>
          <div className="grid gap-4 lg:grid-cols-2">
            <LineBandTable
              title="By opponent front"
              note="The half that is new information — and the table to trust."
              rows={backtest.data.opponent_bands}
            />
            <LineBandTable
              title="By combined matchup rating"
              note="Separates cleanly, but partly because good offenses rush well."
              rows={backtest.data.matchup_bands}
            />
          </div>
        </div>
      )}
    </div>
  )
}
