import { useMemo, useState } from 'react'
import { Info, Upload } from 'lucide-react'
import { useBuildLineups, useSlateCoverage } from '../api/optimizer'
import { getAvailableSeasons, getDefaultSeason, getCurrentNFLWeek } from '../utils/nflDate'
import PageHeader from '../components/ui/PageHeader'
import ErrorCard from '../components/ui/ErrorCard'

const SEASONS = getAvailableSeasons()
const WEEKS = Array.from({ length: 18 }, (_, i) => i + 1)
const OBJECTIVES = [
  { value: 'mean', label: 'Mean — expected points' },
  { value: 'median', label: 'Median' },
  { value: 'floor', label: 'Floor — cash games' },
  { value: 'ceiling', label: 'Ceiling — tournaments' },
]

export default function Optimizer() {
  const [season, setSeason] = useState(getDefaultSeason())
  const [week, setWeek] = useState(() => getCurrentNFLWeek(getDefaultSeason()))
  const [file, setFile] = useState<File | null>(null)
  const [numLineups, setNumLineups] = useState(5)
  const [objective, setObjective] = useState('mean')
  const [salaryCap, setSalaryCap] = useState(60000)
  const [maxUsage, setMaxUsage] = useState(50)
  const [exclude, setExclude] = useState('')

  const coverage = useSlateCoverage(season, week)
  const build = useBuildLineups()
  const result = build.data
  const error = build.error as { response?: { data?: { detail?: string } } } | null
  const detail = error?.response?.data?.detail

  const matchRate = useMemo(() => {
    if (!result || !result.slate_players) return null
    return Math.round((result.matched_to_projections / result.slate_players) * 100)
  }, [result])

  return (
    <div className="p-6">
      <PageHeader
        title="Lineup optimizer"
        subtitle="FanDuel lineups from the week's projections and the slate's salaries"
      />

      <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 mb-6 flex gap-3">
        <Info size={16} className="text-slate-400 shrink-0 mt-0.5" />
        <div className="text-xs text-slate-400 space-y-1.5">
          <p>
            FanDuel publishes salaries only in the slate's own player list, so download that
            CSV from the contest page and drop it here. Everything else — the projections, the
            floor and ceiling — comes from the board.
          </p>
          <p>
            Defenses use FanDuel's <span className="text-slate-200 font-medium">FPPG</span> from
            that file, because the model projects skill positions only.
          </p>
          <p>
            <span className="text-slate-200 font-medium">Mean</span> is the default objective
            because it is the one that was measured: across 2025, mean-optimised lineups beat
            ceiling-optimised ones on every metric, including best-of-five.
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-3 mb-5">
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Season
          <select
            value={season}
            onChange={(e) => setSeason(Number(e.target.value))}
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          >
            {SEASONS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Week
          <select
            value={week}
            onChange={(e) => setWeek(Number(e.target.value))}
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          >
            {WEEKS.map((w) => <option key={w} value={w}>Week {w}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Lineups
          <input
            type="number" min={1} max={50} value={numLineups}
            onChange={(e) => setNumLineups(Number(e.target.value))}
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm w-20"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Objective
          <select
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          >
            {OBJECTIVES.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Salary cap
          <input
            type="number" step={500} value={salaryCap}
            onChange={(e) => setSalaryCap(Number(e.target.value))}
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm w-28"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Max exposure %
          <input
            type="number" min={1} max={100} value={maxUsage}
            onChange={(e) => setMaxUsage(Number(e.target.value))}
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm w-24"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-slate-400 grow min-w-[12rem]">
          Exclude (comma separated)
          <input
            value={exclude}
            onChange={(e) => setExclude(e.target.value)}
            placeholder="Players to keep out of every lineup"
            className="bg-slate-700 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-3 py-1.5 text-sm w-full"
          />
        </label>
      </div>

      {/* Upload + run */}
      <div className="flex flex-wrap items-center gap-3 mb-6">
        <label className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 border border-slate-600 text-white rounded-lg px-3 py-2 text-sm cursor-pointer transition-colors">
          <Upload size={14} />
          {file ? file.name : 'Choose FanDuel slate CSV'}
          <input
            type="file" accept=".csv,text/csv" className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <button
          onClick={() => file && build.mutate({
            file, season, week, num_lineups: numLineups, objective,
            salary_cap: salaryCap, max_usage_percentage: maxUsage, exclude,
          })}
          disabled={!file || build.isPending}
          className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
            !file || build.isPending
              ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
              : 'bg-brand-green text-white hover:brightness-110'
          }`}
        >
          {build.isPending ? 'Building…' : 'Build lineups'}
        </button>
        {coverage.data?.status === 'success' && (
          <span className="text-xs text-slate-400">
            {coverage.data.projected_players} players projected for week {coverage.data.week}
          </span>
        )}
        {coverage.data?.status === 'no_data' && (
          <span className="text-xs text-amber-400">
            Nothing projected for that week yet — the board has to run first.
          </span>
        )}
      </div>

      {detail && <ErrorCard message={detail} />}

      {result && (
        <>
          <p className="text-slate-400 text-sm mb-1">
            {result.count} lineup{result.count === 1 ? '' : 's'} · {season} week {result.week} ·{' '}
            {result.objective} objective ·{' '}
            <span className={matchRate !== null && matchRate < 60 ? 'text-amber-400' : ''}>
              {result.matched_to_projections} of {result.slate_players} slate players matched to
              the board
            </span>
          </p>
          {result.note && <p className="text-amber-400 text-xs mb-3">{result.note}</p>}
          {matchRate !== null && matchRate < 60 && (
            <div className="bg-slate-800 border border-amber-700/40 rounded-xl p-3 mb-4 flex gap-2 text-xs text-amber-200/80">
              <Info size={14} className="shrink-0 mt-0.5" />
              <span>
                Only {matchRate}% of the slate matched. Salaries join to projections by name, so
                this usually means the CSV is for a different week than the board, or a
                showdown slate whose names are formatted differently.
              </span>
            </div>
          )}

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {result.data.map((lineup) => (
              <div key={lineup.lineup} className="bg-slate-800 border border-slate-700 rounded-xl p-4">
                <div className="flex items-baseline justify-between mb-3">
                  <h3 className="text-white font-medium text-sm">Lineup {lineup.lineup}</h3>
                  <span className="text-xs text-slate-400">
                    ${lineup.salary.toLocaleString()} ·{' '}
                    <span className="text-brand-green font-medium">{lineup.projected} pts</span>
                  </span>
                </div>
                <table className="w-full text-xs">
                  <tbody>
                    {lineup.players.map((p, i) => (
                      <tr key={`${p.player_name}-${i}`} className="border-t border-slate-700/60">
                        <td className="py-1.5 pr-2 text-slate-500 w-16">{p.roster_position}</td>
                        <td className="py-1.5 pr-2 text-slate-200">
                          {p.player_name}
                          {!p.from_model && (
                            <span className="text-slate-500" title="FanDuel's FPPG, not a model projection"> *</span>
                          )}
                        </td>
                        <td className="py-1.5 pr-2 text-right text-slate-400 tabular-nums">
                          ${p.salary?.toLocaleString()}
                        </td>
                        <td className="py-1.5 text-right text-slate-200 tabular-nums w-12">
                          {p.projected}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
          <p className="text-slate-500 text-xs mt-4">
            * FanDuel's own FPPG rather than a model projection — defenses, and anyone the name
            join missed.
          </p>
        </>
      )}
    </div>
  )
}
