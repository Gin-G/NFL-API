import { Fragment, useMemo, useState } from 'react'
import { ChevronDown, ChevronRight, Download, Info, Trash2, Upload } from 'lucide-react'
import {
  downloadFanduelUpload,
  useBuildLineups, useForgetSlate, useLineupExposure, useLineupPerformance, useSavedSlates,
  useSlateCoverage,
} from '../api/optimizer'
import { getAvailableSeasons, getDefaultSeason, getCurrentNFLWeek } from '../utils/nflDate'
import PageHeader from '../components/ui/PageHeader'
import ErrorCard from '../components/ui/ErrorCard'

const SEASONS = getAvailableSeasons()
const WEEKS = Array.from({ length: 18 }, (_, i) => i + 1)
const SLATES = [
  { value: 'all', label: 'Thu–Mon (every game)' },
  { value: 'main', label: 'Sunday main (1pm & 4pm)' },
  { value: 'sunday', label: 'Sunday (all, incl. London & SNF)' },
  { value: 'primetime', label: 'Primetime (TNF, SNF, MNF)' },
]
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
  const [slate, setSlate] = useState('all')
  const [salaryCap, setSalaryCap] = useState(60000)
  const [maxUsage, setMaxUsage] = useState(50)
  // How the nine picks may relate to each other — the one thing a board of
  // per-player projections cannot say. 4 is FanDuel's own per-team limit.
  const [maxPerTeam, setMaxPerTeam] = useState(4)
  const [stackQb, setStackQb] = useState(0)
  const [rbWithQb, setRbWithQb] = useState(true)
  const [ownershipWeight, setOwnershipWeight] = useState(0)
  const [contestId, setContestId] = useState('')
  const [exclude, setExclude] = useState('')

  const [exposureScope, setExposureScope] = useState<'build' | 'week'>('build')
  const [openLineups, setOpenLineups] = useState<Set<string>>(new Set())

  const coverage = useSlateCoverage(season, week)
  const slates = useSavedSlates(season, week)
  const performance = useLineupPerformance(season)
  // The same record scoped to the chosen week, which is the only call that
  // returns the lineups themselves rather than the averages.
  const weekRecord = useLineupPerformance(season, week)
  const exposure = useLineupExposure(season, week)
  const forget = useForgetSlate()
  const build = useBuildLineups()
  const result = build.data
  const error = build.error as { response?: { data?: { detail?: string } } } | null
  const detail = error?.response?.data?.detail

  // The slate on file for this week. A build with no new file uses it, which is
  // the whole point: FanDuel's CSV is downloaded once, not once per question.
  const saved = useMemo(() => {
    const rows = slates.data?.data?.filter((s) => s.week === week) ?? []
    const classic = rows.filter((s) => s.contest === 'classic')
    return (classic.length ? classic : rows)[0] ?? null
  }, [slates.data, week])
  const canBuild = Boolean(file || saved)

  // Exposure, either of the lineups just built or of every build of the week.
  // The cap is a promise about one build; pooled is where concentration shows.
  const pooledExposure = exposure.data?.status === 'success' ? exposure.data : null
  const scope = result ? exposureScope : 'week'
  const exposureRows = scope === 'build' ? result?.exposure ?? [] : pooledExposure?.data ?? []
  const exposureLineups = scope === 'build' ? result?.count ?? 0 : pooledExposure?.lineups ?? 0
  const overCap = new Set(scope === 'week' ? pooledExposure?.over_cap ?? [] : [])
  const showsActual = exposureRows.some((r) => r.actual !== null)

  // The chosen week's own numbers: its objectives, and how much of it is scored.
  const weekObjectives = weekRecord.data?.by_objective ?? []
  const weekContests = new Set(weekObjectives.map((o) => o.contest))
  const weekVerdict = weekRecord.data?.objective_verdict ?? null
  const weekLineupCount = weekRecord.data?.totals?.lineups ?? 0
  const weekScored = weekRecord.data?.totals?.scored_lineups ?? 0

  const toggleLineup = (key: string) =>
    setOpenLineups((open) => {
      const next = new Set(open)
      next.has(key) ? next.delete(key) : next.add(key)
      return next
    })

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
            Take the <span className="text-slate-200 font-medium">Thu–Mon</span> export to get
            every player, then pick the slate you are actually entering. The file carries no
            kickoff times, so the windows come from our schedule.
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
          Slate
          <select
            value={slate}
            onChange={(e) => setSlate(e.target.value)}
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          >
            {SLATES.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
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
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Max per team
          <select
            value={maxPerTeam}
            onChange={(e) => setMaxPerTeam(Number(e.target.value))}
            title="At most this many of the nine from one team, the defense included. FanDuel's own limit is 4."
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm w-24"
          >
            {[1, 2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          QB stack
          <select
            value={stackQb}
            onChange={(e) => setStackQb(Number(e.target.value))}
            title="Require this many of the quarterback's own receivers or tight ends."
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm w-28"
          >
            <option value={0}>none</option>
            <option value={1}>+1 pass catcher</option>
            <option value={2}>+2 pass catchers</option>
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          RB with his QB
          <select
            value={rbWithQb ? 'yes' : 'no'}
            onChange={(e) => setRbWithQb(e.target.value === 'yes')}
            title="A running back and his own quarterback compete for the same drives."
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm w-24"
          >
            <option value="yes">allowed</option>
            <option value="no">never</option>
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Fade the field
          <select
            value={ownershipWeight}
            onChange={(e) => setOwnershipWeight(Number(e.target.value))}
            title="Charge each player for how much of the field owns him. Buys contrarian lineups with expected points — a tournament setting, not a better build."
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm w-32"
          >
            <option value={0}>off</option>
            <option value={0.05}>slight</option>
            <option value={0.1}>moderate</option>
            <option value={0.25}>heavy</option>
          </select>
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
          {file ? file.name : saved ? 'Replace slate CSV' : 'Choose FanDuel slate CSV'}
          <input
            type="file" accept=".csv,text/csv" className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <button
          onClick={() => canBuild && build.mutate({
            file, season, week, num_lineups: numLineups, objective,
            salary_cap: salaryCap, max_usage_percentage: maxUsage, exclude, slate,
            max_per_team: maxPerTeam, stack_qb: stackQb, rb_with_qb: rbWithQb,
            ownership_weight: ownershipWeight,
          }, {
            // The file has been saved server-side now, so later builds should
            // use the stored copy rather than re-uploading the same bytes.
            onSuccess: () => setFile(null),
          })}
          disabled={!canBuild || build.isPending}
          className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
            !canBuild || build.isPending
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

      {saved && !file && (
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-3 mb-4 flex flex-wrap items-center gap-2 text-xs text-slate-400">
          <span className="text-slate-200 font-medium">Slate on file</span>
          <span>
            {saved.filename} · {saved.players} players
            {saved.contest === 'showdown' ? ' · Showdown' : ''}
            {saved.teams.length ? ` · ${saved.teams.length} teams` : ''}
            {saved.uploaded_at
              ? ` · uploaded ${new Date(saved.uploaded_at + 'Z').toLocaleString()}`
              : ''}
          </span>
          <span className="text-slate-500">
            Build as often as you like — no re-upload. A new file replaces it.
          </span>
          <button
            onClick={() => forget.mutate(saved.slate_id)}
            disabled={forget.isPending}
            className="ml-auto flex items-center gap-1 text-slate-500 hover:text-red-400 transition-colors"
            title="Forget this slate"
          >
            <Trash2 size={12} /> Forget
          </button>
        </div>
      )}

      {!saved && !file && (
        <p className="text-xs text-slate-500 mb-4">
          Nothing saved for week {week} yet. Upload the contest player list once; every
          build after that reuses it.
        </p>
      )}

      {detail && <ErrorCard message={detail} />}

      {result && (
        <>
          <p className="text-slate-400 text-sm mb-1">
            {result.count} lineup{result.count === 1 ? '' : 's'} · {season} week {result.week} ·{' '}
            {result.objective} objective ·{' '}
            {result.slate !== 'all' && (
              <>
                {result.games.length} game{result.games.length === 1 ? '' : 's'} in the{' '}
                {result.slate} slate ({result.slate_players} of {result.players_in_file} players) ·{' '}
              </>
            )}
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
                {/* table-fixed, so the name column absorbs the slack and a long
                    name truncates inside the card rather than widening the table
                    past it. The numeric columns are given widths for the same
                    reason: without them an auto table hands the name whatever is
                    left after the content, which is nothing. */}
                <table className="w-full table-fixed text-xs">
                  <tbody>
                    {lineup.players.map((p, i) => (
                      <tr key={`${p.player_name}-${i}`} className="border-t border-slate-700/60">
                        <td className="py-1.5 pr-2 text-slate-500 w-11">{p.roster_position}</td>
                        <td className="py-1.5 pr-2 text-slate-200 truncate" title={p.player_name}>
                          {p.player_name}
                          {!p.from_model && (
                            <span className="text-slate-500" title="FanDuel's FPPG, not a model projection"> *</span>
                          )}
                        </td>
                        <td className="py-1.5 pr-2 text-right text-slate-400 tabular-nums w-16 whitespace-nowrap">
                          ${p.salary?.toLocaleString()}
                        </td>
                        <td className="py-1.5 text-right text-slate-200 tabular-nums w-11 whitespace-nowrap">
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
          {result.build_id !== null && (
            <div className="flex flex-wrap items-center gap-3 mt-3">
              <button
                onClick={() => downloadFanduelUpload(result.build_id as number,
                                                     contestId.trim() || undefined)}
                title="Nine columns in FanDuel's order, one row per lineup — bulk entry takes it as is."
                className="flex items-center gap-2 bg-emerald-700 hover:bg-emerald-600 text-white rounded-lg px-3 py-1.5 text-sm transition-colors"
              >
                <Download size={14} />
                Download for FanDuel
              </button>
              <label className="flex items-center gap-2 text-xs text-slate-400">
                Contest ID
                <input
                  value={contestId}
                  onChange={(e) => setContestId(e.target.value)}
                  placeholder="optional"
                  title="A FanDuel player id is <contest>-<player>. Entering a different contest from the one you priced off? Put its id here — the number in the file name — or FanDuel rejects the whole upload."
                  className="bg-slate-700 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-2 py-1 text-xs w-28"
                />
              </label>
              <span className="text-slate-500 text-xs">
                Recorded as build {result.build_id}; it gets scored once the week is played.
              </span>
            </div>
          )}
        </>
      )}

      {/* Who the lineups actually lean on. The cap bounds this within one build;
          pooled across the week nothing bounds it, which is the number worth seeing. */}
      {exposureRows.length > 0 && (
        <div className="mt-10">
          <div className="flex items-baseline justify-between gap-4 mb-1">
            <h2 className="text-white font-medium text-sm">Player exposure</h2>
            {result && pooledExposure && (
              <div className="flex text-xs rounded-lg overflow-hidden border border-slate-700">
                {([
                  ['build', 'This build'],
                  ['week', `All builds · week ${week}`],
                ] as const).map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setExposureScope(value)}
                    className={
                      scope === value
                        ? 'px-3 py-1 bg-slate-700 text-white'
                        : 'px-3 py-1 text-slate-400 hover:text-slate-200'
                    }
                  >
                    {label}
                  </button>
                ))}
              </div>
            )}
          </div>
          <p className="text-xs text-slate-500 mb-3">
            {scope === 'build' ? (
              <>
                Share of the {exposureLineups} lineup{exposureLineups === 1 ? '' : 's'} just
                built, capped at {result?.exposure_cap}% per player.
              </>
            ) : (
              <>
                Share of all {exposureLineups} lineup{exposureLineups === 1 ? '' : 's'} across{' '}
                {pooledExposure?.builds} build{pooledExposure?.builds === 1 ? '' : 's'} this
                week, from {pooledExposure?.players_used} different players. The per-player cap
                binds inside a build only — pooled, nothing stops the same name appearing in
                every one of them.
              </>
            )}
          </p>

          {overCap.size > 0 && (
            <div className="bg-slate-800 border border-amber-700/40 rounded-xl p-3 mb-3 flex gap-2 text-xs text-amber-200/80">
              <Info size={14} className="shrink-0 mt-0.5" />
              <span>
                {Array.from(overCap).join(', ')}{' '}
                {overCap.size === 1 ? 'is' : 'are'} in more of the week than any single build
                allowed ({pooledExposure?.exposure_caps.map((c) => `${c}%`).join(', ')}). No
                build broke its cap; this is the concentration the caps cannot see.
              </span>
            </div>
          )}

          <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-400 border-b border-slate-700">
                  <th className="text-left py-2 px-3 font-medium whitespace-nowrap">Player</th>
                  <th className="text-left py-2 px-2 font-medium whitespace-nowrap">Pos</th>
                  <th className="text-left py-2 px-2 font-medium whitespace-nowrap">Team</th>
                  <th className="text-right py-2 px-2 font-medium whitespace-nowrap">Salary</th>
                  <th className="text-right py-2 px-2 font-medium whitespace-nowrap">Lineups</th>
                  {scope === 'week' && (
                    <th className="text-right py-2 px-2 font-medium whitespace-nowrap">Builds</th>
                  )}
                  <th className="text-left py-2 px-3 font-medium w-40">Exposure</th>
                  <th className="text-right py-2 px-2 font-medium whitespace-nowrap">Proj</th>
                  {showsActual && <th className="text-right py-2 px-3 font-medium whitespace-nowrap">Actual</th>}
                </tr>
              </thead>
              <tbody>
                {exposureRows.map((r) => {
                  const flagged = overCap.has(r.player_name)
                  const slots = r.slots ? Object.entries(r.slots) : []
                  return (
                    <tr
                      key={`${r.player_name}-${r.team}`}
                      className="border-b border-slate-700/40 last:border-0"
                    >
                      <td className={`py-1.5 px-3 ${flagged ? 'text-amber-300' : 'text-slate-200'}`}>
                        {r.player_name}
                        {!r.from_model && (
                          <span
                            className="text-slate-500"
                            title="FanDuel's FPPG, not a model projection"
                          >
                            {' '}*
                          </span>
                        )}
                      </td>
                      <td
                        className="py-1.5 px-2 text-slate-500"
                        title={slots.length > 1
                          ? slots.map(([slot, n]) => `${slot}: ${n}`).join(', ')
                          : undefined}
                      >
                        {r.position}
                        {slots.length > 1 && <span className="text-slate-600"> †</span>}
                      </td>
                      <td className="py-1.5 px-2 text-slate-400">{r.team}</td>
                      <td className="py-1.5 px-2 text-right text-slate-400 tabular-nums whitespace-nowrap">
                        {r.salary === null ? '—' : `$${r.salary.toLocaleString()}`}
                      </td>
                      <td className="py-1.5 px-2 text-right text-slate-300 tabular-nums whitespace-nowrap">
                        {r.lineups}
                      </td>
                      {scope === 'week' && (
                        <td className="py-1.5 px-2 text-right text-slate-400 tabular-nums whitespace-nowrap">
                          {r.builds ?? '—'}
                        </td>
                      )}
                      <td className="py-1.5 px-3">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 flex-1 min-w-[3rem] bg-slate-700 rounded-full overflow-hidden">
                            <div
                              className={`h-full ${flagged ? 'bg-amber-400' : 'bg-brand-green'}`}
                              style={{ width: `${Math.min(r.exposure ?? 0, 100)}%` }}
                            />
                          </div>
                          <span
                            className={`tabular-nums w-11 text-right ${
                              flagged ? 'text-amber-300' : 'text-slate-300'
                            }`}
                          >
                            {r.exposure === null ? '—' : `${r.exposure}%`}
                          </span>
                        </div>
                      </td>
                      <td className="py-1.5 px-2 text-right text-slate-300 tabular-nums whitespace-nowrap">
                        {r.projected ?? '—'}
                      </td>
                      {showsActual && (
                        <td className="py-1.5 px-3 text-right text-slate-200 tabular-nums whitespace-nowrap">
                          {r.actual ?? '—'}
                        </td>
                      )}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="text-slate-500 text-xs mt-2">
            * FanDuel's own FPPG rather than a model projection.
            {exposureRows.some((r) => (r.slots ? Object.keys(r.slots).length > 1 : false)) &&
              ' † filled more than one slot — hover for the split.'}
          </p>
        </div>
      )}

      {/* The optimizer's own record. Built before kickoff, scored afterwards. */}
      <div className="mt-10">
        <h2 className="text-white font-medium text-sm mb-1">Track record</h2>
        <p className="text-xs text-slate-500 mb-3">
          Every build is recorded and scored once the week is played. The baseline is the
          lineup FanDuel's own <span className="text-slate-300">FPPG</span> would have built
          from the same players under the same cap — beating it is the whole claim the
          board makes. <span className="text-slate-300">Best possible</span> is the slate's
          best lineup in hindsight; nobody hits it.
        </p>

        {performance.data?.status === 'no_data' && (
          <p className="text-xs text-slate-500">{performance.data.message}</p>
        )}

        {performance.data?.status === 'success' && (
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 text-left">
                  <th className="pb-2 pr-3 font-medium whitespace-nowrap">Week</th>
                  <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Lineups</th>
                  <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Scored</th>
                  <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Projected</th>
                  <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Actual</th>
                  <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">FPPG baseline</th>
                  <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Best possible</th>
                  <th className="pb-2 font-medium text-right whitespace-nowrap">Beat baseline</th>
                </tr>
              </thead>
              <tbody>
                {performance.data.weeks.map((w) => {
                  const edge = w.actual !== null && w.fppg_baseline !== null
                    ? w.actual - w.fppg_baseline
                    : null
                  return (
                    <tr key={w.week} className="border-t border-slate-700/60">
                      <td className="py-1.5 pr-3 text-slate-200">Week {w.week}</td>
                      <td className="py-1.5 pr-3 text-right text-slate-400 tabular-nums whitespace-nowrap">
                        {w.lineups}
                      </td>
                      <td className="py-1.5 pr-3 text-right text-slate-400 tabular-nums whitespace-nowrap">
                        {w.scored_lineups || '—'}
                      </td>
                      <td className="py-1.5 pr-3 text-right text-slate-400 tabular-nums whitespace-nowrap">
                        {w.projected ?? '—'}
                      </td>
                      <td className={`py-1.5 pr-3 text-right tabular-nums ${
                        edge === null ? 'text-slate-400'
                          : edge > 0 ? 'text-brand-green' : 'text-amber-400'
                      }`}>
                        {w.actual ?? '—'}
                      </td>
                      <td className="py-1.5 pr-3 text-right text-slate-400 tabular-nums whitespace-nowrap">
                        {w.fppg_baseline ?? '—'}
                      </td>
                      <td className="py-1.5 pr-3 text-right text-slate-500 tabular-nums whitespace-nowrap">
                        {w.best_possible ?? '—'}
                      </td>
                      <td className="py-1.5 text-right text-slate-400 tabular-nums whitespace-nowrap">
                        {w.compared_to_fppg ? `${w.beat_fppg} of ${w.compared_to_fppg}` : '—'}
                      </td>
                    </tr>
                  )
                })}
                {performance.data.totals.weeks_scored > 0 && (
                  <tr className="border-t border-slate-600 text-slate-200">
                    <td className="py-2 pr-3 font-medium">
                      {performance.data.totals.weeks_scored} week
                      {performance.data.totals.weeks_scored === 1 ? '' : 's'}
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums whitespace-nowrap">
                      {performance.data.totals.lineups}
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums whitespace-nowrap">
                      {performance.data.totals.scored_lineups}
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums whitespace-nowrap">
                      {performance.data.totals.projected ?? '—'}
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums whitespace-nowrap">
                      {performance.data.totals.actual ?? '—'}
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums whitespace-nowrap">
                      {performance.data.totals.fppg_baseline ?? '—'}
                    </td>
                    <td className="py-2 pr-3 text-right text-slate-500 tabular-nums whitespace-nowrap">
                      {performance.data.totals.best_possible ?? '—'}
                    </td>
                    <td className="py-2 text-right tabular-nums whitespace-nowrap">
                      {performance.data.totals.beat_fppg_rate !== null
                        ? `${Math.round(performance.data.totals.beat_fppg_rate * 100)}%`
                        : '—'}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
            <p className="text-slate-500 text-xs mt-3">
              Only lineups where every slot could be scored are averaged — a player who did
              not play is a real zero, but a defense on a week team stats were unavailable
              leaves a lineup out of the numbers until the next scoring run.
            </p>
          </div>
        )}
      </div>

      {/* Week N itself: which objective won, then every lineup and how it ended up. */}
      <div className="mt-10">
        <h2 className="text-white font-medium text-sm mb-1">Week {week} lineups</h2>

        {weekRecord.data?.status === 'no_data' && (
          <p className="text-xs text-slate-500">
            Nothing built for week {week} yet.
          </p>
        )}

        {weekRecord.data?.status === 'success' && (
          <>
            {weekScored === 0 ? (
              <p className="text-xs text-slate-500 mb-3">
                {weekLineupCount} lineup{weekLineupCount === 1 ? '' : 's'} recorded, none
                scored yet. Scoring needs the week's stats loaded, which happens after the
                last game.
              </p>
            ) : (
              <p className="text-xs text-slate-500 mb-3">
                {weekScored} of {weekLineupCount} lineup
                {weekLineupCount === 1 ? '' : 's'} scored.
              </p>
            )}

            {/* The insight: which objective actually won. */}
            {weekObjectives.length > 0 && (
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 mb-4 overflow-x-auto">
                <h3 className="text-white font-medium text-xs mb-2">Objectives</h3>
                {weekVerdict?.comparable && weekVerdict.best ? (
                  <p className="text-xs text-slate-300 mb-3">
                    <span className="text-brand-green font-medium">{weekVerdict.best}</span>{' '}
                    beat <span className="text-slate-200">{weekVerdict.worst}</span> by{' '}
                    <span className="tabular-nums whitespace-nowrap">{weekVerdict.spread}</span> points a
                    lineup this week
                    {weekContests.size > 1 && ` (${weekVerdict.contest} only)`}.
                    <span className="text-slate-500">
                      {' '}One slate of {weekVerdict.scored_lineups} lineups, though —
                      largely the same players in each, so this is a single observation,
                      not a reason to change the setting.
                    </span>
                  </p>
                ) : (
                  <p className="text-xs text-slate-500 mb-3">
                    {weekVerdict && weekVerdict.objectives_scored < 2
                      ? 'Two objectives have to have scored lineups before one can be said to have beaten the other.'
                      : 'No scored lineups yet this week.'}
                  </p>
                )}
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-slate-500 text-left">
                      <th className="pb-2 pr-3 font-medium whitespace-nowrap">Objective</th>
                      {weekContests.size > 1 && (
                        <th className="pb-2 pr-3 font-medium whitespace-nowrap">Contest</th>
                      )}
                      <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Lineups</th>
                      <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Scored</th>
                      <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Projected</th>
                      <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Actual</th>
                      <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Best</th>
                      <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Worst</th>
                      <th className="pb-2 font-medium text-right whitespace-nowrap">vs baseline</th>
                    </tr>
                  </thead>
                  <tbody>
                    {weekObjectives.map((o) => {
                      const edge = o.actual !== null && o.fppg_baseline !== null
                        ? o.actual - o.fppg_baseline
                        : null
                      return (
                        <tr
                          key={`${o.objective}-${o.contest}`}
                          className="border-t border-slate-700/60"
                        >
                          <td className="py-1.5 pr-3 text-slate-200">{o.objective}</td>
                          {weekContests.size > 1 && (
                            <td className="py-1.5 pr-3 text-slate-500">{o.contest}</td>
                          )}
                          <td className="py-1.5 pr-3 text-right text-slate-400 tabular-nums whitespace-nowrap">
                            {o.lineups}
                          </td>
                          <td className="py-1.5 pr-3 text-right text-slate-400 tabular-nums whitespace-nowrap">
                            {o.scored_lineups}
                          </td>
                          <td className="py-1.5 pr-3 text-right text-slate-400 tabular-nums whitespace-nowrap">
                            {o.projected ?? '—'}
                          </td>
                          <td className="py-1.5 pr-3 text-right text-slate-100 tabular-nums whitespace-nowrap">
                            {o.actual ?? '—'}
                          </td>
                          <td className="py-1.5 pr-3 text-right text-slate-400 tabular-nums whitespace-nowrap">
                            {o.best_lineup_actual ?? '—'}
                          </td>
                          <td className="py-1.5 pr-3 text-right text-slate-400 tabular-nums whitespace-nowrap">
                            {o.worst_lineup_actual ?? '—'}
                          </td>
                          <td
                            className={`py-1.5 text-right tabular-nums ${
                              edge === null
                                ? 'text-slate-500'
                                : edge > 0
                                  ? 'text-brand-green'
                                  : 'text-red-400'
                            }`}
                          >
                            {edge === null ? '—' : `${edge > 0 ? '+' : ''}${edge.toFixed(1)}`}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Every build, and inside it every lineup. */}
            <div className="space-y-4">
              {(weekRecord.data.builds ?? []).map((b) => (
                <div
                  key={b.build_id}
                  className="bg-slate-800 border border-slate-700 rounded-xl p-4 overflow-x-auto"
                >
                  <div className="flex items-baseline justify-between gap-3 mb-3 flex-wrap">
                    <h3 className="text-white font-medium text-xs">
                      {b.objective} · {b.contest} · {b.slate}
                      <span className="text-slate-500 font-normal">
                        {' '}· build {b.build_id} · {b.lineups.length} lineup
                        {b.lineups.length === 1 ? '' : 's'}
                      </span>
                    </h3>
                    <span className="text-xs text-slate-400">
                      {b.fppg_baseline !== null ? (
                        <>
                          FPPG baseline{' '}
                          <span className="text-slate-200 tabular-nums whitespace-nowrap">{b.fppg_baseline}</span>
                        </>
                      ) : (
                        <span className="text-slate-500">no baseline</span>
                      )}
                      {b.best_possible !== null && (
                        <>
                          {' '}· best possible{' '}
                          <span className="text-slate-200 tabular-nums whitespace-nowrap">{b.best_possible}</span>
                        </>
                      )}
                      {b.scored_at === null && (
                        <span className="text-amber-400/80"> · not scored</span>
                      )}
                    </span>
                  </div>

                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-slate-500 text-left">
                        <th className="pb-2 pr-3 font-medium whitespace-nowrap">Lineup</th>
                        <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Salary</th>
                        <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Projected</th>
                        <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">Actual</th>
                        <th className="pb-2 pr-3 font-medium whitespace-nowrap text-right">vs baseline</th>
                        <th className="pb-2 font-medium" />
                      </tr>
                    </thead>
                    <tbody>
                      {b.lineups.map((l) => {
                        const key = `${b.build_id}-${l.lineup}`
                        const open = openLineups.has(key)
                        const edge = l.actual !== null && b.fppg_baseline !== null
                          ? l.actual - b.fppg_baseline
                          : null
                        return (
                          <Fragment key={key}>
                            <tr
                              className="border-t border-slate-700/60 cursor-pointer hover:bg-slate-700/20"
                              onClick={() => toggleLineup(key)}
                            >
                              <td className="py-1.5 pr-3 text-slate-200">
                                Lineup {l.lineup}
                                {(l.unscored_players ?? 0) > 0 && (
                                  <span
                                    className="text-amber-400/80"
                                    title={`${l.unscored_players} slot(s) could not be scored`}
                                  >
                                    {' '}· {l.unscored_players} unscored
                                  </span>
                                )}
                              </td>
                              <td className="py-1.5 pr-3 text-right text-slate-400 tabular-nums whitespace-nowrap">
                                {l.salary === null ? '—' : `$${l.salary.toLocaleString()}`}
                              </td>
                              <td className="py-1.5 pr-3 text-right text-slate-400 tabular-nums whitespace-nowrap">
                                {l.projected ?? '—'}
                              </td>
                              <td className="py-1.5 pr-3 text-right text-slate-100 font-medium tabular-nums whitespace-nowrap">
                                {l.actual ?? '—'}
                              </td>
                              <td
                                className={`py-1.5 pr-3 text-right tabular-nums ${
                                  edge === null
                                    ? 'text-slate-500'
                                    : edge > 0
                                      ? 'text-brand-green'
                                      : 'text-red-400'
                                }`}
                              >
                                {edge === null
                                  ? '—'
                                  : `${edge > 0 ? '+' : ''}${edge.toFixed(1)}`}
                              </td>
                              <td className="py-1.5 text-right text-slate-500">
                                {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                              </td>
                            </tr>
                            {open && (
                              <tr className="border-t border-slate-700/40 bg-slate-900/40">
                                <td colSpan={6} className="px-3 py-2">
                                  {/* min-w-[22rem] + the card's overflow-x-auto:
                                      the roster scrolls with the card on a narrow
                                      screen. Without a min-width a nested table
                                      squeezes its own columns to illegible and
                                      still drags the parent table wider. */}
                                  <table className="w-full min-w-[22rem] table-fixed text-xs">
                                    <tbody>
                                      {l.players.map((p, i) => {
                                        const diff = p.actual !== null && p.projected !== null
                                          ? p.actual - p.projected
                                          : null
                                        return (
                                          <tr key={`${p.player_name}-${i}`}>
                                            <td className="py-1 pr-2 text-slate-500 w-14">
                                              {p.slot}
                                            </td>
                                            <td
                                              className="py-1 pr-2 text-slate-200 truncate"
                                              title={p.player_name}
                                            >
                                              {p.player_name}
                                              {!p.from_model && (
                                                <span
                                                  className="text-slate-500"
                                                  title="FanDuel's FPPG, not a model projection"
                                                >
                                                  {' '}*
                                                </span>
                                              )}
                                            </td>
                                            <td className="py-1 pr-2 text-slate-500 w-10">{p.team}</td>
                                            <td className="py-1 pr-2 text-right text-slate-400 tabular-nums w-16 whitespace-nowrap">
                                              {p.salary === null
                                                ? '—'
                                                : `$${p.salary.toLocaleString()}`}
                                            </td>
                                            <td className="py-1 pr-2 text-right text-slate-400 tabular-nums w-14">
                                              {p.projected ?? '—'}
                                            </td>
                                            <td className="py-1 pr-2 text-right text-slate-100 tabular-nums w-14">
                                              {p.actual ?? '—'}
                                            </td>
                                            <td
                                              className={`py-1 text-right tabular-nums w-16 ${
                                                diff === null
                                                  ? 'text-slate-600'
                                                  : diff > 0
                                                    ? 'text-brand-green'
                                                    : 'text-red-400'
                                              }`}
                                            >
                                              {diff === null
                                                ? '—'
                                                : `${diff > 0 ? '+' : ''}${diff.toFixed(1)}`}
                                            </td>
                                          </tr>
                                        )
                                      })}
                                    </tbody>
                                  </table>
                                </td>
                              </tr>
                            )}
                          </Fragment>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
