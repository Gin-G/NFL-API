import { useState } from 'react'
import { Info } from 'lucide-react'
import {
  useCoverageCorners,
  useCoverageSecondaries,
  useCoverageMatchups,
} from '../api/coverage'
import { getAvailableSeasons, getDefaultSeason, getCurrentNFLWeek } from '../utils/nflDate'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'
import GradeScore from '../components/ui/GradeScore'

const SEASONS = getAvailableSeasons()
const WEEKS = Array.from({ length: 18 }, (_, i) => i + 1)

const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(0)}%`
const one = (v: number | null | undefined) =>
  v === null || v === undefined ? '—' : v.toFixed(1)

export default function CoverageReport() {
  const defaultSeason = getDefaultSeason()
  const [season, setSeason] = useState(defaultSeason)
  const [week, setWeek] = useState(() => getCurrentNFLWeek(defaultSeason))
  const [tab, setTab] = useState<'matchups' | 'secondaries' | 'corners'>('matchups')

  const matchups = useCoverageMatchups(season, week)
  const secondaries = useCoverageSecondaries(season, week)
  const corners = useCoverageCorners(season, week)

  const matchupRows = matchups.data?.matchups ?? []
  const secondaryRows = secondaries.data?.secondaries ?? []
  const cornerRows = corners.data?.corners ?? []
  const loading = matchups.isLoading || secondaries.isLoading || corners.isLoading
  const error = matchups.error || secondaries.error || corners.error

  return (
    <div className="p-6">
      <PageHeader
        title="Coverage Report"
        subtitle="Which corners get beaten and who lines up across — 0–100, where 50 is league average and high is good coverage"
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
          {(['matchups', 'secondaries', 'corners'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1.5 text-sm ${
                tab === t ? 'bg-slate-600 text-white' : 'bg-slate-800 text-slate-400'
              }`}
            >
              {t === 'matchups' ? 'This week' : t === 'secondaries' ? 'Secondaries' : 'Corners'}
            </button>
          ))}
        </div>
      </div>

      {/* How to read the number — the part that keeps it honest. */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 mb-6 flex gap-3">
        <Info size={16} className="text-slate-400 shrink-0 mt-0.5" />
        <div className="text-xs text-slate-400 space-y-1.5">
          <p>
            <span className="text-slate-200 font-medium">This page does not move the
            projections.</span>{' '}
            Every grade here was fitted against the board's own error leave-one-season-out and
            scored on both MAE and top-24 ordering. Nothing paid. Wide receiver was{' '}
            <span className="text-slate-300">actively harmful</span>; tight end and running back
            came out at zero. So the projected points shown are unadjusted, and the leans are
            notes for you, not arithmetic.
          </p>
          <p>
            <span className="text-slate-200 font-medium">Completion rate is the grade; yards is
            not.</span>{' '}
            Correlating a corner against himself across his career, how often he is thrown at
            (0.73), how deep (0.73) and the share he allows complete (0.52) all persist —{' '}
            <span className="text-slate-300">yards allowed per target does not (0.19)</span>. A
            corner sees a median of 26 targets a season and yards per completion rides on a few
            long plays, so yardage is shown for reference and should not be ranked on.
          </p>
          <p>
            <span className="text-slate-200 font-medium">Read volume next to the grade.</span> A
            corner who is avoided and a corner who is good look identical in a rate; only his
            target count separates them. And the <span className="text-slate-300">spread</span>{' '}
            column matters as much as the average — a secondary can grade out fine and still have
            one side worth attacking.
          </p>
          <p>
            Nobody publishes the coverage assignment, so this can say a secondary has a weak
            corner and cannot say which receiver draws him. Grades for a week use only charting
            from earlier weeks.
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
          message="Could not load coverage ratings."
          onRetry={() => {
            matchups.refetch()
            secondaries.refetch()
            corners.refetch()
          }}
        />
      )}

      {!loading && !error && tab === 'matchups' && (
        matchupRows.length === 0 ? (
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 text-sm text-slate-400">
            No coverage matchups for {season} week {week} yet. Grades need a few weeks of charting
            in the season, and the week needs a schedule.
          </div>
        ) : (
          <div className="space-y-3">
            {matchupRows.map((row) => (
              <div
                key={`${row.team}-${row.opponent}`}
                className="bg-slate-800 border border-slate-700 rounded-xl p-4"
              >
                <div className="flex flex-wrap items-center gap-3 mb-3">
                  <span className="font-semibold text-slate-100">{row.team}</span>
                  <span className="text-xs text-slate-500">
                    {row.home ? 'vs' : 'at'} {row.opponent}
                  </span>
                  <GradeScore value={row.opp_cover_grade} size="sm" />
                  <span className="text-xs text-slate-400">{row.opp_cover_band}</span>
                  <span className="text-xs text-slate-500 ml-auto">
                    spread {pct(row.opp_cover_spread)}
                    {row.opp_weakest_corner && (
                      <> · softest corner{' '}
                        <span className="text-slate-300">{row.opp_weakest_corner}</span>
                      </>
                    )}
                  </span>
                </div>

                {row.opp_corners && row.opp_corners.length > 0 && (
                  <div className="text-xs text-slate-500 mb-3">
                    {row.opp_corners.map((c) => (
                      <span key={c.pfr_player_id ?? c.player_name} className="mr-4">
                        {c.player_name} — {pct(c.cmp_pct_allowed)} on {c.targets ?? 0} tgt,
                        {' '}{one(c.adot)} adot, grade {one(c.cover_grade)}
                      </span>
                    ))}
                  </div>
                )}

                {row.receivers.length === 0 ? (
                  <p className="text-xs text-slate-500">
                    No projections for {row.team} this week.
                  </p>
                ) : (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-xs text-slate-500 text-left">
                        <th className="font-normal pb-1">Receiver</th>
                        <th className="font-normal pb-1">Pos</th>
                        <th className="font-normal pb-1 text-right">Proj</th>
                        <th className="font-normal pb-1 text-right">Ceiling</th>
                        <th className="font-normal pb-1 pl-4">Lean (advisory)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {row.receivers.map((r) => (
                        <tr key={r.player_id} className="border-t border-slate-700/50">
                          <td className="py-1 text-slate-200">{r.player_name}</td>
                          <td className="py-1 text-slate-500 text-xs">{r.position}</td>
                          <td className="py-1 text-right tabular-nums text-slate-300">
                            {one(r.projected_points)}
                          </td>
                          <td className="py-1 text-right tabular-nums text-slate-500">
                            {one(r.ceiling)}
                          </td>
                          <td className="py-1 pl-4 text-xs text-slate-400">{r.coverage_lean}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            ))}
          </div>
        )
      )}

      {!loading && !error && tab === 'secondaries' && (
        secondaryRows.length === 0 ? (
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 text-sm text-slate-400">
            No secondary grades for {season} through week {week}.
          </div>
        ) : (
          <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-slate-500 text-left border-b border-slate-700">
                  <th className="font-normal p-3">#</th>
                  <th className="font-normal p-3">Team</th>
                  <th className="font-normal p-3">Grade</th>
                  <th className="font-normal p-3">Band</th>
                  <th className="font-normal p-3 text-right">Cmp allowed</th>
                  <th className="font-normal p-3 text-right">Spread</th>
                  <th className="font-normal p-3 text-right">aDOT</th>
                  <th className="font-normal p-3">Softest corner</th>
                </tr>
              </thead>
              <tbody>
                {secondaryRows.map((s, i) => (
                  <tr key={s.team} className="border-b border-slate-700/40">
                    <td className="p-3 text-xs text-slate-500 tabular-nums">{i + 1}</td>
                    <td className="p-3 font-medium text-slate-200">{s.team}</td>
                    <td className="p-3"><GradeScore value={s.cover_grade} size="sm" /></td>
                    <td className="p-3 text-xs text-slate-400">{s.band}</td>
                    <td className="p-3 text-right tabular-nums text-slate-300">
                      {pct(s.cover_rate)}
                    </td>
                    <td className="p-3 text-right tabular-nums text-slate-300">
                      {pct(s.cover_spread)}
                    </td>
                    <td className="p-3 text-right tabular-nums text-slate-500">{one(s.adot)}</td>
                    <td className="p-3 text-xs text-slate-400">{s.weakest_corner ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      )}

      {!loading && !error && tab === 'corners' && (
        cornerRows.length === 0 ? (
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 text-sm text-slate-400">
            No corner grades for {season} through week {week}. A corner needs charted targets
            before he can be graded.
          </div>
        ) : (
          <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-slate-500 text-left border-b border-slate-700">
                  <th className="font-normal p-3">#</th>
                  <th className="font-normal p-3">Corner</th>
                  <th className="font-normal p-3">Team</th>
                  <th className="font-normal p-3">Grade</th>
                  <th className="font-normal p-3 text-right">Tgt</th>
                  <th className="font-normal p-3 text-right">Tgt/g</th>
                  <th className="font-normal p-3 text-right">aDOT</th>
                  <th className="font-normal p-3 text-right">Cmp %</th>
                  <th className="font-normal p-3 text-right text-slate-600">Yds/tgt*</th>
                </tr>
              </thead>
              <tbody>
                {cornerRows.map((c, i) => (
                  <tr key={c.pfr_player_id} className="border-b border-slate-700/40">
                    <td className="p-3 text-xs text-slate-500 tabular-nums">{i + 1}</td>
                    <td className="p-3 text-slate-200">{c.player_name}</td>
                    <td className="p-3 text-xs text-slate-400">{c.team ?? '—'}</td>
                    <td className="p-3"><GradeScore value={c.cover_grade} size="sm" /></td>
                    <td className="p-3 text-right tabular-nums text-slate-300">{c.targets}</td>
                    <td className="p-3 text-right tabular-nums text-slate-400">
                      {one(c.targets_per_game)}
                    </td>
                    <td className="p-3 text-right tabular-nums text-slate-400">{one(c.adot)}</td>
                    <td className="p-3 text-right tabular-nums text-slate-300">
                      {pct(c.cmp_pct_allowed)}
                    </td>
                    <td className="p-3 text-right tabular-nums text-slate-600">
                      {one(c.yards_per_target)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-slate-500 p-3 border-t border-slate-700">
              * Yards per target is greyed out because it is not a trait — a corner's own career
              odd/even correlation is 0.19, against 0.52 for completion rate. It is here so you can
              see it, not so you can sort by it.
            </p>
          </div>
        )
      )}
    </div>
  )
}
