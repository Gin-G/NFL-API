import { useRef, useState } from 'react'
import { Info, Upload } from 'lucide-react'
import { useTargets, useUploadPicks } from '../api/targets'
import type { TargetPileScore, TargetRow } from '../api/types'
import { getAvailableSeasons, getDefaultSeason, getCurrentNFLWeek } from '../utils/nflDate'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'

const SEASONS = getAvailableSeasons()
const WEEKS = Array.from({ length: 18 }, (_, i) => i + 1)

const TAG_LABEL: Record<string, string> = {
  cash: 'Cash',
  gpp: 'GPP',
  punt: 'Punt',
  favorite_cash: '★ Cash',
  favorite_gpp: '★ GPP',
  mention: 'Mention',
}

function fmt(v: number | null | undefined, digits = 1) {
  return v === null || v === undefined ? '—' : v.toFixed(digits)
}

function money(v: number | null) {
  return v === null ? '—' : `$${v.toLocaleString()}`
}

function PileTable({ rows, played }: { rows: TargetRow[]; played: boolean }) {
  if (!rows.length) {
    return <p className="text-slate-500 text-sm px-1 py-3">None this week.</p>
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-slate-400 text-xs border-b border-slate-700">
            <th className="text-left font-medium py-2 pr-3">Player</th>
            <th className="text-left font-medium py-2 pr-3">Their call</th>
            <th className="text-right font-medium py-2 pr-3">Salary</th>
            <th className="text-right font-medium py-2 pr-3">Our proj</th>
            <th className="text-right font-medium py-2 pr-3">Range</th>
            <th className="text-right font-medium py-2 pr-3">Pts rank</th>
            <th className="text-right font-medium py-2 pr-3">Value</th>
            <th className="text-right font-medium py-2 pr-3">Value rank</th>
            {played && <th className="text-right font-medium py-2 pr-3">Actual</th>}
            {played && <th className="text-right font-medium py-2">vs board</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={(r.player_id ?? r.player) + r.position} className="border-b border-slate-800 align-top">
              <td className="py-2 pr-3">
                <div className="text-white font-medium">{r.player}</div>
                <div className="text-slate-500 text-xs">
                  {r.position} · {r.team ?? '—'} · {r.articles.join(', ')}
                </div>
                <div className="text-slate-500 text-xs mt-0.5">{r.why}</div>
              </td>
              <td className="py-2 pr-3">
                <div className="flex flex-wrap gap-1">
                  {r.tags.length ? (
                    r.tags.map((t) => (
                      <span
                        key={t}
                        className="px-2 py-0.5 rounded-full border border-slate-600 text-slate-200 text-xs whitespace-nowrap"
                      >
                        {TAG_LABEL[t] ?? t}
                      </span>
                    ))
                  ) : (
                    <span className="text-slate-500 text-xs">—</span>
                  )}
                </div>
              </td>
              <td className="py-2 pr-3 text-right tabular-nums text-slate-300">{money(r.salary)}</td>
              <td className="py-2 pr-3 text-right tabular-nums text-white">{fmt(r.projected)}</td>
              <td className="py-2 pr-3 text-right tabular-nums text-slate-400 whitespace-nowrap">
                {r.floor === null ? '—' : `${fmt(r.floor)}–${fmt(r.ceiling)}`}
              </td>
              <td className="py-2 pr-3 text-right tabular-nums text-slate-300">
                {r.proj_rank === null ? '—' : `#${r.proj_rank}`}
              </td>
              <td className="py-2 pr-3 text-right tabular-nums text-slate-300">{fmt(r.value, 2)}</td>
              <td className="py-2 pr-3 text-right tabular-nums text-slate-300">
                {r.value_rank === null ? '—' : `#${r.value_rank}`}
              </td>
              {played && (
                <td className="py-2 pr-3 text-right tabular-nums text-white">{fmt(r.actual)}</td>
              )}
              {played && (
                <td className="py-2 text-right tabular-nums text-slate-300">
                  {r.vs_board === null ? '—' : `${r.vs_board > 0 ? '+' : ''}${fmt(r.vs_board)}`}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ScoreLine({ label, s }: { label: string; s?: TargetPileScore }) {
  if (!s) return null
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
      <div className="text-slate-400 text-xs">{label}, after the games</div>
      <div className="text-white text-lg font-semibold tabular-nums mt-1">
        {s.mean_vs_board > 0 ? '+' : ''}
        {s.mean_vs_board.toFixed(1)} vs our board
      </div>
      <div className="text-slate-400 text-xs mt-1 tabular-nums">
        {s.n} played · scored {s.mean_actual.toFixed(1)} against our {s.mean_projected.toFixed(1)} ·{' '}
        {s.beat_board} of {s.n} beat our number
      </div>
    </div>
  )
}

export default function Targets() {
  const defaultSeason = getDefaultSeason()
  const [season, setSeason] = useState(defaultSeason)
  const [week, setWeek] = useState(() => getCurrentNFLWeek(defaultSeason))
  const targets = useTargets(season, week)
  const upload = useUploadPicks(season, week)
  const fileRef = useRef<HTMLInputElement>(null)

  const data = targets.data
  const rows = data ? [...data.agree, ...data.skeptical] : []
  const played = rows.some((r) => r.actual !== null)

  return (
    <div className="p-6">
      <PageHeader
        title="Players to Target"
        subtitle="The week's expert picks beside our own board — the ones we agree on, and the ones we're skeptical of"
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
        <input
          ref={fileRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) upload.mutate(f)
            e.target.value = ''
          }}
        />
        <button
          onClick={() => fileRef.current?.click()}
          disabled={upload.isPending}
          className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 border border-slate-600 text-white text-sm rounded-lg disabled:opacity-50"
        >
          <Upload size={14} />
          {upload.isPending ? 'Uploading…' : 'Upload picks CSV'}
        </button>
        {upload.data && (
          <span className="text-slate-400 text-xs">
            Stored {upload.data.stored} picks
            {upload.data.unmatched.length > 0 &&
              ` · not on our board: ${upload.data.unmatched.join(', ')}`}
          </span>
        )}
        {upload.error && (
          <span className="text-red-400 text-xs">
            {(upload.error as { response?: { data?: { detail?: string } } }).response?.data?.detail ??
              'Upload failed'}
          </span>
        )}
      </div>

      <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 mb-6 flex gap-3">
        <Info className="text-slate-400 shrink-0 mt-0.5" size={16} />
        <p className="text-slate-400 text-sm">
          <span className="text-slate-200">Agree</span> means our board would play him too: inside our top{' '}
          {data?.depth ? Object.entries(data.depth).map(([k, v]) => `${v} ${k}s`).join(', ') : 'N'} by
          projected points or by points per $1,000.{' '}
          <span className="text-slate-200">Skeptical</span> means outside both. Display only — nothing here
          changes a projection or the optimizer. Personal use: picks come from articles you subscribe to,
          uploaded by hand.
        </p>
      </div>

      {targets.isLoading && <SkeletonCard />}
      {targets.error && <ErrorCard message="Could not load this week's targets." onRetry={() => targets.refetch()} />}

      {data && data.count === 0 && (
        <p className="text-slate-400 text-sm">
          No picks uploaded for week {week} yet. Upload a CSV with columns{' '}
          <code className="text-slate-300">player, team, position, tags, article, author, salary_fd</code>.
        </p>
      )}

      {data && data.count > 0 && (
        <div className="space-y-8">
          {data.scored && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <ScoreLine label="Agree" s={data.scored.agree} />
              <ScoreLine label="Skeptical" s={data.scored.skeptical} />
            </div>
          )}
          <section>
            <h2 className="text-white font-semibold mb-2">
              We agree <span className="text-slate-500 font-normal">({data.agree.length})</span>
            </h2>
            <div className="bg-slate-900/40 border border-slate-700 rounded-xl p-3">
              <PileTable rows={data.agree} played={played} />
            </div>
          </section>
          <section>
            <h2 className="text-white font-semibold mb-2">
              Skeptical <span className="text-slate-500 font-normal">({data.skeptical.length})</span>
            </h2>
            <p className="text-slate-500 text-xs mb-2">
              They like him; our board doesn't. After the games, this is the pile that says which of us knew
              something.
            </p>
            <div className="bg-slate-900/40 border border-slate-700 rounded-xl p-3">
              <PileTable rows={data.skeptical} played={played} />
            </div>
          </section>
          {data.no_board.length > 0 && (
            <section>
              <h2 className="text-white font-semibold mb-2">
                Not on our board <span className="text-slate-500 font-normal">({data.no_board.length})</span>
              </h2>
              <div className="bg-slate-900/40 border border-slate-700 rounded-xl p-3">
                <PileTable rows={data.no_board} played={played} />
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  )
}
