import { useMemo, useState } from 'react'
import {
  CartesianGrid, Cell, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts'
import { Info, Table2, ScatterChart as ScatterIcon } from 'lucide-react'
import { useConsistency } from '../api/optimizer'
import type { ConsistencyPlayer } from '../api/types'

/** Who gets the ball steadily.
 *
 *  One position at a time, deliberately. A quarterback sees about 38
 *  opportunities a game and a receiver about 8, so a shared x-axis would
 *  squash three positions into the left margin — and a scatter is an
 *  all-pairs form, where four categorical hues cannot be told apart safely.
 *  Filtering to one position gives one series: no legend needed, one hue, and
 *  an axis that uses its whole width.
 *
 *  The answer is the top-right of the plot: high volume, steady volume. The
 *  bubble's size is points per game, so productivity is visible without
 *  entering the ranking — it cannot, because scoring spread does not persist.
 */

const POSITIONS = ['RB', 'WR', 'TE', 'QB']

/** Validated against this surface (#1e293b, dark): lightness band, chroma
 *  floor and 3:1 contrast all pass. One series, so no CVD pair to clear. */
const SERIES = '#3987e5'
const SURFACE = '#1e293b'
const GRID = '#334155'
const INK_MUTED = '#94a3b8'
const INK_PRIMARY = '#f1f5f9'

function TipRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-6">
      <span className="text-slate-400">{label}</span>
      <span className="text-slate-100 tabular-nums">{value}</span>
    </div>
  )
}

function ChartTooltip({ active, payload }: {
  active?: boolean; payload?: { payload: ConsistencyPlayer }[]
}) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload
  const trend = p.opportunity_trend
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs space-y-1 shadow-xl max-w-[18rem]">
      <div className="text-slate-100 font-medium">
        {p.player}{' '}
        <span className="text-slate-500">{p.position} · {p.team}</span>
      </div>
      <TipRow label="Opportunities / game"
              value={`${p.opportunities_per_game.toFixed(1)} (${p.opportunity_low.toFixed(0)}–${p.opportunity_high.toFixed(0)})`} />
      <TipRow label="Volume steadiness" value={`${p.volume_steadiness.toFixed(0)}/100`} />
      {trend !== null && Math.abs(trend) >= 0.5 && (
        <TipRow label="Lately"
                value={`${trend > 0 ? '+' : ''}${trend.toFixed(1)} a game (${p.opportunities_recent.toFixed(1)})`} />
      )}
      <div className="border-t border-slate-700 pt-1 mt-1">
        <TipRow label="Points / game" value={p.points_per_game.toFixed(1)} />
        <TipRow label="Has scored"
                value={`${p.points_low.toFixed(1)}–${p.points_high.toFixed(1)}`} />
        <TipRow label="Cleared 15" value={`${(p.hit_rate_15 * 100).toFixed(0)}% of games`} />
        {p.salary !== null && (
          <TipRow label="Salary" value={`$${p.salary.toLocaleString()}`} />
        )}
      </div>
      <p className="text-slate-500 pt-0.5">{p.games} games</p>
    </div>
  )
}

export default function ConsistencyChart({ season, week }: {
  season: number; week?: number
}) {
  const [position, setPosition] = useState('RB')
  const [minGames, setMinGames] = useState(4)
  const [asTable, setAsTable] = useState(false)
  const query = useConsistency(season, week, { positions: position, minGames })

  const rows = query.data?.data ?? []
  // Label the two EXTREMES, not the top three.
  //
  // The top three by steadiness sit on top of each other by construction —
  // rendered, their names overlapped into one illegible smear. The steadiest
  // and the highest-volume player are the chart's actual endpoints, they are
  // usually different people, and they are far apart on the plot, so the
  // labels cannot collide. Everyone else is a hover or the table.
  const labelled = useMemo(() => {
    if (!rows.length) return new Map<string, string>()
    const steadiest = rows.reduce((a, b) =>
      b.volume_steadiness > a.volume_steadiness ? b : a)
    const busiest = rows.reduce((a, b) =>
      b.opportunities_per_game > a.opportunities_per_game ? b : a)
    const out = new Map<string, string>()
    out.set(steadiest.player, steadiest.player)
    out.set(busiest.player, busiest.player)
    return out
  }, [rows])

  const thin = rows.length > 0 && rows.every((r) => r.games < 6)

  return (
    <div className="space-y-3">
      {/* One filter row above everything it scopes, never inside the card. */}
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <div className="flex items-center gap-1">
          {POSITIONS.map((value) => (
            <button
              key={value}
              onClick={() => setPosition(value)}
              className={`rounded-md px-2.5 py-1 transition-colors ${
                position === value
                  ? 'bg-slate-600 text-white'
                  : 'bg-slate-700/50 text-slate-400 hover:text-slate-200'
              }`}
            >
              {value}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-slate-400">
          Min games
          <input
            type="number" min={2} max={18} value={minGames}
            onChange={(e) => setMinGames(Number(e.target.value) || 4)}
            className="bg-slate-700 border border-slate-600 text-white rounded px-2 py-1 w-14 text-xs tabular-nums"
          />
        </label>
        <button
          onClick={() => setAsTable((t) => !t)}
          className="flex items-center gap-1.5 text-slate-400 hover:text-slate-200 transition-colors ml-auto"
        >
          {asTable ? <ScatterIcon size={13} /> : <Table2 size={13} />}
          {asTable ? 'Chart' : 'Table'}
        </button>
      </div>

      {query.isLoading && <p className="text-sm text-slate-500">Loading…</p>}
      {query.isError && (
        <p className="text-sm text-slate-400">
          {(query.error as { response?: { data?: { detail?: string } } })
            ?.response?.data?.detail ?? 'Consistency could not be loaded.'}
        </p>
      )}

      {!query.isLoading && !query.isError && rows.length === 0 && (
        <p className="text-sm text-slate-500">
          No {position} has {minGames} games yet.
        </p>
      )}

      {rows.length > 0 && thin && (
        <p className="text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2 flex gap-2">
          <Info size={14} className="shrink-0 mt-0.5" />
          <span>
            Every player here has under six games. A spread needs games: over
            2026's nine weeks the reliability of volume dispersion falls to
            +0.07, CI [−0.08, +0.22]. Read the volume before the steadiness.
          </span>
        </p>
      )}

      {rows.length > 0 && !asTable && (
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
          <h4 className="text-xs text-slate-300 mb-0.5">
            {position}s — volume, and how steady it is
          </h4>
          <p className="text-xs text-slate-500 mb-3">
            Top-right is the answer: plenty of the ball, every week. Bubble size
            is points per game.
          </p>
          <ResponsiveContainer width="100%" height={340}>
            <ScatterChart margin={{ top: 8, right: 24, bottom: 28, left: 8 }}>
              <CartesianGrid stroke={GRID} strokeWidth={1} />
              <XAxis
                type="number" dataKey="opportunities_per_game"
                name="Opportunities per game"
                tick={{ fill: INK_MUTED, fontSize: 11 }}
                stroke={GRID}
                label={{ value: 'Opportunities per game', position: 'bottom',
                         offset: 10, fill: INK_MUTED, fontSize: 11 }}
              />
              <YAxis
                type="number" dataKey="volume_steadiness" domain={[0, 100]}
                name="Volume steadiness"
                tick={{ fill: INK_MUTED, fontSize: 11 }}
                stroke={GRID}
                label={{ value: 'Volume steadiness', angle: -90,
                         position: 'insideLeft', fill: INK_MUTED, fontSize: 11 }}
              />
              {/* Area ∝ points per game, so productivity reads without
                  entering the ranking. */}
              <ZAxis type="number" dataKey="points_per_game" range={[80, 520]} />
              <Tooltip
                content={<ChartTooltip />}
                cursor={{ strokeDasharray: '0', stroke: GRID }}
              />
              <Scatter
                data={rows}
                fill={SERIES}
                // A 2px ring in the surface colour keeps overlapping dots
                // legible and widens the hit target.
                stroke={SURFACE}
                strokeWidth={2}
                isAnimationActive={false}
                label={({ x, y, index }: { x: number; y: number; index: number }) => {
                  const row = rows[index]
                  if (!row || !labelled.has(row.player)) return <g key={index} />
                  return (
                    <text
                      key={index} x={x} y={y - 14} textAnchor="middle"
                      fill={INK_PRIMARY} fontSize={10}
                    >
                      {row.player}
                    </text>
                  )
                }}
              >
                {rows.map((r) => <Cell key={r.player} fill={SERIES} />)}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {rows.length > 0 && asTable && (
        <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700">
                <th className="text-left py-2 px-3 font-medium">Player</th>
                <th className="text-right py-2 px-2 font-medium">Gm</th>
                <th className="text-right py-2 px-2 font-medium">Opp/gm</th>
                <th className="text-right py-2 px-2 font-medium">Range</th>
                <th className="text-right py-2 px-2 font-medium">Steadiness</th>
                <th className="text-right py-2 px-2 font-medium">Lately</th>
                <th className="text-right py-2 px-2 font-medium">Pts/gm</th>
                <th className="text-right py-2 px-2 font-medium">Scored</th>
                <th className="text-right py-2 px-3 font-medium">15+</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.player} className="border-b border-slate-700/40 last:border-0">
                  <td className="py-1.5 px-3 text-slate-200 whitespace-nowrap">
                    {r.player} <span className="text-slate-500">{r.team}</span>
                  </td>
                  <td className="py-1.5 px-2 text-right text-slate-400 tabular-nums">{r.games}</td>
                  <td className="py-1.5 px-2 text-right text-slate-200 tabular-nums">
                    {r.opportunities_per_game.toFixed(1)}
                  </td>
                  <td className="py-1.5 px-2 text-right text-slate-400 tabular-nums whitespace-nowrap">
                    {r.opportunity_low.toFixed(0)}–{r.opportunity_high.toFixed(0)}
                  </td>
                  <td className="py-1.5 px-2 text-right text-slate-200 tabular-nums">
                    {r.volume_steadiness.toFixed(0)}
                  </td>
                  <td className="py-1.5 px-2 text-right tabular-nums whitespace-nowrap">
                    {r.opportunity_trend === null ? (
                      <span className="text-slate-600">—</span>
                    ) : (
                      <span className={r.opportunity_trend > 1 ? 'text-slate-200'
                        : r.opportunity_trend < -1 ? 'text-amber-300' : 'text-slate-500'}>
                        {r.opportunity_trend > 0 ? '+' : ''}
                        {r.opportunity_trend.toFixed(1)}
                      </span>
                    )}
                  </td>
                  <td className="py-1.5 px-2 text-right text-slate-300 tabular-nums">
                    {r.points_per_game.toFixed(1)}
                  </td>
                  <td className="py-1.5 px-2 text-right text-slate-400 tabular-nums whitespace-nowrap">
                    {r.points_low.toFixed(1)}–{r.points_high.toFixed(1)}
                  </td>
                  <td className="py-1.5 px-3 text-right text-slate-400 tabular-nums">
                    {(r.hit_rate_15 * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {query.data && (
        <details className="text-xs text-slate-500">
          <summary className="cursor-pointer hover:text-slate-300">
            Why this ranks on volume and not on scoring
          </summary>
          <div className="mt-2 space-y-2 pl-1 border-l border-slate-700 pl-3">
            <p>{query.data.why_not_scoring}</p>
            <p>{query.data.what_does_measure}</p>
            <p>{query.data.best_single_predictor}</p>
            <p className="text-slate-600">{query.data.small_sample_warning}</p>
          </div>
        </details>
      )}
    </div>
  )
}
