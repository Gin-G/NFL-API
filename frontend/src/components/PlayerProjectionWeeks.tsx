import { useMemo } from 'react'
import { usePlayerProjections } from '../api/projections'
import type { WeeklyProjection } from '../api/types'
import SkeletonCard from './ui/SkeletonCard'
import { statsForPosition, fmtStat, fmtPoints, fmtRange } from '../utils/projectionStats'

/** Flags worth surfacing — a plain veteran projection needs no badge. */
const TYPE_LABELS: Record<string, { label: string; className: string }> = {
  rookie_ml: { label: 'Rookie prior', className: 'bg-slate-700 text-slate-300' },
  injured_out: { label: 'Out', className: 'bg-red-900/60 text-red-300' },
}

/** Floor→ceiling band for one week, scaled against this player's best week. */
function WeekRangeBar({
  floor,
  ceiling,
  mean,
  max,
}: {
  floor: number | null
  ceiling: number | null
  mean: number | null
  max: number
}) {
  if (floor == null || ceiling == null || max <= 0) return null
  const left = Math.max(0, Math.min(100, (floor / max) * 100))
  const right = Math.max(0, Math.min(100, (ceiling / max) * 100))
  const tick = mean == null ? null : Math.max(0, Math.min(100, (mean / max) * 100))

  return (
    <div className="relative h-1.5 w-full bg-slate-700 rounded-full overflow-hidden">
      <div
        className="absolute inset-y-0 bg-brand-green/60 rounded-full"
        style={{ left: `${left}%`, width: `${Math.max(1, right - left)}%` }}
      />
      {tick != null && (
        <div className="absolute inset-y-0 w-0.5 bg-white/80" style={{ left: `calc(${tick}% - 1px)` }} />
      )}
    </div>
  )
}

function SummaryStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[10px] text-slate-400 uppercase tracking-wide">{label}</p>
      <p className="text-white text-sm font-semibold tabular-nums mt-0.5">{value}</p>
    </div>
  )
}

interface Props {
  playerId: string
  position: string
  season: number
}

/**
 * Week-by-week projection detail for one player. The season page shows summed
 * totals; this is where the matchup-driven variation between weeks shows up.
 */
export default function PlayerProjectionWeeks({ playerId, position, season }: Props) {
  const { data, isLoading, error } = usePlayerProjections(playerId, season)

  const weeks: WeeklyProjection[] = useMemo(() => {
    const rows = data?.data ?? []
    return [...rows].sort((a, b) => a.week - b.week)
  }, [data])

  const totals = useMemo(() => {
    const sum = (pick: (w: WeeklyProjection) => number | null) =>
      weeks.reduce((acc, w) => acc + (pick(w) ?? 0), 0)
    return {
      points: sum((w) => w.projected_points),
      floor: sum((w) => w.floor),
      ceiling: sum((w) => w.ceiling),
    }
  }, [weeks])

  const maxCeiling = useMemo(
    () => weeks.reduce((m, w) => Math.max(m, w.ceiling ?? 0), 0),
    [weeks],
  )

  const stats = statsForPosition(position)

  if (isLoading) return <SkeletonCard rows={6} />

  if (error) {
    return <p className="text-slate-400 text-sm">Failed to load weekly projections.</p>
  }

  if (data?.status === 'no_data' || !weeks.length) {
    return (
      <div className="text-center py-10 text-slate-500">
        <p className="text-sm">No weekly projections cached for {season}.</p>
        <p className="text-xs mt-1 text-slate-600">
          Run the projections job to generate them.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
        <div className="grid grid-cols-4 gap-3">
          <SummaryStat label="Proj Pts" value={fmtPoints(totals.points)} />
          <SummaryStat label="PPG" value={fmtPoints(totals.points / weeks.length)} />
          <SummaryStat label="Weeks" value={String(weeks.length)} />
          <SummaryStat label="Floor–Ceil" value={fmtRange(totals.floor, totals.ceiling)} />
        </div>
        <p className="text-[11px] text-slate-500 mt-3">
          Weekly projections vary with the matchup — the season total is just their sum. The band
          is the 10th to 90th percentile outcome.
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700 text-slate-400 text-left">
              <th className="py-2 pr-3 font-medium">Wk</th>
              <th className="py-2 pr-3 font-medium">Proj</th>
              <th className="py-2 pr-4 font-medium min-w-[6rem]">Floor–Ceiling</th>
              {stats.map((s) => (
                <th key={s.key} className="py-2 pr-3 font-medium" title={s.hint}>
                  {s.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {weeks.map((w) => {
              const flag = w.prediction_type ? TYPE_LABELS[w.prediction_type] : undefined
              return (
                <tr key={w.week} className="border-b border-slate-800">
                  <td className="py-2 pr-3 text-slate-400 tabular-nums">
                    {w.week}
                    {flag && (
                      <span
                        className={`ml-1.5 px-1 py-0.5 rounded text-[9px] uppercase tracking-wide ${flag.className}`}
                      >
                        {flag.label}
                      </span>
                    )}
                  </td>
                  <td className="py-2 pr-3 text-white font-medium tabular-nums">
                    {fmtPoints(w.projected_points)}
                  </td>
                  <td className="py-2 pr-4 min-w-[6rem]" title={`${fmtRange(w.floor, w.ceiling)} points`}>
                    <span className="text-slate-400 text-xs tabular-nums">
                      {fmtRange(w.floor, w.ceiling)}
                    </span>
                    <div className="mt-1">
                      <WeekRangeBar
                        floor={w.floor}
                        ceiling={w.ceiling}
                        mean={w.projected_points}
                        max={maxCeiling}
                      />
                    </div>
                  </td>
                  {stats.map((s) => (
                    <td key={s.key} className="py-2 pr-3 text-slate-300 tabular-nums">
                      {fmtStat(w[s.weeklyKey], s.decimals)}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
