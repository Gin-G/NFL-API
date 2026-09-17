import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp } from 'lucide-react'
import type { WeeklyProjection } from '../api/types'
import { statsForPosition, fmtStat, fmtPoints, type ProjectionStat } from '../utils/projectionStats'

type SortKey = keyof WeeklyProjection
const TEXT_KEYS = new Set<SortKey>(['player_name', 'team', 'position'])

interface Props {
  rows: WeeklyProjection[]
  position: string
  onSelect?: (row: WeeklyProjection) => void
}

/**
 * One week of the board.
 *
 * Deliberately not the season table with a filter on it. A week is a different
 * question — these are the points expected in one game, with a floor and
 * ceiling around them — and the season view's per-game average would read as
 * the same number while meaning something else entirely.
 */
export default function WeeklyProjectionsTable({ rows, position, onSelect }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('projected_points')
  const [asc, setAsc] = useState(false)

  const stats: ProjectionStat[] = useMemo(
    () => statsForPosition(position),
    [position],
  )

  const sorted = useMemo(() => {
    const out = [...rows]
    out.sort((a, b) => {
      const x = a[sortKey]
      const y = b[sortKey]
      if (x == null && y == null) return 0
      if (x == null) return 1          // nulls last, whichever way it sorts
      if (y == null) return -1
      if (typeof x === 'string' || typeof y === 'string') {
        const c = String(x).localeCompare(String(y))
        return asc ? c : -c
      }
      return asc ? Number(x) - Number(y) : Number(y) - Number(x)
    })
    return out
  }, [rows, sortKey, asc])

  function sortBy(key: SortKey) {
    if (key === sortKey) {
      setAsc(!asc)
      return
    }
    setSortKey(key)
    setAsc(TEXT_KEYS.has(key))
  }

  function Header({ k, label, hint, className }: {
    k: SortKey; label: string; hint?: string; className?: string
  }) {
    return (
      <th
        className={`px-3 py-2 text-left font-medium text-slate-300 cursor-pointer select-none hover:text-white ${className ?? ''}`}
        onClick={() => sortBy(k)}
        title={hint}
      >
        <span className="inline-flex items-center gap-1">
          {label}
          {sortKey === k && (asc ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
        </span>
      </th>
    )
  }

  if (!rows.length) {
    return (
      <div className="text-sm text-slate-400 py-8 text-center">
        No projections cached for this week yet.
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="border-b border-slate-700">
          <tr>
            <Header k="player_name" label="Player" />
            <Header k="team" label="Team" className="hidden sm:table-cell" />
            <Header k="position" label="Pos" className="hidden sm:table-cell" />
            <Header k="projected_points" label="Proj Pts"
                    hint="Projected fantasy points for this week alone" />
            <Header k="floor" label="Floor" hint="10th percentile"
                    className="hidden md:table-cell" />
            <Header k="ceiling" label="Ceiling" hint="90th percentile"
                    className="hidden md:table-cell" />
            {stats.map((s) => (
              <Header key={s.weeklyKey} k={s.weeklyKey as SortKey} label={s.label}
                      hint={s.hint} className="hidden lg:table-cell" />
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr
              key={`${r.player_id}-${r.week}`}
              className={`border-b border-slate-800 hover:bg-slate-800/50 ${onSelect ? 'cursor-pointer' : ''}`}
              onClick={() => onSelect?.(r)}
            >
              <td className="px-3 py-2 text-white">{r.player_name}</td>
              <td className="px-3 py-2 text-slate-400 hidden sm:table-cell">{r.team ?? '—'}</td>
              <td className="px-3 py-2 text-slate-400 hidden sm:table-cell">{r.position}</td>
              <td className="px-3 py-2 text-white tabular-nums">{fmtPoints(r.projected_points)}</td>
              <td className="px-3 py-2 text-slate-400 tabular-nums hidden md:table-cell">
                {fmtPoints(r.floor)}
              </td>
              <td className="px-3 py-2 text-slate-400 tabular-nums hidden md:table-cell">
                {fmtPoints(r.ceiling)}
              </td>
              {/* weeklyKey, not key: the season table calls the same stat
                  `interceptions` where a week calls it `passing_interceptions`. */}
              {stats.map((s) => (
                <td key={s.weeklyKey} className="px-3 py-2 text-slate-400 tabular-nums hidden lg:table-cell">
                  {fmtStat(r[s.weeklyKey] as number | null, s.decimals)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
