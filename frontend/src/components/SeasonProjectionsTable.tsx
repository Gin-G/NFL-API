import { useEffect, useMemo, useState } from 'react'
import { ArrowDown, ArrowUp } from 'lucide-react'
import type { SeasonProjectionEntry } from '../api/types'
import {
  statsForPosition,
  fmtStat,
  fmtPoints,
  fmtRange,
  type ProjectionStat,
} from '../utils/projectionStats'

type SortKey = keyof SeasonProjectionEntry

interface Column {
  key: SortKey
  label: string
  hint?: string
  className?: string
}

/** Keys that sort alphabetically ascending by default; everything else desc. */
const TEXT_KEYS = new Set<SortKey>(['player_name', 'team', 'position'])

/** Columns shown for every position. */
const BASE_COLUMNS: Column[] = [
  { key: 'player_name', label: 'Player' },
  { key: 'team', label: 'Team', className: 'hidden sm:table-cell' },
  { key: 'games', label: 'G', hint: 'Weeks projected', className: 'hidden sm:table-cell' },
  {
    key: 'total_points',
    label: 'Proj Pts',
    hint: 'Projected fantasy points for the full season',
  },
  { key: 'ppg', label: 'PPG', hint: 'Projected points per game' },
  {
    key: 'ceiling_total',
    label: 'Floor–Ceiling',
    hint: '10th to 90th percentile season total — the band, not a guarantee. Sorts by ceiling.',
  },
]

/**
 * Floor-to-ceiling band with a tick at the mean projection, scaled against the
 * best ceiling on screen so rows are comparable at a glance.
 */
function RangeBar({
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
  const width = Math.max(1, right - left)
  const tick = mean == null ? null : Math.max(0, Math.min(100, (mean / max) * 100))

  return (
    <div className="relative h-1.5 w-full bg-slate-700 rounded-full overflow-hidden mt-1">
      <div
        className="absolute inset-y-0 bg-brand-green/60 rounded-full"
        style={{ left: `${left}%`, width: `${width}%` }}
      />
      {tick != null && (
        <div
          className="absolute inset-y-0 w-0.5 bg-white/80"
          style={{ left: `calc(${tick}% - 1px)` }}
        />
      )}
    </div>
  )
}

/** Compact "4512 Pass Yds · 31.2 Pass TD" line, used when positions are mixed. */
function StatLine({ row }: { row: SeasonProjectionEntry }) {
  const stats = statsForPosition(row.position).slice(0, 3)
  if (!stats.length) return <span className="text-slate-600">—</span>
  return (
    <span className="text-slate-400 text-xs whitespace-nowrap">
      {stats.map((s, i) => (
        <span key={s.key}>
          {i > 0 && <span className="text-slate-600"> · </span>}
          <span className="text-slate-300 tabular-nums">{fmtStat(row[s.key], s.decimals)}</span>{' '}
          {s.label}
        </span>
      ))}
    </span>
  )
}

/** Sortable column header — arrow shows the active key and direction. */
function SortHeader({
  label,
  sortableKey,
  activeKey,
  desc,
  onSort,
  hint,
  className,
}: {
  label: string
  sortableKey: SortKey
  activeKey: SortKey
  desc: boolean
  onSort: (key: SortKey) => void
  hint?: string
  className?: string
}) {
  const active = activeKey === sortableKey
  return (
    <th className={`py-2 pr-4 font-medium ${className ?? ''}`}>
      <button
        onClick={() => onSort(sortableKey)}
        title={hint}
        className={`inline-flex items-center gap-1 hover:text-white transition-colors ${
          active ? 'text-white' : ''
        }`}
      >
        {label}
        {active && (desc ? <ArrowDown size={12} /> : <ArrowUp size={12} />)}
      </button>
    </th>
  )
}

interface Props {
  rows: SeasonProjectionEntry[]
  /** Active position filter — '' means all positions are mixed together. */
  position: string
  onSelect: (row: SeasonProjectionEntry) => void
}

/** Sortable season-projection leaderboard with position-aware stat columns. */
export default function SeasonProjectionsTable({ rows, position, onSelect }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('total_points')
  const [desc, setDesc] = useState(true)

  // Stat columns change with the position, so a sort on (say) passing yards has
  // no visible header once you switch to RB — fall back to the default sort.
  useEffect(() => {
    setSortKey('total_points')
    setDesc(true)
  }, [position])

  // Only one position on screen → show its stat columns. Mixed positions get a
  // per-row stat line instead, since a shared set of columns would show
  // meaningless cross-position noise (passing yards for a WR, and so on).
  const statColumns: ProjectionStat[] = useMemo(() => statsForPosition(position), [position])
  const mixed = !position

  const sorted = useMemo(() => {
    const out = [...rows]
    out.sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      // Missing values sort last in both directions.
      if (av == null && bv == null) return 0
      if (av == null) return 1
      if (bv == null) return -1
      const cmp =
        typeof av === 'string' && typeof bv === 'string'
          ? av.localeCompare(bv)
          : (av as number) - (bv as number)
      return desc ? -cmp : cmp
    })
    return out
  }, [rows, sortKey, desc])

  const maxCeiling = useMemo(
    () => rows.reduce((m, r) => Math.max(m, r.ceiling_total ?? 0), 0),
    [rows],
  )

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setDesc((d) => !d)
    } else {
      setSortKey(key)
      // Names read best A→Z; every number reads best biggest-first.
      setDesc(!TEXT_KEYS.has(key))
    }
  }

  if (!rows.length) {
    return <p className="text-slate-400 text-sm">No projected players match these filters.</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700 text-slate-400 text-left">
            <th className="py-2 pr-3 font-medium w-8">#</th>
            {mixed && (
              <SortHeader
                label="Pos"
                sortableKey="position"
                activeKey={sortKey}
                desc={desc}
                onSort={toggleSort}
              />
            )}
            {BASE_COLUMNS.map((col) => (
              <SortHeader
                key={col.key}
                label={col.label}
                sortableKey={col.key}
                activeKey={sortKey}
                desc={desc}
                onSort={toggleSort}
                hint={col.hint}
                className={col.className}
              />
            ))}
            {mixed ? (
              <th className="py-2 pr-4 font-medium hidden lg:table-cell">Projected stats</th>
            ) : (
              statColumns.map((s) => (
                <SortHeader
                  key={s.key}
                  label={s.label}
                  sortableKey={s.key}
                  activeKey={sortKey}
                  desc={desc}
                  onSort={toggleSort}
                  hint={s.hint}
                />
              ))
            )}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => (
            <tr
              key={row.player_id}
              onClick={() => onSelect(row)}
              className="border-b border-slate-800 hover:bg-slate-800/70 cursor-pointer transition-colors"
            >
              <td className="py-2 pr-3 text-slate-500 tabular-nums">{i + 1}</td>
              {mixed && <td className="py-2 pr-4 text-slate-300">{row.position}</td>}
              <td className="py-2 pr-4 text-white font-medium hover:text-brand-green transition-colors whitespace-nowrap">
                {row.player_name}
              </td>
              <td className="py-2 pr-4 text-slate-400 hidden sm:table-cell">{row.team ?? '—'}</td>
              <td className="py-2 pr-4 text-slate-400 tabular-nums hidden sm:table-cell">
                {row.games}
              </td>
              <td className="py-2 pr-4 text-white font-semibold tabular-nums">
                {fmtPoints(row.total_points)}
              </td>
              <td className="py-2 pr-4 text-slate-300 tabular-nums">{fmtPoints(row.ppg)}</td>
              <td
                className="py-2 pr-4 min-w-[7rem]"
                title={`${fmtRange(row.floor_total, row.ceiling_total)} points (10th–90th percentile)`}
              >
                <span className="text-slate-400 text-xs tabular-nums">
                  {fmtRange(row.floor_total, row.ceiling_total)}
                </span>
                <RangeBar
                  floor={row.floor_total}
                  ceiling={row.ceiling_total}
                  mean={row.total_points}
                  max={maxCeiling}
                />
              </td>
              {mixed ? (
                <td className="py-2 pr-4 hidden lg:table-cell">
                  <StatLine row={row} />
                </td>
              ) : (
                statColumns.map((s) => (
                  <td key={s.key} className="py-2 pr-4 text-slate-300 tabular-nums">
                    {fmtStat(row[s.key], s.decimals)}
                  </td>
                ))
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
