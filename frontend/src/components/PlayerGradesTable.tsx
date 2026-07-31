import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp } from 'lucide-react'
import type { PlayerGradeEntry } from '../api/types'
import GradeScore from './ui/GradeScore'
import GradeMeter from './ui/GradeMeter'

type SortKey = keyof Pick<
  PlayerGradeEntry,
  'player_name' | 'position' | 'games' | 'grade' | 'opportunity_grade' | 'efficiency_grade' | 'fppg'
>

interface Column {
  key: SortKey
  label: string
  defaultDesc: boolean
  hint?: string
  className?: string
}

const COLUMNS: Column[] = [
  { key: 'player_name', label: 'Player', defaultDesc: false },
  { key: 'position', label: 'Pos', defaultDesc: false },
  { key: 'games', label: 'G', defaultDesc: true, hint: 'Games played', className: 'hidden sm:table-cell' },
  {
    key: 'grade',
    label: 'Grade',
    defaultDesc: true,
    hint: '0-100, 50 = average for this position. Compare a WR to other WRs, not to QBs.',
  },
  {
    key: 'opportunity_grade',
    label: 'Opportunity',
    defaultDesc: true,
    hint: 'Volume and role (WOPR, target share, carries) — the stable, predictive part of the grade.',
  },
  {
    key: 'efficiency_grade',
    label: 'Efficiency',
    defaultDesc: true,
    hint: 'Per-play production — noisier and less predictive than opportunity.',
  },
  { key: 'fppg', label: 'FPPG', defaultDesc: true, hint: 'Fantasy points per game' },
]

interface Props {
  players: PlayerGradeEntry[]
  onSelect: (p: PlayerGradeEntry) => void
}

/** Sortable player grade leaderboard. */
export default function PlayerGradesTable({ players, onSelect }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('grade')
  const [desc, setDesc] = useState(true)

  const sorted = useMemo(() => {
    const rows = [...players]
    rows.sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      const cmp =
        typeof av === 'string' && typeof bv === 'string'
          ? av.localeCompare(bv)
          : (av as number) - (bv as number)
      return desc ? -cmp : cmp
    })
    return rows
  }, [players, sortKey, desc])

  function toggleSort(col: Column) {
    if (col.key === sortKey) {
      setDesc((d) => !d)
    } else {
      setSortKey(col.key)
      setDesc(col.defaultDesc)
    }
  }

  if (!players.length) return <p className="text-slate-400 text-sm">No players match these filters.</p>

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700 text-slate-400 text-left">
            <th className="py-2 pr-3 font-medium w-8">#</th>
            {COLUMNS.map((col) => (
              <th key={col.key} className={`py-2 pr-4 font-medium ${col.className ?? ''}`}>
                <button
                  onClick={() => toggleSort(col)}
                  title={col.hint}
                  className={`inline-flex items-center gap-1 hover:text-white transition-colors ${
                    sortKey === col.key ? 'text-white' : ''
                  }`}
                >
                  {col.label}
                  {sortKey === col.key && (desc ? <ArrowDown size={12} /> : <ArrowUp size={12} />)}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((p, i) => (
            <tr
              key={p.player_id}
              onClick={() => onSelect(p)}
              className="border-b border-slate-800 hover:bg-slate-800/70 cursor-pointer transition-colors"
            >
              <td className="py-2 pr-3 text-slate-500 tabular-nums">{i + 1}</td>
              <td className="py-2 pr-4 text-white font-medium hover:text-brand-green transition-colors">
                {p.player_name}
              </td>
              <td className="py-2 pr-4 text-slate-300">{p.position}</td>
              <td className="py-2 pr-4 text-slate-400 tabular-nums hidden sm:table-cell">{p.games}</td>
              <td className="py-2 pr-4">
                <GradeScore value={p.grade} size="sm" />
              </td>
              <td className="py-2 pr-4 min-w-[6.5rem]">
                <GradeMeter value={p.opportunity_grade} />
              </td>
              <td className="py-2 pr-4 min-w-[6.5rem]">
                <GradeMeter
                  value={p.efficiency_grade}
                  lowConfidence
                  title="Per-play efficiency is noisier than opportunity — read with less confidence."
                />
              </td>
              <td className="py-2 pr-4 text-slate-300 tabular-nums">{p.fppg.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
