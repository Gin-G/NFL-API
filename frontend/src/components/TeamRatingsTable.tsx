import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowDown, ArrowUp } from 'lucide-react'
import { useTeams } from '../api/teams'
import type { TeamRating } from '../api/types'
import GradeMeter from './ui/GradeMeter'

type SortKey = keyof Pick<
  TeamRating,
  'team' | 'games' | 'offense_grade' | 'defense_grade' | 'offense_rating' | 'defense_rating'
>

interface Column {
  key: SortKey
  label: string
  /** Which direction a fresh click on this column should sort. */
  defaultDesc: boolean
  hint?: string
  className?: string
}

const COLUMNS: Column[] = [
  { key: 'team', label: 'Team', defaultDesc: false },
  { key: 'games', label: 'G', defaultDesc: true, hint: 'Games graded', className: 'hidden sm:table-cell' },
  {
    key: 'offense_grade',
    label: 'Offense',
    defaultDesc: true,
    hint: '0-100, 50 = league average. Higher = better offense.',
  },
  {
    key: 'defense_grade',
    label: 'Defense',
    defaultDesc: true,
    hint: '0-100, 50 = league average. Higher = stingier defense (fewer points allowed). Noisier than the offense grade.',
  },
  {
    key: 'offense_rating',
    label: 'Off pts',
    defaultDesc: true,
    hint: 'Opponent-adjusted points scored per game vs league average',
    className: 'hidden md:table-cell',
  },
  {
    key: 'defense_rating',
    label: 'Def pts',
    defaultDesc: false,
    hint: 'Opponent-adjusted points allowed per game vs league average (lower is better)',
    className: 'hidden md:table-cell',
  },
]

function ptsFmt(v: number): string {
  return (v >= 0 ? '+' : '') + v.toFixed(1)
}

/** Sortable 32-team grade table. */
export default function TeamRatingsTable({ ratings }: { ratings: TeamRating[] }) {
  const navigate = useNavigate()
  const { data: teamsData } = useTeams()
  const [sortKey, setSortKey] = useState<SortKey>('offense_grade')
  const [desc, setDesc] = useState(true)

  const teamMeta = useMemo(() => {
    const map: Record<string, { name: string; logo: string | null }> = {}
    for (const t of teamsData?.data ?? []) {
      map[t.team_abbr] = { name: t.team_name, logo: t.team_logo_espn }
    }
    return map
  }, [teamsData])

  const sorted = useMemo(() => {
    const rows = [...ratings]
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
  }, [ratings, sortKey, desc])

  function toggleSort(col: Column) {
    if (col.key === sortKey) {
      setDesc((d) => !d)
    } else {
      setSortKey(col.key)
      setDesc(col.defaultDesc)
    }
  }

  if (!ratings.length) return <p className="text-slate-400 text-sm">No team ratings available.</p>

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
                  {sortKey === col.key &&
                    (desc ? <ArrowDown size={12} /> : <ArrowUp size={12} />)}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => {
            const meta = teamMeta[r.team]
            return (
              <tr
                key={r.team}
                onClick={() => navigate(`/teams/${r.team}`)}
                className="border-b border-slate-800 hover:bg-slate-800/70 cursor-pointer transition-colors"
              >
                <td className="py-2 pr-3 text-slate-500 tabular-nums">{i + 1}</td>
                <td className="py-2 pr-4">
                  <div className="flex items-center gap-2">
                    {meta?.logo && (
                      <img src={meta.logo} alt="" className="w-6 h-6 object-contain shrink-0" />
                    )}
                    <div className="min-w-0">
                      <span className="text-white font-medium">{r.team}</span>
                      <span className="hidden lg:inline text-slate-400 ml-2 truncate">
                        {meta?.name ?? ''}
                      </span>
                    </div>
                  </div>
                </td>
                <td className="py-2 pr-4 text-slate-400 tabular-nums hidden sm:table-cell">
                  {r.games}
                </td>
                <td className="py-2 pr-4 min-w-[7rem]">
                  <GradeMeter value={r.offense_grade} />
                </td>
                <td className="py-2 pr-4 min-w-[7rem]">
                  <GradeMeter
                    value={r.defense_grade}
                    lowConfidence
                    title="Defense grades are noisier than offense grades — read with less confidence."
                  />
                </td>
                <td className="py-2 pr-4 tabular-nums hidden md:table-cell text-slate-300">
                  {ptsFmt(r.offense_rating)}
                </td>
                <td className="py-2 pr-4 tabular-nums hidden md:table-cell text-slate-300">
                  {ptsFmt(r.defense_rating)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
