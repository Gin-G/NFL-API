import type { LineMatchup } from '../api/types'
import GradeScore from './ui/GradeScore'

/** Band label for a 0-100 matchup rating — mirrors trenches.BANDS on the API. */
function band(rating: number | null): { label: string; tone: string } {
  if (rating === null) return { label: '—', tone: 'text-slate-500' }
  if (rating >= 70) return { label: 'Smash', tone: 'text-green-400' }
  if (rating >= 58) return { label: 'Lean up', tone: 'text-emerald-400' }
  if (rating >= 42) return { label: 'Neutral', tone: 'text-slate-400' }
  if (rating >= 30) return { label: 'Lean down', tone: 'text-orange-400' }
  return { label: 'Fade', tone: 'text-red-400' }
}

interface Props {
  rows: LineMatchup[]
  basis?: string
}

/**
 * One row per offense, ordered by the run matchup. The opponent's front gets
 * its own column rather than being folded into the rating alone, because it is
 * the half of the matchup that the projection does not already know.
 */
export default function LineMatchupTable({ rows, basis }: Props) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-slate-900/60 text-xs uppercase tracking-wide text-slate-400">
            <tr>
              <th className="text-left px-4 py-2 font-medium">Offense</th>
              <th className="text-left px-3 py-2 font-medium">vs</th>
              <th className="text-center px-3 py-2 font-medium">Run matchup</th>
              <th className="text-center px-3 py-2 font-medium">Own run block</th>
              <th className="text-center px-3 py-2 font-medium">Opp run stop</th>
              <th className="text-center px-3 py-2 font-medium">Pass matchup</th>
              <th className="text-center px-3 py-2 font-medium">Own protect</th>
              <th className="text-center px-3 py-2 font-medium">Opp rush</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const run = band(row.run_matchup)
              return (
                <tr
                  key={`${row.team}-${row.opponent}`}
                  className="border-t border-slate-700/60 hover:bg-slate-700/30"
                >
                  <td className="px-4 py-2">
                    <span className="font-medium text-slate-100">{row.team}</span>
                    <span className="ml-2 text-xs text-slate-500">
                      {row.home ? 'home' : 'away'}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-slate-400">{row.opponent}</td>
                  <td className="px-3 py-2 text-center">
                    <div className="inline-flex flex-col items-center gap-0.5">
                      <GradeScore value={row.run_matchup} size="md" />
                      <span className={`text-[10px] uppercase tracking-wide ${run.tone}`}>
                        {run.label}
                      </span>
                    </div>
                  </td>
                  <td className="px-3 py-2 text-center">
                    <GradeScore value={row.run_block} size="sm" />
                  </td>
                  <td className="px-3 py-2 text-center">
                    <GradeScore value={row.run_stop} size="sm" />
                  </td>
                  <td className="px-3 py-2 text-center">
                    <GradeScore value={row.pass_matchup} size="md" />
                  </td>
                  <td className="px-3 py-2 text-center">
                    <GradeScore value={row.pass_block} size="sm" />
                  </td>
                  <td className="px-3 py-2 text-center">
                    <GradeScore value={row.pass_rush} size="sm" />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {basis && (
        <div className="px-4 py-2 border-t border-slate-700/60 text-xs text-slate-500">
          Grades basis: {basis}. A blend means last season's regressed grade is still carrying part
          of the weight.
        </div>
      )}
    </div>
  )
}
