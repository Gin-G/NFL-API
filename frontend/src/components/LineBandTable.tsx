import type { LineBand } from '../api/types'

interface Props {
  title: string
  note?: string
  rows: LineBand[]
}

/**
 * What a band of ratings historically produced. `ypc_vs_own` is the column that
 * means anything on its own: a team's yards per carry in the game minus its own
 * average for the season, which takes team quality out and leaves the matchup.
 */
export default function LineBandTable({ title, note, rows }: Props) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-700/60">
        <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
        {note && <p className="text-xs text-slate-500 mt-0.5">{note}</p>}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-slate-900/60 text-xs uppercase tracking-wide text-slate-400">
            <tr>
              <th className="text-left px-4 py-2 font-medium">Band</th>
              <th className="text-right px-3 py-2 font-medium">Games</th>
              <th className="text-right px-3 py-2 font-medium">Yds/carry</th>
              <th className="text-right px-3 py-2 font-medium">vs own avg</th>
              <th className="text-right px-3 py-2 font-medium">Rush yds</th>
              <th className="text-right px-3 py-2 font-medium">Stuffed</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.band} className="border-t border-slate-700/60">
                <td className="px-4 py-2 capitalize text-slate-200">{row.band}</td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-400">
                  {row.team_games}
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-100">
                  {row.ypc.toFixed(2)}
                </td>
                <td
                  className={`px-3 py-2 text-right tabular-nums ${
                    row.ypc_vs_own > 0 ? 'text-green-400' : 'text-orange-400'
                  }`}
                >
                  {row.ypc_vs_own > 0 ? '+' : ''}
                  {row.ypc_vs_own.toFixed(2)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-300">
                  {row.rush_yards.toFixed(1)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-400">
                  {(row.stuff_rate * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
