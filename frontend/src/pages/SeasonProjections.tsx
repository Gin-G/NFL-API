import { useMemo, useState } from 'react'
import { Info } from 'lucide-react'
import { useSeasonProjections } from '../api/projections'
import { getAvailableSeasons, getDefaultSeason } from '../utils/nflDate'
import type { SeasonProjectionEntry } from '../api/types'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'
import SeasonProjectionsTable from '../components/SeasonProjectionsTable'
import PlayerProjectionWeeks from '../components/PlayerProjectionWeeks'
import PlayerPanel from '../components/PlayerPanel'

const SEASONS = getAvailableSeasons()
/** '' is the "all positions" tab. */
const POSITION_TABS = ['', 'QB', 'RB', 'WR', 'TE']
const LIMITS = [50, 100, 300, 500]

export default function SeasonProjections() {
  const [season, setSeason] = useState(getDefaultSeason())
  const [position, setPosition] = useState('')
  const [limit, setLimit] = useState(300)
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<SeasonProjectionEntry | null>(null)

  const { data, isLoading, error, refetch } = useSeasonProjections(
    season,
    position || undefined,
    limit,
  )

  const rows = data?.data ?? []
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return rows
    return rows.filter(
      (r) =>
        r.player_name.toLowerCase().includes(q) ||
        (r.team ?? '').toLowerCase().includes(q),
    )
  }, [rows, search])

  return (
    <div className="p-6">
      <PageHeader
        title="Season Projections"
        subtitle="Projected full-season fantasy totals — the sum of every projected week"
      />

      {/* Filters */}
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
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          aria-label="How many players"
        >
          {LIMITS.map((n) => (
            <option key={n} value={n}>Top {n} by projected points</option>
          ))}
        </select>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search player or team"
          className="bg-slate-700 border border-slate-600 text-white placeholder-slate-400 rounded-lg px-3 py-1.5 text-sm w-56"
          aria-label="Search player or team"
        />
      </div>

      {/* Position tabs */}
      <div className="flex gap-1 mb-6 bg-slate-800 rounded-lg p-1 w-fit">
        {POSITION_TABS.map((p) => (
          <button
            key={p || 'all'}
            onClick={() => setPosition(p)}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
              position === p ? 'bg-brand-green text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            {p || 'All'}
          </button>
        ))}
      </div>

      {/* How to read these projections */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 mb-6 flex gap-3">
        <Info size={16} className="text-slate-400 shrink-0 mt-0.5" />
        <div className="text-xs text-slate-400 space-y-1.5">
          <p>
            Totals are the sum of a player's <span className="text-slate-200 font-medium">weekly</span>{' '}
            projections — click a row to see the week-by-week detail, where the matchup-driven
            variation lives.
          </p>
          <p>
            <span className="text-slate-200 font-medium">Floor–Ceiling</span> is the 10th to 90th
            percentile season outcome. A wide band means a volatile projection, not a better one;
            the bar's white tick marks the mean.
          </p>
          <p>
            Only the stats that matter for a position are shown — the model emits every stat for
            every player, so a receiver's stray "passing yards" is noise, not a projection.
          </p>
        </div>
      </div>

      {isLoading && (
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <SkeletonCard key={i} rows={2} />
          ))}
        </div>
      )}

      {error && <ErrorCard message="Failed to load season projections" onRetry={() => refetch()} />}

      {data?.status === 'no_data' && (
        <p className="text-slate-400 text-sm">
          No projections cached for {season}. Run the projections job to generate them.
        </p>
      )}

      {data && data.status !== 'no_data' && (
        <>
          <p className="text-slate-400 text-sm mb-3">
            {filtered.length} projected {position ? `${position}s` : 'players'} · {season} season
            totals{search && ` matching "${search}"`} — click a player for their weekly projections
          </p>
          <SeasonProjectionsTable rows={filtered} position={position} onSelect={setSelected} />
        </>
      )}

      {selected && (
        <PlayerPanel
          playerId={selected.player_id}
          playerName={selected.player_name}
          subtitle={`${selected.position}${selected.team ? ` · ${selected.team}` : ''} · ${season} projections`}
          season={season}
          showBreakdown={false}
          onClose={() => setSelected(null)}
        >
          <PlayerProjectionWeeks
            playerId={selected.player_id}
            position={selected.position}
            season={season}
          />
        </PlayerPanel>
      )}
    </div>
  )
}
