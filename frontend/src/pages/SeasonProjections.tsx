import { useMemo, useState } from 'react'
import { Info } from 'lucide-react'
import { useSeasonProjections, useWeeklyProjections } from '../api/projections'
import { getAvailableSeasons, getDefaultSeason, getCurrentNFLWeek } from '../utils/nflDate'
import type { SeasonProjectionEntry } from '../api/types'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'
import SeasonProjectionsTable from '../components/SeasonProjectionsTable'
import WeeklyProjectionsTable from '../components/WeeklyProjectionsTable'
import PlayerProjectionWeeks from '../components/PlayerProjectionWeeks'
import PlayerPanel from '../components/PlayerPanel'

const SEASONS = getAvailableSeasons()
/** '' is the "all positions" tab. */
const POSITION_TABS = ['', 'QB', 'RB', 'WR', 'TE']
const LIMITS = [50, 100, 300, 500]
const WEEKS = Array.from({ length: 18 }, (_, i) => i + 1)

export default function SeasonProjections() {
  const [season, setSeason] = useState(getDefaultSeason())
  const [position, setPosition] = useState('')
  const [limit, setLimit] = useState(300)
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<SeasonProjectionEntry | null>(null)
  const [scope, setScope] = useState<'season' | 'week'>('season')
  const [week, setWeek] = useState(() => getCurrentNFLWeek(getDefaultSeason()))

  const seasonQ = useSeasonProjections(season, position || undefined, limit)
  // Both hooks are always mounted; react-query caches per key, so flipping
  // scope is instant rather than a refetch. Only the active one is rendered.
  const weekQ = useWeeklyProjections(season, week, position || undefined, limit)
  const active = scope === 'season' ? seasonQ : weekQ
  const { data, isLoading, error, refetch } = active

  // When the week on screen was last computed, and whether that predates the
  // season. A week the job has not come back to still carries its preseason
  // number, and a projection with no football behind it should not look like
  // one that has seen five games.
  const stamp = weekQ.data?.data?.[0]?.computed_at ?? null
  const computedAt = stamp ? new Date(stamp).toLocaleDateString() : null
  const stale = !!stamp && new Date(stamp) < new Date(`${season}-09-01`)

  const matches = (name: string, team: string | null) => {
    const q = search.trim().toLowerCase()
    if (!q) return true
    return name.toLowerCase().includes(q) || (team ?? '').toLowerCase().includes(q)
  }
  const seasonRows = useMemo(
    () => (seasonQ.data?.data ?? []).filter((r) => matches(r.player_name, r.team)),
    [seasonQ.data, search],
  )
  const weekRows = useMemo(
    () => (weekQ.data?.data ?? []).filter((r) => matches(r.player_name, r.team)),
    [weekQ.data, search],
  )

  return (
    <div className="p-6">
      <PageHeader
        title="Projections"
        subtitle={
          scope === 'season'
            ? 'Projected full-season fantasy totals — the sum of every projected week'
            : `Week ${week} — the points expected in one game, not a season average`
        }
      />

      {/* Season or a single week. The season view is the sum of the weekly
          ones, so this is the same projection read at two scales. */}
      <div className="flex gap-1 mb-4 bg-slate-800 rounded-lg p-1 w-fit">
        {(['season', 'week'] as const).map((s) => (
          <button
            key={s}
            onClick={() => setScope(s)}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
              scope === s ? 'bg-brand-green text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            {s === 'season' ? 'Season' : 'By week'}
          </button>
        ))}
      </div>

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
        {scope === 'week' && (
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
        )}
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

      {error && (
        <ErrorCard
          message={`Failed to load ${scope === 'season' ? 'season' : `week ${week}`} projections`}
          onRetry={() => refetch()}
        />
      )}

      {data?.status === 'no_data' && (
        <p className="text-slate-400 text-sm">
          {scope === 'season'
            ? `No projections cached for ${season}. Run the projections job to generate them.`
            : `No projections cached for ${season} week ${week} yet. The job projects the
               upcoming week and everything after it, so a past week stops being refreshed
               once it has been played.`}
        </p>
      )}

      {data && data.status !== 'no_data' && scope === 'season' && (
        <>
          <p className="text-slate-400 text-sm mb-3">
            {seasonRows.length} projected {position ? `${position}s` : 'players'} · {season} season
            totals{search && ` matching "${search}"`} — click a player for their weekly projections
          </p>
          <SeasonProjectionsTable rows={seasonRows} position={position} onSelect={setSelected} />
        </>
      )}

      {data && data.status !== 'no_data' && scope === 'week' && (
        <>
          <p className="text-slate-400 text-sm mb-3">
            {weekRows.length} projected {position ? `${position}s` : 'players'} · {season} week{' '}
            {week}{search && ` matching "${search}"`}
            {computedAt && (
              <>
                {' · '}
                <span className={stale ? 'text-amber-400' : 'text-slate-300'}>
                  computed {computedAt}
                </span>
              </>
            )}
          </p>
          {stale && (
            <div className="bg-slate-800 border border-amber-700/40 rounded-xl p-3 mb-4 flex gap-2 text-xs text-amber-200/80">
              <Info size={14} className="shrink-0 mt-0.5" />
              <span>
                This is the preseason projection for week {week}, not a refreshed one. The job
                republishes the <span className="font-medium">upcoming</span> week each Wednesday;
                later weeks keep their preseason number until their turn comes round, so nothing
                here has seen a snap of this season.
              </span>
            </div>
          )}
          <WeeklyProjectionsTable rows={weekRows} position={position} />
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
