import { useState } from 'react'
import { Info } from 'lucide-react'
import { useTeamRatings } from '../api/grades'
import { getAvailableSeasons, getDefaultSeason } from '../utils/nflDate'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'
import GradeScaleLegend from '../components/ui/GradeScaleLegend'
import TeamRatingsTable from '../components/TeamRatingsTable'
import MatchupHelper from '../components/MatchupHelper'

const SEASONS = getAvailableSeasons()
const WEEKS = Array.from({ length: 18 }, (_, i) => i + 1)

export default function Ratings() {
  const [season, setSeason] = useState(getDefaultSeason())
  const [throughWeek, setThroughWeek] = useState<number | undefined>(undefined)

  const { data, isLoading, error, refetch } = useTeamRatings(season, throughWeek)

  const ratings = data?.data ?? []
  // Every grade is prior-only until games are played — say so rather than
  // presenting a preseason projection as a measured result.
  const preseasonOnly = ratings.length > 0 && ratings.every((r) => r.games === 0)
  const maxGames = ratings.reduce((m, r) => Math.max(m, r.games), 0)

  return (
    <div className="p-6">
      <PageHeader
        title="Team Ratings"
        subtitle="Opponent-adjusted offense and defense grades — 0–100, where 50 is league average"
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
          value={throughWeek ?? ''}
          onChange={(e) => setThroughWeek(e.target.value ? Number(e.target.value) : undefined)}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          aria-label="Grade through week"
        >
          <option value="">All completed games</option>
          {WEEKS.map((w) => (
            <option key={w} value={w}>Through week {w}</option>
          ))}
        </select>
        <GradeScaleLegend />
      </div>

      {/* How to read these grades */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 mb-6 flex gap-3">
        <Info size={16} className="text-slate-400 shrink-0 mt-0.5" />
        <div className="text-xs text-slate-400 space-y-1.5">
          <p>
            <span className="text-slate-200 font-medium">Offense grade</span> — higher is a better
            offense. This is the more trustworthy of the two: team offense is a stable,
            predictable trait.
          </p>
          <p>
            <span className="text-slate-200 font-medium">Defense grade</span> — higher means a
            stingier defense (fewer points allowed).{' '}
            <span className="text-slate-300">
              Defense is noisier year to year, so read these with less confidence
            </span>{' '}
            — they're shown dimmed for that reason.
          </p>
          <p>
            Grades blend a preseason prior (last season regressed toward the mean) with in-season
            results, and the prior fades as games are played. Use{' '}
            <span className="text-slate-300">Through week N</span> for a point-in-time view of how
            grades moved over a season.
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

      {error && <ErrorCard message="Failed to load team ratings" onRetry={() => refetch()} />}

      {data?.status === 'no_data' && (
        <p className="text-slate-400 text-sm">{data.message}</p>
      )}

      {data && data.status !== 'no_data' && (
        <>
          {preseasonOnly && (
            <div className="bg-slate-800 border border-brand-gold/40 rounded-xl px-4 py-3 mb-6 text-xs text-slate-300">
              No {season} games have been graded yet — every grade below is the preseason prior
              (last season's rating regressed toward the mean). They'll update as games are played.
            </div>
          )}

          <p className="text-slate-400 text-sm mb-6">
            {data.total_teams} teams ·{' '}
            {throughWeek ? `graded through week ${throughWeek}` : 'all completed games'}
            {maxGames > 0 && ` · up to ${maxGames} game${maxGames === 1 ? '' : 's'} graded per team`}
          </p>

          <MatchupHelper ratings={ratings} />

          <h2 className="text-sm font-semibold text-slate-300 mb-3">
            All teams <span className="text-slate-500 font-normal">— click a column to sort</span>
          </h2>
          <TeamRatingsTable ratings={ratings} />
        </>
      )}
    </div>
  )
}
