import { useState } from 'react'
import { useCoaches, useCoachAnalysis, useCoachGrades, useTeamStaff } from '../api/coaches'
import { useTeams } from '../api/teams'
import { getAvailableSeasons } from '../utils/nflDate'
import type { Coach, Team } from '../api/types'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'
import GradeBadge from '../components/ui/GradeBadge'
import WinPctBarChart from '../components/charts/WinPctBarChart'
import CoachBreakdown from '../components/CoachBreakdown'
import { X, TrendingUp, ChevronRight } from 'lucide-react'

type Tab = 'overview' | 'breakdown'

// Load all available seasons so coaches show their full career history
const ALL_YEARS = getAvailableSeasons(1999)

// Static division structure — current 32 franchises by active abbreviation
const NFL_STRUCTURE: Record<string, Record<string, string[]>> = {
  AFC: {
    East:  ['BUF', 'MIA', 'NE', 'NYJ'],
    North: ['BAL', 'CIN', 'CLE', 'PIT'],
    South: ['HOU', 'IND', 'JAX', 'TEN'],
    West:  ['DEN', 'KC', 'LAC', 'LV'],
  },
  NFC: {
    East:  ['DAL', 'NYG', 'PHI', 'WAS'],
    North: ['CHI', 'DET', 'GB', 'MIN'],
    South: ['ATL', 'CAR', 'NO', 'TB'],
    West:  ['ARI', 'LA', 'SEA', 'SF'],
  },
}

/** Shorten team name to city/region; keep nickname to disambiguate NY/LA pairs. */
function teamCity(team: Team): string {
  const { team_abbr, team_name } = team
  if (team_abbr === 'NYG') return 'NY Giants'
  if (team_abbr === 'NYJ') return 'NY Jets'
  if (team_abbr === 'LAC') return 'LA Chargers'
  if (team_abbr === 'LA' || team_abbr === 'LAR') return 'LA Rams'
  // Strip the last word (the nickname): "New England Patriots" → "New England"
  const parts = team_name.split(' ')
  return parts.length > 1 ? parts.slice(0, -1).join(' ') : team_name
}

// ── Detail panel ──────────────────────────────────────────────────────────────

function CoachDetailPanel({
  coachName,
  years,
  onClose,
}: {
  coachName: string
  years: number[]
  onClose: () => void
}) {
  const [tab, setTab] = useState<Tab>('overview')
  const { data: analysisData, isLoading: analysisLoading } = useCoachAnalysis(coachName, years)
  const { data: gradesData, isLoading: gradesLoading } = useCoachGrades(coachName, years)
  const { data: staffData } = useTeamStaff()

  const staffEntry = staffData?.configured
    ? staffData.data.find((s) => s.head_coach?.toLowerCase() === coachName.toLowerCase())
    : undefined

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-black/60">
      <div className="min-h-full flex items-start justify-center p-4">
        <div className="bg-slate-800 rounded-2xl w-full max-w-3xl border border-slate-700 p-6 relative my-8">
          <button onClick={onClose} className="absolute top-4 right-4 text-slate-400 hover:text-white">
            <X size={20} />
          </button>

          <h2 className="text-xl font-bold text-white mb-1">{coachName}</h2>

          {staffEntry && (
            <div className="flex flex-wrap gap-3 mb-3">
              {staffEntry.offensive_coordinator && (
                <div className="bg-slate-700/60 rounded-lg px-3 py-1.5 text-xs">
                  <span className="text-slate-400 mr-1">OC</span>
                  <span className="text-white font-medium">{staffEntry.offensive_coordinator}</span>
                </div>
              )}
              {staffEntry.defensive_coordinator && (
                <div className="bg-slate-700/60 rounded-lg px-3 py-1.5 text-xs">
                  <span className="text-slate-400 mr-1">DC</span>
                  <span className="text-white font-medium">{staffEntry.defensive_coordinator}</span>
                </div>
              )}
              {staffEntry.special_teams_coordinator && (
                <div className="bg-slate-700/60 rounded-lg px-3 py-1.5 text-xs">
                  <span className="text-slate-400 mr-1">STC</span>
                  <span className="text-white font-medium">{staffEntry.special_teams_coordinator}</span>
                </div>
              )}
            </div>
          )}

          <div className="flex gap-1 mb-5 border-b border-slate-700">
            {(['overview', 'breakdown'] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-2 text-sm font-medium capitalize transition-colors border-b-2 -mb-px ${
                  tab === t
                    ? 'border-brand-green text-white'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                {t === 'breakdown' ? 'Scheme Breakdown' : 'Overview'}
              </button>
            ))}
          </div>

          {tab === 'overview' && (
            <>
              {(analysisLoading || gradesLoading) && <SkeletonCard rows={6} />}
              {analysisData && (
                <div className="mb-6">
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-2">
                    Season Records
                  </h3>
                  {analysisData.seasons.length > 0 ? (
                    <WinPctBarChart data={analysisData.seasons} />
                  ) : (
                    <p className="text-slate-500 text-sm">No season data available.</p>
                  )}
                </div>
              )}
              {gradesData?.grades && gradesData.grades.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">
                    Performance Grades
                  </h3>
                  <div className="space-y-3">
                    {gradesData.grades.map((g) => (
                      <div key={g.season} className="bg-slate-700/50 rounded-xl p-4">
                        <div className="flex items-center justify-between mb-2">
                          <div>
                            <span className="font-semibold text-white">{g.season}</span>
                            <span className="text-slate-400 text-sm ml-2">
                              {g.teams.join(', ')} · {g.record}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="text-center">
                              <p className="text-xs text-slate-500 mb-1">Win</p>
                              <GradeBadge grade={g.win_letter_grade} />
                            </div>
                            {g.roster_quality_grade && (
                              <div className="text-center">
                                <p className="text-xs text-slate-500 mb-1">Roster</p>
                                <GradeBadge grade={g.roster_quality_grade} />
                              </div>
                            )}
                          </div>
                        </div>
                        <div className="flex gap-4 text-xs text-slate-400">
                          <span>Win %: {g.win_percentage.toFixed(1)}%</span>
                          <span>Win score: {g.win_score.toFixed(1)}</span>
                          {g.roster_quality_score !== undefined && (
                            <span>Roster score: {g.roster_quality_score.toFixed(1)}</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {analysisData && Object.keys(analysisData.offensive_analysis).length > 0 && (
                <div className="mt-6">
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">
                    Roster Quality Breakdown
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {Object.values(analysisData.offensive_analysis).map((oa) => (
                      <div key={`${oa.team}-${oa.season}`} className="bg-slate-700/50 rounded-lg p-3 text-xs">
                        <p className="font-semibold text-slate-300 mb-2">{oa.team} {oa.season} — Offense</p>
                        <div className="space-y-1 text-slate-400">
                          <p>QB avg: {oa.qb_avg_grade?.toFixed(1) ?? '—'}</p>
                          <p>RB avg: {oa.rb_avg_grade?.toFixed(1) ?? '—'}</p>
                          <p>WR/TE avg: {oa.wr_te_avg_grade?.toFixed(1) ?? '—'}</p>
                        </div>
                      </div>
                    ))}
                    {Object.values(analysisData.defensive_analysis).map((da) => (
                      <div key={`${da.team}-${da.season}-def`} className="bg-slate-700/50 rounded-lg p-3 text-xs">
                        <p className="font-semibold text-slate-300 mb-2">{da.team} {da.season} — Defense</p>
                        <div className="space-y-1 text-slate-400">
                          <p>Defense avg: {da.defense_avg_grade?.toFixed(1) ?? '—'}</p>
                          <p>Overall avg: {da.overall_avg_grade?.toFixed(1) ?? '—'}</p>
                          <p>Tier: {da.roster_tier ?? '—'}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {tab === 'breakdown' && <CoachBreakdown coachName={coachName} />}
        </div>
      </div>
    </div>
  )
}

// ── Legacy coach card ─────────────────────────────────────────────────────────

function CoachCard({ coach, onClick }: { coach: Coach; onClick: () => void }) {
  const latestSeason = coach.seasons[coach.seasons.length - 1]
  const totalWins = coach.seasons.reduce((acc, s) => acc + s.wins, 0)
  const totalLosses = coach.seasons.reduce((acc, s) => acc + s.losses, 0)
  const overallWinPct =
    totalWins + totalLosses > 0
      ? ((totalWins / (totalWins + totalLosses)) * 100).toFixed(1)
      : '—'

  return (
    <button
      onClick={onClick}
      className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-500 rounded-xl p-4 text-left transition-colors w-full"
    >
      <div className="flex items-start justify-between gap-2 mb-3">
        <div>
          <p className="font-semibold text-white">{coach.name}</p>
          {latestSeason && (
            <p className="text-xs text-slate-400 mt-0.5">
              {latestSeason.teams.join(', ')} · {latestSeason.season}
            </p>
          )}
        </div>
        <TrendingUp size={16} className="text-slate-500 shrink-0 mt-1" />
      </div>
      <div className="flex gap-4 text-sm">
        <div>
          <p className="text-slate-400 text-xs">Overall</p>
          <p className="text-white font-bold">{totalWins}–{totalLosses}</p>
        </div>
        <div>
          <p className="text-slate-400 text-xs">Win %</p>
          <p className="text-white font-bold">{overallWinPct}%</p>
        </div>
        {latestSeason && (
          <div>
            <p className="text-slate-400 text-xs">{latestSeason.season}</p>
            <p className="text-white font-bold">{latestSeason.record}</p>
          </div>
        )}
      </div>
    </button>
  )
}

// ── League board (top section) ────────────────────────────────────────────────

function LeagueBoard({
  coaches,
  teams,
  onSelectCoach,
}: {
  coaches: Coach[]
  teams: Team[]
  onSelectCoach: (name: string) => void
}) {
  // Build abbr → team and abbr → coach lookups
  const teamByAbbr = Object.fromEntries(teams.map((t) => [t.team_abbr, t]))
  const coachByTeam = Object.fromEntries(
    coaches
      .filter((c) => c.is_active)
      .map((c) => {
        const latestTeam = c.seasons[c.seasons.length - 1]?.teams[0] ?? ''
        return [latestTeam, c]
      })
  )

  return (
    <div className="space-y-8 mb-12">
      {Object.entries(NFL_STRUCTURE).map(([conf, divisions]) => (
        <div key={conf}>
          {/* Conference header */}
          <div className="flex items-center gap-3 mb-4">
            <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${
              conf === 'AFC'
                ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                : 'bg-red-500/20 text-red-300 border border-red-500/30'
            }`}>
              {conf}
            </span>
            <div className="flex-1 h-px bg-slate-700" />
          </div>

          {/* 4 divisions side by side */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(divisions).map(([divName, abbrs]) => (
              <div
                key={divName}
                className="bg-slate-800/60 border border-slate-700 rounded-xl overflow-hidden"
              >
                {/* Division label */}
                <div className="px-4 py-2 bg-slate-700/50 border-b border-slate-700">
                  <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    {conf} {divName}
                  </span>
                </div>

                {/* Team rows */}
                <div className="divide-y divide-slate-700/50">
                  {abbrs.map((abbr) => {
                    const team = teamByAbbr[abbr]
                    const coach = coachByTeam[abbr]
                    const city = team ? teamCity(team) : abbr

                    return (
                      <div key={abbr} className="px-4 py-3 flex items-center justify-between gap-2">
                        <div className="min-w-0">
                          <p className="text-xs text-slate-400 font-medium truncate">{city}</p>
                          {coach ? (
                            <button
                              onClick={() => onSelectCoach(coach.name)}
                              className="text-sm font-semibold text-white hover:text-brand-green transition-colors text-left truncate flex items-center gap-1 group"
                            >
                              {coach.name}
                              <ChevronRight
                                size={12}
                                className="text-slate-500 group-hover:text-brand-green shrink-0 transition-colors"
                              />
                            </button>
                          ) : (
                            <p className="text-sm text-slate-500 italic">TBD</p>
                          )}
                        </div>
                        {coach && (
                          <div className="text-right shrink-0">
                            <p className="text-xs text-slate-500">
                              {coach.seasons[coach.seasons.length - 1]?.record ?? ''}
                            </p>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Coaches() {
  const years = ALL_YEARS
  const [selected, setSelected] = useState<string | null>(null)
  const [search, setSearch] = useState('')

  const { data, isLoading, error, refetch } = useCoaches(years)
  const { data: teamsData } = useTeams()

  const allCoaches = data?.data ?? []
  const activeCoaches = allCoaches.filter((c) => c.is_active)
  const teams = teamsData?.data ?? []

  const filteredLegacy = allCoaches
    .filter((c) => !c.is_active)
    .filter((c) => c.name.toLowerCase().includes(search.toLowerCase()))

  return (
    <div className="p-6">
      <PageHeader title="Coaches" subtitle="Current rosters by division · historical records below" />

      {/* ── League board ── */}
      {isLoading && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-12">
          {Array.from({ length: 8 }).map((_, i) => <SkeletonCard key={i} rows={5} />)}
        </div>
      )}
      {error && <ErrorCard message="Failed to load coaches" onRetry={() => refetch()} />}

      {!isLoading && !error && (
        <LeagueBoard
          coaches={activeCoaches}
          teams={teams}
          onSelectCoach={setSelected}
        />
      )}

      {/* ── Former head coaches ── */}
      <div className="border-t border-slate-700 pt-8">
        <div className="flex items-center gap-4 mb-5">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wide whitespace-nowrap">
            Former Head Coaches
          </h2>
          <input
            placeholder="Search..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm w-44 placeholder-slate-500"
          />
          {data && (
            <span className="text-slate-500 text-sm">
              {filteredLegacy.length} coaches
            </span>
          )}
        </div>

        {isLoading && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} rows={4} />)}
          </div>
        )}

        {!isLoading && !error && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredLegacy.map((c) => (
              <CoachCard key={c.name} coach={c} onClick={() => setSelected(c.name)} />
            ))}
          </div>
        )}
      </div>

      {selected && (
        <CoachDetailPanel
          coachName={selected}
          years={years}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  )
}
