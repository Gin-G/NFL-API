import { useState } from 'react'
import { useCoaches, useCoachAnalysis, useCoachGrades, useTeamStaff } from '../api/coaches'
import { getAvailableSeasons } from '../utils/nflDate'
import type { Coach } from '../api/types'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'
import GradeBadge from '../components/ui/GradeBadge'
import WinPctBarChart from '../components/charts/WinPctBarChart'
import CoachBreakdown from '../components/CoachBreakdown'
import { X, TrendingUp } from 'lucide-react'

type Tab = 'overview' | 'breakdown'

// Load all available seasons so coaches show their full career history
const ALL_YEARS = getAvailableSeasons(1999)

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

  // Find the team entry where this coach is the head coach
  const staffEntry = staffData?.configured
    ? staffData.data.find(
        (s) => s.head_coach?.toLowerCase() === coachName.toLowerCase()
      )
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

        {/* Tab bar */}
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

        {/* Overview tab */}
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

        {/* Scheme Breakdown tab */}
        {tab === 'breakdown' && <CoachBreakdown coachName={coachName} />}
      </div>
      </div>
    </div>
  )
}

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

export default function Coaches() {
  const years = ALL_YEARS
  const [selected, setSelected] = useState<string | null>(null)
  const [search, setSearch] = useState('')

  const { data, isLoading, error, refetch } = useCoaches(years)

  const allCoaches = (data?.data ?? []).filter((c) =>
    c.name.toLowerCase().includes(search.toLowerCase())
  )
  const activeCoaches = allCoaches.filter((c) => c.is_active)
  const legacyCoaches = allCoaches.filter((c) => !c.is_active)

  return (
    <div className="p-6">
      <PageHeader title="Coaches" subtitle="Head coach records and performance grades" />

      <div className="flex gap-3 mb-6">
        <input
          placeholder="Search coach..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm w-48 placeholder-slate-500"
        />
        {data && (
          <span className="text-slate-400 text-sm self-center">
            {allCoaches.length} coaches
          </span>
        )}
      </div>

      {isLoading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 9 }).map((_, i) => <SkeletonCard key={i} rows={4} />)}
        </div>
      )}

      {error && <ErrorCard message="Failed to load coaches" onRetry={() => refetch()} />}

      {!isLoading && !error && (
        <>
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">
            Active Head Coaches ({activeCoaches.length})
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
            {activeCoaches.map((c) => (
              <CoachCard key={c.name} coach={c} onClick={() => setSelected(c.name)} />
            ))}
          </div>

          {legacyCoaches.length > 0 && (
            <>
              <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">
                Former Head Coaches ({legacyCoaches.length})
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {legacyCoaches.map((c) => (
                  <CoachCard key={c.name} coach={c} onClick={() => setSelected(c.name)} />
                ))}
              </div>
            </>
          )}
        </>
      )}

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
