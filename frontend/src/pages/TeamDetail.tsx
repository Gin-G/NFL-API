import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ChevronLeft } from 'lucide-react'
import { useTeam, useTeamDepthChart, useTeamSnapCounts } from '../api/teams'
import { useTeamRating } from '../api/grades'
import { useTeamStaff } from '../api/coaches'
import { useSchedules } from '../api/schedules'
import { getAvailableSeasons, getDefaultSeason } from '../utils/nflDate'
import type { DepthChartEntry, SnapCountEntry, Game } from '../api/types'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'
import GradeScore from '../components/ui/GradeScore'

const SEASONS = getAvailableSeasons()

type Tab = 'overview' | 'depth-chart' | 'snap-counts' | 'schedule'

// ─── Overview tab ─────────────────────────────────────────────────────────────

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-slate-400">{label}</span>
      <span className="text-white">{value}</span>
    </div>
  )
}

/** Compact offense/defense grade card, linking through to the full ratings page. */
function TeamGradesCard({ abbr, season }: { abbr: string; season: number }) {
  const { data } = useTeamRating(season, abbr)
  const rating = data?.data
  if (!rating) return null

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
          {season} Grades
        </p>
        <Link to="/ratings" className="text-xs text-brand-green hover:underline">
          All teams
        </Link>
      </div>
      <div className="flex items-center gap-6">
        <GradeScore value={rating.offense_grade} caption="Offense" />
        <GradeScore
          value={rating.defense_grade}
          caption="Defense"
          title="Higher = stingier defense. Noisier than the offense grade."
        />
        <p className="text-[11px] text-slate-500">
          0–100, 50 = league average ·{' '}
          {rating.games === 0
            ? 'preseason prior — no games graded yet'
            : `${rating.games} game${rating.games === 1 ? '' : 's'} graded`}
        </p>
      </div>
    </div>
  )
}

function OverviewTab({ abbr, season }: { abbr: string; season: number }) {
  const { data, isLoading, error, refetch } = useTeam(abbr)
  const { data: staffData } = useTeamStaff(abbr)

  const staff = staffData?.configured
    ? staffData.data.find((s) => s.team_abbr.toUpperCase() === abbr.toUpperCase())
    : undefined

  const hasStaff =
    staff &&
    (staff.head_coach ||
      staff.offensive_coordinator ||
      staff.defensive_coordinator ||
      staff.special_teams_coordinator)

  if (isLoading) return <SkeletonCard rows={8} />
  if (error) return <ErrorCard message="Failed to load team" onRetry={() => refetch()} />
  if (!data?.data) return <p className="text-slate-400 text-sm">No data available.</p>

  const t = data.data

  return (
    <div className="max-w-lg space-y-6">
      <div className="flex items-center gap-6">
        {t.team_logo_espn && (
          <img src={t.team_logo_espn} alt={t.team_abbr} className="w-24 h-24 object-contain" />
        )}
        <div>
          <h2 className="text-2xl font-bold text-white">{t.team_name}</h2>
          <p className="text-slate-400">{t.team_conf} · {t.team_division}</p>
          {t.team_color && (
            <div className="flex items-center gap-2 mt-2">
              <span
                className="w-5 h-5 rounded border border-slate-600"
                style={{ backgroundColor: t.team_color }}
              />
              {t.team_color2 && (
                <span
                  className="w-5 h-5 rounded border border-slate-600"
                  style={{ backgroundColor: t.team_color2 }}
                />
              )}
            </div>
          )}
        </div>
      </div>

      <TeamGradesCard abbr={abbr} season={season} />

      <div className="bg-slate-800 rounded-xl p-4 space-y-2 border border-slate-700">
        <Row label="Abbreviation" value={t.team_abbr} />
        <Row label="Conference" value={t.team_conf ?? '—'} />
        <Row label="Division" value={t.team_division ?? '—'} />
      </div>

      {hasStaff && (
        <div className="bg-slate-800 rounded-xl p-4 space-y-2 border border-slate-700">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">
            Coaching Staff
          </p>
          {staff!.head_coach && <Row label="Head Coach" value={staff!.head_coach} />}
          {staff!.offensive_coordinator && (
            <Row label="Off. Coordinator" value={staff!.offensive_coordinator} />
          )}
          {staff!.defensive_coordinator && (
            <Row label="Def. Coordinator" value={staff!.defensive_coordinator} />
          )}
          {staff!.special_teams_coordinator && (
            <Row label="ST Coordinator" value={staff!.special_teams_coordinator} />
          )}
        </div>
      )}
    </div>
  )
}

// ─── Depth Chart tab ──────────────────────────────────────────────────────────

function DepthChartTab({ abbr, season }: { abbr: string; season: number }) {
  const [week, setWeek] = useState<number | undefined>(undefined)
  const { data, isLoading, error, refetch } = useTeamDepthChart(abbr, season, week)

  // Group by position
  const byPosition: Record<string, DepthChartEntry[]> = {}
  for (const entry of data?.data ?? []) {
    const pos = entry.position ?? 'OTHER'
    if (!byPosition[pos]) byPosition[pos] = []
    byPosition[pos].push(entry)
  }

  // Available weeks from data (if loaded without week filter)
  const weeks = data?.data
    ? [...new Set(data.data.map((e) => e.week))].sort((a, b) => a - b)
    : []

  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <label className="text-sm text-slate-400">Week</label>
        <select
          value={week ?? ''}
          onChange={(e) => setWeek(e.target.value ? Number(e.target.value) : undefined)}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
        >
          <option value="">Latest</option>
          {weeks.map((w) => (
            <option key={w} value={w}>Week {w}</option>
          ))}
        </select>
      </div>

      {isLoading && <SkeletonCard rows={6} />}
      {error && <ErrorCard message="Failed to load depth chart" onRetry={() => refetch()} />}

      {data?.status === 'no_data' && (
        <p className="text-slate-400 text-sm">{data.message ?? 'No depth chart data available.'}</p>
      )}

      {data?.status === 'success' && (
        <div className="space-y-6">
          {Object.entries(byPosition).map(([pos, entries]) => (
            <div key={pos}>
              <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wide mb-2">{pos}</h3>
              <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700 text-slate-400 text-left">
                      <th className="py-2 px-4 font-medium w-16">Depth</th>
                      <th className="py-2 px-4 font-medium">Player</th>
                      <th className="py-2 px-4 font-medium">Position</th>
                    </tr>
                  </thead>
                  <tbody>
                    {entries.map((e, i) => (
                      <tr key={`${e.player_id}-${i}`} className="border-b border-slate-700/50 last:border-0">
                        <td className="py-2 px-4 text-slate-400">{e.depth_team}</td>
                        <td className="py-2 px-4 text-white font-medium">{e.full_name ?? e.player_id}</td>
                        <td className="py-2 px-4 text-slate-300">{e.depth_chart_position ?? pos}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Snap Counts tab ──────────────────────────────────────────────────────────

function SnapCountsTab({ abbr, season }: { abbr: string; season: number }) {
  const [week, setWeek] = useState<number | undefined>(undefined)
  const { data, isLoading, error, refetch } = useTeamSnapCounts(abbr, season, week)

  const weeks = data?.data
    ? [...new Set(data.data.map((e) => e.week))].sort((a, b) => a - b)
    : []

  const rows = data?.data ?? []
  const displayed = week !== undefined ? rows.filter((r) => r.week === week) : rows

  function pct(v: number): string {
    return `${Math.round(v * 100)}%`
  }

  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <label className="text-sm text-slate-400">Week</label>
        <select
          value={week ?? ''}
          onChange={(e) => setWeek(e.target.value ? Number(e.target.value) : undefined)}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
        >
          <option value="">All Weeks</option>
          {weeks.map((w) => (
            <option key={w} value={w}>Week {w}</option>
          ))}
        </select>
      </div>

      {isLoading && <SkeletonCard rows={6} />}
      {error && <ErrorCard message="Failed to load snap counts" onRetry={() => refetch()} />}

      {data?.status === 'no_data' && (
        <p className="text-slate-400 text-sm">{data.message ?? 'No snap count data available.'}</p>
      )}

      {data?.status === 'success' && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700 text-slate-400 text-left">
                <th className="py-2 pr-3 font-medium">Player</th>
                <th className="py-2 pr-3 font-medium">Pos</th>
                {week === undefined && <th className="py-2 pr-3 font-medium">Wk</th>}
                <th className="py-2 pr-3 font-medium text-right">Off</th>
                <th className="py-2 pr-3 font-medium text-right">Off%</th>
                <th className="py-2 pr-3 font-medium text-right">Def</th>
                <th className="py-2 pr-3 font-medium text-right">Def%</th>
                <th className="py-2 pr-3 font-medium text-right">ST</th>
                <th className="py-2 font-medium text-right">ST%</th>
              </tr>
            </thead>
            <tbody>
              {displayed.map((r: SnapCountEntry, i) => (
                <tr key={`${r.player_id}-${r.week}-${i}`} className="border-b border-slate-800 hover:bg-slate-800/50">
                  <td className="py-1.5 pr-3 text-white font-medium">{r.player_name ?? r.player_id}</td>
                  <td className="py-1.5 pr-3 text-slate-300">{r.position ?? '—'}</td>
                  {week === undefined && <td className="py-1.5 pr-3 text-slate-400">{r.week}</td>}
                  <td className="py-1.5 pr-3 text-slate-300 text-right">{r.offense_snaps}</td>
                  <td className="py-1.5 pr-3 text-slate-400 text-right">{pct(r.offense_pct)}</td>
                  <td className="py-1.5 pr-3 text-slate-300 text-right">{r.defense_snaps}</td>
                  <td className="py-1.5 pr-3 text-slate-400 text-right">{pct(r.defense_pct)}</td>
                  <td className="py-1.5 pr-3 text-slate-300 text-right">{r.st_snaps}</td>
                  <td className="py-1.5 text-slate-400 text-right">{pct(r.st_pct)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ─── Schedule tab ─────────────────────────────────────────────────────────────

function ScheduleTab({ abbr, season }: { abbr: string; season: number }) {
  const { data, isLoading, error, refetch } = useSchedules(season, undefined, abbr)

  if (isLoading) return <SkeletonCard rows={6} />
  if (error) return <ErrorCard message="Failed to load schedule" onRetry={() => refetch()} />
  if (!data?.data.length) return <p className="text-slate-400 text-sm">No games found.</p>

  const games = data.data

  function gameResult(g: Game): string {
    if (g.home_score == null || g.away_score == null) return '—'
    const isHome = g.home_team === abbr
    const myScore = isHome ? g.home_score : g.away_score
    const oppScore = isHome ? g.away_score : g.home_score
    if (myScore > oppScore) return 'W'
    if (myScore < oppScore) return 'L'
    return 'T'
  }

  function oppTeam(g: Game): string {
    return g.home_team === abbr ? `vs ${g.away_team}` : `@ ${g.home_team}`
  }

  function scoreStr(g: Game): string {
    if (g.home_score == null || g.away_score == null) return '—'
    const isHome = g.home_team === abbr
    const myScore = isHome ? g.home_score : g.away_score
    const oppScore = isHome ? g.away_score : g.home_score
    return `${myScore}–${oppScore}`
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700 text-slate-400 text-left">
            <th className="py-2 pr-4 font-medium">Week</th>
            <th className="py-2 pr-4 font-medium">Opponent</th>
            <th className="py-2 pr-4 font-medium text-right">Score</th>
            <th className="py-2 pr-4 font-medium text-right">Result</th>
            <th className="py-2 font-medium">Stadium</th>
          </tr>
        </thead>
        <tbody>
          {games.map((g) => {
            const result = gameResult(g)
            return (
              <tr key={g.game_id ?? `${g.week}`} className="border-b border-slate-800 hover:bg-slate-800/50">
                <td className="py-2 pr-4 text-slate-400">{g.week}</td>
                <td className="py-2 pr-4 text-white font-medium">{oppTeam(g)}</td>
                <td className="py-2 pr-4 text-slate-300 text-right">{scoreStr(g)}</td>
                <td className={`py-2 pr-4 text-right font-semibold ${result === 'W' ? 'text-green-400' : result === 'L' ? 'text-red-400' : 'text-slate-400'}`}>
                  {result}
                </td>
                <td className="py-2 text-slate-400 truncate max-w-[180px]">{g.stadium ?? '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ─── Main page ────────────────────────────────────────────────────────────────

const TABS: { id: Tab; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'depth-chart', label: 'Depth Chart' },
  { id: 'snap-counts', label: 'Snap Counts' },
  { id: 'schedule', label: 'Schedule' },
]

export default function TeamDetail() {
  const { abbr } = useParams<{ abbr: string }>()
  const [activeTab, setActiveTab] = useState<Tab>('overview')
  const [season, setSeason] = useState(getDefaultSeason())

  if (!abbr) return null

  const upperAbbr = abbr.toUpperCase()

  return (
    <div className="p-6">
      {/* Back link */}
      <Link
        to="/teams"
        className="inline-flex items-center gap-1 text-sm text-slate-400 hover:text-white mb-4"
      >
        <ChevronLeft size={16} />
        All Teams
      </Link>

      {/* Season selector */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white">{upperAbbr}</h1>
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-400">Season</label>
          <select
            value={season}
            onChange={(e) => setSeason(Number(e.target.value))}
            className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          >
            {SEASONS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-6 border-b border-slate-700">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium rounded-t transition-colors ${
              activeTab === tab.id
                ? 'text-white border-b-2 border-brand-green -mb-px'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'overview' && <OverviewTab abbr={upperAbbr} season={season} />}
      {activeTab === 'depth-chart' && <DepthChartTab abbr={upperAbbr} season={season} />}
      {activeTab === 'snap-counts' && <SnapCountsTab abbr={upperAbbr} season={season} />}
      {activeTab === 'schedule' && <ScheduleTab abbr={upperAbbr} season={season} />}
    </div>
  )
}
