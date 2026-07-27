import { useNavigate } from 'react-router-dom'
import { useTeams } from '../api/teams'
import type { Team } from '../api/types'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'

const CONF_ORDER = ['AFC', 'NFC']
const DIV_ORDER = ['East', 'North', 'South', 'West']

function groupTeams(teams: Team[]) {
  const map: Record<string, Record<string, Team[]>> = {}
  for (const t of teams) {
    const conf = t.team_conf ?? 'Other'
    const div = (t.team_division ?? '').replace(conf, '').trim() || 'Other'
    if (!map[conf]) map[conf] = {}
    if (!map[conf][div]) map[conf][div] = []
    map[conf][div].push(t)
  }
  return map
}

function TeamCard({ team, onClick }: { team: Team; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="bg-slate-800 hover:bg-slate-750 border border-slate-700 hover:border-slate-500 rounded-xl p-3 flex items-center gap-3 transition-colors w-full text-left"
    >
      {team.team_logo_espn ? (
        <img
          src={team.team_logo_espn}
          alt={team.team_abbr}
          className="w-10 h-10 object-contain shrink-0"
        />
      ) : (
        <div
          className="w-10 h-10 rounded-full shrink-0"
          style={{ backgroundColor: team.team_color ?? '#334155' }}
        />
      )}
      <div className="min-w-0">
        <p className="font-semibold text-white text-sm truncate">{team.team_name}</p>
        <p className="text-slate-400 text-xs">{team.team_abbr}</p>
      </div>
    </button>
  )
}

export default function Teams() {
  const navigate = useNavigate()
  const { data, isLoading, error, refetch } = useTeams()

  if (isLoading) {
    return (
      <div className="p-6">
        <PageHeader title="Teams" />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {Array.from({ length: 8 }).map((_, i) => <SkeletonCard key={i} rows={2} />)}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6">
        <PageHeader title="Teams" />
        <ErrorCard message="Failed to load teams" onRetry={() => refetch()} />
      </div>
    )
  }

  const grouped = groupTeams(data?.data ?? [])

  return (
    <div className="p-6">
      <PageHeader title="Teams" subtitle={`${data?.total_teams ?? 0} NFL teams`} />

      {CONF_ORDER.map((conf) => (
        <div key={conf} className="mb-8">
          <h2 className="text-lg font-bold text-slate-200 mb-4 border-b border-slate-700 pb-2">
            {conf}
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            {DIV_ORDER.map((div) => {
              const teams = grouped[conf]?.[div]
              if (!teams?.length) return null
              return (
                <div key={div}>
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-2">
                    {conf} {div}
                  </h3>
                  <div className="space-y-2">
                    {teams.map((t) => (
                      <TeamCard key={t.team_abbr} team={t} onClick={() => navigate(`/teams/${t.team_abbr}`)} />
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      ))}

    </div>
  )
}
