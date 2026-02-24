import { useState } from 'react'
import { useTeams, useTeam } from '../api/teams'
import { useTeamStaff } from '../api/coaches'
import type { Team } from '../api/types'
import PageHeader from '../components/ui/PageHeader'
import SkeletonCard from '../components/ui/SkeletonCard'
import ErrorCard from '../components/ui/ErrorCard'
import { X } from 'lucide-react'

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

function TeamDetailPanel({ abbr, onClose }: { abbr: string; onClose: () => void }) {
  const { data, isLoading, error, refetch } = useTeam(abbr)
  const { data: staffData } = useTeamStaff(abbr)

  // Pick the entry for this team (filter is applied server-side but guard client-side too)
  const staff = staffData?.configured
    ? staffData.data.find((s) => s.team_abbr.toUpperCase() === abbr.toUpperCase())
    : undefined

  const hasStaff =
    staff &&
    (staff.head_coach ||
      staff.offensive_coordinator ||
      staff.defensive_coordinator ||
      staff.special_teams_coordinator)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60">
      <div className="bg-slate-800 rounded-2xl w-full max-w-md border border-slate-700 p-6 relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-white"
        >
          <X size={20} />
        </button>

        {isLoading && <SkeletonCard rows={6} />}
        {error && (
          <ErrorCard message="Failed to load team details" onRetry={() => refetch()} />
        )}
        {data?.data && (
          <>
            <div className="flex items-center gap-4 mb-4">
              {data.data.team_logo_espn && (
                <img
                  src={data.data.team_logo_espn}
                  alt={data.data.team_abbr}
                  className="w-16 h-16 object-contain"
                />
              )}
              <div>
                <h2 className="text-xl font-bold text-white">{data.data.team_name}</h2>
                <p className="text-slate-400 text-sm">{data.data.team_conf} · {data.data.team_division}</p>
              </div>
            </div>

            <div className="space-y-2 text-sm">
              <Row label="Abbreviation" value={data.data.team_abbr} />
              <Row label="Nickname" value={data.data.team_nick ?? '—'} />
              <Row label="Conference" value={data.data.team_conf ?? '—'} />
              <Row label="Division" value={data.data.team_division ?? '—'} />
              {data.data.team_color && (
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Primary Color</span>
                  <span
                    className="w-6 h-6 rounded border border-slate-600"
                    style={{ backgroundColor: data.data.team_color }}
                  />
                </div>
              )}
            </div>

            {hasStaff && (
              <div className="mt-4 pt-4 border-t border-slate-700">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
                  Coaching Staff
                </p>
                <div className="space-y-2 text-sm">
                  {staff!.head_coach && (
                    <Row label="Head Coach" value={staff!.head_coach} />
                  )}
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
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-400">{label}</span>
      <span className="text-white">{value}</span>
    </div>
  )
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
  const { data, isLoading, error, refetch } = useTeams()
  const [selected, setSelected] = useState<string | null>(null)

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
                      <TeamCard key={t.team_abbr} team={t} onClick={() => setSelected(t.team_abbr)} />
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      ))}

      {selected && (
        <TeamDetailPanel abbr={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  )
}
