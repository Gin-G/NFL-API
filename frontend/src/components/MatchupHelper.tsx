import { useEffect, useMemo, useState } from 'react'
import { ArrowLeftRight } from 'lucide-react'
import { useTeams } from '../api/teams'
import type { TeamRating } from '../api/types'
import GradeScore from './ui/GradeScore'
import GradeMeter from './ui/GradeMeter'
import { gradeTone, fmtEdge } from '../utils/gradeScale'

/** One side of a matchup: an offense graded against the opposing defense. */
function MatchupSide({
  offense,
  defense,
  offenseRank,
  defenseRank,
  total,
  logos,
}: {
  offense: TeamRating
  defense: TeamRating
  offenseRank: number
  defenseRank: number
  total: number
  logos: Record<string, string | null>
}) {
  const edge = offense.offense_grade - defense.defense_grade
  const tone = gradeTone(50 + edge / 2) // color the edge on the same scale, softened

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
      <div className="flex items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2 min-w-0">
          {logos[offense.team] && (
            <img src={logos[offense.team]!} alt="" className="w-7 h-7 object-contain shrink-0" />
          )}
          <div className="min-w-0">
            <p className="text-white text-sm font-semibold">{offense.team} offense</p>
            <p className="text-slate-500 text-[11px]">#{offenseRank} of {total} offense</p>
          </div>
        </div>
        <span className="text-slate-500 text-xs shrink-0">vs</span>
        <div className="flex items-center gap-2 min-w-0">
          <div className="min-w-0 text-right">
            <p className="text-white text-sm font-semibold">{defense.team} defense</p>
            <p className="text-slate-500 text-[11px]">#{defenseRank} of {total} defense</p>
          </div>
          {logos[defense.team] && (
            <img src={logos[defense.team]!} alt="" className="w-7 h-7 object-contain shrink-0" />
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="flex flex-col items-center gap-2">
          <GradeScore value={offense.offense_grade} size="lg" />
          <span className="text-[10px] text-slate-400 uppercase tracking-wide">Offense grade</span>
          <GradeMeter value={offense.offense_grade} showValue={false} />
        </div>
        <div className="flex flex-col items-center gap-2">
          <GradeScore
            value={defense.defense_grade}
            size="lg"
            title="Higher = stingier defense. Defense grades are noisier than offense grades."
          />
          <span className="text-[10px] text-slate-400 uppercase tracking-wide">Defense grade</span>
          <GradeMeter value={defense.defense_grade} showValue={false} lowConfidence />
        </div>
      </div>

      <div className="border-t border-slate-700 pt-3 flex items-baseline justify-between">
        <span className="text-xs text-slate-400">
          {offense.team} offense edge
        </span>
        <span className="text-sm font-bold tabular-nums" style={{ color: tone.text }}>
          {fmtEdge(edge)}
        </span>
      </div>
      <p className="text-[11px] text-slate-500 mt-1">
        Offense grade minus the opposing defense grade — a rough directional read, not a
        projection.
      </p>
    </div>
  )
}

/**
 * Pick two teams and compare each side's offense against the other's defense.
 */
export default function MatchupHelper({ ratings }: { ratings: TeamRating[] }) {
  const { data: teamsData } = useTeams()

  const byTeam = useMemo(() => {
    const map: Record<string, TeamRating> = {}
    for (const r of ratings) map[r.team] = r
    return map
  }, [ratings])

  const teamList = useMemo(() => ratings.map((r) => r.team).sort(), [ratings])

  // League ranks (1 = best) for each grade, so a grade has context.
  const ranks = useMemo(() => {
    const off = [...ratings].sort((a, b) => b.offense_grade - a.offense_grade)
    const def = [...ratings].sort((a, b) => b.defense_grade - a.defense_grade)
    const map: Record<string, { off: number; def: number }> = {}
    off.forEach((r, i) => {
      map[r.team] = { off: i + 1, def: 0 }
    })
    def.forEach((r, i) => {
      if (map[r.team]) map[r.team].def = i + 1
    })
    return map
  }, [ratings])

  const [teamA, setTeamA] = useState('')
  const [teamB, setTeamB] = useState('')

  // Seed (and re-seed, when the season's team set changes) with the best
  // offense against the best defense — a matchup worth looking at by default.
  useEffect(() => {
    if (!ratings.length) return
    const bestOffense = [...ratings].sort((a, b) => b.offense_grade - a.offense_grade)[0].team
    const bestDefense =
      [...ratings].sort((a, b) => b.defense_grade - a.defense_grade).find((r) => r.team !== bestOffense)
        ?.team ?? bestOffense
    setTeamA((cur) => (teamList.includes(cur) ? cur : bestOffense))
    setTeamB((cur) => (teamList.includes(cur) ? cur : bestDefense))
  }, [ratings, teamList])

  const logos = useMemo(() => {
    const map: Record<string, string | null> = {}
    for (const t of teamsData?.data ?? []) map[t.team_abbr] = t.team_logo_espn
    return map
  }, [teamsData])

  const a = byTeam[teamA]
  const b = byTeam[teamB]

  if (!ratings.length) return null

  return (
    <section className="mb-8">
      <h2 className="text-sm font-semibold text-slate-300 mb-3">Matchup helper</h2>

      <div className="flex flex-wrap items-center gap-3 mb-4">
        <select
          value={teamA}
          onChange={(e) => setTeamA(e.target.value)}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          aria-label="Team A"
        >
          {teamList.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <button
          onClick={() => {
            setTeamA(teamB)
            setTeamB(teamA)
          }}
          className="text-slate-400 hover:text-white transition-colors p-1.5 rounded-lg hover:bg-slate-800"
          title="Swap teams"
        >
          <ArrowLeftRight size={16} />
        </button>
        <select
          value={teamB}
          onChange={(e) => setTeamB(e.target.value)}
          className="bg-slate-700 border border-slate-600 text-white rounded-lg px-3 py-1.5 text-sm"
          aria-label="Team B"
        >
          {teamList.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      {a && b ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <MatchupSide
            offense={a}
            defense={b}
            offenseRank={ranks[a.team]?.off ?? 0}
            defenseRank={ranks[b.team]?.def ?? 0}
            total={ratings.length}
            logos={logos}
          />
          <MatchupSide
            offense={b}
            defense={a}
            offenseRank={ranks[b.team]?.off ?? 0}
            defenseRank={ranks[a.team]?.def ?? 0}
            total={ratings.length}
            logos={logos}
          />
        </div>
      ) : (
        <p className="text-slate-400 text-sm">Pick two teams to compare.</p>
      )}
    </section>
  )
}
