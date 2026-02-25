import { useState } from 'react'
import { useCoachBreakdown } from '../api/coaches'
import type { CoachBreakdownEntry, RunLocationEntry, PassDirectionEntry } from '../api/types'
import SkeletonCard from './ui/SkeletonCard'

// ─── Formatters ───────────────────────────────────────────────────────────────

function pct(v: number | null, decimals = 0): string {
  if (v == null) return '—'
  return (v * 100).toFixed(decimals) + '%'
}

function epaFmt(v: number | null): string {
  if (v == null) return '—'
  return (v >= 0 ? '+' : '') + v.toFixed(2)
}

function numFmt(v: number | null, d = 1): string {
  if (v == null) return '—'
  return v.toFixed(d)
}

function epaColor(v: number | null, invert = false): string {
  if (v == null) return 'text-slate-400'
  const positive = invert ? v <= 0 : v >= 0
  return positive ? 'text-emerald-400' : 'text-red-400'
}

// ─── Reusable mini-components ─────────────────────────────────────────────────

function MiniBar({
  value,
  color = 'bg-emerald-500',
}: {
  value: number
  color?: string
}) {
  return (
    <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden flex-1">
      <div
        className={`h-full ${color} rounded-full`}
        style={{ width: `${Math.min(100, value * 100)}%` }}
      />
    </div>
  )
}

function BarRow({
  label,
  value,
  color,
  suffix,
}: {
  label: string
  value: number | null
  color?: string
  suffix?: string
}) {
  if (value == null) return null
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-slate-400 w-32 shrink-0 truncate">{label}</span>
      <MiniBar value={value} color={color} />
      <span className="text-white w-10 text-right shrink-0 tabular-nums">
        {pct(value)}{suffix}
      </span>
    </div>
  )
}

function StatPill({
  label,
  value,
  sub,
  valueClass,
}: {
  label: string
  value: string
  sub?: string
  valueClass?: string
}) {
  return (
    <div className="bg-slate-800 rounded-lg px-3 py-2 text-center border border-slate-700/50">
      <p className="text-xs text-slate-500 mb-0.5 leading-tight">{label}</p>
      <p className={`font-bold text-sm tabular-nums ${valueClass ?? 'text-white'}`}>{value}</p>
      {sub && <p className="text-xs text-slate-600 mt-0.5">{sub}</p>}
    </div>
  )
}

function SectionLabel({ title }: { title: string }) {
  return (
    <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-2 mt-3">
      {title}
    </p>
  )
}

// ─── Insights (strengths / weaknesses / tendencies) ───────────────────────────

function Insights({
  strengths,
  weaknesses,
  tendencies,
}: {
  strengths: string[]
  weaknesses: string[]
  tendencies: string[]
}) {
  if (!strengths.length && !weaknesses.length && !tendencies.length) return null
  return (
    <div className="space-y-2 mb-4">
      {strengths.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {strengths.map((s, i) => (
            <span
              key={i}
              className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/25 text-xs px-2 py-0.5 rounded-full"
            >
              ✓ {s}
            </span>
          ))}
        </div>
      )}
      {weaknesses.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {weaknesses.map((w, i) => (
            <span
              key={i}
              className="bg-red-500/10 text-red-400 border border-red-500/25 text-xs px-2 py-0.5 rounded-full"
            >
              ✗ {w}
            </span>
          ))}
        </div>
      )}
      {tendencies.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {tendencies.map((t, i) => (
            <span
              key={i}
              className="bg-blue-500/10 text-blue-400 border border-blue-500/25 text-xs px-2 py-0.5 rounded-full"
            >
              → {t}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Entry view (one season) ─────────────────────────────────────────────────

function BreakdownView({ entry }: { entry: CoachBreakdownEntry }) {
  const { offense, defense } = entry
  const { formation, personnel, run_scheme, passing } = offense
  const { scheme } = defense

  const topPersonnel = Object.entries(personnel).slice(0, 5)
  const topFormations = Object.entries(formation.formation_breakdown).slice(0, 6)
  const topDefPersonnel = Object.entries(scheme.personnel_breakdown).slice(0, 5)

  // Compute totals for relative bar widths
  const runTotal = Object.values(run_scheme.by_location as Record<string, RunLocationEntry>).reduce(
    (a, b) => a + b.plays,
    0,
  )
  const passTotal = Object.values(passing.by_direction as Record<string, PassDirectionEntry>).reduce(
    (a, b) => a + b.plays,
    0,
  )

  return (
    <div className="space-y-3">
      <Insights
        strengths={entry.strengths}
        weaknesses={entry.weaknesses}
        tendencies={entry.tendencies}
      />

      {/* ── OFFENSE ─────────────────────────────────────────────────────────── */}
      <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/40">
        <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-widest mb-3">
          Offense · {offense.total_plays} plays
        </h3>

        {/* Key metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
          <StatPill label="Pass Rate" value={pct(offense.pass_rate)} />
          <StatPill
            label="EPA / Play"
            value={epaFmt(offense.avg_epa_per_play)}
            valueClass={epaColor(offense.avg_epa_per_play)}
          />
          <StatPill label="3rd Down Conv" value={pct(offense.third_down_conversion_rate)} />
          <StatPill
            label="4th Down Conv"
            value={offense.fourth_down_attempts > 0 ? pct(offense.fourth_down_conversion_rate) : '—'}
            sub={offense.fourth_down_attempts > 0 ? `${offense.fourth_down_attempts} att` : undefined}
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1">
          {/* Formation */}
          <div>
            <SectionLabel title="Formation" />
            <div className="space-y-1.5">
              <BarRow label="Shotgun" value={formation.shotgun_rate} color="bg-sky-500" />
              <BarRow label="Play Action" value={formation.play_action_rate} color="bg-violet-500" />
              <BarRow label="No Huddle" value={formation.no_huddle_rate} color="bg-amber-500" />
            </div>
            {topFormations.length > 0 && (
              <>
                <p className="text-xs text-slate-500 mt-2 mb-1.5">By formation</p>
                <div className="space-y-1.5">
                  {topFormations.map(([name, f]) => (
                    <div key={name} className="flex items-center gap-2 text-xs">
                      <span className="text-slate-400 w-28 shrink-0 truncate">{name}</span>
                      <MiniBar value={f.pct_of_plays} color="bg-sky-500/50" />
                      <span className="text-slate-300 w-8 text-right shrink-0 tabular-nums">
                        {pct(f.pct_of_plays)}
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Personnel */}
          <div>
            <SectionLabel title="Personnel" />
            {topPersonnel.length > 0 ? (
              <div className="space-y-2">
                {topPersonnel.map(([name, p]) => (
                  <div key={name}>
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-slate-300 w-36 shrink-0 truncate">{name}</span>
                      <MiniBar value={p.pct_of_plays} color="bg-emerald-500/60" />
                      <span className="text-white w-8 text-right shrink-0 tabular-nums">
                        {pct(p.pct_of_plays)}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500 mt-0.5">
                      {pct(p.pass_rate)} pass · {epaFmt(p.avg_epa)} EPA
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-slate-500">No personnel data available</p>
            )}
          </div>
        </div>

        {/* Run scheme + Passing */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 mt-1">
          {/* Run scheme */}
          <div>
            <SectionLabel title="Run Scheme" />
            <div className="space-y-1.5">
              <BarRow label="Inside runs" value={run_scheme.inside_rate} color="bg-orange-500" />
              <BarRow label="Outside runs" value={run_scheme.outside_rate} color="bg-orange-400" />
            </div>
            {runTotal > 0 && Object.keys(run_scheme.by_location).length > 0 && (
              <>
                <p className="text-xs text-slate-500 mt-2 mb-1.5">By direction</p>
                <div className="space-y-1.5">
                  {Object.entries(run_scheme.by_location as Record<string, RunLocationEntry>).map(
                    ([loc, data]) => {
                      const locPct = runTotal > 0 ? data.plays / runTotal : 0
                      return (
                        <div key={loc} className="flex items-center gap-2 text-xs">
                          <span className="text-slate-400 w-16 shrink-0 capitalize">{loc}</span>
                          <MiniBar value={locPct} color="bg-orange-500/50" />
                          <span className="text-slate-300 w-8 text-right shrink-0 tabular-nums">
                            {(locPct * 100).toFixed(0)}%
                          </span>
                          {data.avg_yards != null && (
                            <span className="text-slate-500 w-14 text-right shrink-0 tabular-nums">
                              {numFmt(data.avg_yards)} yds
                            </span>
                          )}
                        </div>
                      )
                    },
                  )}
                </div>
              </>
            )}
          </div>

          {/* Passing detail */}
          <div>
            <SectionLabel title="Passing" />
            <div className="grid grid-cols-2 gap-1.5 mb-2">
              <StatPill label="Avg Air Yds" value={numFmt(passing.avg_air_yards)} />
              <StatPill label="Avg YAC" value={numFmt(passing.avg_yac)} />
              <StatPill
                label="Dropback EPA"
                value={epaFmt(passing.dropback_epa)}
                valueClass={epaColor(passing.dropback_epa)}
              />
              <StatPill label="Scramble %" value={pct(passing.scramble_rate)} />
            </div>
            <div className="space-y-1.5">
              <BarRow label="Deep (>15 yds)" value={passing.deep_pass_rate} color="bg-purple-500" />
              <BarRow label="Intermediate" value={passing.intermediate_rate} color="bg-purple-400" />
              <BarRow label="Screen / Short" value={passing.screen_rate} color="bg-purple-300" />
            </div>
            {passTotal > 0 && Object.keys(passing.by_direction).length > 0 && (
              <>
                <p className="text-xs text-slate-500 mt-2 mb-1.5">By direction</p>
                <div className="space-y-1.5">
                  {Object.entries(passing.by_direction as Record<string, PassDirectionEntry>).map(
                    ([dir, data]) => {
                      const dirPct = passTotal > 0 ? data.plays / passTotal : 0
                      return (
                        <div key={dir} className="flex items-center gap-2 text-xs">
                          <span className="text-slate-400 w-16 shrink-0 capitalize">{dir}</span>
                          <MiniBar value={dirPct} color="bg-purple-500/50" />
                          <span className="text-slate-300 w-8 text-right shrink-0 tabular-nums">
                            {(dirPct * 100).toFixed(0)}%
                          </span>
                          {data.avg_epa != null && (
                            <span
                              className={`w-14 text-right shrink-0 tabular-nums text-xs ${epaColor(data.avg_epa)}`}
                            >
                              {epaFmt(data.avg_epa)}
                            </span>
                          )}
                        </div>
                      )
                    },
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* ── DEFENSE ─────────────────────────────────────────────────────────── */}
      <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/40">
        <h3 className="text-xs font-bold text-sky-400 uppercase tracking-widest mb-3">
          Defense · {defense.total_plays} plays
        </h3>

        {/* Key metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
          <StatPill
            label="EPA Allowed"
            value={epaFmt(defense.avg_epa_allowed_per_play)}
            valueClass={epaColor(defense.avg_epa_allowed_per_play, true)}
          />
          <StatPill label="3rd Down Stop" value={pct(defense.third_down_stop_rate)} />
          <StatPill label="RZ TD Allowed" value={pct(defense.red_zone_td_rate_allowed)} />
          <StatPill label="Sack Rate" value={pct(defense.sack_rate)} />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1">
          {/* Pressure */}
          <div>
            <SectionLabel title="Pressure & Box" />
            <div className="space-y-1.5">
              <BarRow label="Blitz rate" value={scheme.blitz_rate} color="bg-red-500" />
              <BarRow label="QB hit rate" value={scheme.qb_hit_rate} color="bg-red-400" />
            </div>
            {scheme.avg_defenders_in_box != null && (
              <p className="text-xs text-slate-400 mt-2">
                Avg defenders in box:{' '}
                <span className="text-white font-medium">
                  {numFmt(scheme.avg_defenders_in_box)}
                </span>
              </p>
            )}
            {(scheme.blitz_epa_allowed != null || scheme.non_blitz_epa_allowed != null) && (
              <div className="mt-2 space-y-1">
                {scheme.blitz_epa_allowed != null && (
                  <p className="text-xs text-slate-400">
                    Blitz EPA allowed:{' '}
                    <span className={epaColor(scheme.blitz_epa_allowed, true)}>
                      {epaFmt(scheme.blitz_epa_allowed)}
                    </span>
                  </p>
                )}
                {scheme.non_blitz_epa_allowed != null && (
                  <p className="text-xs text-slate-400">
                    Non-blitz EPA:{' '}
                    <span className={epaColor(scheme.non_blitz_epa_allowed, true)}>
                      {epaFmt(scheme.non_blitz_epa_allowed)}
                    </span>
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Defense personnel */}
          <div>
            <SectionLabel title="Personnel" />
            {topDefPersonnel.length > 0 ? (
              <div className="space-y-2">
                {topDefPersonnel.map(([name, d]) => (
                  <div key={name}>
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-slate-300 w-28 shrink-0 truncate">{name}</span>
                      <MiniBar value={d.pct_of_plays} color="bg-sky-500/60" />
                      <span className="text-white w-8 text-right shrink-0 tabular-nums">
                        {pct(d.pct_of_plays)}
                      </span>
                    </div>
                    {d.avg_epa_allowed != null && (
                      <p className="text-xs text-slate-500 mt-0.5">
                        EPA allowed:{' '}
                        <span className={epaColor(d.avg_epa_allowed, true)}>
                          {epaFmt(d.avg_epa_allowed)}
                        </span>
                      </p>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-slate-500">No personnel data available</p>
            )}
          </div>
        </div>
      </div>

      {/* ── SAMPLE PLAYS ────────────────────────────────────────────────────── */}
      {offense.fourth_down_sample.length > 0 && (
        <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/40">
          <h3 className="text-xs font-bold text-amber-400 uppercase tracking-widest mb-3">
            4th Down Sample
          </h3>
          <div className="space-y-2">
            {offense.fourth_down_sample.map((play, i) => (
              <div key={i} className="bg-slate-800/60 rounded-lg p-2.5 text-xs">
                <p className="text-slate-300 line-clamp-2">{play.desc ?? '—'}</p>
                <div className="flex gap-3 mt-1 text-slate-500">
                  {play.week != null && <span>Wk {play.week}</span>}
                  {play.yards_gained != null && (
                    <span>
                      {play.yards_gained > 0 ? '+' : ''}
                      {play.yards_gained} yds
                    </span>
                  )}
                  {play.epa != null && (
                    <span className={epaColor(play.epa)}>{epaFmt(play.epa)} EPA</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Main exported component ─────────────────────────────────────────────────

export default function CoachBreakdown({ coachName }: { coachName: string }) {
  const { data, isLoading } = useCoachBreakdown(coachName)
  const [selectedIdx, setSelectedIdx] = useState(0)

  if (isLoading) return <SkeletonCard rows={8} />

  if (!data || data.status === 'no_data' || !data.data || data.data.length === 0) {
    return (
      <div className="text-center py-12 text-slate-500">
        <p className="text-sm">Play-by-play breakdown not yet available.</p>
        <p className="text-xs mt-1 text-slate-600">
          Requires the PBP data loader to run first.
        </p>
      </div>
    )
  }

  const entry = data.data[Math.min(selectedIdx, data.data.length - 1)]

  return (
    <div>
      {/* Season / team selector */}
      {data.data.length > 1 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {data.data.map((d, i) => (
            <button
              key={`${d.season}-${d.team}`}
              onClick={() => setSelectedIdx(i)}
              className={`px-3 py-1 text-xs rounded-full border transition-colors ${
                i === selectedIdx
                  ? 'border-brand-green text-brand-green bg-brand-green/10'
                  : 'border-slate-600 text-slate-400 hover:border-slate-400'
              }`}
            >
              {d.season} · {d.team}
            </button>
          ))}
        </div>
      )}

      <BreakdownView entry={entry} />
    </div>
  )
}
