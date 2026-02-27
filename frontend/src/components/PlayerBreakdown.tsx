import { usePlayerBreakdown } from '../api/players'
import type {
  PlayerBreakdownEntry,
  QBPassingMetrics,
  RushingMetrics,
  ReceivingMetrics,
  DefensiveMetrics,
  DirectionEntry,
} from '../api/types'
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

function MiniBar({ value, color = 'bg-emerald-500' }: { value: number; color?: string }) {
  return (
    <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden flex-1">
      <div className={`h-full ${color} rounded-full`} style={{ width: `${Math.min(100, value * 100)}%` }} />
    </div>
  )
}

function BarRow({ label, value, color, max }: { label: string; value: number | null; color?: string; max?: number }) {
  if (value == null) return null
  const barVal = max ? Math.min(1, value / max) : Math.min(1, value)
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-slate-400 w-32 shrink-0 truncate">{label}</span>
      <MiniBar value={barVal} color={color} />
      <span className="text-white w-10 text-right shrink-0 tabular-nums">
        {max ? value.toFixed(1) : pct(value)}
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

function DirectionBreakdown({
  data,
  label,
  color,
}: {
  data: { left: DirectionEntry; middle: DirectionEntry; right: DirectionEntry }
  label: string
  color: string
}) {
  const dirs = ['left', 'middle', 'right'] as const
  const total = dirs.reduce((sum, d) => {
    const e = data[d]
    return sum + (e.attempts ?? e.carries ?? e.targets ?? 0)
  }, 0)
  if (total === 0) return null

  return (
    <>
      <p className="text-xs text-slate-500 mt-2 mb-1.5">By {label} direction</p>
      <div className="space-y-1.5">
        {dirs.map((dir) => {
          const e = data[dir]
          const count = e.attempts ?? e.carries ?? e.targets ?? 0
          const frac = total > 0 ? count / total : 0
          return (
            <div key={dir} className="flex items-center gap-2 text-xs">
              <span className="text-slate-400 w-16 shrink-0 capitalize">{dir}</span>
              <MiniBar value={frac} color={color} />
              <span className="text-slate-300 w-8 text-right shrink-0 tabular-nums">
                {(frac * 100).toFixed(0)}%
              </span>
              {e.epa != null && (
                <span className={`w-14 text-right shrink-0 tabular-nums ${epaColor(e.epa)}`}>
                  {epaFmt(e.epa)} EPA
                </span>
              )}
            </div>
          )
        })}
      </div>
    </>
  )
}

// ─── Position-specific sections ───────────────────────────────────────────────

function PassingSection({ m }: { m: QBPassingMetrics }) {
  return (
    <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/40">
      <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-widest mb-3">
        Passing · {m.dropbacks} dropbacks
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
        <StatPill label="Comp %" value={pct(m.comp_pct, 1)} />
        <StatPill
          label="EPA / Dropback"
          value={epaFmt(m.epa_per_dropback)}
          valueClass={epaColor(m.epa_per_dropback)}
        />
        <StatPill label="Pass Yards" value={String(m.pass_yards)} sub={`${m.pass_tds} TD · ${m.interceptions} INT`} />
        <StatPill label="Avg Air Yds" value={numFmt(m.avg_air_yards)} sub={`Avg YAC: ${numFmt(m.avg_yac)}`} />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6">
        <div>
          <SectionLabel title="Tendencies" />
          <div className="space-y-1.5">
            <BarRow label="Deep (>15 yd)" value={m.deep_rate} color="bg-violet-500" />
            <BarRow label="Screen / Short" value={m.screen_rate} color="bg-violet-300" />
            {m.play_action_rate != null && (
              <BarRow label="Play Action" value={m.play_action_rate} color="bg-sky-500" />
            )}
            <BarRow label="Sack Rate" value={m.sack_rate} color="bg-red-400" />
            <BarRow label="Scramble Rate" value={m.scramble_rate} color="bg-amber-400" />
            <BarRow label="3rd Down Conv" value={m.third_down_conv_rate} color="bg-emerald-500" />
          </div>
        </div>
        <div>
          <SectionLabel title="Red Zone" />
          <div className="grid grid-cols-2 gap-1.5 mb-2">
            <StatPill label="RZ Dropbacks" value={String(m.rz_dropbacks)} />
            <StatPill
              label="RZ EPA"
              value={epaFmt(m.rz_epa)}
              valueClass={epaColor(m.rz_epa)}
            />
          </div>
          <DirectionBreakdown data={m.pass_direction} label="pass" color="bg-violet-500/50" />
        </div>
      </div>
    </div>
  )
}

function RushingSection({ m, compact = false }: { m: RushingMetrics; compact?: boolean }) {
  return (
    <div className={`bg-slate-900/50 rounded-xl p-4 border border-slate-700/40 ${compact ? '' : ''}`}>
      <h3 className="text-xs font-bold text-orange-400 uppercase tracking-widest mb-3">
        Rushing · {m.carries} carries
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
        <StatPill label="YPC" value={numFmt(m.yards_per_carry)} sub={`${m.rush_yards} yds`} />
        <StatPill
          label="EPA / Carry"
          value={epaFmt(m.epa_per_carry)}
          valueClass={epaColor(m.epa_per_carry)}
        />
        <StatPill label="Success Rate" value={pct(m.success_rate, 1)} sub={`${m.rush_tds} TDs`} />
        <StatPill label="Explosive %" value={pct(m.explosive_rate, 1)} sub={`Neg: ${pct(m.negative_rate, 1)}`} />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6">
        <div>
          <SectionLabel title="Run Type" />
          <div className="space-y-1.5">
            <BarRow label="Inside runs" value={m.inside_rate} color="bg-orange-500" />
            <BarRow label="Outside runs" value={m.outside_rate} color="bg-orange-300" />
          </div>
          <DirectionBreakdown data={m.by_direction} label="run" color="bg-orange-500/50" />
        </div>
        <div>
          <SectionLabel title="Red Zone" />
          <div className="grid grid-cols-2 gap-1.5">
            <StatPill label="RZ Carries" value={String(m.rz_carries)} />
            <StatPill
              label="RZ EPA"
              value={epaFmt(m.rz_epa)}
              valueClass={epaColor(m.rz_epa)}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

function ReceivingSection({ m, compact = false }: { m: ReceivingMetrics; compact?: boolean }) {
  return (
    <div className={`bg-slate-900/50 rounded-xl p-4 border border-slate-700/40 ${compact ? '' : ''}`}>
      <h3 className="text-xs font-bold text-sky-400 uppercase tracking-widest mb-3">
        Receiving · {m.targets} targets
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
        <StatPill label="Catch Rate" value={pct(m.catch_rate, 1)} sub={`${m.receptions} rec`} />
        <StatPill
          label="EPA / Target"
          value={epaFmt(m.epa_per_target)}
          valueClass={epaColor(m.epa_per_target)}
        />
        <StatPill label="Rec Yards" value={String(m.rec_yards)} sub={`${m.rec_tds} TDs`} />
        <StatPill label="Avg Air Yds" value={numFmt(m.avg_air_yards)} sub={`YAC: ${numFmt(m.avg_yac)}`} />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6">
        <div>
          <SectionLabel title="Target Profile" />
          <div className="space-y-1.5">
            <BarRow label="Deep (>15 yd)" value={m.deep_target_rate} color="bg-sky-500" />
            <BarRow label="Short (≤5 yd)" value={m.short_target_rate} color="bg-sky-300" />
          </div>
          <div className="mt-2">
            <p className="text-xs text-slate-500 mb-1">
              YPT: <span className="text-white">{numFmt(m.yards_per_target)}</span>
              {' · '}
              YPR: <span className="text-white">{numFmt(m.yards_per_reception)}</span>
            </p>
          </div>
          <DirectionBreakdown data={m.by_direction} label="target" color="bg-sky-500/50" />
        </div>
        <div>
          <SectionLabel title="Red Zone" />
          <div className="grid grid-cols-2 gap-1.5">
            <StatPill label="RZ Targets" value={String(m.rz_targets)} />
            <StatPill label="RZ TDs" value={String(m.rz_tds)} valueClass="text-emerald-400" />
          </div>
        </div>
      </div>
    </div>
  )
}

function DefensiveSection({ m }: { m: DefensiveMetrics }) {
  const maxStat = Math.max(
    m.total_tackles, m.sacks * 5, m.passes_defended * 3, m.interceptions * 10, 1
  )
  return (
    <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/40">
      <h3 className="text-xs font-bold text-red-400 uppercase tracking-widest mb-3">
        Defense
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
        <StatPill label="Sacks" value={m.sacks.toFixed(1)} sub={`${m.qb_hits} QB hits`} />
        <StatPill label="Pressures" value={m.pressures.toFixed(1)} />
        <StatPill label="Tackles" value={numFmt(m.total_tackles, 1)} sub={`${m.solo_tackles} solo`} />
        <StatPill label="TFLs" value={String(m.tackles_for_loss)} />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6">
        <div>
          <SectionLabel title="Pass Defense" />
          <div className="space-y-1.5">
            <BarRow label="Sacks" value={m.sacks} max={maxStat} color="bg-red-500" />
            <BarRow label="QB Hits" value={m.qb_hits} max={maxStat} color="bg-red-400" />
            <BarRow label="Pass Breakups" value={m.passes_defended} max={maxStat} color="bg-sky-400" />
            <BarRow label="Interceptions" value={m.interceptions} max={maxStat} color="bg-emerald-400" />
          </div>
        </div>
        <div>
          <SectionLabel title="Run Defense" />
          <div className="space-y-1.5">
            <BarRow label="Solo Tackles" value={m.solo_tackles} max={maxStat} color="bg-orange-400" />
            <BarRow label="Assist Tackles" value={m.assist_tackles} max={maxStat} color="bg-orange-300" />
            <BarRow label="Tackles for Loss" value={m.tackles_for_loss} max={maxStat} color="bg-amber-400" />
            <BarRow label="Forced Fumbles" value={m.forced_fumbles} max={maxStat} color="bg-yellow-400" />
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Entry view (one season) ─────────────────────────────────────────────────

function EntryView({ entry }: { entry: PlayerBreakdownEntry }) {
  const { passing, rushing, receiving, defense } = entry
  const isQB = entry.position === 'QB' || !!passing
  const isDef = ['DE', 'DT', 'LB', 'CB', 'S', 'DB', 'FS', 'SS', 'ILB', 'OLB', 'DL'].includes(entry.position)
  const isRB = entry.position === 'RB'

  return (
    <div className="space-y-3">
      {/* QB: passing first, then optional rushing */}
      {isQB && passing && <PassingSection m={passing} />}
      {isQB && rushing && <RushingSection m={rushing} compact />}

      {/* RB: rushing first, then optional receiving */}
      {!isQB && isRB && rushing && <RushingSection m={rushing} />}
      {!isQB && isRB && receiving && <ReceivingSection m={receiving} compact />}

      {/* WR/TE: receiving */}
      {!isQB && !isRB && !isDef && receiving && <ReceivingSection m={receiving} />}

      {/* DEF positions */}
      {isDef && defense && <DefensiveSection m={defense} />}

      {/* Fallback: show whatever data exists */}
      {!isQB && !isRB && !isDef && !receiving && !defense && rushing && <RushingSection m={rushing} />}
      {!passing && !rushing && !receiving && !defense && (
        <p className="text-slate-500 text-sm text-center py-6">
          No PBP data available for this player in {entry.season}.
        </p>
      )}
    </div>
  )
}

// ─── Main exported component ─────────────────────────────────────────────────

export default function PlayerBreakdown({
  playerId,
  season,
}: {
  playerId: string
  season: number
}) {
  const { data, isLoading } = usePlayerBreakdown(playerId, season)

  if (isLoading) return <SkeletonCard rows={8} />

  if (!data) {
    return (
      <div className="text-center py-12 text-slate-500">
        <p className="text-sm">Unable to load breakdown data.</p>
      </div>
    )
  }

  if (data.status === 'computing') {
    return (
      <div className="text-center py-12 text-slate-500">
        <p className="text-sm">Analytics are being computed…</p>
        {data.pct_complete != null && (
          <p className="text-xs mt-1">{data.pct_complete}% complete</p>
        )}
      </div>
    )
  }

  if (data.status === 'no_data' || !data.data || data.data.length === 0) {
    return (
      <div className="text-center py-12 text-slate-500">
        <p className="text-sm">Play-by-play breakdown not yet available.</p>
        <p className="text-xs mt-1 text-slate-600">
          Run the player analytics job to generate data.
        </p>
      </div>
    )
  }

  const entry = data.data[0]

  return <EntryView entry={entry} />
}
