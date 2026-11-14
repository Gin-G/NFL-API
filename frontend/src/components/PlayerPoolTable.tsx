import { useMemo, useState } from 'react'
import { Lock, Ban, RotateCcw, Search } from 'lucide-react'
import { usePlayerPool, useSetPoolRules, useClearPoolRules } from '../api/optimizer'
import type { PoolPlayer, PoolRuleWrite } from '../api/types'

const EXPERT_TAG: Record<string, string> = {
  cash: 'Cash',
  gpp: 'GPP',
  punt: 'Punt',
  favorite_cash: '★ Cash',
  favorite_gpp: '★ GPP',
}

/** The uploaded slate as a table you can act on.
 *
 *  The build settings apply to everyone at once: the exposure cap holds the
 *  whole slate to one number and can neither require a player nor name one. So
 *  knowing you want a particular back left nothing to type. Here each row
 *  carries its own instruction — in every lineup, in none, or in a share of
 *  them — stored against the slate, so every later build obeys it without
 *  being told again.
 */

const POSITIONS = ['QB', 'RB', 'WR', 'TE', 'D']

/** Shares worth one click. 100 is the lock, 0 is the fade. */
const QUICK_SHARES = [25, 50, 75]

function money(value: number | null) {
  return value === null ? '—' : `$${value.toLocaleString()}`
}

function num(value: number | null, digits = 1) {
  return value === null ? '—' : value.toFixed(digits)
}

/** The rule as a short phrase. "40%" alone reads as a cap, which it is not. */
function ruleLabel(player: PoolPlayer) {
  const rule = player.rule
  if (!rule) return null
  if (rule.kind === 'lock') return 'Every lineup'
  if (rule.kind === 'exclude') return 'Excluded'
  if (rule.kind === 'target') return `${rule.min_exposure}% of lineups`
  if (rule.kind === 'band') return `${rule.min_exposure}–${rule.max_exposure}%`
  if (rule.kind === 'floor') return `at least ${rule.min_exposure}%`
  return `at most ${rule.max_exposure}%`
}

export default function PlayerPoolTable({
  season, week, enabled = true,
}: { season: number; week: number; enabled?: boolean }) {
  const pool = usePlayerPool(season, week, enabled)
  const setRules = useSetPoolRules(season, week)
  const clearRules = useClearPoolRules(season, week)
  const [position, setPosition] = useState('')
  const [search, setSearch] = useState('')
  const [onlyRules, setOnlyRules] = useState(false)
  const [share, setShare] = useState<Record<string, string>>({})

  const error = (setRules.error ?? clearRules.error) as
    { response?: { data?: { detail?: string } } } | null
  const detail = error?.response?.data?.detail

  const rows = useMemo(() => {
    const all = pool.data?.data ?? []
    const needle = search.trim().toLowerCase()
    return all.filter((p) => (
      (!position || p.position === position)
      && (!needle || p.nickname.toLowerCase().includes(needle))
      && (!onlyRules || p.rule !== null)
    ))
  }, [pool.data, position, search, onlyRules])

  const apply = (rule: PoolRuleWrite) => setRules.mutate([rule])

  if (!enabled) return null
  if (pool.isLoading) {
    return <p className="text-sm text-slate-500">Loading the slate…</p>
  }
  if (pool.isError) {
    const message = (pool.error as { response?: { data?: { detail?: string } } })
      ?.response?.data?.detail
    return (
      <p className="text-sm text-slate-400">
        {message ?? 'The player pool could not be loaded.'}
      </p>
    )
  }
  const data = pool.data
  if (!data || data.status !== 'success') return null

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-2 text-slate-400">
          <Search size={13} className="text-slate-500" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Find a player"
            className="bg-slate-700 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-2.5 py-1.5 text-xs w-44"
          />
        </label>
        <div className="flex items-center gap-1">
          {['', ...POSITIONS].map((value) => (
            <button
              key={value || 'all'}
              onClick={() => setPosition(value)}
              className={`rounded-md px-2 py-1 transition-colors ${
                position === value
                  ? 'bg-slate-600 text-white'
                  : 'bg-slate-700/50 text-slate-400 hover:text-slate-200'
              }`}
            >
              {value || 'All'}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-1.5 text-slate-400 cursor-pointer">
          <input
            type="checkbox" checked={onlyRules}
            onChange={(e) => setOnlyRules(e.target.checked)}
            className="accent-brand-green"
          />
          Only players with a rule ({data.rules_set})
        </label>
        {data.rules_set > 0 && (
          <button
            onClick={() => clearRules.mutate(undefined)}
            disabled={clearRules.isPending}
            className="flex items-center gap-1.5 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <RotateCcw size={12} />
            Clear the pool
          </button>
        )}
        <span className="text-slate-500 ml-auto">
          {rows.length} of {data.players_in_file} in {data.slate.filename}
        </span>
      </div>

      {detail && (
        <p className="text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
          {detail}
        </p>
      )}

      <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-400 border-b border-slate-700">
              <th className="text-left py-2 px-3 font-medium whitespace-nowrap">Player</th>
              <th className="text-left py-2 px-2 font-medium whitespace-nowrap">Pos</th>
              <th className="text-left py-2 px-2 font-medium whitespace-nowrap">Opp</th>
              <th className="text-right py-2 px-2 font-medium whitespace-nowrap">Salary</th>
              <th className="text-right py-2 px-2 font-medium whitespace-nowrap">Proj</th>
              <th className="text-right py-2 px-2 font-medium whitespace-nowrap">Pts/$1k</th>
              <th
                className="text-right py-2 px-2 font-medium whitespace-nowrap"
                title="The lowest, highest and median he has actually scored. Shown because it reads well, never used to rank — as a band it covers less of the next result than the model's floor and ceiling do."
              >
                Has scored
              </th>
              <th
                className="text-left py-2 px-2 font-medium whitespace-nowrap"
                title="What this week's expert articles said about him (Players to Target). For reading only — the builder never uses it; set a rule if you agree."
              >
                Experts
              </th>
              <th
                className="text-right py-2 px-2 font-medium whitespace-nowrap"
                title="His share of every lineup already entered this week, pooled across builds"
              >
                In the week
              </th>
              <th className="text-left py-2 px-3 font-medium whitespace-nowrap">Rule</th>
              <th className="text-right py-2 px-3 font-medium whitespace-nowrap">Set</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => {
              const label = ruleLabel(p)
              const locked = p.rule?.kind === 'lock'
              const excluded = p.rule?.kind === 'exclude'
              return (
                <tr
                  key={p.nickname}
                  className={`border-b border-slate-700/40 last:border-0 ${
                    excluded ? 'opacity-50' : ''
                  }`}
                >
                  <td className="py-1.5 px-3 whitespace-nowrap">
                    <span className={locked ? 'text-brand-green' : 'text-slate-200'}>
                      {p.nickname}
                    </span>
                    {!p.from_model && (
                      <span
                        className="text-slate-500"
                        title="FanDuel's FPPG, not a model projection — the model projects skill positions only"
                      >
                        {' '}*
                      </span>
                    )}
                    {p.injury_flag && (
                      <span
                        className={`ml-1.5 rounded px-1 py-0.5 text-[0.65rem] ${
                          p.rosterable
                            ? 'bg-amber-500/15 text-amber-300'
                            : 'bg-red-500/15 text-red-300'
                        }`}
                        title={p.rosterable
                          ? 'FanDuel lists him questionable; the board does not discount for that'
                          : 'FanDuel will not pay him, so no lineup can hold him'}
                      >
                        {p.injury_flag}
                      </span>
                    )}
                  </td>
                  <td className="py-1.5 px-2 text-slate-500">{p.position}</td>
                  <td className="py-1.5 px-2 text-slate-400 whitespace-nowrap">
                    {p.opponent ?? '—'}
                  </td>
                  <td className="py-1.5 px-2 text-right text-slate-400 tabular-nums whitespace-nowrap">
                    {money(p.salary)}
                  </td>
                  <td className="py-1.5 px-2 text-right text-slate-300 tabular-nums">
                    {num(p.projected_points)}
                  </td>
                  <td className="py-1.5 px-2 text-right text-slate-300 tabular-nums">
                    {num(p.value, 2)}
                  </td>
                  <td
                    className="py-1.5 px-2 text-right text-slate-400 tabular-nums whitespace-nowrap"
                    title={p.history
                      ? `median ${p.history.median} over ${p.history.games} games`
                      : 'No games on record before this week'}
                  >
                    {p.history ? `${p.history.low}–${p.history.high}` : '—'}
                  </td>
                  <td
                    className="py-1.5 px-2 whitespace-nowrap"
                    title={p.expert ? p.expert.articles.join(', ') : 'No expert pick this week'}
                  >
                    {p.expert && p.expert.tags.length ? (
                      <span className="flex gap-1">
                        {p.expert.tags.filter((t) => t !== 'mention').map((t) => (
                          <span
                            key={t}
                            className={`rounded px-1 py-0.5 text-[0.65rem] ${
                              t.startsWith('favorite')
                                ? 'bg-sky-500/20 text-sky-200'
                                : 'bg-slate-700 text-slate-300'
                            }`}
                          >
                            {EXPERT_TAG[t] ?? t}
                          </span>
                        ))}
                      </span>
                    ) : (
                      <span className="text-slate-600">{p.expert ? 'noted' : '—'}</span>
                    )}
                  </td>
                  <td className="py-1.5 px-2 text-right tabular-nums whitespace-nowrap">
                    {p.entered_exposure === null ? (
                      <span className="text-slate-600">—</span>
                    ) : (
                      <span className={p.entered_exposure >= 50
                        ? 'text-amber-300' : 'text-slate-400'}>
                        {p.entered_exposure}%
                      </span>
                    )}
                  </td>
                  <td className="py-1.5 px-3 whitespace-nowrap">
                    {label ? (
                      <button
                        onClick={() => clearRules.mutate(p.nickname)}
                        title="Clear this rule"
                        className={`rounded px-1.5 py-0.5 text-[0.7rem] transition-colors ${
                          locked
                            ? 'bg-brand-green/15 text-brand-green hover:bg-brand-green/25'
                            : excluded
                              ? 'bg-slate-600/40 text-slate-300 hover:bg-slate-600/60'
                              : 'bg-sky-500/15 text-sky-300 hover:bg-sky-500/25'
                        }`}
                      >
                        {label} ×
                      </button>
                    ) : (
                      <span className="text-slate-600">—</span>
                    )}
                    {p.rule?.note && (
                      <span className="ml-1.5 text-slate-500">{p.rule.note}</span>
                    )}
                  </td>
                  <td className="py-1.5 px-3">
                    <div className="flex items-center justify-end gap-1">
                      <button
                        onClick={() => apply({ nickname: p.nickname, rule: 'lock' })}
                        disabled={!p.rosterable || setRules.isPending}
                        title={p.rosterable
                          ? 'In every lineup'
                          : `FanDuel lists him ${p.injury_flag}, so he cannot be required`}
                        className="rounded p-1 text-slate-400 hover:text-brand-green hover:bg-slate-700 disabled:opacity-30 disabled:hover:text-slate-400 disabled:hover:bg-transparent transition-colors"
                      >
                        <Lock size={12} />
                      </button>
                      {QUICK_SHARES.map((pct) => (
                        <button
                          key={pct}
                          onClick={() => apply({ nickname: p.nickname,
                                                 min_exposure: pct, max_exposure: pct })}
                          disabled={!p.rosterable || setRules.isPending}
                          title={`In about ${pct}% of the lineups`}
                          className="rounded px-1 py-1 text-[0.7rem] text-slate-500 hover:text-sky-300 hover:bg-slate-700 disabled:opacity-30 transition-colors tabular-nums"
                        >
                          {pct}
                        </button>
                      ))}
                      <input
                        value={share[p.nickname] ?? ''}
                        onChange={(e) => setShare((s) => ({ ...s, [p.nickname]: e.target.value }))}
                        onKeyDown={(e) => {
                          if (e.key !== 'Enter') return
                          const pct = Number(share[p.nickname])
                          if (!Number.isFinite(pct) || pct < 0 || pct > 100) return
                          apply({ nickname: p.nickname, min_exposure: pct, max_exposure: pct })
                          setShare((s) => ({ ...s, [p.nickname]: '' }))
                        }}
                        disabled={!p.rosterable}
                        placeholder="%"
                        title="A share of the lineups, 0 to 100. Enter to apply."
                        className="bg-slate-700 border border-slate-600 text-white placeholder-slate-600 rounded px-1.5 py-0.5 text-[0.7rem] w-10 text-center tabular-nums disabled:opacity-30"
                      />
                      <button
                        onClick={() => apply({ nickname: p.nickname, rule: 'exclude' })}
                        disabled={setRules.isPending}
                        title="In none of them"
                        className="rounded p-1 text-slate-400 hover:text-red-300 hover:bg-slate-700 disabled:opacity-30 transition-colors"
                      >
                        <Ban size={12} />
                      </button>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-slate-500">
        A rule is a share of the lineups, held as a floor as well as a ceiling —
        the point of typing 40 is to get 40, not "up to 40". The lock is 100% and
        the fade is 0%.{' '}
        {data.lineups_entered > 0
          ? `${data.lineups_entered} lineup${data.lineups_entered === 1 ? '' : 's'} already entered this week; "In the week" is each player's share of them.`
          : 'Nothing is entered for this week yet.'}{' '}
        <span className="text-slate-600">
          * means FanDuel's FPPG rather than a model projection.
        </span>
      </p>
    </div>
  )
}
