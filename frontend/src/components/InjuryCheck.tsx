import { useState } from 'react'
import { ArrowRight, RotateCcw, Stethoscope, UserX } from 'lucide-react'
import { useClearPoolRules, useInjuryCheck, useSetPoolRules } from '../api/optimizer'
import type { InjuryChange } from '../api/types'
import RefreshProjectionsButton from './RefreshProjectionsButton'

/**
 * Re-check injury status against ESPN and show who changed.
 *
 * nflverse, which the board is built from, carries the clubs' Friday
 * designation and nothing after it. A Sunday inactive -- Kenneth Walker in
 * 2026 week 10 -- is on ESPN and never there. This lists each slate player whose
 * ESPN status this week disagrees with the board, and who moves up if he sits.
 * It applies nothing by itself: "Rule out" is the pool's Out rule (instant, on
 * the stored board), and Refresh projections reruns the board, which reads the
 * same feed.
 */
export default function InjuryCheck({ season, week }: { season: number; week?: number }) {
  const check = useInjuryCheck(season, week)
  const setRules = useSetPoolRules(season, week)
  const clearRules = useClearPoolRules(season, week)
  const [showQ, setShowQ] = useState(false)
  const data = check.data
  const busy = setRules.isPending || clearRules.isPending

  const act = (c: InjuryChange) => {
    const done = () => check.mutate(false)
    if (c.kind === 'out' && !c.applied) {
      setRules.mutate([{ nickname: c.nickname, rule: 'out' }], { onSuccess: done })
    } else if (c.applied) {
      clearRules.mutate(c.nickname, { onSuccess: done })
    }
  }

  const main = data?.changes.filter((c) => c.kind !== 'questionable') ?? []
  const questionable = data?.changes.filter((c) => c.kind === 'questionable') ?? []
  const err = check.error as { response?: { data?: { detail?: string } } } | null

  return (
    <div className="mb-6 rounded-lg border border-slate-700 bg-slate-800/50 p-3">
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={() => check.mutate(true)}
          disabled={check.isPending}
          className="flex items-center gap-2 rounded-lg bg-slate-700 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-slate-600 disabled:opacity-50"
          title="Compare ESPN's injury news this week with the board"
        >
          <Stethoscope size={14} className={check.isPending ? 'animate-pulse' : undefined} />
          {check.isPending ? 'Checking…' : 'Check injuries'}
        </button>
        <RefreshProjectionsButton />
        {data && (
          <span className="text-xs text-slate-400">
            ESPN, {fmtTime(data.checked_at)} ·{' '}
            {data.pending
              ? <span className="text-amber-400">{data.pending} need a decision</span>
              : 'nothing waiting'}
            {data.board_computed_at && <> · board computed {fmtTime(data.board_computed_at)}</>}
          </span>
        )}
        {err && (
          <span className="text-xs text-amber-400" role="alert">
            {err.response?.data?.detail ?? 'The injury check failed'}
          </span>
        )}
      </div>

      {data && main.length === 0 && (
        <p className="mt-2 text-xs text-slate-500">
          ESPN has no status this week that the board does not already reflect.
        </p>
      )}

      {main.length > 0 && (
        <ul className="mt-3 divide-y divide-slate-700/60">
          {main.map((c) => <Row key={c.nickname} c={c} busy={busy} onAct={act} />)}
        </ul>
      )}

      {questionable.length > 0 && (
        <div className="mt-2">
          <button onClick={() => setShowQ((v) => !v)}
                  className="text-xs text-slate-400 hover:text-slate-200">
            {showQ ? 'Hide' : 'Show'} {questionable.length} questionable
          </button>
          {showQ && (
            <ul className="mt-1 divide-y divide-slate-700/60">
              {questionable.map((c) => <Row key={c.nickname} c={c} busy={busy} onAct={act} />)}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

const BADGE: Record<string, string> = {
  out: 'bg-red-500/15 text-red-300',
  cleared: 'bg-brand-green/15 text-brand-green',
  questionable: 'bg-slate-600/40 text-slate-300',
}
const LABEL: Record<string, string> = { out: 'Out', cleared: 'Back', questionable: 'Q' }

function Row({ c, busy, onAct }: { c: InjuryChange; busy: boolean;
                                   onAct: (c: InjuryChange) => void }) {
  const canRuleOut = c.kind === 'out' && !c.applied && !c.started
  const canUndo = c.applied && !c.started && c.kind === 'cleared'
  return (
    <li className={`flex flex-wrap items-start gap-x-3 gap-y-1 py-2 text-sm ${c.started ? 'opacity-50' : ''}`}>
      <span className={`mt-0.5 rounded px-1.5 py-0.5 text-[11px] font-semibold ${BADGE[c.kind]}`}>
        {c.espn_status === 'DOUBTFUL' ? 'Doubtful' : LABEL[c.kind]}
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-slate-100">
          {c.nickname}{' '}
          <span className="text-xs text-slate-400">
            {c.position} · {c.team}{c.salary ? ` · $${c.salary.toLocaleString()}` : ''}
            {c.board_points !== null && ` · board ${c.board_points.toFixed(1)}`}
          </span>
        </div>
        {c.comment && <div className="truncate text-xs text-slate-500" title={c.comment}>{c.comment}</div>}
        {c.promotes.length > 0 && (
          <div className="mt-0.5 flex flex-wrap gap-x-3 text-xs text-slate-300">
            {c.promotes.slice(0, 3).map((p) => (
              <span key={p.nickname} className="flex items-center gap-1">
                <ArrowRight size={11} className="text-brand-green" />
                {p.nickname} {p.from.toFixed(1)} → <span className="text-brand-green">{p.to.toFixed(1)}</span>
                {p.salary ? <span className="text-slate-500">${p.salary.toLocaleString()}</span> : null}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="flex items-center gap-2 text-xs">
        {c.started && <span className="text-slate-500">kicked off</span>}
        {c.applied && c.kind === 'out' && <span className="text-slate-400">ruled out</span>}
        {canRuleOut && (
          <button onClick={() => onAct(c)} disabled={busy}
                  className="flex items-center gap-1 rounded bg-red-500/20 px-2 py-1 text-red-200 hover:bg-red-500/30 disabled:opacity-50">
            <UserX size={12} /> Rule out
          </button>
        )}
        {canUndo && (
          <button onClick={() => onAct(c)} disabled={busy}
                  className="flex items-center gap-1 rounded bg-slate-700 px-2 py-1 text-slate-200 hover:bg-slate-600 disabled:opacity-50">
            <RotateCcw size={12} /> Undo out
          </button>
        )}
      </div>
    </li>
  )
}

function fmtTime(iso: string | null) {
  if (!iso) return '—'
  const d = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`)
  return d.toLocaleString(undefined, { weekday: 'short', hour: 'numeric', minute: '2-digit' })
}
