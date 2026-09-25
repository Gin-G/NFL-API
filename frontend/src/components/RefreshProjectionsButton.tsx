import { useEffect, useRef, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import {
  useProjectionJobStatus,
  useProjectionRefreshState,
  useStartProjectionRefresh,
} from '../api/projections'

/**
 * Recompute the board now, for news that lands between scheduled runs — a
 * Friday injury designation, a depth-chart change.
 *
 * The run happens in a Kubernetes job, not in the API, and reuses the model
 * already trained for this week, so it is minutes rather than the better part
 * of an hour. While it runs this polls the job and refreshes the tables when it
 * finishes. The endpoint is rate limited rather than authenticated, so the
 * disabled state here is a courtesy, not the protection.
 */
export default function RefreshProjectionsButton() {
  const queryClient = useQueryClient()
  const stateQ = useProjectionRefreshState()
  const running = stateQ.data?.running ?? false
  const statusQ = useProjectionJobStatus(running)
  const start = useStartProjectionRefresh()
  const [justFinished, setJustFinished] = useState(false)
  const wasRunning = useRef(false)

  // When a run ends, pull the new numbers in rather than leaving yesterday's on
  // screen behind a "done" message.
  useEffect(() => {
    if (wasRunning.current && !running) {
      queryClient.invalidateQueries({ queryKey: ['weeklyProjections'] })
      queryClient.invalidateQueries({ queryKey: ['seasonProjections'] })
      setJustFinished(true)
      const timer = setTimeout(() => setJustFinished(false), 30_000)
      return () => clearTimeout(timer)
    }
    wasRunning.current = running
  }, [running, queryClient])

  const failed = stateQ.data?.job_status === 'failed'
  const pct = statusQ.data?.pct_complete ?? 0
  const disabled = start.isPending || !(stateQ.data?.can_refresh ?? false)
  const startError = start.error as { response?: { data?: { detail?: string } } } | null
  const message = running
    ? `Recomputing… ${Math.round(pct)}%`
    : startError?.response?.data?.detail ?? stateQ.data?.reason

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={() => start.mutate()}
        disabled={disabled}
        title={stateQ.data?.reason ?? 'Recompute projections from the latest injuries and rosters'}
        className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
          disabled
            ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
            : 'bg-brand-green text-white hover:brightness-110'
        }`}
      >
        <RefreshCw size={14} className={running ? 'animate-spin' : undefined} />
        {running ? 'Recomputing' : 'Refresh projections'}
      </button>
      {message && (
        <span
          className={`text-xs ${startError || failed ? 'text-amber-400' : 'text-slate-400'}`}
          role={startError ? 'alert' : undefined}
        >
          {message}
        </span>
      )}
      {justFinished && !running && !startError && (
        <span className="text-xs text-brand-green">Projections updated</span>
      )}
    </div>
  )
}
