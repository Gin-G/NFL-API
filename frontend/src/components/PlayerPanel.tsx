import type { ReactNode } from 'react'
import { X } from 'lucide-react'
import PlayerBreakdown from './PlayerBreakdown'

interface PlayerPanelProps {
  playerId: string
  playerName: string
  /** Small line under the name, e.g. "WR · KC · 2025". */
  subtitle?: string
  season: number
  onClose: () => void
  /**
   * Show the play-by-play breakdown under the children. Turn it off for
   * forward-looking views (projections), where a season has no plays yet.
   */
  showBreakdown?: boolean
  /** Optional content rendered above the play-by-play breakdown. */
  children?: ReactNode
}

/**
 * Slide-over player detail panel: header + optional extra content + the PBP
 * breakdown. Shared by the Players page and the player grade leaderboard.
 */
export default function PlayerPanel({
  playerId,
  playerName,
  subtitle,
  season,
  onClose,
  showBreakdown = true,
  children,
}: PlayerPanelProps) {
  return (
    <div className="fixed inset-0 z-50 flex justify-end" onClick={onClose}>
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" />

      {/* Panel */}
      <div
        className="relative w-full max-w-xl bg-slate-900 border-l border-slate-700 overflow-y-auto h-full"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 bg-slate-900 border-b border-slate-700 px-5 py-4 flex items-start justify-between z-10">
          <div>
            <h2 className="text-white font-semibold text-base">{playerName}</h2>
            {subtitle && <p className="text-slate-400 text-xs mt-0.5">{subtitle}</p>}
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors ml-4 mt-0.5"
          >
            <X size={18} />
          </button>
        </div>

        <div className="px-5 py-4 space-y-5">
          {children}
          {showBreakdown && <PlayerBreakdown playerId={playerId} season={season} />}
        </div>
      </div>
    </div>
  )
}
