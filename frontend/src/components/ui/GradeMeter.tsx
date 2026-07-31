import { gradeTone, fmtGrade } from '../../utils/gradeScale'

interface GradeMeterProps {
  value: number | null | undefined
  /** Label shown above the bar. */
  label?: string
  /** Show the numeric grade next to the bar. */
  showValue?: boolean
  /**
   * Dims the bar for grades that carry less confidence (team defense grades,
   * per-play efficiency grades). Also surfaced as a hover title.
   */
  lowConfidence?: boolean
  title?: string
}

/**
 * Diverging 0-100 gauge centred on 50 (league / positional average): the fill
 * grows left from the midpoint for below-average grades and right for
 * above-average ones, so "how far from average" is the thing you read.
 */
export default function GradeMeter({
  value,
  label,
  showValue = true,
  lowConfidence = false,
  title,
}: GradeMeterProps) {
  const tone = gradeTone(value)
  const grade = value ?? 50
  // Half-widths measured from the 50% midpoint.
  const offset = Math.min(50, Math.abs(grade - 50))
  const above = grade >= 50

  return (
    <div className="w-full" title={title ?? tone.label}>
      {(label || showValue) && (
        <div className="flex items-baseline justify-between gap-2 mb-1">
          {label && (
            <span className="text-[10px] text-slate-400 uppercase tracking-wide">{label}</span>
          )}
          {showValue && (
            <span className="text-xs font-semibold tabular-nums" style={{ color: tone.text }}>
              {fmtGrade(value)}
            </span>
          )}
        </div>
      )}
      <div className="relative h-2 bg-slate-700/70 rounded-full overflow-hidden">
        {/* Midpoint (50 = average) */}
        <div className="absolute inset-y-0 left-1/2 w-px bg-slate-500 z-10" />
        {value != null && (
          <div
            className={`absolute inset-y-0 rounded-full ${lowConfidence ? 'opacity-60' : ''}`}
            style={{
              backgroundColor: tone.hex,
              left: above ? '50%' : `${50 - offset}%`,
              width: `${offset}%`,
            }}
          />
        )}
      </div>
    </div>
  )
}
