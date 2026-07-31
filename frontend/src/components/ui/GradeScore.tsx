import { gradeTone, fmtGrade } from '../../utils/gradeScale'

interface GradeScoreProps {
  value: number | null | undefined
  /** Small caption under the number, e.g. "Offense". */
  caption?: string
  size?: 'sm' | 'md' | 'lg'
  /** Adds the plain-English bucket ("Above average") as a hover title. */
  title?: string
}

const SIZES = {
  sm: 'text-xs px-1.5 py-0.5 min-w-[2.75rem]',
  md: 'text-sm px-2 py-1 min-w-[3.25rem]',
  lg: 'text-2xl px-3 py-1.5 min-w-[4.5rem]',
}

/**
 * Colored 0-100 grade badge. Shares its color scale with GradeMeter and the
 * player leaderboard via utils/gradeScale.
 */
export default function GradeScore({ value, caption, size = 'md', title }: GradeScoreProps) {
  const tone = gradeTone(value)
  return (
    <div className="inline-flex flex-col items-center gap-0.5">
      <span
        className={`inline-flex items-center justify-center rounded-md font-bold tabular-nums border ${SIZES[size]}`}
        style={{ backgroundColor: tone.bg, borderColor: tone.border, color: tone.text }}
        title={title ?? tone.label}
      >
        {fmtGrade(value)}
      </span>
      {caption && <span className="text-[10px] text-slate-400 uppercase tracking-wide">{caption}</span>}
    </div>
  )
}
