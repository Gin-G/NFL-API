/**
 * Shared color scale for 0-100 grades (50 = average).
 *
 * Used by both the team ratings table and the player grade leaderboard so a
 * "62" always looks the same shade of green wherever it appears.
 *
 * The scale is diverging and centred on 50:
 *   30 or below → full red      (well below average)
 *   ~50         → neutral gray  (average)
 *   70 or above → full green    (well above average)
 * so anything under 40 reads clearly red and anything over 60 clearly green,
 * with a gray band around the middle where the differences aren't meaningful.
 */

const RED: RGB = [220, 38, 38] // red-600
// A true neutral gray, not slate: mixing through a blue-tinted gray turns the
// mid-range purple on the red side and teal on the green side.
const NEUTRAL: RGB = [122, 122, 125]
const GREEN: RGB = [22, 163, 74] // green-600
/** Slate-toned neutral for "no grade", to sit quietly against the UI. */
const NO_GRADE: RGB = [100, 116, 139] // slate-500

/** Grade distance from 50 at which the scale reaches full saturation. */
const FULL_SCALE = 20

type RGB = [number, number, number]

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

function mix(a: RGB, b: RGB, t: number): RGB {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ]
}

function rgba([r, g, b]: RGB, alpha: number): string {
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

function hex([r, g, b]: RGB): string {
  return '#' + [r, g, b].map((c) => c.toString(16).padStart(2, '0')).join('')
}

/** The base color for a grade, as an [r,g,b] triple. */
function gradeRGB(grade: number): RGB {
  const t = clamp((grade - 50) / FULL_SCALE, -1, 1)
  return t >= 0 ? mix(NEUTRAL, GREEN, t) : mix(NEUTRAL, RED, -t)
}

export interface GradeTone {
  /** Solid scale color — for bar fills and dots. */
  hex: string
  /** Translucent version of the same color — for badge backgrounds on dark UI. */
  bg: string
  /** Border color for badges. */
  border: string
  /** Lightened color, readable as text on the dark slate background. */
  text: string
  /** Plain-English bucket, e.g. "Above average". */
  label: string
}

/** Full color treatment for a grade. Null/undefined grades render neutral. */
export function gradeTone(grade: number | null | undefined): GradeTone {
  if (grade == null || Number.isNaN(grade)) {
    return {
      hex: hex(NO_GRADE),
      bg: rgba(NO_GRADE, 0.15),
      border: rgba(NO_GRADE, 0.4),
      text: '#94a3b8',
      label: 'No grade',
    }
  }
  const base = gradeRGB(grade)
  return {
    hex: hex(base),
    bg: rgba(base, 0.18),
    border: rgba(base, 0.5),
    // Lighten toward white so the number stays legible on slate-800/900.
    text: hex(mix(base, [255, 255, 255], 0.45)),
    label: gradeLabel(grade),
  }
}

/** Plain-English bucket for a grade. */
export function gradeLabel(grade: number | null | undefined): string {
  if (grade == null || Number.isNaN(grade)) return 'No grade'
  if (grade >= 70) return 'Elite'
  if (grade >= 60) return 'Above average'
  if (grade > 55) return 'Slightly above average'
  if (grade >= 45) return 'Average'
  if (grade > 40) return 'Slightly below average'
  if (grade > 30) return 'Below average'
  return 'Poor'
}

/** Format a grade for display. */
export function fmtGrade(grade: number | null | undefined): string {
  if (grade == null || Number.isNaN(grade)) return '—'
  return grade.toFixed(1)
}

/**
 * Signed difference between two grades, e.g. an offense grade vs the opposing
 * defense grade. Positive = the first side has the edge.
 */
export function fmtEdge(diff: number): string {
  return (diff >= 0 ? '+' : '') + diff.toFixed(1)
}
