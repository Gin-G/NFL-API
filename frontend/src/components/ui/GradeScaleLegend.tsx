import { gradeTone } from '../../utils/gradeScale'

/** Swatch key for the shared 0-100 grade color scale. */
export default function GradeScaleLegend({ average = 'league average' }: { average?: string }) {
  const stops: [number, string][] = [
    [32, 'below 40'],
    [50, `~50 ${average}`],
    [68, 'above 60'],
  ]
  return (
    <div className="flex items-center gap-3 text-[11px] text-slate-400">
      {stops.map(([grade, label]) => (
        <span key={label} className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: gradeTone(grade).hex }} />
          {label}
        </span>
      ))}
    </div>
  )
}
