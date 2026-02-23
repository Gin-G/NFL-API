export default function SkeletonCard({ rows = 1 }: { rows?: number }) {
  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 animate-pulse space-y-3">
      <div className="h-3 w-24 bg-slate-700 rounded" />
      <div className="h-7 w-16 bg-slate-700 rounded" />
      {Array.from({ length: rows - 1 }).map((_, i) => (
        <div key={i} className="h-3 w-full bg-slate-700 rounded" />
      ))}
    </div>
  )
}
