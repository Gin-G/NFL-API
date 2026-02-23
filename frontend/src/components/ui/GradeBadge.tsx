function gradeColor(grade: string): string {
  const g = grade.toUpperCase()
  if (g.startsWith('A')) return 'bg-green-700 text-green-100'
  if (g.startsWith('B')) return 'bg-blue-700 text-blue-100'
  if (g.startsWith('C')) return 'bg-yellow-700 text-yellow-100'
  if (g.startsWith('D')) return 'bg-orange-700 text-orange-100'
  return 'bg-red-800 text-red-100'
}

export default function GradeBadge({ grade }: { grade: string }) {
  return (
    <span
      className={`inline-flex items-center justify-center w-10 h-8 rounded font-bold text-sm ${gradeColor(grade)}`}
    >
      {grade}
    </span>
  )
}
