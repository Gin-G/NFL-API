import { AlertCircle } from 'lucide-react'

interface ErrorCardProps {
  message?: string
  onRetry?: () => void
}

export default function ErrorCard({ message = 'Something went wrong', onRetry }: ErrorCardProps) {
  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-red-800 flex flex-col items-center gap-3 text-center">
      <AlertCircle className="text-brand-red" size={28} />
      <p className="text-slate-300 text-sm">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-4 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg transition-colors"
        >
          Retry
        </button>
      )}
    </div>
  )
}
