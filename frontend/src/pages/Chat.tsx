import { useState, useRef, useEffect, FormEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { MessageCircle, Send, AlertTriangle } from 'lucide-react'
import { sendChatMessage } from '../api/chat'
import type { ChatMessage } from '../api/types'
import PageHeader from '../components/ui/PageHeader'

function TypingIndicator() {
  return (
    <div className="flex items-end gap-2 mb-4">
      <div className="bg-slate-800 border border-brand-green rounded-2xl rounded-bl-sm px-4 py-3">
        <div className="flex gap-1 items-center h-4">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="block w-2 h-2 bg-slate-400 rounded-full animate-bounce"
              style={{ animationDelay: `${i * 150}ms` }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

function UnavailableBanner() {
  return (
    <div className="flex items-center gap-3 px-4 py-3 mb-6 rounded-lg bg-yellow-900/40 text-yellow-300 border border-yellow-700 text-sm">
      <AlertTriangle size={18} className="shrink-0" />
      <span>
        Chat is unavailable — <code className="font-mono text-xs">ANTHROPIC_API_KEY</code> is not
        configured on the server.
      </span>
    </div>
  )
}

export default function Chat() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [history, setHistory] = useState<ChatMessage[]>([])
  const [unavailable, setUnavailable] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  const mutation = useMutation({
    mutationFn: ({ message, hist }: { message: string; hist: ChatMessage[] }) =>
      sendChatMessage(message, hist),
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.response },
      ])
      setHistory(data.history)
    },
    onError: (err: unknown) => {
      const status = (err as { response?: { status?: number } })?.response?.status
      if (status === 503) {
        setUnavailable(true)
      }
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content:
            status === 503
              ? 'Chat is currently unavailable. Please check back later.'
              : 'Sorry, something went wrong. Please try again.',
        },
      ])
    },
  })

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, mutation.isPending])

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const text = input.trim()
    if (!text || mutation.isPending) return
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    mutation.mutate({ message: text, hist: history })
  }

  return (
    <div className="flex flex-col h-full p-6">
      <PageHeader title="Chat" subtitle="Ask questions about NFL stats, schedules, and more" />

      {unavailable && <UnavailableBanner />}

      {/* Message list */}
      <div className="flex-1 overflow-y-auto min-h-0 space-y-2 mb-4 pr-1">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-slate-500 gap-3 py-16">
            <MessageCircle size={48} className="opacity-30" />
            <p className="text-sm">Ask me anything about NFL data</p>
            <div className="flex flex-wrap gap-2 justify-center mt-2">
              {[
                'Who had the most rushing yards in week 4 of 2024?',
                'How did the Chiefs do in 2024?',
                'Who are the top QBs of 2024?',
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => setInput(suggestion)}
                  className="text-xs px-3 py-1.5 rounded-full bg-slate-700 hover:bg-slate-600 text-slate-300 transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, idx) =>
          msg.role === 'user' ? (
            <div key={idx} className="flex justify-end mb-4">
              <div className="max-w-[75%] bg-slate-700 text-white rounded-2xl rounded-br-sm px-4 py-3 text-sm whitespace-pre-wrap">
                {msg.content}
              </div>
            </div>
          ) : (
            <div key={idx} className="flex justify-start mb-4">
              <div className="max-w-[75%] bg-slate-800 border border-brand-green text-slate-200 rounded-2xl rounded-bl-sm px-4 py-3 text-sm whitespace-pre-wrap">
                {msg.content}
              </div>
            </div>
          ),
        )}

        {mutation.isPending && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      {/* Input form */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about NFL stats, schedules, players…"
          disabled={mutation.isPending}
          className="flex-1 bg-slate-800 border border-slate-600 rounded-xl px-4 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-brand-green disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!input.trim() || mutation.isPending}
          className="bg-brand-green hover:bg-green-600 disabled:opacity-40 text-white rounded-xl px-4 py-2.5 transition-colors"
        >
          <Send size={18} />
        </button>
      </form>
    </div>
  )
}
