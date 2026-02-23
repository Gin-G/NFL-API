import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'

interface WinPctDataPoint {
  season: number
  win_percentage: number
  record: string
}

interface WinPctBarChartProps {
  data: WinPctDataPoint[]
}

export default function WinPctBarChart({ data }: WinPctBarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis dataKey="season" tick={{ fill: '#94a3b8', fontSize: 12 }} />
        <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" />
        <Tooltip
          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
          labelStyle={{ color: '#f1f5f9' }}
          formatter={(value: number, _name: string, item) => [
            `${value.toFixed(1)}% (${(item.payload as WinPctDataPoint | undefined)?.record ?? ''})`,
            'Win %',
          ]}
        />
        <Bar dataKey="win_percentage" radius={[4, 4, 0, 0]}>
          {data.map((entry) => (
            <Cell
              key={entry.season}
              fill={entry.win_percentage >= 60 ? '#1a7a4a' : entry.win_percentage >= 40 ? '#d4b94e' : '#c8102e'}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
