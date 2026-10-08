import { useMutation, useQuery } from '@tanstack/react-query'
import { apiClient } from './client'
import type { LineupResponse, SlateCoverage } from './types'

/** What the board holds for a week, before any CSV is uploaded. */
export function useSlateCoverage(season: number, week?: number) {
  return useQuery({
    queryKey: ['slateCoverage', season, week],
    queryFn: async () => {
      const params: Record<string, number> = { season }
      if (week !== undefined) params.week = week
      const { data } = await apiClient.get<SlateCoverage>('/optimizer/slate-coverage', { params })
      return data
    },
    staleTime: 1000 * 60 * 10,
  })
}

export interface LineupRequest {
  file: File
  season: number
  week: number
  num_lineups: number
  objective: string
  salary_cap: number
  max_usage_percentage: number
  exclude: string
}

/** Upload a FanDuel slate CSV and get lineups back. */
export function useBuildLineups() {
  return useMutation({
    mutationFn: async (req: LineupRequest) => {
      const form = new FormData()
      form.append('file', req.file)
      Object.entries(req).forEach(([key, value]) => {
        if (key !== 'file') form.append(key, String(value))
      })
      const { data } = await apiClient.post<LineupResponse>('/optimizer/lineups', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      return data
    },
  })
}
