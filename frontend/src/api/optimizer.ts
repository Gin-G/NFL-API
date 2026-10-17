import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'
import type { LineupPerformance, LineupResponse, SlateCoverage, SlateList } from './types'

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

/** The slate CSVs on file. FanDuel publishes pricing nowhere fetchable, so the
 *  upload is the one manual step — and it only has to happen once per slate. */
export function useSavedSlates(season: number, week?: number) {
  return useQuery({
    queryKey: ['savedSlates', season, week],
    queryFn: async () => {
      const params: Record<string, number> = { season }
      if (week !== undefined) params.week = week
      const { data } = await apiClient.get<SlateList>('/optimizer/slates', { params })
      return data
    },
    staleTime: 1000 * 60,
  })
}

export function useForgetSlate() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (slateId: number) => {
      const { data } = await apiClient.delete(`/optimizer/slates/${slateId}`)
      return data
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['savedSlates'] }),
  })
}

/** How the lineups this optimizer built actually scored. */
export function useLineupPerformance(season: number, week?: number) {
  return useQuery({
    queryKey: ['lineupPerformance', season, week],
    queryFn: async () => {
      const params: Record<string, number> = { season }
      if (week !== undefined) params.week = week
      const { data } = await apiClient.get<LineupPerformance>('/optimizer/performance', { params })
      return data
    },
    staleTime: 1000 * 60 * 5,
  })
}

export interface LineupRequest {
  /** Omitted to build from the week's saved slate. */
  file?: File | null
  season: number
  week: number
  num_lineups: number
  objective: string
  salary_cap: number
  max_usage_percentage: number
  exclude: string
  /** all | main | sunday | primetime — the export is always Thu-Mon. */
  slate: string
  /** A specific saved slate, for a Showdown file alongside a classic list. */
  slate_id?: number
}

/** Build lineups: from an uploaded FanDuel slate CSV, or from the saved one. */
export function useBuildLineups() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (req: LineupRequest) => {
      const form = new FormData()
      if (req.file) form.append('file', req.file)
      Object.entries(req).forEach(([key, value]) => {
        if (key !== 'file' && value !== undefined && value !== null) {
          form.append(key, String(value))
        }
      })
      const { data } = await apiClient.post<LineupResponse>('/optimizer/lineups', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      return data
    },
    onSuccess: () => {
      // The upload was saved and the build recorded, so both lists moved.
      queryClient.invalidateQueries({ queryKey: ['savedSlates'] })
      queryClient.invalidateQueries({ queryKey: ['lineupPerformance'] })
    },
  })
}
