import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'
import type { PicksUploadResponse, TargetsResponse } from './types'

/** Expert picks for a week, in agree / skeptical / no-board piles. */
export function useTargets(season: number, week: number) {
  return useQuery({
    queryKey: ['targets', season, week],
    queryFn: async () => {
      const { data } = await apiClient.get<TargetsResponse>('/targets/', {
        params: { season, week },
      })
      return data
    },
  })
}

/** Upload a picks CSV; re-uploading an article replaces it. */
export function useUploadPicks(season: number, week: number) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (file: File) => {
      const form = new FormData()
      form.append('file', file)
      form.append('season', String(season))
      form.append('week', String(week))
      const { data } = await apiClient.post<PicksUploadResponse>('/targets/picks', form)
      return data
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['targets', season, week] }),
  })
}
