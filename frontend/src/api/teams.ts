import { useQuery } from '@tanstack/react-query'
import { apiClient } from './client'
import type { TeamsResponse, TeamDetailResponse } from './types'

export function useTeams() {
  return useQuery({
    queryKey: ['teams'],
    queryFn: async () => {
      const { data } = await apiClient.get<TeamsResponse>('/teams/')
      return data
    },
    staleTime: 1000 * 60 * 60, // teams rarely change
  })
}

export function useTeam(abbr: string) {
  return useQuery({
    queryKey: ['team', abbr],
    queryFn: async () => {
      const { data } = await apiClient.get<TeamDetailResponse>(`/teams/${abbr}`)
      return data
    },
    enabled: !!abbr,
    staleTime: 1000 * 60 * 60,
  })
}
