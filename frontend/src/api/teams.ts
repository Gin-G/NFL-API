import { useQuery } from '@tanstack/react-query'
import { apiClient } from './client'
import type { TeamsResponse, TeamDetailResponse, DepthChartResponse, SnapCountsResponse } from './types'

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

export function useTeamDepthChart(abbr: string, season: number, week?: number) {
  return useQuery({
    queryKey: ['teamDepthChart', abbr, season, week],
    queryFn: async () => {
      const params: Record<string, string | number> = { season }
      if (week !== undefined) params.week = week
      const { data } = await apiClient.get<DepthChartResponse>(`/teams/${abbr}/depth-chart`, { params })
      return data
    },
    enabled: !!abbr,
    staleTime: 1000 * 60 * 60,
  })
}

export function useTeamSnapCounts(abbr: string, season: number, week?: number) {
  return useQuery({
    queryKey: ['teamSnapCounts', abbr, season, week],
    queryFn: async () => {
      const params: Record<string, string | number> = { season }
      if (week !== undefined) params.week = week
      const { data } = await apiClient.get<SnapCountsResponse>(`/teams/${abbr}/snap-counts`, { params })
      return data
    },
    enabled: !!abbr,
    staleTime: 1000 * 60 * 60,
  })
}
