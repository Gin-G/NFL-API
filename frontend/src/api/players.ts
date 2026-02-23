import { useQuery } from '@tanstack/react-query'
import { apiClient } from './client'
import { getCurrentNFLSeason } from '../utils/nflDate'
import type {
  RostersResponse,
  PlayerStatsResponse,
  PlayerDetailResponse,
  PlayerGradesResponse,
} from './types'

export function useRosters(season: number, team?: string, position?: string, week?: number) {
  return useQuery({
    queryKey: ['rosters', season, team, position, week],
    queryFn: async () => {
      const params: Record<string, string | number> = { season }
      if (team) params.team = team
      if (position) params.position = position
      if (week !== undefined) params.week = week
      const { data } = await apiClient.get<RostersResponse>('/players/rosters', { params })
      return data
    },
    staleTime: 1000 * 60 * 10,
  })
}

export function usePlayerStats(
  season: number,
  position?: string,
  team?: string,
  week?: number,
  playerId?: string
) {
  return useQuery({
    queryKey: ['playerStats', season, position, team, week, playerId],
    queryFn: async () => {
      const params: Record<string, string | number> = { season }
      if (position) params.position = position
      if (team) params.team = team
      if (week !== undefined) params.week = week
      if (playerId) params.player_id = playerId
      const { data } = await apiClient.get<PlayerStatsResponse>('/players/stats', { params })
      return data
    },
    staleTime: 1000 * 60 * 5,
  })
}

export function usePlayerDetail(playerId: string, season: number) {
  return useQuery({
    queryKey: ['playerDetail', playerId, season],
    queryFn: async () => {
      const { data } = await apiClient.get<PlayerDetailResponse>(`/players/${playerId}`, {
        params: { season },
      })
      return data
    },
    enabled: !!playerId,
  })
}

export function usePlayerGrades(years: number[] = [getCurrentNFLSeason()], minGames = 3, limit = 20) {
  return useQuery({
    queryKey: ['playerGrades', years, minGames, limit],
    queryFn: async () => {
      const params = new URLSearchParams()
      years.forEach((y) => params.append('years', String(y)))
      params.set('min_games', String(minGames))
      params.set('limit', String(limit))
      const { data } = await apiClient.get<PlayerGradesResponse>(`/players/grades?${params}`)
      return data
    },
    staleTime: 1000 * 60 * 15,
  })
}
