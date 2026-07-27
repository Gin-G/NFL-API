import { useQuery } from '@tanstack/react-query'
import { apiClient } from './client'
import type { PBPResponse } from './types'

export function useGamePBP(gameId: string | null) {
  return useQuery({
    queryKey: ['gamePBP', gameId],
    queryFn: async () => {
      const { data } = await apiClient.get<PBPResponse>('/pbp/', { params: { game_id: gameId } })
      return data
    },
    enabled: !!gameId,
    staleTime: 1000 * 60 * 60 * 24, // completed games don't change
  })
}
