import { useQuery } from '@tanstack/react-query'
import { apiClient } from './client'
import type {
  SeasonProjectionsResponse,
  PlayerProjectionsResponse,
  WeeklyProjectionsResponse,
} from './types'

/**
 * Season-long projected totals, best-projected first.
 *
 * The endpoint answers 200 with `status: 'no_data'` when nothing is cached for
 * the season, so there is no 404 to translate here.
 */
export function useSeasonProjections(season: number, position?: string, limit = 300) {
  return useQuery({
    queryKey: ['seasonProjections', season, position, limit],
    queryFn: async () => {
      const params: Record<string, string | number> = { limit }
      if (position) params.position = position
      const { data } = await apiClient.get<SeasonProjectionsResponse>(
        `/projections/season/${season}`,
        { params },
      )
      return data
    },
    staleTime: 1000 * 60 * 30,
  })
}

/** Every cached weekly projection for one player in a season. */
export function usePlayerProjections(playerId: string | null, season: number) {
  return useQuery({
    queryKey: ['playerProjections', playerId, season],
    queryFn: async () => {
      const { data } = await apiClient.get<PlayerProjectionsResponse>(
        `/projections/player/${playerId}`,
        { params: { season } },
      )
      return data
    },
    enabled: !!playerId,
    staleTime: 1000 * 60 * 30,
  })
}

/**
 * One week's projections for every player, best-projected first.
 *
 * `asOf` reads the archive rather than the current projection: the value is
 * how many weeks were complete when the number was computed, so `asOf: 0` is
 * the preseason view of that week. Omit it for what we think today.
 */
export function useWeeklyProjections(
  season: number,
  week: number,
  position?: string,
  limit = 300,
  asOf?: number,
) {
  return useQuery({
    queryKey: ['weeklyProjections', season, week, position, limit, asOf],
    queryFn: async () => {
      const params: Record<string, string | number> = { season, week, limit }
      if (position) params.position = position
      if (asOf !== undefined) params.as_of = asOf
      const { data } = await apiClient.get<WeeklyProjectionsResponse>(
        '/projections/',
        { params },
      )
      return data
    },
    staleTime: 1000 * 60 * 30,
  })
}
