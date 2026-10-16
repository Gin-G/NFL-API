import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import { apiClient } from './client'
import type {
  CoverageCornersResponse,
  CoverageSecondariesResponse,
  CoverageMatchupsResponse,
} from './types'

/**
 * Coverage ratings. These 404 for a season PFR has not charted and for the
 * first weeks of one, before any corner has enough targets — both ordinary
 * states, so the hooks return empty shapes and let the page explain.
 */
function emptyOn404<T>(err: unknown, fallback: T): T {
  if (axios.isAxiosError(err) && err.response?.status === 404) return fallback
  throw err
}

/** Every graded cornerback, worst coverage first. */
export function useCoverageCorners(season: number, throughWeek?: number, minTargets = 10) {
  return useQuery({
    queryKey: ['coverageCorners', season, throughWeek, minTargets],
    queryFn: async () => {
      const params: Record<string, number> = { season, min_targets: minTargets }
      if (throughWeek !== undefined) params.through_week = throughWeek
      try {
        const { data } = await apiClient.get<CoverageCornersResponse>('/coverage/corners', {
          params,
        })
        return data
      } catch (err) {
        return emptyOn404(err, {
          season,
          through_week: throughWeek ?? null,
          scale: '',
          shrinkage_targets: 0,
          not_a_trait: '',
          bands: [],
          corners: [],
        } as CoverageCornersResponse)
      }
    },
    staleTime: 1000 * 60 * 15,
  })
}

/** Every secondary, softest first, with its weakest regular corner. */
export function useCoverageSecondaries(season: number, throughWeek?: number) {
  return useQuery({
    queryKey: ['coverageSecondaries', season, throughWeek],
    queryFn: async () => {
      const params: Record<string, number> = { season }
      if (throughWeek !== undefined) params.through_week = throughWeek
      try {
        const { data } = await apiClient.get<CoverageSecondariesResponse>(
          '/coverage/secondaries',
          { params },
        )
        return data
      } catch (err) {
        return emptyOn404(err, {
          season,
          through_week: throughWeek ?? null,
          scale: '',
          bands: [],
          secondaries: [],
        } as CoverageSecondariesResponse)
      }
    },
    staleTime: 1000 * 60 * 15,
  })
}

/**
 * This week's games from the receiving side, with the receivers each secondary
 * faces. The `coverage_lean` on each receiver is a label for a human: the
 * projection itself is not adjusted by anything on this page, which is a
 * measured decision rather than an unfinished one.
 */
export function useCoverageMatchups(season: number, week: number) {
  return useQuery({
    queryKey: ['coverageMatchups', season, week],
    queryFn: async () => {
      try {
        const { data } = await apiClient.get<CoverageMatchupsResponse>('/coverage/matchups', {
          params: { season, week },
        })
        return data
      } catch (err) {
        return emptyOn404(err, {
          season,
          week,
          bands: [],
          lean_is_advisory: true,
          matchups: [],
        } as CoverageMatchupsResponse)
      }
    },
    enabled: week > 0,
    staleTime: 1000 * 60 * 15,
  })
}
