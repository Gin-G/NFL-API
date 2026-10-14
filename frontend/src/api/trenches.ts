import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import { apiClient } from './client'
import type {
  LineUnitsResponse,
  LineMatchupsResponse,
  LineBacktestResponse,
} from './types'

/**
 * Line-of-scrimmage ratings. These 404 for a season the loader has not
 * ingested and for the first weeks of a season, before there is enough
 * football to rate anything — both are ordinary states, so the hooks return
 * empty shapes and let the page explain rather than throwing an error card.
 */
function emptyOn404<T>(err: unknown, fallback: T): T {
  if (axios.isAxiosError(err) && err.response?.status === 404) return fallback
  throw err
}

/** 0–100 grades for all four line units, optionally as they stood entering a week. */
export function useLineUnits(season: number, throughWeek?: number) {
  return useQuery({
    queryKey: ['lineUnits', season, throughWeek],
    queryFn: async () => {
      const params: Record<string, number> = {}
      if (throughWeek !== undefined) params.through_week = throughWeek
      try {
        const { data } = await apiClient.get<LineUnitsResponse>('/trenches/ratings', {
          params: { season, ...params },
        })
        return data
      } catch (err) {
        return emptyOn404(err, {
          season,
          through_week: throughWeek ?? null,
          scale: '',
          units: [],
        } as LineUnitsResponse)
      }
    },
    staleTime: 1000 * 60 * 15,
  })
}

/** Both sides of every game in a week, run and pass, rated before kickoff. */
export function useLineMatchups(season: number, week: number) {
  return useQuery({
    queryKey: ['lineMatchups', season, week],
    queryFn: async () => {
      try {
        const { data } = await apiClient.get<LineMatchupsResponse>('/trenches/matchups', {
          params: { season, week },
        })
        return data
      } catch (err) {
        return emptyOn404(err, {
          season,
          week,
          basis: '',
          bands: [],
          matchups: [],
        } as LineMatchupsResponse)
      }
    },
    enabled: week > 0,
    staleTime: 1000 * 60 * 15,
  })
}

/**
 * What each rating band historically delivered. Shown next to the ratings on
 * purpose: a 0–100 number invites more confidence than this one has earned,
 * and the only honest answer to "is 87 a lot?" is what the eighties did.
 */
export function useLineBacktest(seasons = 3) {
  return useQuery({
    queryKey: ['lineBacktest', seasons],
    queryFn: async () => {
      try {
        const { data } = await apiClient.get<LineBacktestResponse>('/trenches/backtest', {
          params: { seasons },
        })
        return data
      } catch (err) {
        return emptyOn404(err, null as unknown as LineBacktestResponse)
      }
    },
    staleTime: 1000 * 60 * 60,
  })
}
