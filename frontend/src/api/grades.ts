import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import { apiClient } from './client'
import type {
  TeamRatingsResponse,
  TeamRatingResponse,
  PlayerGradesListResponse,
} from './types'

/**
 * The grading endpoints return 404 when a season has nothing to grade (e.g.
 * player grades before a season kicks off). Translate that into the same
 * `status: 'no_data'` shape the rest of the API uses, so pages can show an
 * explanation instead of a generic error card.
 */
function noDataOn404<T>(err: unknown, fallback: T): T {
  if (axios.isAxiosError(err) && err.response?.status === 404) return fallback
  throw err
}

/** Team offense/defense grades for a season, optionally as of a given week. */
export function useTeamRatings(season: number, throughWeek?: number) {
  return useQuery({
    queryKey: ['teamRatings', season, throughWeek],
    queryFn: async () => {
      const params: Record<string, number> = {}
      if (throughWeek !== undefined) params.through_week = throughWeek
      try {
        const { data } = await apiClient.get<TeamRatingsResponse>(`/ratings/${season}`, { params })
        return data
      } catch (err) {
        return noDataOn404(err, {
          status: 'no_data',
          season,
          through_week: throughWeek ?? null,
          total_teams: 0,
          data: [],
          message: `No schedule or score data available for ${season}.`,
        } as TeamRatingsResponse)
      }
    },
    staleTime: 1000 * 60 * 15,
  })
}

/** Grades for a single team. */
export function useTeamRating(season: number, team: string, throughWeek?: number) {
  return useQuery({
    queryKey: ['teamRating', season, team, throughWeek],
    queryFn: async () => {
      const params: Record<string, number> = {}
      if (throughWeek !== undefined) params.through_week = throughWeek
      try {
        const { data } = await apiClient.get<TeamRatingResponse>(`/ratings/${season}/${team}`, {
          params,
        })
        return data
      } catch (err) {
        return noDataOn404(err, {
          status: 'no_data',
          season,
          data: null,
          message: `No rating for ${team} in ${season}.`,
        } as TeamRatingResponse)
      }
    },
    enabled: !!team,
    staleTime: 1000 * 60 * 15,
  })
}

/** Player grade leaderboard — top `limit` by grade, optionally one position. */
export function usePlayerGradeLeaderboard(
  season: number,
  position?: string,
  throughWeek?: number,
  limit = 100,
) {
  return useQuery({
    queryKey: ['playerGradeLeaderboard', season, position, throughWeek, limit],
    queryFn: async () => {
      const params: Record<string, string | number> = { limit }
      if (position) params.position = position
      if (throughWeek !== undefined) params.through_week = throughWeek
      try {
        const { data } = await apiClient.get<PlayerGradesListResponse>(
          `/player-grades/${season}`,
          { params },
        )
        return data
      } catch (err) {
        return noDataOn404(err, {
          status: 'no_data',
          season,
          through_week: throughWeek ?? null,
          total: 0,
          data: [],
          message: `No player stats to grade for ${season} yet.`,
        } as PlayerGradesListResponse)
      }
    },
    staleTime: 1000 * 60 * 15,
  })
}
