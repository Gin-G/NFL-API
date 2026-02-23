import { useQuery } from '@tanstack/react-query'
import { apiClient } from './client'
import type { SchedulesResponse, WeeklyScheduleResponse } from './types'

export function useSchedules(season: number, week?: number, team?: string) {
  return useQuery({
    queryKey: ['schedules', season, week, team],
    queryFn: async () => {
      const params: Record<string, string | number> = { season }
      if (week !== undefined) params.week = week
      if (team) params.team = team
      const { data } = await apiClient.get<SchedulesResponse>('/schedules/', { params })
      return data
    },
    staleTime: 1000 * 60 * 5,
  })
}

export function useWeeklySchedule(season: number, week: number) {
  return useQuery({
    queryKey: ['weeklySchedule', season, week],
    queryFn: async () => {
      const { data } = await apiClient.get<WeeklyScheduleResponse>(
        `/schedules/${season}/week/${week}`
      )
      return data
    },
    staleTime: 1000 * 60 * 5,
  })
}
