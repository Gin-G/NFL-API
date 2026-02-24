import { useQuery } from '@tanstack/react-query'
import { apiClient } from './client'
import { getCurrentNFLSeason } from '../utils/nflDate'
import type {
  CoachesResponse,
  CoachAnalysisResponse,
  CoachGradesResponse,
  CoachCompareResponse,
  TeamStaffResponse,
} from './types'

function yearsParams(years: number[]) {
  const p = new URLSearchParams()
  years.forEach((y) => p.append('years', String(y)))
  return p.toString()
}

export function useCoaches(years: number[] = [getCurrentNFLSeason()], season?: number) {
  return useQuery({
    queryKey: ['coaches', years, season],
    queryFn: async () => {
      const base = `/coaches/?${yearsParams(years)}`
      const url = season !== undefined ? `${base}&season=${season}` : base
      const { data } = await apiClient.get<CoachesResponse>(url)
      return data
    },
    staleTime: 1000 * 60 * 10,
  })
}

export function useCoachAnalysis(coachName: string, years: number[] = [getCurrentNFLSeason()], season?: number) {
  return useQuery({
    queryKey: ['coachAnalysis', coachName, years, season],
    queryFn: async () => {
      const base = `/coaches/${encodeURIComponent(coachName)}/analysis?${yearsParams(years)}`
      const url = season !== undefined ? `${base}&season=${season}` : base
      const { data } = await apiClient.get<CoachAnalysisResponse>(url)
      return data
    },
    enabled: !!coachName,
    staleTime: 1000 * 60 * 10,
  })
}

export function useCoachGrades(coachName: string, years: number[] = [getCurrentNFLSeason()], season?: number) {
  return useQuery({
    queryKey: ['coachGrades', coachName, years, season],
    queryFn: async () => {
      const base = `/coaches/${encodeURIComponent(coachName)}/grades?${yearsParams(years)}`
      const url = season !== undefined ? `${base}&season=${season}` : base
      const { data } = await apiClient.get<CoachGradesResponse>(url)
      return data
    },
    enabled: !!coachName,
    staleTime: 1000 * 60 * 10,
  })
}

/** Fetch HC/OC/DC/STC for all teams (or one team via teamAbbr). */
export function useTeamStaff(teamAbbr?: string) {
  return useQuery({
    queryKey: ['teamStaff', teamAbbr ?? 'all'],
    queryFn: async () => {
      const url = teamAbbr
        ? `/coaches/staff?team=${encodeURIComponent(teamAbbr)}`
        : '/coaches/staff'
      const { data } = await apiClient.get<TeamStaffResponse>(url)
      return data
    },
    // Staff rarely changes — cache for 24 hours
    staleTime: 1000 * 60 * 60 * 24,
  })
}

export function useCoachCompare(coachNames: string[], years: number[] = [getCurrentNFLSeason()], season?: number) {
  return useQuery({
    queryKey: ['coachCompare', coachNames, years, season],
    queryFn: async () => {
      const params = new URLSearchParams()
      coachNames.forEach((n) => params.append('coach_names', n))
      years.forEach((y) => params.append('years', String(y)))
      if (season !== undefined) params.set('season', String(season))
      const { data } = await apiClient.post<CoachCompareResponse>(
        `/coaches/compare?${params}`
      )
      return data
    },
    enabled: coachNames.length >= 2,
    staleTime: 1000 * 60 * 10,
  })
}
