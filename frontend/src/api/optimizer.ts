import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'
import type {
  ExposureResponse,
  LineupPerformance,
  LineupResponse,
  SlateCoverage,
  SlateList,
} from './types'

/** What the board holds for a week, before any CSV is uploaded. */
export function useSlateCoverage(season: number, week?: number) {
  return useQuery({
    queryKey: ['slateCoverage', season, week],
    queryFn: async () => {
      const params: Record<string, number> = { season }
      if (week !== undefined) params.week = week
      const { data } = await apiClient.get<SlateCoverage>('/optimizer/slate-coverage', { params })
      return data
    },
    staleTime: 1000 * 60 * 10,
  })
}

/** The slate CSVs on file. FanDuel publishes pricing nowhere fetchable, so the
 *  upload is the one manual step — and it only has to happen once per slate. */
export function useSavedSlates(season: number, week?: number) {
  return useQuery({
    queryKey: ['savedSlates', season, week],
    queryFn: async () => {
      const params: Record<string, number> = { season }
      if (week !== undefined) params.week = week
      const { data } = await apiClient.get<SlateList>('/optimizer/slates', { params })
      return data
    },
    staleTime: 1000 * 60,
  })
}

export function useForgetSlate() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (slateId: number) => {
      const { data } = await apiClient.delete(`/optimizer/slates/${slateId}`)
      return data
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['savedSlates'] }),
  })
}

/** How the lineups this optimizer built actually scored. */
export function useLineupPerformance(season: number, week?: number) {
  return useQuery({
    queryKey: ['lineupPerformance', season, week],
    queryFn: async () => {
      const params: Record<string, number> = { season }
      if (week !== undefined) params.week = week
      const { data } = await apiClient.get<LineupPerformance>('/optimizer/performance', { params })
      return data
    },
    staleTime: 1000 * 60 * 5,
  })
}

/** Player exposure pooled across every recorded build. The per-build cap binds
 *  inside one build only, so this is the only place concentration shows up. */
export function useLineupExposure(season: number, week?: number) {
  return useQuery({
    queryKey: ['lineupExposure', season, week],
    queryFn: async () => {
      const params: Record<string, number> = { season }
      if (week !== undefined) params.week = week
      const { data } = await apiClient.get<ExposureResponse>('/optimizer/exposure', { params })
      return data
    },
    staleTime: 1000 * 60 * 5,
  })
}

export interface LineupRequest {
  /** Omitted to build from the week's saved slate. */
  file?: File | null
  season: number
  week: number
  num_lineups: number
  objective: string
  salary_cap: number
  max_usage_percentage: number
  exclude: string
  /** all | main | sunday | primetime — the export is always Thu-Mon. */
  slate: string
  /** At most this many of the nine from one team, the defense included.
   *  4 is FanDuel's own limit. */
  max_per_team: number
  /** Require this many WR/TE from the quarterback's team. */
  stack_qb: number
  /** False keeps a running back off his own quarterback. */
  rb_with_qb: boolean
  /** Points charged per percent of projected ownership, tilting the build
   *  toward players the field has not found. 0 ignores the field. It buys that
   *  tilt with expected points, so it is a tournament setting: on the 2026
   *  week 6 main slate a charge of 0.10 took the best five from 125.1 actual
   *  down to 109.3. */
  ownership_weight?: number
  /** A specific saved slate, for a Showdown file alongside a classic list. */
  slate_id?: number
}

/** Build lineups: from an uploaded FanDuel slate CSV, or from the saved one. */
/** Download a build as the CSV FanDuel's bulk entry accepts.
 *
 * `contestId` rewrites the contest half of every FanDuel id. An id is
 * "<contest>-<player>", so lineups priced from one contest's player list are
 * rejected wholesale by another — the number in the file name of the list or
 * upload template you downloaded is the one to pass. */
export async function downloadFanduelUpload(buildId: number, contestId?: string) {
  const { data, headers } = await apiClient.get(
    `/optimizer/builds/${buildId}/upload.csv`,
    { params: contestId ? { contest_id: contestId } : {}, responseType: 'blob' },
  )
  const match = /filename="([^"]+)"/.exec(String(headers['content-disposition'] ?? ''))
  const url = URL.createObjectURL(data as Blob)
  const link = document.createElement('a')
  link.href = url
  link.download = match?.[1] ?? `fanduel-build${buildId}.csv`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

export function useBuildLineups() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (req: LineupRequest) => {
      const form = new FormData()
      if (req.file) form.append('file', req.file)
      Object.entries(req).forEach(([key, value]) => {
        if (key !== 'file' && value !== undefined && value !== null) {
          form.append(key, String(value))
        }
      })
      const { data } = await apiClient.post<LineupResponse>('/optimizer/lineups', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      return data
    },
    // The slate is saved BEFORE the solve, so a build that fits nothing still
    // leaves the file on file — hence onSettled, not onSuccess. Nothing is
    // recorded for a failed build, so the record only moves on success.
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['savedSlates'] }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['lineupPerformance'] })
      queryClient.invalidateQueries({ queryKey: ['lineupExposure'] })
    },
  })
}
