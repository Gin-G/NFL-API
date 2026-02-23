// ─── Teams ────────────────────────────────────────────────────────────────────

export interface Team {
  team_abbr: string
  team_name: string
  team_nick: string
  team_conf: string
  team_division: string
  team_color: string | null
  team_color2: string | null
  team_logo_espn: string | null
  team_logo_wikipedia: string | null
  team_wordmark: string | null
}

export interface TeamsResponse {
  status: string
  total_teams: number
  data: Team[]
}

export interface TeamDetailResponse {
  status: string
  data: Team | null
}

// ─── Schedules ────────────────────────────────────────────────────────────────

export interface Game {
  game_id: string | null
  season: number
  week: number
  game_type: string | null
  gameday: string | null
  weekday: string | null
  gametime: string | null
  away_team: string
  home_team: string
  away_score: number | null
  home_score: number | null
  location: string | null
  result: number | null
  overtime: number | null
  away_qb_name: string | null
  home_qb_name: string | null
  away_coach: string | null
  home_coach: string | null
  stadium: string | null
  spread_line: number | null
  total_line: number | null
  div_game: number | null
  roof: string | null
  surface: string | null
}

export interface SchedulesResponse {
  status: string
  season: number
  total_games: number
  data: Game[]
}

export interface WeeklyScheduleResponse {
  status: string
  season: number
  week: number
  games: Game[]
}

// ─── Players ─────────────────────────────────────────────────────────────────

export interface RosterPlayer {
  player_id: string
  player_name: string
  player_display_name: string | null
  position: string
  team: string
  jersey_number: number | null
  height: string | null
  weight: number | null
  college: string | null
  rookie_year: number | null
  headshot_url: string | null
  season: number
  week: number
}

export interface PlayerStat {
  player_id: string
  player_name: string
  player_display_name: string | null
  position: string
  recent_team: string
  season: number
  week: number
  completions: number | null
  attempts: number | null
  passing_yards: number | null
  passing_tds: number | null
  interceptions: number | null
  carries: number | null
  rushing_yards: number | null
  rushing_tds: number | null
  receptions: number | null
  targets: number | null
  receiving_yards: number | null
  receiving_tds: number | null
  fantasy_points: number | null
  fantasy_points_ppr: number | null
}

export interface PlayerGrade {
  player_name: string
  position: string
  numeric_grade: number | null
  letter_grade: string | null
  grade_category: string | null
  [key: string]: unknown
}

export interface RostersResponse {
  status: string
  season: number
  total_players: number
  data: RosterPlayer[]
}

export interface PlayerStatsResponse {
  status: string
  season: number
  total_records: number
  data: PlayerStat[]
}

export interface PlayerDetailResponse {
  status: string
  player_info: Partial<RosterPlayer>
  season_stats: PlayerStat[]
  games_played: number
}

export interface PlayerGradesResponse {
  status: string
  total_players?: number
  message?: string
  data: PlayerGrade[]
}

// ─── Coaches ─────────────────────────────────────────────────────────────────

export interface CoachSeason {
  season: number
  teams: string[]
  record: string
  wins: number
  losses: number
  win_percentage: number
  games_coached: number
}

export interface Coach {
  name: string
  seasons: CoachSeason[]
}

export interface CoachesResponse {
  status: string
  total_coaches: number
  data: Coach[]
}

export interface OffensiveAnalysis {
  team: string
  season: number
  qb_avg_grade: number | null
  rb_avg_grade: number | null
  wr_te_avg_grade: number | null
}

export interface DefensiveAnalysis {
  team: string
  season: number
  defense_avg_grade: number | null
  overall_avg_grade: number | null
  roster_tier: string | null
}

export interface CoachAnalysisResponse {
  status: string
  coach: string
  season: number | null
  seasons: CoachSeason[]
  offensive_analysis: Record<string, OffensiveAnalysis>
  defensive_analysis: Record<string, DefensiveAnalysis>
}

export interface CoachGradeEntry {
  season: number
  teams: string[]
  record: string
  win_percentage: number
  win_score: number
  win_letter_grade: string
  roster_quality_score?: number
  roster_quality_grade?: string
}

export interface CoachGradesResponse {
  status: string
  coach: string
  season: number | null
  grades: CoachGradeEntry[] | null
}

export interface CoachCompareResponse {
  status: string
  coaches: string[]
  season: number | null
  comparison_matrix: {
    win_percentage: Record<string, { score: number; letter_grade: string }>
  }
  detailed_data: Record<
    string,
    {
      seasons: CoachSeason[]
      overall_record: string
      overall_win_percentage: number
      win_score: number
      win_letter_grade: string
    }
  >
}

// ─── Health ───────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string
  timestamp: string
  systems: {
    player_grading: boolean
    coaching_analytics: boolean
  }
}
