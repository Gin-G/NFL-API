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
  message?: string
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
  is_active: boolean
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

// ─── Coaching Staff (SportRadar) ──────────────────────────────────────────────

export interface StaffMember {
  team_abbr: string
  team_name: string
  head_coach: string | null
  offensive_coordinator: string | null
  defensive_coordinator: string | null
  special_teams_coordinator: string | null
}

export interface TeamStaffResponse {
  status: string
  configured: boolean
  total_teams?: number
  message?: string
  data: StaffMember[]
}

// ─── Coach Breakdown (PBP analytics) ─────────────────────────────────────────

export interface FormationEntry {
  plays: number
  pct_of_plays: number
  pass_rate: number | null
  avg_epa: number | null
  success_rate: number | null
}

export interface PersonnelEntry {
  plays: number
  pct_of_plays: number
  pass_rate: number | null
  avg_epa: number | null
  success_rate: number | null
  avg_yards: number | null
}

export interface RunLocationEntry {
  plays: number
  avg_yards: number | null
  avg_epa: number | null
  success_rate: number | null
}

export interface PassDirectionEntry {
  plays: number
  avg_epa: number | null
  success_rate: number | null
  avg_air_yards: number | null
}

export interface DefPersonnelEntry {
  plays: number
  pct_of_plays: number
  avg_epa_allowed: number | null
  opponent_success_rate: number | null
}

export interface FormationBreakdown {
  shotgun_rate: number | null
  no_huddle_rate: number | null
  play_action_rate: number | null
  formation_breakdown: Record<string, FormationEntry>
}

export interface RunScheme {
  total_runs: number
  inside_rate: number | null
  outside_rate: number | null
  by_location: Record<string, RunLocationEntry>
}

export interface PassDetail {
  avg_air_yards: number | null
  deep_pass_rate: number | null
  screen_rate: number | null
  intermediate_rate: number | null
  scramble_rate: number | null
  dropback_epa: number | null
  avg_yac: number | null
  by_direction: Record<string, PassDirectionEntry>
}

export interface DefenseScheme {
  blitz_rate: number | null
  avg_defenders_in_box: number | null
  blitz_epa_allowed: number | null
  non_blitz_epa_allowed: number | null
  qb_hit_rate: number | null
  sack_rate: number | null
  personnel_breakdown: Record<string, DefPersonnelEntry>
}

export interface SamplePlay {
  desc: string | null
  yards_gained: number | null
  epa: number | null
  week: number | null
  game_id: string | null
}

export interface BreakdownOffense {
  total_plays: number
  pass_rate: number | null
  pass_rate_by_down: Record<string, number | null>
  avg_epa_per_play: number | null
  red_zone_pass_rate: number | null
  fourth_down_attempts: number
  fourth_down_conversion_rate: number | null
  two_point_attempts: number
  two_point_success_rate: number | null
  third_down_conversion_rate: number | null
  formation: FormationBreakdown
  personnel: Record<string, PersonnelEntry>
  run_scheme: RunScheme
  passing: PassDetail
  fourth_down_sample: SamplePlay[]
  third_down_sample: SamplePlay[]
}

export interface BreakdownDefense {
  total_plays: number
  avg_epa_allowed_per_play: number | null
  third_down_stop_rate: number | null
  red_zone_td_rate_allowed: number | null
  sack_rate: number | null
  scheme: DefenseScheme
}

export interface CoachBreakdownEntry {
  season: number
  team: string
  offense: BreakdownOffense
  defense: BreakdownDefense
  strengths: string[]
  weaknesses: string[]
  tendencies: string[]
}

export interface CoachBreakdownResponse {
  status: string
  coach: string
  season: number | null
  data: CoachBreakdownEntry[]
  message?: string
}

// ─── Player Breakdown (PBP analytics) ────────────────────────────────────────

export interface DirectionEntry {
  attempts?: number
  carries?: number
  targets?: number
  epa: number | null
  pct: number | null
}

export interface QBPassingMetrics {
  player_name: string
  team: string
  dropbacks: number
  completions: number
  comp_pct: number | null
  epa_per_dropback: number | null
  pass_yards: number
  pass_tds: number
  interceptions: number
  sacks_taken: number
  sack_rate: number | null
  scramble_rate: number | null
  avg_air_yards: number | null
  deep_rate: number | null
  screen_rate: number | null
  avg_yac: number | null
  third_down_conv_rate: number | null
  rz_epa: number | null
  rz_dropbacks: number
  play_action_rate?: number | null
  pass_direction: {
    left: DirectionEntry
    middle: DirectionEntry
    right: DirectionEntry
  }
}

export interface RushingMetrics {
  player_name: string
  team: string
  carries: number
  rush_yards: number
  yards_per_carry: number | null
  epa_per_carry: number | null
  success_rate: number | null
  rush_tds: number
  explosive_rate: number | null
  negative_rate: number | null
  rz_carries: number
  rz_epa: number | null
  inside_rate: number | null
  outside_rate: number | null
  by_direction: {
    left: DirectionEntry
    middle: DirectionEntry
    right: DirectionEntry
  }
}

export interface ReceivingMetrics {
  player_name: string
  team: string
  targets: number
  receptions: number
  catch_rate: number | null
  rec_yards: number
  yards_per_target: number | null
  yards_per_reception: number | null
  epa_per_target: number | null
  avg_air_yards: number | null
  avg_yac: number | null
  rec_tds: number
  deep_target_rate: number | null
  short_target_rate: number | null
  rz_targets: number
  rz_tds: number
  by_direction: {
    left: DirectionEntry
    middle: DirectionEntry
    right: DirectionEntry
  }
}

export interface DefensiveMetrics {
  team: string
  sacks: number
  qb_hits: number
  pressures: number
  solo_tackles: number
  assist_tackles: number
  total_tackles: number
  interceptions: number
  passes_defended: number
  forced_fumbles: number
  fumble_recoveries: number
  tackles_for_loss: number
}

export interface PlayerBreakdownEntry {
  season: number
  team: string
  position: string
  player_name: string
  computed_at: string | null
  passing?: QBPassingMetrics
  rushing?: RushingMetrics
  receiving?: ReceivingMetrics
  defense?: DefensiveMetrics
}

export interface PlayerBreakdownResponse {
  status: string
  player_id: string
  season: number
  data: PlayerBreakdownEntry[]
  message?: string
  pct_complete?: number
  current_season?: number
}

// ─── Chat ─────────────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatResponse {
  response: string
  history: ChatMessage[]
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
