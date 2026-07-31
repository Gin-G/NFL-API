import type { SeasonProjectionEntry, WeeklyProjection } from '../api/types'

/**
 * Which projected stats are worth showing for each position.
 *
 * The projections come from a single multi-output model, so every row carries
 * every stat — a WR has a few projected passing yards, a QB a few receptions.
 * That cross-position leakage is noise, not signal, so each position only
 * renders the stats that actually describe its role.
 */

/** Season-totals field names (the season endpoint flattens INTs to `interceptions`). */
export type SeasonStatKey = Extract<
  keyof SeasonProjectionEntry,
  | 'passing_yards'
  | 'passing_tds'
  | 'interceptions'
  | 'rushing_yards'
  | 'rushing_tds'
  | 'receiving_yards'
  | 'receptions'
  | 'receiving_tds'
>

/** Matching field names on a weekly projection row. */
export type WeeklyStatKey = Extract<
  keyof WeeklyProjection,
  | 'passing_yards'
  | 'passing_tds'
  | 'passing_interceptions'
  | 'rushing_yards'
  | 'rushing_tds'
  | 'receiving_yards'
  | 'receptions'
  | 'receiving_tds'
>

export interface ProjectionStat {
  key: SeasonStatKey
  weeklyKey: WeeklyStatKey
  label: string
  hint: string
  /** Yards read as whole numbers; TDs and receptions keep one decimal. */
  decimals: 0 | 1
}

const PASS_YDS: ProjectionStat = {
  key: 'passing_yards',
  weeklyKey: 'passing_yards',
  label: 'Pass Yds',
  hint: 'Projected passing yards',
  decimals: 0,
}
const PASS_TDS: ProjectionStat = {
  key: 'passing_tds',
  weeklyKey: 'passing_tds',
  label: 'Pass TD',
  hint: 'Projected passing touchdowns',
  decimals: 1,
}
const INTS: ProjectionStat = {
  key: 'interceptions',
  weeklyKey: 'passing_interceptions',
  label: 'INT',
  hint: 'Projected interceptions thrown',
  decimals: 1,
}
const RUSH_YDS: ProjectionStat = {
  key: 'rushing_yards',
  weeklyKey: 'rushing_yards',
  label: 'Rush Yds',
  hint: 'Projected rushing yards',
  decimals: 0,
}
const RUSH_TDS: ProjectionStat = {
  key: 'rushing_tds',
  weeklyKey: 'rushing_tds',
  label: 'Rush TD',
  hint: 'Projected rushing touchdowns',
  decimals: 1,
}
const REC: ProjectionStat = {
  key: 'receptions',
  weeklyKey: 'receptions',
  label: 'Rec',
  hint: 'Projected receptions',
  decimals: 1,
}
const REC_YDS: ProjectionStat = {
  key: 'receiving_yards',
  weeklyKey: 'receiving_yards',
  label: 'Rec Yds',
  hint: 'Projected receiving yards',
  decimals: 0,
}
const REC_TDS: ProjectionStat = {
  key: 'receiving_tds',
  weeklyKey: 'receiving_tds',
  label: 'Rec TD',
  hint: 'Projected receiving touchdowns',
  decimals: 1,
}

export const POSITION_STATS: Record<string, ProjectionStat[]> = {
  QB: [PASS_YDS, PASS_TDS, INTS, RUSH_YDS, RUSH_TDS],
  RB: [RUSH_YDS, RUSH_TDS, REC, REC_YDS, REC_TDS],
  WR: [REC, REC_YDS, REC_TDS, RUSH_YDS],
  TE: [REC, REC_YDS, REC_TDS, RUSH_YDS],
}

/**
 * Stat columns for a position. Unknown or mixed positions get none: with every
 * position in one table there is no honest set of stat columns to show.
 */
export function statsForPosition(position?: string | null): ProjectionStat[] {
  return POSITION_STATS[(position ?? '').toUpperCase()] ?? []
}

/** A projected stat, or an em dash when the model produced nothing. */
export function fmtStat(value: number | null | undefined, decimals: 0 | 1): string {
  if (value == null) return '—'
  return value.toFixed(decimals)
}

/** Fantasy points — always one decimal. */
export function fmtPoints(value: number | null | undefined): string {
  if (value == null) return '—'
  return value.toFixed(1)
}

/** Floor–ceiling band. Whole points: the spread is the message, not the digits. */
export function fmtRange(low: number | null | undefined, high: number | null | undefined): string {
  if (low == null || high == null) return '—'
  return `${Math.round(low)}–${Math.round(high)}`
}
