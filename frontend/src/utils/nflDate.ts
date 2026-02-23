/**
 * NFL season date utilities.
 *
 * The NFL season is named for the year it *starts* (September).
 * January and February are the playoffs/Super Bowl window for the
 * previous calendar year's season.  March–August is the off-season.
 *
 * Examples at runtime date 2026-02-23:
 *   getCurrentNFLSeason() → 2025
 *   getCurrentNFLWeek(2025) → 18  (season finished, show last regular-season week)
 *   getAvailableSeasons()  → [2025, 2024, 2023, 2022, 2021, 2020]
 */

/**
 * Return the season year of the current or most recently completed
 * NFL season.  Works correctly in any future year.
 */
export function getCurrentNFLSeason(date: Date = new Date()): number {
  const month = date.getMonth() + 1 // 1-12
  const year = date.getFullYear()
  // Sep–Dec: new season is underway — season year = current calendar year
  // Jan–Aug: we are in the off-season or playoff tail of the previous season
  return month >= 9 ? year : year - 1
}

/**
 * Return an appropriate NFL week number to display for the given season.
 *
 * - Past seasons (season < current) → 18 (last regular-season week)
 * - Current season, Jan–Feb          → 18 (playoffs / Super Bowl window)
 * - Current season, Sep–Dec          → approximate week based on elapsed days
 * - Season hasn't started yet        → 1
 */
export function getCurrentNFLWeek(season: number, date: Date = new Date()): number {
  const currentSeason = getCurrentNFLSeason(date)

  // Past season — show the final regular-season week
  if (season < currentSeason) return 18

  const month = date.getMonth() + 1
  const year = date.getFullYear()

  // Jan / Feb belong to the playoff window of the previous year's season.
  // By the time we reach this branch, season === currentSeason, meaning
  // we are in the Jan/Feb tail of that season — it is complete or nearly so.
  if (month <= 2) return 18

  // Before the season has started (before September of the season year)
  if (year < season || (year === season && month < 9)) return 1

  // Approximate: NFL week 1 kicks off around September 5 of the season year
  const seasonStart = new Date(season, 8, 5) // month index 8 = September
  const msPerWeek = 7 * 24 * 60 * 60 * 1000
  const elapsed = date.getTime() - seasonStart.getTime()
  if (elapsed < 0) return 1
  return Math.min(18, Math.max(1, Math.floor(elapsed / msPerWeek) + 1))
}

/**
 * Return a descending list of available seasons from the current season
 * back to firstSeason (default 2020).
 */
export function getAvailableSeasons(firstSeason = 2020): number[] {
  const current = getCurrentNFLSeason()
  return Array.from({ length: current - firstSeason + 1 }, (_, i) => current - i)
}
