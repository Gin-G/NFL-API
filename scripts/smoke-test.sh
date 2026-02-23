#!/usr/bin/env bash
# NFL Analytics API + Dashboard smoke test
#
# Usage:
#   ./scripts/smoke-test.sh                        # defaults to http://localhost:8000
#   ./scripts/smoke-test.sh https://nfl-api.nickknows.net
#
# Exits 0 if all checks pass, 1 if any fail.

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

# Prefer system curl over conda/brew shims that may have libssl warnings
CURL=$(which -a curl 2>/dev/null | grep -v miniforge | grep -v homebrew | head -1 || echo curl)

# ─── Helpers ──────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0
declare -a ERRORS=()

pass() { echo -e "  ${GREEN}PASS${NC}  $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}FAIL${NC}  $1"; FAIL=$((FAIL + 1)); ERRORS+=("$1"); }
skip() { echo -e "  ${YELLOW}SKIP${NC}  $1"; SKIP=$((SKIP + 1)); }

section() { echo -e "\n${BOLD}── $1 ${NC}$(printf '─%.0s' {1..50})"; }

# curl wrapper: returns body on 2xx, prints error and returns "" on failure
get() {
  local url="$BASE_URL$1"
  local timeout="${2:-30}"
  $CURL -sf --max-time "$timeout" "$url" 2>/dev/null || true
}

# Returns HTTP status code only
status_code() {
  $CURL -o /dev/null -sw '%{http_code}' --max-time "${2:-15}" "$BASE_URL$1" 2>/dev/null || echo "000"
}

# Validate JSON field with a python one-liner; prints nothing on success
check_json() {
  local body="$1"
  local expr="$2"   # python expression that raises if wrong
  local label="$3"
  if echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); $expr" 2>/dev/null; then
    pass "$label"
  else
    fail "$label"
  fi
}

echo -e "${BOLD}NFL Analytics smoke test${NC}"
echo "Target: $BASE_URL"
echo "$(date)"

# ─── Frontend ─────────────────────────────────────────────────────────────────

section "Frontend HTML"

HTML=$(get "/dashboard/")
if [ -z "$HTML" ]; then
  fail "GET /dashboard/ — request failed or returned empty body"
else
  echo "$HTML" | grep -q "<!DOCTYPE html>" \
    && pass "GET /dashboard/ — HTTP 200, HTML document returned" \
    || fail "GET /dashboard/ — missing DOCTYPE"

  echo "$HTML" | grep -q '<div id="root">' \
    && pass "GET /dashboard/ — contains <div id=\"root\">" \
    || fail "GET /dashboard/ — missing <div id=\"root\">"

  echo "$HTML" | grep -q "NFL Analytics Dashboard" \
    && pass "GET /dashboard/ — <title> is 'NFL Analytics Dashboard'" \
    || fail "GET /dashboard/ — wrong or missing <title>"

  echo "$HTML" | grep -qE 'src="[^"]+\.js"' \
    && pass "GET /dashboard/ — HTML references a JS bundle" \
    || fail "GET /dashboard/ — no JS bundle reference in HTML"

  # Extract JS bundle path and verify it loads
  JS_PATH=$(echo "$HTML" | grep -oP 'src="/dashboard/assets/[^"]+\.js"' | head -1 | grep -oP '"/[^"]+"' | tr -d '"')
  if [ -n "$JS_PATH" ]; then
    JS_STATUS=$(status_code "$JS_PATH")
    [ "$JS_STATUS" = "200" ] \
      && pass "GET $JS_PATH — JS bundle served (HTTP 200)" \
      || fail "GET $JS_PATH — JS bundle returned HTTP $JS_STATUS"

    # Check Content-Type header
    JS_CT=$($CURL -sI --max-time 10 "$BASE_URL$JS_PATH" 2>/dev/null | grep -i "content-type" | head -1)
    echo "$JS_CT" | grep -qi "javascript" \
      && pass "JS bundle — Content-Type contains 'javascript'" \
      || fail "JS bundle — unexpected Content-Type: $JS_CT"
  else
    skip "JS bundle URL check — could not parse URL from HTML"
  fi

  # CSS bundle
  CSS_PATH=$(echo "$HTML" | grep -oP 'href="/dashboard/assets/[^"]+\.css"' | head -1 | grep -oP '"/[^"]+"' | tr -d '"')
  if [ -n "$CSS_PATH" ]; then
    CSS_STATUS=$(status_code "$CSS_PATH")
    [ "$CSS_STATUS" = "200" ] \
      && pass "GET $CSS_PATH — CSS bundle served (HTTP 200)" \
      || fail "GET $CSS_PATH — CSS bundle returned HTTP $CSS_STATUS"
  else
    skip "CSS bundle check — no CSS link found (may be inlined)"
  fi
fi

# SPA deep-link fallback (html=True in StaticFiles)
section "Frontend SPA deep links"

for SUBPATH in /dashboard/teams /dashboard/players /dashboard/coaches /dashboard/schedule; do
  SUB_HTML=$(get "$SUBPATH")
  if [ -z "$SUB_HTML" ]; then
    fail "GET $SUBPATH — request failed"
  else
    echo "$SUB_HTML" | grep -q '<div id="root">' \
      && pass "GET $SUBPATH — SPA fallback serves root div" \
      || fail "GET $SUBPATH — missing root div (SPA fallback broken)"
  fi
done

# Unknown asset must 404, not fallback to index.html
UNKNOWN_STATUS=$(status_code "/dashboard/assets/nonexistent-xyz.js")
[ "$UNKNOWN_STATUS" = "404" ] \
  && pass "GET /dashboard/assets/nonexistent-xyz.js — 404 (correct)" \
  || fail "GET /dashboard/assets/nonexistent-xyz.js — expected 404, got $UNKNOWN_STATUS"

# ─── API: Core ────────────────────────────────────────────────────────────────

section "API: Core"

ROOT=$(get "/")
if [ -z "$ROOT" ]; then
  fail "GET / — request failed"
else
  check_json "$ROOT" "assert d['status'] == 'running'" "GET / — status is 'running'"
  check_json "$ROOT" "assert d['version'] == '1.0.0'" "GET / — version is '1.0.0'"
  check_json "$ROOT" \
    "assert all(k in d['endpoints'] for k in ('teams','schedules','players','coaches','docs','health'))" \
    "GET / — all endpoint keys present"
fi

HEALTH=$(get "/health")
if [ -z "$HEALTH" ]; then
  fail "GET /health — request failed"
else
  check_json "$HEALTH" "assert d['status'] == 'healthy'" "GET /health — status is 'healthy'"
  check_json "$HEALTH" "assert 'timestamp' in d" "GET /health — timestamp present"
  check_json "$HEALTH" \
    "assert 'player_grading' in d['systems'] and 'coaching_analytics' in d['systems']" \
    "GET /health — systems keys present"
fi

# ─── API: Teams ───────────────────────────────────────────────────────────────

section "API: Teams"

TEAMS=$(get "/teams/")
if [ -z "$TEAMS" ]; then
  fail "GET /teams/ — request failed"
else
  check_json "$TEAMS" "assert d['status'] == 'success'" "GET /teams/ — status success"
  check_json "$TEAMS" "assert d['total_teams'] > 0" "GET /teams/ — total_teams > 0"
  check_json "$TEAMS" "assert isinstance(d['data'], list) and len(d['data']) > 0" \
    "GET /teams/ — data is a non-empty list"
  check_json "$TEAMS" \
    "t=d['data'][0]; assert all(k in t for k in ('team_abbr','team_name','team_conf','team_division'))" \
    "GET /teams/ — team objects have required fields"
  # Verify at least one team per conference
  check_json "$TEAMS" \
    "confs={t['team_conf'] for t in d['data'] if t.get('team_conf')}; assert 'AFC' in confs and 'NFC' in confs" \
    "GET /teams/ — both AFC and NFC teams present"
fi

TEAM_KC=$(get "/teams/KC")
if [ -z "$TEAM_KC" ]; then
  fail "GET /teams/KC — request failed"
else
  check_json "$TEAM_KC" "assert d['data']['team_abbr'] == 'KC'" "GET /teams/KC — team_abbr is KC"
  check_json "$TEAM_KC" "assert d['data']['team_conf'] == 'AFC'" "GET /teams/KC — team_conf is AFC"
fi

# Case-insensitive team lookup
TEAM_KC_LOWER=$(get "/teams/kc")
if [ -n "$TEAM_KC_LOWER" ]; then
  check_json "$TEAM_KC_LOWER" "assert d['data']['team_abbr'] == 'KC'" \
    "GET /teams/kc — lowercase abbr normalised to KC"
fi

UNKNOWN_TEAM_STATUS=$(status_code "/teams/ZZZ")
[ "$UNKNOWN_TEAM_STATUS" = "404" ] \
  && pass "GET /teams/ZZZ — 404 for unknown team" \
  || fail "GET /teams/ZZZ — expected 404, got $UNKNOWN_TEAM_STATUS"

# ─── API: Schedules ───────────────────────────────────────────────────────────

section "API: Schedules"

SCHED=$(get "/schedules/?season=2024&week=1" 60)
if [ -z "$SCHED" ]; then
  fail "GET /schedules/?season=2024&week=1 — request failed"
else
  check_json "$SCHED" "assert d['status'] == 'success'" \
    "GET /schedules/?season=2024&week=1 — status success"
  check_json "$SCHED" "assert d['total_games'] > 0" \
    "GET /schedules/?season=2024&week=1 — total_games > 0"
  check_json "$SCHED" \
    "g=d['data'][0]; assert all(k in g for k in ('home_team','away_team','week','season'))" \
    "GET /schedules/ — game objects have required fields"
  check_json "$SCHED" \
    "assert all(g['week'] == 1 for g in d['data'])" \
    "GET /schedules/?week=1 — all games are week 1"
fi

SCHED_WEEK=$(get "/schedules/2024/week/1" 60)
if [ -z "$SCHED_WEEK" ]; then
  fail "GET /schedules/2024/week/1 — request failed"
else
  check_json "$SCHED_WEEK" "assert d['season'] == 2024 and d['week'] == 1" \
    "GET /schedules/2024/week/1 — season and week match path params"
  check_json "$SCHED_WEEK" "assert isinstance(d['games'], list)" \
    "GET /schedules/2024/week/1 — games is a list"
fi

# Team filter
SCHED_KC=$(get "/schedules/?season=2024&team=KC" 60)
if [ -n "$SCHED_KC" ]; then
  check_json "$SCHED_KC" \
    "games=d['data']; assert all(g.get('home_team')=='KC' or g.get('away_team')=='KC' for g in games)" \
    "GET /schedules/?team=KC — all games involve KC"
fi

# ─── API: Players ─────────────────────────────────────────────────────────────

section "API: Players"

ROSTER=$(get "/players/rosters?season=2024&team=KC" 60)
if [ -z "$ROSTER" ]; then
  fail "GET /players/rosters?season=2024&team=KC — request failed"
else
  check_json "$ROSTER" "assert d['status'] == 'success'" \
    "GET /players/rosters — status success"
  check_json "$ROSTER" "assert d['total_players'] > 0" \
    "GET /players/rosters — total_players > 0"
  check_json "$ROSTER" \
    "p=d['data'][0]; assert all(k in p for k in ('player_id','player_name','position','team'))" \
    "GET /players/rosters — player objects have required fields"
  check_json "$ROSTER" \
    "assert all(p['team'] == 'KC' for p in d['data'])" \
    "GET /players/rosters?team=KC — all players are on KC"
fi

STATS=$(get "/players/stats?season=2024&position=QB&week=1" 60)
if [ -z "$STATS" ]; then
  fail "GET /players/stats?season=2024&position=QB&week=1 — request failed"
else
  check_json "$STATS" "assert d['status'] == 'success'" \
    "GET /players/stats — status success"
  check_json "$STATS" "assert d['total_records'] > 0" \
    "GET /players/stats?season=2024&position=QB&week=1 — total_records > 0"
  check_json "$STATS" \
    "p=d['data'][0]; assert all(k in p for k in ('player_id','player_name','season','week'))" \
    "GET /players/stats — stat records have required fields"
  check_json "$STATS" \
    "assert all(p['position'] == 'QB' for p in d['data'])" \
    "GET /players/stats?position=QB — all records are QBs"
fi

MAHOMES=$(get "/players/00-0033873?season=2024" 60)
if [ -z "$MAHOMES" ]; then
  fail "GET /players/00-0033873 — request failed"
else
  check_json "$MAHOMES" "assert d['status'] == 'success'" \
    "GET /players/00-0033873 — status success"
  check_json "$MAHOMES" "assert 'player_info' in d and 'season_stats' in d" \
    "GET /players/00-0033873 — player_info and season_stats present"
  check_json "$MAHOMES" "assert d['games_played'] > 0" \
    "GET /players/00-0033873 — games_played > 0"
fi

# ─── API: Coaches ─────────────────────────────────────────────────────────────

section "API: Coaches"

COACHES=$(get "/coaches/?years=2024" 180)
if [ -z "$COACHES" ]; then
  fail "GET /coaches/?years=2024 — request failed"
else
  check_json "$COACHES" "assert d['status'] == 'success'" \
    "GET /coaches/ — status success"
  check_json "$COACHES" "assert d['total_coaches'] > 0" \
    "GET /coaches/ — total_coaches > 0"
  check_json "$COACHES" \
    "c=d['data'][0]; assert 'name' in c and 'seasons' in c" \
    "GET /coaches/ — coach objects have name and seasons"
  check_json "$COACHES" \
    "s=d['data'][0]['seasons'][0]; assert all(k in s for k in ('season','record','wins','losses','win_percentage','games_coached'))" \
    "GET /coaches/ — season records have all required fields"
fi

ANALYSIS=$(get "/coaches/Andy%20Reid/analysis?years=2024" 180)
if [ -z "$ANALYSIS" ]; then
  fail "GET /coaches/Andy Reid/analysis — request failed"
else
  check_json "$ANALYSIS" "assert d['status'] == 'success'" \
    "GET /coaches/Andy Reid/analysis — status success"
  check_json "$ANALYSIS" "assert d['coach'] == 'Andy Reid'" \
    "GET /coaches/Andy Reid/analysis — coach name matches"
  check_json "$ANALYSIS" "assert 'offensive_analysis' in d and 'defensive_analysis' in d" \
    "GET /coaches/Andy Reid/analysis — has offense and defense sections"
  check_json "$ANALYSIS" "assert isinstance(d['seasons'], list) and len(d['seasons']) > 0" \
    "GET /coaches/Andy Reid/analysis — seasons list is non-empty"
fi

GRADES=$(get "/coaches/Andy%20Reid/grades?years=2024" 180)
if [ -z "$GRADES" ]; then
  fail "GET /coaches/Andy Reid/grades — request failed"
else
  check_json "$GRADES" "assert d['status'] == 'success'" \
    "GET /coaches/Andy Reid/grades — status success"
  check_json "$GRADES" "assert isinstance(d['grades'], list) and len(d['grades']) > 0" \
    "GET /coaches/Andy Reid/grades — grades list is non-empty"
  check_json "$GRADES" \
    "g=d['grades'][0]; assert all(k in g for k in ('season','record','win_percentage','win_score','win_letter_grade'))" \
    "GET /coaches/Andy Reid/grades — grade entries have all required fields"
  check_json "$GRADES" \
    "g=d['grades'][0]; assert g['win_letter_grade'] in ('A+','A','A-','B+','B','B-','C+','C','C-','D+','D','D-','F')" \
    "GET /coaches/Andy Reid/grades — win_letter_grade is a valid letter grade"
fi

UNKNOWN_COACH_STATUS=$(status_code "/coaches/Nobody%20Here/grades?years=2024" 30)
[ "$UNKNOWN_COACH_STATUS" = "404" ] \
  && pass "GET /coaches/Nobody Here/grades — 404 for unknown coach" \
  || fail "GET /coaches/Nobody Here/grades — expected 404, got $UNKNOWN_COACH_STATUS"

# ─── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "$(printf '═%.0s' {1..60})"
echo -e "${BOLD}Results:${NC}  ${GREEN}${PASS} passed${NC}  |  ${RED}${FAIL} failed${NC}  |  ${YELLOW}${SKIP} skipped${NC}"

if [ "${#ERRORS[@]}" -gt 0 ]; then
  echo ""
  echo -e "${RED}Failures:${NC}"
  for err in "${ERRORS[@]}"; do
    echo "  • $err"
  done
  echo ""
  exit 1
fi

echo ""
