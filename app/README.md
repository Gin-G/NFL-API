# NFL Analytics API - Complete Endpoint Documentation

## 🏈 Overview

The NFL Analytics API provides comprehensive access to NFL data, advanced player grading, coaching analytics, and performance metrics. Built with FastAPI and powered by official NFL data sources.

**Base URL:** `http://localhost:8000` (development) | `https://api.nfl-analytics.com` (production)

**API Documentation:** `/docs` (Swagger UI) | `/redoc` (ReDoc)

---

## 📋 Quick Start

```bash
# Start the API locally
make dev

# Or with Docker
make up

# Health check
curl http://localhost:8000/health
```

---

## 🔗 Core Endpoints

### System & Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root endpoint with API information |
| `GET` | `/health` | Health check and system status |
| `GET` | `/stats/summary` | Comprehensive statistics summary |

**Example:**
```bash
curl http://localhost:8000/health
```

---

## 🏟️ Team Endpoints

### Get All Teams
```http
GET /teams
```

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "team_abbr": "KC",
      "team_name": "Kansas City Chiefs",
      "team_conf": "AFC",
      "team_division": "West"
    }
  ]
}
```

### Get Team Details
```http
GET /teams/{team_abbr}
```

**Parameters:**
- `team_abbr` (path): Team abbreviation (e.g., "KC", "NE")

---

## 📅 Schedule Endpoints

### Get Schedules
```http
GET /schedules?season=2023&week=1&team=KC
```

**Query Parameters:**
- `season` (optional): Season year (default: 2023)
- `week` (optional): Specific week number
- `team` (optional): Team abbreviation

### Get Weekly Schedule
```http
GET /schedules/{season}/week/{week}
```

**Example:**
```bash
curl "http://localhost:8000/schedules/2023/week/1"
```

---

## 👥 Player Endpoints

### Get Player Rosters
```http
GET /players/rosters?season=2023&team=KC&position=QB
```

**Query Parameters:**
- `season` (optional): Season year
- `week` (optional): Specific week
- `team` (optional): Team abbreviation
- `position` (optional): Player position

### Get Player Statistics
```http
GET /players/stats?season=2023&position=QB&team=KC
```

**Query Parameters:**
- `season` (optional): Season year
- `player_id` (optional): Specific player ID
- `position` (optional): Player position
- `team` (optional): Team abbreviation
- `week` (optional): Specific week

### Get Player Details
```http
GET /players/{player_id}?season=2023
```

**Example Response:**
```json
{
  "status": "success",
  "player_info": {
    "player_id": "00-0026498",
    "player_name": "Patrick Mahomes",
    "position": "QB",
    "team": "KC",
    "height": "6-3",
    "weight": 230,
    "college": "Texas Tech"
  },
  "season_stats": [...],
  "games_played": 17
}
```

---

## 📊 Player Grading Endpoints

### Get Player Grades
```http
GET /grades/players?years=2023&player_type=OFFENSE&position_group=QB&limit=10
```

**Query Parameters:**
- `years` (array): Years to analyze (default: [2023])
- `min_games` (int): Minimum games played (default: 3)
- `player_type` (optional): "OFFENSE" or "DEFENSE"
- `position_group` (optional): Position group filter
- `limit` (int): Maximum results (default: 50)

**Position Groups:**
- **Offense:** QB, RB, WR_TE
- **Defense:** PASS_RUSHER, LINEBACKER, SECONDARY

**Example Response:**
```json
{
  "status": "success",
  "total_players": 15,
  "data": [
    {
      "player_name": "Josh Allen",
      "position": "QB",
      "position_group": "QB",
      "player_type": "OFFENSE",
      "avg_grade": 82.4,
      "letter_grade": "B+",
      "games_played": 17,
      "consistency": 78.2,
      "over_performances": 4
    }
  ]
}
```

### Get Player Grade Details
```http
GET /grades/players/{player_name}?years=2023
```

**Example:**
```bash
curl "http://localhost:8000/grades/players/Patrick%20Mahomes?years=2023"
```

### Get Top Players by Position
```http
GET /grades/players/top/{position_group}?years=2023&limit=10
```

**Position Groups:** QB, RB, WR_TE, PASS_RUSHER, LINEBACKER, SECONDARY

---

## 🎯 Coaching Endpoints

### Get All Coaches
```http
GET /coaches?season=2023&years=2023,2024
```

**Query Parameters:**
- `season` (optional): Filter by specific season
- `years` (array): Years to load data for

### Get Coach Analysis
```http
GET /coaches/{coach_name}/analysis?season=2023
```

**Example Response:**
```json
{
  "status": "success",
  "coach": "Andy Reid",
  "season": 2023,
  "offensive_analysis": {
    "play_type_pct": {
      "pass": 62.5,
      "run": 37.5
    },
    "passing_efficiency": {
      "yards_per_attempt": 7.8,
      "completion_pct": 67.2,
      "td_pct": 5.1,
      "int_pct": 1.8
    }
  },
  "defensive_analysis": {...},
  "situational_analysis": {...}
}
```

### Get Coach Grades
```http
GET /coaches/{coach_name}/grades?season=2023
```

**Example Response:**
```json
{
  "status": "success",
  "coach": "Andy Reid",
  "specialty": "offense",
  "team_record": {
    "wins": 14,
    "losses": 3,
    "win_pct": 82.4
  },
  "grades": {
    "offensive_overall": {
      "score": 85.3,
      "letter_grade": "B"
    },
    "defensive_overall": {
      "score": 76.8,
      "letter_grade": "C+"
    },
    "overall": {
      "score": 82.1,
      "letter_grade": "B-"
    }
  }
}
```

### Compare Coaches
```http
POST /coaches/compare?season=2023
```

**Request Body:**
```json
{
  "coach_names": ["Andy Reid", "Bill Belichick", "Sean McVay"]
}
```

---

## 📈 Play-by-Play Endpoints

### Get Play-by-Play Data
```http
GET /pbp?season=2023&week=1&team=KC&limit=100
```

**Query Parameters:**
- `season` (required): Season year
- `week` (optional): Specific week
- `game_id` (optional): Specific game ID
- `team` (optional): Team abbreviation
- `limit` (int): Maximum plays to return (default: 100)

---

## 🔍 Search Endpoints

### Search Players
```http
GET /search/players?query=Mahomes&season=2023&limit=20
```

**Query Parameters:**
- `query` (required): Search query for player name
- `season` (optional): Season year (default: 2023)
- `limit` (optional): Maximum results (default: 20)

### Search Coaches
```http
GET /search/coaches?query=Reid&years=2023,2024
```

**Query Parameters:**
- `query` (required): Search query for coach name
- `years` (array): Years to search (default: [2023, 2024])

---

## 📊 Response Format

### Standard Success Response
```json
{
  "status": "success",
  "data": [...],
  "total_records": 50,
  "parameters": {...}
}
```

### Error Response
```json
{
  "status": "error",
  "message": "Player not found",
  "detail": "Player 'John Doe' not found in database",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 🎭 Grade Scale

| Letter | Range | Description |
|--------|-------|-------------|
| A+ | 95-100 | Elite, championship-level |
| A | 90-94.9 | Outstanding performance |
| A- | 85-89.9 | Excellent |
| B+ | 80-84.9 | Very good |
| B | 75-79.9 | Good |
| B- | 70-74.9 | Above average |
| C+ | 65-69.9 | Average plus |
| C | 55-64.9 | Average |
| C- | 50-54.9 | Below average |
| D+/D/D- | 35-49.9 | Poor |
| F | 0-34.9 | Very poor |

---

## 🔧 Query Examples

### Top Quarterbacks in 2023
```bash
curl "http://localhost:8000/grades/players/top/QB?years=2023&limit=10"
```

### Defensive Players with Outlier Performances
```bash
curl "http://localhost:8000/grades/players?years=2023&player_type=DEFENSE&min_games=5"
```

### Coach Performance Comparison
```bash
curl -X POST "http://localhost:8000/coaches/compare" \
  -H "Content-Type: application/json" \
  -d '{"coach_names": ["Andy Reid", "Sean McVay"]}'
```

### Player Season Details
```bash
curl "http://localhost:8000/grades/players/Josh%20Allen?years=2023&min_games=1"
```

### Team Schedule
```bash
curl "http://localhost:8000/schedules?season=2023&team=KC"
```

---

## 🚀 Development

### Running Locally
```bash
# Install dependencies
make install

# Start development server
make dev

# Run tests
make test

# Build Docker image
make build

# Start with Docker Compose
make up
```

### Docker Deployment
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f nfl-api

# Health check
curl http://localhost:8000/health
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
make k8s-deploy

# Check status
make k8s-status

# View logs
make k8s-logs
```

---

## 📈 Performance Features

- **Caching:** Redis-based caching for improved response times
- **Auto-scaling:** Kubernetes HPA for dynamic scaling
- **Health Checks:** Comprehensive health monitoring
- **Rate Limiting:** Built-in API rate limiting
- **Database:** PostgreSQL for data persistence
- **Monitoring:** Integrated monitoring and logging

---

## 🛡️ Security

- **CORS:** Configurable CORS policies
- **Validation:** Pydantic model validation
- **Error Handling:** Comprehensive error responses
- **Secrets Management:** Kubernetes secrets for sensitive data

---

## 📚 Additional Resources

- **API Documentation:** `/docs` (Swagger UI)
- **Alternative Docs:** `/redoc` (ReDoc)
- **Health Check:** `/health`
- **Stats Summary:** `/stats/summary`

---

## 🤝 Support

For questions, issues, or feature requests:
- Check the `/docs` endpoint for interactive API documentation
- Review the health status at `/health`
- Use the search endpoints to discover available data

---

## 🎯 Common Use Cases

### 1. Fantasy Football Analysis
```bash
# Get top-performing RBs
curl "http://localhost:8000/grades/players/top/RB?years=2023&limit=20"

# Check player consistency
curl "http://localhost:8000/grades/players/Christian%20McCaffrey?years=2023"
```

### 2. Coaching Evaluation
```bash
# Compare offensive coordinators
curl -X POST "http://localhost:8000/coaches/compare" \
  -H "Content-Type: application/json" \
  -d '{"coach_names": ["Kyle Shanahan", "Sean McVay", "Andy Reid"]}'

# Analyze specific coach performance
curl "http://localhost:8000/coaches/Andy%20Reid/grades?season=2023"
```

### 3. Team Analysis
```bash
# Get team roster
curl "http://localhost:8000/players/rosters?team=KC&season=2023"

# Get team schedule
curl "http://localhost:8000/schedules?team=KC&season=2023"

# Get team play-by-play data
curl "http://localhost:8000/pbp?team=KC&season=2023&week=1"
```

### 4. Player Scouting
```bash
# Search for players
curl "http://localhost:8000/search/players?query=Smith&season=2023"

# Get detailed player stats
curl "http://localhost:8000/players/00-0036355?season=2023"

# Check defensive player grades
curl "http://localhost:8000/grades/players?player_type=DEFENSE&position_group=PASS_RUSHER"
```

---

## 🔄 Data Flow

```
NFL Data Sources → nfl_data_py → FastAPI Endpoints → JSON Response
                     ↓
                 Grading Algorithms → Performance Analysis → Letter Grades
                     ↓
                 PostgreSQL Cache → Redis Cache → Optimized Responses
```

---

## 📊 Available Data

### Seasons Covered
- **2023 Season:** Complete regular season + playoffs
- **2024 Season:** Current season (updated weekly)
- **Historical:** Expandable to previous seasons

### Player Types
- **Offensive Players:** QB, RB, WR, TE
- **Defensive Players:** DE, DT, LB, CB, S

### Statistics Included
- **Basic Stats:** Passing, rushing, receiving, defensive
- **Advanced Metrics:** Efficiency, consistency, situational performance
- **Grading:** AI-powered performance evaluation
- **Trends:** Week-by-week performance tracking

---

## 🚀 Performance Optimization

### Caching Strategy
```bash
# Data cached for 1 hour by default
# Player grades: Cached after first calculation
# Team data: Cached for 6 hours
# Schedule data: Cached for 24 hours
```

### Scaling Features
- **Horizontal Pod Autoscaling:** 2-10 replicas based on CPU/memory
- **Database Connection Pooling:** Optimized PostgreSQL connections
- **Redis Caching:** Fast response times for repeated queries
- **CDN Ready:** Static assets can be served via CDN

---

## 🔐 Authentication (Future Feature)

```bash
# Future API key authentication
curl -H "X-API-Key: your-api-key" "http://localhost:8000/grades/players"

# Rate limiting per API key
# Free tier: 1000 requests/day
# Pro tier: 10000 requests/day
```

---

## 📈 Monitoring Endpoints

### Health Checks
```bash
# Basic health
curl http://localhost:8000/health

# Detailed status (future)
curl http://localhost:8000/status

# Metrics (future)
curl http://localhost:8000/metrics
```

### Performance Metrics
```bash
# API response times
# Database connection status
# Cache hit rates
# Error rates by endpoint
```