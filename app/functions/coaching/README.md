# NFL Coaching Grading System

A comprehensive Python system for evaluating NFL head coaching performance using roster-aware analytics. The system adjusts coaching grades based on the talent level of key contributors, providing objective assessments of coaching effectiveness independent of roster quality.

## Overview

This system combines play-by-play analysis, team performance metrics, and player talent evaluation to generate coaching grades that account for the quality of available talent. It focuses on **key contributors only** (approximately 25 players per team) rather than evaluating entire rosters, providing more accurate assessments of coaching effectiveness.

## Core Functionality

### RosterAwareCoachingAnalytics Class

The main class that handles all coaching analysis functionality.

```python
from fixed_roster_coaching import RosterAwareCoachingAnalytics

# Initialize for specific seasons
analytics = RosterAwareCoachingAnalytics(years=[2023])

# Load data and calculate grades
analytics.load_data()
analytics.extract_coaching_info()
analytics.calculate_player_grades()
```

## Key Functions

### Data Loading and Preparation

#### `load_data()`
Loads NFL data from nfl_data_py sources.

**NFL Data Sources:**
- `nfl.import_pbp_data(years)` - Play-by-play data for defensive analysis
- `nfl.import_schedules(years)` - Game schedules for coaching assignments
- `nfl.import_weekly_data(years)` - Weekly player statistics for offensive analysis

**Key Columns Used:**
- **Play-by-Play**: `sack`, `sack_player_name`, `interception`, `interception_player_name`, `solo_tackle_1_player_name`, `assist_tackle_1_player_name`, `pass_defense_1_player_name`, `forced_fumble_player_1_player_name`, `defteam`, `season`, `week`
- **Schedule**: `home_coach`, `away_coach`, `home_team`, `away_team`, `home_score`, `away_score`, `game_id`, `season`, `week`
- **Weekly Stats**: `passing_yards`, `passing_tds`, `interceptions`, `attempts`, `completions`, `rushing_yards`, `rushing_tds`, `carries`, `receiving_yards`, `receiving_tds`, `receptions`, `targets`, `position`, `recent_team`

#### `extract_coaching_info()`
Extracts coaching assignments and team records from schedule data.

**Creates coaching data structure:**
```python
{
    ('Coach Name', season): {
        'name': str,
        'season': int,
        'teams': set,
        'games': [list of game records]
    }
}
```

### Player Grading System

#### `calculate_player_grades()`
Generates performance grades for all players (25-95 scale).

**Offensive Player Grading:**
- **QB**: Base 50 + passing yards (0-15) + completion % (0-15) + TDs (0-15) + INT penalty (0 to -10)
- **RB**: Base 50 + rushing yards (0-20) + YPC bonus (0-10) + TDs (0-15) + receiving (0-10)  
- **WR/TE**: Base 50 + receiving yards (0-25) + receptions (0-15) + TDs (0-15) + catch rate (0-5)

**Defensive Player Grading:**
- Base 55 + sacks (10 pts each) + interceptions (12 pts each) + tackles (up to 15 pts) + pass deflections (4 pts each) + forced fumbles (8 pts each)

#### `_identify_key_contributors(player_stats)`
Filters to key contributors only based on games played and position.

**Selection Criteria:**
- **QB**: Top 2 by games played
- **RB**: Top 3 with 4+ games  
- **WR/TE**: Top 6 with 6+ games
- **Defense**: Top 15 with 8+ games

### Roster Analysis

#### `analyze_roster_quality(team, season, key_contributors_only=True)`
Evaluates roster talent focusing on players who actually impact games.

**Returns analysis dictionary with:**
- Overall roster grade and tier classification
- Elite (78+), Good (70-77), Average (62-69), Below Average (<62) player counts
- Position-specific averages and best players
- Top 5 players by grade

**Roster Tier Classifications:**
- **Elite** (72+): Championship-level talent
- **Good** (68-71): Solid talent with few weaknesses  
- **Average** (64-67): Decent talent requiring strategic coaching
- **Below Average** (60-63): Limited talent needing creative coaching
- **Poor** (<60): Talent-deficient roster requiring system to overcome limitations

### Coaching Grade Calculation

#### `calculate_base_coaching_grade(coach_name, season)`
Calculates base coaching performance from team record.

**Grade Scale Based on Win Percentage:**
- 80%+ wins: 90-100 grade
- 70-79%: 80-90 grade
- 60-69%: 75-80 grade  
- 50-59%: 65-75 grade
- 40-49%: 55-65 grade
- <40%: 40-55 grade

#### `calculate_roster_adjusted_grade(coach_name, season)`
Applies roster-based adjustment to coaching grade.

**Adjustment Logic:**
- League average for key contributors: 67
- Maximum adjustment: ±12 points
- Better roster = higher expectations (negative adjustment)
- Efficiency = base performance - expected performance

**Returns:**
```python
{
    'base_grade': float,           # Raw coaching performance
    'adjusted_grade': float,       # Roster-adjusted grade  
    'roster_quality': dict,        # Full roster analysis
    'adjustment': float,           # Points adjusted
    'efficiency': float,           # Performance vs expectations
    'record': str                  # Win-loss record
}
```

### Analysis and Reporting

#### `print_roster_aware_report(coach_name, season)`
Generates comprehensive coaching report including:
- Base and adjusted coaching grades
- Roster quality analysis with key contributors
- Position group breakdowns
- Top 5 players
- Coaching efficiency interpretation

#### `compare_coaches_roster_aware(coach_names, season)`
Side-by-side comparison of multiple coaches with:
- Base vs adjusted grades
- Roster quality tiers
- Elite/good player counts
- Efficiency rankings
- Performance insights

#### `find_coaching_overperformers(season, min_efficiency=15)`
#### `find_coaching_underperformers(season, max_efficiency=-5)`
Identifies coaches significantly exceeding or falling short of roster expectations.

#### `analyze_roster_vs_performance(season)`
League-wide analysis correlating roster quality with team performance.

## Usage Examples

### Basic Analysis
```python
# Initialize system
analytics = RosterAwareCoachingAnalytics(years=[2023])
analytics.load_data()
analytics.extract_coaching_info() 
analytics.calculate_player_grades()

# Analyze specific coach
analytics.print_roster_aware_report('Andy Reid', 2023)
```

### Advanced Analysis
```python
# Compare multiple coaches
coaches = ['Andy Reid', 'Kyle Shanahan', 'Mike Tomlin']
analytics.compare_coaches_roster_aware(coaches, 2023)

# Find efficiency leaders
overperformers = analytics.find_coaching_overperformers(2023)
underperformers = analytics.find_coaching_underperformers(2023)

# League-wide roster correlation
analytics.analyze_roster_vs_performance(2023)
```

### Get Available Coaches
```python
coaches = analytics.get_available_coaches(season=2023)
print(f"Available coaches: {coaches}")
```

## Data Requirements

### Dependencies
```python
import pandas as pd
import numpy as np
import nfl_data_py as nfl
```

### nfl_data_py Functions Called
1. **`nfl.import_pbp_data(years)`** - Play-by-play data for defensive statistics
2. **`nfl.import_schedules(years)`** - Game schedules for coaching assignments  
3. **`nfl.import_weekly_data(years)`** - Weekly player performance data

### Critical Columns

**Play-by-Play Data (Defensive Stats):**
- `sack`, `sack_player_id`, `sack_player_name`
- `half_sack_1_player_id`, `half_sack_1_player_name`, `half_sack_2_player_id`, `half_sack_2_player_name`
- `interception`, `interception_player_id`, `interception_player_name`
- `solo_tackle_1_player_id`, `solo_tackle_1_player_name`
- `assist_tackle_1_player_id`, `assist_tackle_1_player_name`, `assist_tackle_2_player_id`, `assist_tackle_2_player_name`
- `pass_defense_1_player_id`, `pass_defense_1_player_name`
- `forced_fumble_player_1_player_id`, `forced_fumble_player_1_player_name`
- `defteam`, `season`, `week`

**Schedule Data (Coaching Info):**
- `home_coach`, `away_coach`
- `home_team`, `away_team`
- `home_score`, `away_score`
- `game_id`, `season`, `week`

**Weekly Data (Offensive Stats):**
- `player_id`, `player_name`/`player_display_name`
- `position`, `recent_team`
- `passing_yards`, `passing_tds`, `interceptions`, `attempts`, `completions`
- `rushing_yards`, `rushing_tds`, `carries`
- `receiving_yards`, `receiving_tds`, `receptions`, `targets`
- `season`, `week`

## Output Format

### Grade Scale
- **A+ (95-100)**: Elite championship-level coaching
- **A (90-94)**: Outstanding coaching performance  
- **B+ (85-89)**: Very good coaching, above average
- **B (80-84)**: Good coaching performance
- **B- (75-79)**: Above average coaching
- **C+ (70-74)**: Average coaching with some strengths
- **C (60-69)**: Average NFL coaching performance
- **D (50-59)**: Below average, needs improvement
- **F (<50)**: Poor coaching requiring significant changes

### Efficiency Interpretation
- **+15 or higher**: Elite coaching - maximizing talent at every position
- **+8 to +14**: Excellent coaching - exceeding roster expectations
- **+2 to +7**: Good coaching - solid development and scheme fit
- **-5 to +1**: Average coaching - performing as expected
- **-12 to -6**: Below average - not maximizing talent
- **Below -12**: Poor coaching - significant underutilization

## System Features

### Roster-Aware Adjustments
- Focuses on key contributors (≈25 players) who determine success
- Adjusts coaching grades based on available talent level
- Provides coaching efficiency metrics vs roster expectations

### Comprehensive Analysis
- Position-specific roster breakdowns
- Elite/good/average player identification  
- Top performer and underperformer identification
- League-wide correlation analysis

### Production-Ready Design
- Error handling for missing data
- Caching to avoid redundant calculations
- Modular design for easy integration
- Comprehensive debugging output

## Limitations

### Data Constraints
- Requires minimum 3 games for player qualification
- Defensive stats limited to available play-by-play columns
- Coaching assignments based on schedule data accuracy
- Season-level analysis only (no mid-season coaching changes)

### Methodology Limitations
- Grades reflect outcomes, not pure coaching ability
- Does not account for injuries to key players
- Limited to play-calling and roster management aspects
- No consideration of player development over multiple seasons

## Integration Notes

This system is designed to integrate with the broader NFL analytics infrastructure. The coaching grades can be combined with player performance grades and team analytics for comprehensive evaluation frameworks.

The roster-aware approach provides more accurate coaching assessments than traditional win-loss based evaluations, making it suitable for front office decision-making and coaching performance analysis.