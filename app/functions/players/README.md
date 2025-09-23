# NFL Player Grading Functions

## Overview

Comprehensive NFL player grading system that evaluates all position groups including **individual offensive/defensive line players** and **context-aware adjustments** for skill positions based on line performance. Built using `nfl_data_py` with realistic grade scales and position-specific metrics.

## Core Classes

### `EnhancedNFLPlayerGrader`

Main class that handles all player grading functionality including traditional skill positions, defensive players, and line players with contextual adjustments.

```python
from app.functions.players.grading import EnhancedNFLPlayerGrader

# Initialize for specific seasons
grader = EnhancedNFLPlayerGrader(years=[2023])

# Calculate all player grades
all_grades = grader.calculate_all_grades(min_games=3)
```

## Data Sources & NFL Data Py Functions

### Required NFL Data Functions
```python
import nfl_data_py as nfl

# Data loading functions called automatically
self.weekly_data = nfl.import_weekly_data(self.years)        # Skill position stats
self.pbp_data = nfl.import_pbp_data(self.years)             # Play-by-play for line analysis
self.rosters = nfl.import_weekly_rosters(self.years)        # Position mapping
self.snap_counts = nfl.import_snap_counts(self.years)       # Individual line player snaps
```

### Key Columns Used

#### From `weekly_data` (Skill Positions)
```python
# QB Columns
['player_id', 'player_name', 'position', 'team', 'season', 'week',
 'attempts', 'completions', 'passing_yards', 'passing_tds', 'interceptions']

# RB Columns  
['carries', 'rushing_yards', 'rushing_tds', 'receiving_yards', 
 'receptions', 'receiving_tds', 'targets']

# WR/TE Columns
['receiving_yards', 'receiving_tds', 'receptions', 'targets']
```

#### From `pbp_data` (Line Performance)
```python
# O-Line Analysis
['posteam', 'season', 'week', 'play_type', 'sack', 'qb_hit', 
 'rushing_yards', 'passing_yards']

# D-Line Analysis  
['defteam', 'season', 'week', 'play_type', 'sack', 'qb_hit',
 'rushing_yards']

# Defensive Players
['sack_player_name', 'sack_player_id', 'half_sack_1_player_name',
 'solo_tackle_1_player_name', 'assist_tackle_1_player_name',
 'interception_player_name', 'pass_defense_1_player_name',
 'forced_fumble_player_1_player_name', 'qb_hit_1_player_name']
```

#### From `snap_counts` (Individual Line Players)
```python
['player', 'position', 'team', 'season', 'week', 'offense_snaps', 
 'offense_pct', 'defense_snaps', 'defense_pct']
```

#### From `rosters` (Position Mapping)
```python
['player_id', 'player_name', 'position', 'team', 'week', 'season']
```

## Core Functions

### `calculate_all_grades(min_games=3)`

Main function that calculates all player types and returns comprehensive results.

```python
all_grades = grader.calculate_all_grades(min_games=3)

# Returns dictionary with:
all_grades = {
    'team_oline_grades': DataFrame,      # Team O-Line unit performance
    'team_dline_grades': DataFrame,      # Team D-Line unit performance
    'individual_oline_grades': DataFrame, # Individual O-Line player grades
    'individual_dline_grades': DataFrame, # Individual D-Line player grades
    'enhanced_qb_grades': DataFrame,     # QBs with O-Line adjustments
    'enhanced_rb_grades': DataFrame,     # RBs with O-Line adjustments
    'offensive_grades': DataFrame,       # WR/TE traditional grades
    'defensive_grades': DataFrame        # LB/DB traditional grades
}
```

### Line Grading Functions

#### `calculate_team_oline_grades(min_plays=40)`
Calculates team offensive line unit grades from play-by-play data.

**Input Data**: `pbp_data` filtered to pass/run plays  
**Key Metrics**:
- Pass protection success rate (no pressure allowed)
- Pressure rate (sacks + QB hits)
- Run success rate (≥4 yards gained)
- Sacks allowed per game

**Output Columns**:
```python
['team', 'season', 'week', 'pass_protection_grade', 'run_blocking_grade',
 'overall_oline_grade', 'letter_grade', 'total_plays', 'pass_pro_success_rate',
 'pressure_rate', 'run_success_rate', 'sacks_allowed']
```

#### `calculate_team_dline_grades(min_plays=40)`
Calculates team defensive line unit grades from play-by-play data.

**Input Data**: `pbp_data` filtered to opposing team's offensive plays  
**Key Metrics**:
- Pressure generation rate (sacks + QB hits)
- Run stuff rate (≤2 yards allowed)
- Sacks generated per game
- Negative plays created

**Output Columns**:
```python
['team', 'season', 'week', 'pass_rush_grade', 'run_defense_grade',
 'overall_dline_grade', 'letter_grade', 'total_plays', 'pressure_rate',
 'run_stuff_rate', 'sacks', 'negative_play_rate']
```

#### `calculate_individual_oline_grades(min_games=3)`
Grades individual offensive linemen based on team performance and snap participation.

**Input Data**: `snap_counts` + `rosters` + team O-Line performance  
**Positions Covered**: `['C', 'G', 'LG', 'RG', 'T', 'LT', 'RT', 'OL']`

**Position-Specific Weighting**:
- **Tackles (LT/RT)**: 70% pass protection, 30% run blocking
- **Center**: 50% pass protection, 50% run blocking  
- **Guards (LG/RG)**: 30% pass protection, 70% run blocking

**Output Columns**:
```python
['player_id', 'player_name', 'position', 'team', 'season', 'week',
 'snaps', 'snap_pct', 'individual_grade', 'letter_grade',
 'team_pass_pro', 'team_run_success', 'team_pressure_allowed']
```

#### `calculate_individual_dline_grades(min_games=3)`
Grades individual defensive linemen based on team performance and snap participation.

**Input Data**: `snap_counts` + `rosters` + team D-Line performance  
**Positions Covered**: `['DE', 'DT', 'NT', 'EDGE', 'DL']`

**Position-Specific Weighting**:
- **Edge (DE/EDGE)**: 80% pass rush, 20% run defense
- **Interior (DT/NT)**: 30% pass rush, 70% run defense

**Output Columns**:
```python
['player_id', 'player_name', 'position', 'team', 'season', 'week',
 'snaps', 'snap_pct', 'individual_grade', 'letter_grade',
 'team_pressure_rate', 'team_run_stuff_rate', 'team_negative_plays']
```

### Enhanced Skill Position Functions

#### `calculate_enhanced_qb_grades_with_oline(team_oline_grades, min_games=3)`
Calculates QB grades with realistic O-Line adjustments.

**Input Data**: `weekly_data` (QB stats) + `team_oline_grades`  
**Base Grading Factors**:
- Passing yards (25 points max)
- Completion percentage (30 points max, starts at 50%)
- Touchdown passes (12 points each)
- Interceptions (-8 points each)
- Yards per attempt bonus (4 points per YPA above 6.0)

**O-Line Adjustment**:
```python
# Conservative adjustments based on pass protection grade
if oline_grade >= 85: adjustment = 1.06  # +6% max
elif oline_grade >= 75: adjustment = 1.03  # +3%
elif oline_grade >= 65: adjustment = 1.0   # No change
elif oline_grade >= 55: adjustment = 0.97  # -3%
else: adjustment = 0.93  # -7% max
```

**Output Columns**:
```python
['player_id', 'player_name', 'position', 'team', 'season', 'week',
 'base_grade', 'oline_grade', 'oline_adjustment', 'adjusted_grade',
 'grade_improvement', 'oline_tier', 'attempts', 'completions',
 'passing_yards', 'passing_tds', 'interceptions']
```

#### `calculate_enhanced_rb_grades_with_oline(team_oline_grades, min_games=3)`
Calculates RB grades with realistic O-Line adjustments.

**Input Data**: `weekly_data` (RB stats) + `team_oline_grades`  
**Base Grading Factors**:
- Rushing yards (30 points max)
- Yards per carry (25 points max, baseline 3.5 YPC)
- Rushing touchdowns (10 points each)
- Receiving contribution (yards + receptions + TDs)

**O-Line Adjustment**:
```python
# More aggressive adjustments for RBs (more O-Line dependent)
if oline_grade >= 85: adjustment = 1.08  # +8% max
elif oline_grade >= 75: adjustment = 1.05  # +5%
elif oline_grade >= 65: adjustment = 1.0   # No change
elif oline_grade >= 55: adjustment = 0.95  # -5%
else: adjustment = 0.90  # -10% max
```

**Output Columns**:
```python
['player_id', 'player_name', 'position', 'team', 'season', 'week',
 'base_grade', 'oline_grade', 'oline_adjustment', 'adjusted_grade',
 'grade_improvement', 'oline_tier', 'carries', 'rushing_yards',
 'rushing_tds', 'receiving_yards', 'receptions']
```

### Traditional Grading Functions

#### `calculate_offensive_grades(min_games=3)`
Traditional grading for WR/TE positions.

**Input Data**: `weekly_data` filtered to WR/TE positions  
**Grading Factors**:
- Receiving yards (40 points max)
- Receptions (3 points each)
- Receiving touchdowns (12 points each)
- Catch rate (15 points max)

#### `calculate_defensive_grades(min_games=3)`
Traditional grading for defensive players (excluding line).

**Input Data**: `defensive_weekly` (aggregated from play-by-play)  
**Position Groups**:
- **LINEBACKER**: Emphasizes tackles, versatility
- **SECONDARY**: Emphasizes interceptions, pass deflections

**Key Metrics**:
- Total tackles, tackles for loss
- Sacks, QB hits
- Interceptions, pass deflections
- Forced fumbles

## Grade Scales & Validation

### Realistic Grade Ranges
```python
# Team Units
team_oline_grades: 60-95 (average ~75-80)
team_dline_grades: 60-90 (average ~70-75)

# Individual Players  
individual_line_grades: 65-90 (based on team performance + snaps)
enhanced_qb_grades: 0-100 (with ±7% O-Line adjustment)
enhanced_rb_grades: 0-100 (with ±10% O-Line adjustment)
traditional_grades: 0-100 (standard distribution)
```

### Letter Grade Scale
```python
grade_scale = {
    'A+': (95, 100), 'A': (90, 94.9), 'A-': (85, 89.9),
    'B+': (80, 84.9), 'B': (75, 79.9), 'B-': (70, 74.9),
    'C+': (65, 69.9), 'C': (55, 64.9), 'C-': (50, 54.9),
    'D+': (45, 49.9), 'D': (40, 44.9), 'D-': (35, 39.9),
    'F': (0, 34.9)
}
```

## Error Handling & Fallbacks

### Snap Count Data Issues
When snap count data is unavailable or has column mismatches:

```python
# Automatic fallback methods
calculate_individual_oline_grades() → _calculate_oline_grades_from_pbp()
calculate_individual_dline_grades() → _calculate_dline_grades_from_pbp()

# Creates estimated individual grades based on team performance
```

### Data Validation
```python
# Built-in validation checks
assert 35 <= oline_grade <= 95, "O-Line grades outside realistic range"
assert -10 <= grade_improvement <= 10, "Player adjustments too extreme"
assert 0 <= final_grade <= 100, "Final grades outside valid range"
```

## Usage Examples

### Basic Analysis
```python
from app.functions.players.grading import EnhancedNFLPlayerGrader

# Initialize and run complete analysis
grader = EnhancedNFLPlayerGrader(years=[2023])
all_grades = grader.calculate_all_grades(min_games=3)

# Generate comprehensive report
grader.generate_line_report(all_grades)
```

### Accessing Specific Grade Types
```python
# Team line units
team_oline = all_grades['team_oline_grades']
team_dline = all_grades['team_dline_grades']

# Individual line players
individual_oline = all_grades['individual_oline_grades']
individual_dline = all_grades['individual_dline_grades']

# Enhanced skill positions
enhanced_qbs = all_grades['enhanced_qb_grades']
enhanced_rbs = all_grades['enhanced_rb_grades']

# Traditional positions
wr_te_grades = all_grades['offensive_grades']
def_grades = all_grades['defensive_grades']
```

### Team-Specific Analysis
```python
# Analyze specific team's performance
team = 'KC'

# O-Line unit performance
chiefs_oline = team_oline[team_oline['team'] == team]
avg_oline_grade = chiefs_oline['overall_oline_grade'].mean()

# Individual Chiefs linemen
chiefs_linemen = individual_oline[individual_oline['team'] == team]

# Chiefs QBs with O-Line context
chiefs_qbs = enhanced_qbs[enhanced_qbs['team'] == team]
qb_oline_impact = chiefs_qbs['grade_improvement'].mean()

print(f"Chiefs O-Line Grade: {avg_oline_grade:.1f}")
print(f"QB O-Line Impact: {qb_oline_impact:+.1f} points")
```

### Player Dependency Analysis
```python
# Find most O-Line dependent players
qb_impact = enhanced_qbs.groupby('player_name').agg({
    'base_grade': 'mean',
    'adjusted_grade': 'mean',
    'grade_improvement': 'mean',
    'oline_grade': 'mean'
}).round(1)

# Players most helped by good O-Line
helped_players = qb_impact.nlargest(10, 'grade_improvement')
print("Most O-Line Dependent QBs:")
print(helped_players[['adjusted_grade', 'grade_improvement', 'oline_grade']])
```

## Output Data Structure

All grade DataFrames include these common columns:
- `player_id` / `team`: Unique identifier
- `player_name`: Display name  
- `position`: Specific position (QB, RB, LT, DE, etc.)
- `season`, `week`: Time identifiers
- `*_grade`: Numeric grade (0-100)
- `letter_grade`: Letter grade (A+ through F)

Enhanced grades additionally include:
- `base_grade`: Grade before O-Line adjustments
- `oline_grade`: Supporting O-Line quality
- `oline_adjustment`: Adjustment factor applied
- `grade_improvement`: Points gained/lost from O-Line context

## Performance Notes

- **Memory Usage**: ~4GB RAM recommended for full season analysis
- **Processing Time**: ~30-60 seconds for single season
- **Data Dependencies**: Requires internet connection for initial nfl_data_py downloads
- **Caching**: nfl_data_py automatically caches downloaded data locally

## Dependencies

```python
# Required packages
nfl_data_py  # NFL data access
pandas       # Data manipulation
numpy        # Numerical operations
matplotlib   # Plotting (optional)
seaborn      # Statistical visualization (optional)
scipy        # Statistical analysis (optional)
```

## File Structure

```
app/functions/players/grading.py
├── EnhancedNFLPlayerGrader (main class)
├── Data loading methods (_load_data, _prepare_*)  
├── Team line grading (calculate_team_*_grades)
├── Individual line grading (calculate_individual_*_grades)
├── Enhanced skill grading (calculate_enhanced_*_grades_with_oline)
├── Traditional grading (calculate_offensive/defensive_grades)
├── Helper methods (_calculate_*_grade, _numeric_to_letter_grade)
├── Reporting (generate_line_report)
└── Fallback methods (_calculate_*_grades_from_pbp)
```