# NFL Player Performance Grading System

A comprehensive Python system for grading NFL player performance using `nfl_data_py`, covering both offensive and defensive players with position-specific analytics and outlier detection.

## 🏈 Features

### Comprehensive Player Coverage
- **Offensive Players**: QB, RB, WR, TE grading using weekly statistical data
- **Defensive Players**: Pass rushers, linebackers, secondary using play-by-play data
- **Unified Grading Scale**: A-F letter grades with numerical scores (0-100)

### Advanced Analytics
- **Position-Specific Grading**: Different criteria for each position group
- **Outlier Detection**: Identifies boom/bust performances using statistical analysis
- **Consistency Metrics**: Tracks player reliability vs variance
- **Performance Trends**: Season-long grade tracking and analysis

### Rich Data Sources
- **Weekly Stats**: Offensive player statistics (passing, rushing, receiving)
- **Play-by-Play Data**: Defensive statistics (sacks, tackles, interceptions, etc.)
- **Coverage Schemes**: Defensive formation and coverage analysis (Cover 1, 2, 3, etc.)

## 📊 Output Data

The system generates comprehensive performance data including:

### Sample Results (2023 Season)
```
Top QBs by Grade:
K.Cousins    77.8 (B+)    - 8 games
D.Prescott   75.2 (B)     - 18 games  
B.Purdy      75.2 (B)     - 19 games

Top Defensive Players:
L.David (LB)      83.8 (B+)   - 17 games
K.Mack (EDGE)     82.3 (B+)   - 9 games
C.Ward (DB)       72.0 (B-)   - 16 games

Season Leaders:
Sacks: J.Allen (23.0), T.Watt (19.0), T.Hendrickson (17.5)
Tackles: R.Smith (170), B.Wagner (164), Z.Franklin (158)
Interceptions: D.Bland (9), G.Stone (7), J.Johnson (7)
```

## 🚀 Quick Start

### Installation
```bash
pip install nfl_data_py pandas numpy matplotlib seaborn scipy
```

### Basic Usage
```python
from enhanced_nfl_grading import EnhancedNFLPlayerGrader

# Initialize system for 2023 season
grader = EnhancedNFLPlayerGrader(years=[2023])

# Calculate all player grades
all_grades = grader.calculate_all_grades(min_games=3)

# Identify outlier performances  
outliers = grader.identify_performance_outliers(all_grades)

# Get top performers
top_qbs = grader.get_top_performers(outliers, position_group='QB', n=10)
top_defense = grader.get_top_performers(outliers, player_type='DEFENSE', n=10)
```

## 📈 Grading Methodology

### Offensive Players

#### Quarterbacks (QB)
- **Passing Yards**: Compared to position average (25 points max)
- **Completion %**: Bonus scoring above 50% completion (60 points max)
- **Touchdowns**: 12 points per passing TD
- **Interceptions**: -8 point penalty per INT
- **Efficiency**: Yards per attempt bonus (4 points per YPA above 6.0)
- **Volume**: Attempt bonus up to 10 points

#### Running Backs (RB)  
- **Rushing Yards**: Volume scoring vs position average (35 points max)
- **Yards Per Carry**: Efficiency scoring vs 4.2 YPC baseline (25 points max)
- **Rushing TDs**: 10 points per TD
- **Receiving**: Receiving yards (0.15x) + receptions (2x)
- **Receiving TDs**: 8 points per receiving TD

#### Wide Receivers & Tight Ends (WR/TE)
- **Receiving Yards**: Volume scoring vs position average (40 points max)
- **Receptions**: 3 points per reception
- **Receiving TDs**: 12 points per TD
- **Catch Rate**: Target efficiency scoring (15 points max)

### Defensive Players

#### Pass Rushers (DE, OLB, DT)
- **Sacks**: Primary metric (15 points per relative performance)
- **QB Hits**: Pressure generation (10 points max)
- **Tackles for Loss**: Disruptive plays (8 points max)
- **Total Tackles**: Run defense contribution (7 points max)
- **Forced Fumbles**: 10 points per forced fumble
- **Base Score**: 20 points for position participation

#### Linebackers (ILB, MLB)
- **Total Tackles**: Primary responsibility (20 points max)
- **Tackles for Loss**: Impact plays (10 points max)
- **Sacks**: Pass rush value (12 points per sack)
- **Interceptions**: Coverage success (15 points per INT)
- **Pass Deflections**: Coverage impact (3 points per PD)
- **Forced Fumbles**: Turnover creation (8 points per FF)
- **Base Score**: 15 points for position participation

#### Secondary (CB, S, FS, SS)
- **Interceptions**: Primary coverage metric (20 points per INT)
- **Pass Deflections**: Coverage effectiveness (15 points max)
- **Total Tackles**: Run support (8 points max)
- **Forced Fumbles**: Turnover creation (12 points per FF)
- **Base Score**: 25 points for position participation

## 🎯 Key Functions

### Core Analysis
```python
# Calculate grades for all players
all_grades = grader.calculate_all_grades(min_games=3)

# Find performance outliers (boom/bust games)
outliers = grader.identify_performance_outliers(all_grades, std_threshold=1.5)

# Generate defensive statistics summary
grader.generate_defensive_report()
```

### Performance Rankings
```python
# Top performers by player type
top_offense = grader.get_top_performers(outliers, player_type='OFFENSE', n=10)
top_defense = grader.get_top_performers(outliers, player_type='DEFENSE', n=10)

# Position-specific rankings
top_qbs = grader.get_top_performers(outliers, position_group='QB', n=10)
top_pass_rushers = grader.get_top_performers(outliers, position_group='PASS_RUSHER', n=10)

# Consistency rankings
consistent_players = grader.get_top_performers(outliers, metric='consistency', n=10)
boom_bust_players = grader.get_top_performers(outliers, metric='over_performances', n=10)
```

### Individual Player Analysis
```python
# Filter to specific player
player_data = outliers[outliers['player_name'].str.contains('Josh Allen', case=False)]

# View season performance
print(f"Average Grade: {player_data['numeric_grade'].mean():.1f}")
print(f"Best Game: {player_data['numeric_grade'].max():.1f}")
print(f"Worst Game: {player_data['numeric_grade'].min():.1f}")
print(f"Consistency: {100 - player_data['numeric_grade'].std():.1f}")
```

## 📋 Data Requirements

### Dependencies
- `nfl_data_py`: NFL data access
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib`: Plotting
- `seaborn`: Statistical visualization
- `scipy`: Statistical analysis

### Data Sources
- **Weekly Player Stats**: `nfl.import_weekly_data()`
- **Play-by-Play Data**: `nfl.import_pbp_data()`
- **Player Rosters**: `nfl.import_weekly_rosters()`

### System Requirements
- Python 3.7+
- 4GB+ RAM (for play-by-play data processing)
- Internet connection (for initial data download)

## 🔧 Configuration

### Adjustable Parameters
```python
# Minimum games for qualification
min_games = 3

# Outlier detection sensitivity  
std_threshold = 1.5

# Years to analyze
years = [2022, 2023]

# Initialize with custom settings
grader = EnhancedNFLPlayerGrader(years=years)
```

### Grade Scale
```python
grade_scale = {
    'A+': (95, 100),   'A': (90, 94.9),   'A-': (85, 89.9),
    'B+': (80, 84.9),  'B': (75, 79.9),   'B-': (70, 74.9), 
    'C+': (65, 69.9),  'C': (55, 64.9),   'C-': (50, 54.9),
    'D+': (45, 49.9),  'D': (40, 44.9),   'D-': (35, 39.9),
    'F': (0, 34.9)
}
```

## 🏗️ Architecture

### DevOps-Friendly Design
- **Modular Components**: Separate offensive/defensive processing
- **Error Handling**: Comprehensive logging and fallback methods
- **Scalable Processing**: Efficient pandas vectorization
- **Container Ready**: Easy Docker deployment
- **Configuration**: Environment-based settings support

### Container Deployment
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "enhanced_nfl_grading.py"]
```

### Kubernetes Integration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nfl-grading
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nfl-grading
  template:
    metadata:
      labels:
        app: nfl-grading
    spec:
      containers:
      - name: nfl-grading
        image: nfl-grading:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
```

## 📊 Sample Outputs

### Position Rankings
```
Top 5 Quarterbacks (2023):
Player          Grade   Games   Consistency
K.Cousins       77.8    8       80.6
D.Prescott      75.2    18      73.1  
B.Purdy        75.2    19      74.6
J.Goff         74.3    20      76.8
T.Tagovailoa   72.9    17      71.2
```

### Defensive Leaders
```
Top Pass Rushers (2023):
Player          Grade   Sacks   QB Hits   Games
K.Mack         82.3    17.0    22        9
J.Bosa         79.6    10.5    16        18
T.Watt         78.4    19.0    36        17
M.Crosby       77.9    14.5    18        17
```

### Outlier Performances
```
Biggest Over-Performances (2023):
Player          Week    Grade   vs Avg    Performance Type
T.Hill          Week 3  98.7    +20.5     Over-Performance
J.Jefferson     Week 8  96.4    +18.9     Over-Performance  
C.McCaffrey     Week 12 94.8    +19.7     Over-Performance
```

## 🤝 Contributing

### Development Setup
```bash
git clone <repository>
cd nfl-grading-system
pip install -r requirements.txt
python test_grading_fixes.py  # Run tests
```

### Adding New Metrics
1. Extend position-specific grading functions
2. Update data preparation methods
3. Add corresponding unit tests
4. Update documentation

### Enhancement Ideas
- **Advanced Coverage Metrics**: Utilize coverage scheme data
- **Situational Grading**: Performance by down/distance
- **Opponent Adjustments**: Strength of schedule factors
- **Injury Context**: Performance relative to injury status
- **Weather Factors**: Environmental impact analysis

## 📝 License

This project uses NFL data provided by `nfl_data_py` under their respective licenses. The grading system code is available under MIT License.

## 🏆 Acknowledgments

- **nflfastR team**: For comprehensive NFL play-by-play data
- **nfl_data_py maintainers**: For the Python data access layer
- **NFL community**: For open data initiatives and statistical innovation

---

**Ready to analyze NFL performance like never before!** 🏈📊