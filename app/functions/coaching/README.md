# NFL Coaching Analytics System

A comprehensive Python-based system for analyzing and grading NFL head coaches using advanced statistical methods and play-by-play data.

## Overview

This system evaluates NFL coaches across multiple dimensions including offensive play-calling, defensive performance, situational decision-making, and overall team management. It uses real NFL data to provide objective, data-driven coaching assessments with school-style letter grades.

## Features

### Core Analysis
- **Offensive Tendencies**: Play-calling balance, formation usage, efficiency metrics
- **Defensive Performance**: Yards allowed, pressure generation, red zone defense
- **Situational Analysis**: 4th quarter performance, two-minute drill, clutch situations
- **Personnel Usage**: Formation preferences, tempo, play-action frequency

### Grading System
- **Letter Grades**: A+ through F scale for easy interpretation
- **Weighted Scoring**: Specialist coaches get appropriate emphasis (offensive/defensive)
- **Context Awareness**: Team success and sample size considerations
- **Realistic Thresholds**: Based on actual NFL performance standards

### Reporting Features
- **Individual Coach Reports**: Comprehensive analysis with strengths/weaknesses
- **Multi-Coach Comparisons**: Side-by-side performance evaluations
- **Interactive Mode**: Command-line interface for exploring data
- **Coaching Insights**: Strategic recommendations and observations

## Installation

### Requirements
```bash
pip install pandas numpy nfl_data_py matplotlib seaborn
```

### Dependencies
- **nfl_data_py**: Official NFL data source
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Visualization support

## Usage

### Basic Analysis
```bash
python coach_grading.py
```
Runs automated analysis on available coaches with full reports.

### Interactive Mode
```bash
python coach_grading.py interactive
```
Provides menu-driven interface for custom analysis:
- List all available coaches
- Analyze specific coaches by name
- Compare multiple coaches
- Find top/bottom performers by category

### Debug Mode
```bash
python coach_grading.py debug
```
Shows available data columns for troubleshooting.

## Example Output

```
============================================================
COACHING REPORT: Andy Reid
Season: 2023
============================================================
Season: 2023
Team(s): KC
Record: 15-6 (71.4%)
Specialty: Offense
Elite team performance - Championship caliber

                   OFFENSIVE PERFORMANCE                    
------------------------------------------------------------
Play Balance                          85.8 (B) Very Good
Passing Ypa                          100.0 (A+) Outstanding
Completion Pct                        84.7 (B) Very Good
Passing Tds                           82.5 (B-) Very Good
Interceptions                         81.6 (B-) Very Good
Rushing Ypc                           86.9 (B) Very Good
Red Zone Usage                        75.8 (C) Average
=============================================
OFFENSIVE OVERALL                     85.3 (B) Very Good

============================================================
OVERALL COACHING GRADE                80.2 (B-)
Very good coaching - Above average
============================================================
```

## Grading Methodology

### Performance Thresholds
- **Passing YPA**: 7.5+ = A, 7.0+ = B, 6.5+ = C
- **Completion %**: 65+ = A, 60+ = B, 55+ = C
- **Defensive YPA Allowed**: <6.5 = A, <7.0 = B, <7.5 = C
- **Win Rate Context**: Minimal adjustments (±3%) based on team success

### Weighting System
- **Offensive Specialists**: 60% offense, 40% defense
- **Defensive Specialists**: 40% offense, 60% defense  
- **Balanced Coaches**: 50% offense, 50% defense

### Grade Scale
- **A+ (97-100)**: Elite, championship-level performance
- **A (93-96)**: Outstanding performance
- **B (80-89)**: Above average to very good
- **C (70-79)**: Average NFL performance
- **D (60-69)**: Below average
- **F (<60)**: Poor performance requiring improvement

## Key Improvements (v2.0)

### Fixed Grading Issues
- **Realistic Thresholds**: Adjusted for actual NFL performance standards
- **Proper Averaging**: Fixed mathematical errors in overall grade calculation
- **Moderate Context**: Reduced extreme team success adjustments
- **Balanced Weighting**: Simplified specialist vs. balanced coach treatment

### Enhanced Analysis
- **Situational Performance**: 4th quarter, two-minute drill, goal line
- **Advanced Metrics**: Personnel groupings, formation tendencies
- **Coaching Insights**: Strategic recommendations and observations
- **Comparison Tools**: Multi-coach side-by-side analysis

## Data Sources

- **Play-by-Play Data**: Official NFL statistics via nfl_data_py
- **Schedule Data**: Game results, coaching assignments
- **Coverage**: 2023-2024 seasons (expandable)
- **Update Frequency**: Weekly during NFL season

## Technical Details

### Architecture
- **Object-Oriented Design**: Modular NFLCoachingAnalytics class
- **Data Pipeline**: Load → Extract → Analyze → Grade → Report
- **Error Handling**: Graceful degradation for missing data
- **Performance**: Optimized for large datasets

### Extensibility
- **Custom Metrics**: Easy to add new performance categories
- **Coach Database**: Expandable specialty classifications
- **Season Range**: Configurable year selection
- **Export Options**: Ready for CSV/JSON output integration

## Limitations

### Data Constraints
- **Sample Size**: Minimum games required for reliable grading
- **Injury Impact**: Player availability not considered
- **Coordinator Changes**: Mid-season staff changes not tracked
- **Historical Context**: Limited to recent seasons

### Methodology
- **Correlation vs. Causation**: Grades reflect outcomes, not coaching ability alone
- **Positional Bias**: Some metrics favor certain coaching styles
- **Context Missing**: Roster quality, salary cap constraints not included

## Contributing

### Adding New Metrics
1. Extend `analyze_offensive_tendencies()` or `analyze_defensive_performance()`
2. Update `grade_coach_performance()` with new thresholds
3. Modify `print_coach_report()` for display

### Expanding Coach Database
Update `_determine_coach_specialty()` with additional coach classifications.

### Season Updates
Modify `years` parameter in initialization to include new seasons.

## Future Enhancements

### Planned Features
- **Player Impact Analysis**: Adjust for roster changes
- **Injury Adjustments**: Account for key player availability
- **Historical Trends**: Multi-season coaching progression
- **Predictive Modeling**: Future performance forecasting

### Advanced Analytics
- **EPA-based Grading**: Expected Points Added methodology
- **Win Probability**: Situational decision-making analysis
- **Opponent Adjustments**: Strength of schedule considerations
- **Advanced Metrics**: DVOA, PFF integration potential

## License

This project uses publicly available NFL data through the nfl_data_py package. Ensure compliance with NFL data usage policies for any commercial applications.

## Contact

For questions, improvements, or bug reports, please refer to the code documentation and error handling built into the system.