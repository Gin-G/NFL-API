#!/usr/bin/env python3
"""
Fixed NFL Roster-Aware Coaching Analytics System
Addresses issues with player grade alignment and roster evaluation logic
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class RosterAwareCoachingAnalytics:
    """Fixed coaching analytics system with proper roster evaluation"""
    
    def __init__(self, years=None):
        if years is None:
            years = [2023]
        self.years = years
        self.pbp_data = None
        self.schedule_data = None
        self.coaching_data = {}
        self.player_grades = None
        
        print("NFL Roster-Aware Coaching Analytics System (FIXED)")
        print("=" * 60)
    
    def load_data(self):
        """Load all necessary data"""
        print("Loading NFL data...")
        
        # Load play-by-play data
        print("- Loading play-by-play data...")
        pbp_list = []
        for year in self.years:
            try:
                year_pbp = nfl.import_pbp_data([year])
                pbp_list.append(year_pbp)
                print(f"  - {year}: {len(year_pbp)} plays loaded")
            except Exception as e:
                print(f"  - Error loading {year}: {e}")
        
        if pbp_list:
            self.pbp_data = pd.concat(pbp_list, ignore_index=True)
        
        # Load schedule data
        print("- Loading schedule data...")
        schedule_list = []
        for year in self.years:
            try:
                year_schedule = nfl.import_schedules([year])
                schedule_list.append(year_schedule)
                print(f"  - {year}: {len(year_schedule)} games loaded")
            except Exception as e:
                print(f"  - Error loading schedule for {year}: {e}")
        
        if schedule_list:
            self.schedule_data = pd.concat(schedule_list, ignore_index=True)
        
        print("Data loading complete!")
    
    def calculate_player_grades(self):
        """Calculate simplified but properly scaled player grades"""
        print("Calculating player grades for roster analysis...")
        
        # Load weekly data for offensive players
        weekly_data = nfl.import_weekly_data(self.years)
        
        # Filter to meaningful performances
        weekly_data = weekly_data[
            (weekly_data['attempts'] > 0) |
            (weekly_data['carries'] > 0) |
            (weekly_data['targets'] > 0)
        ]
        
        grades_list = []
        
        # Process each player-game
        for _, row in weekly_data.iterrows():
            try:
                # Determine position group
                if row['position'] in ['QB']:
                    grade = self._calculate_simple_qb_grade(row)
                    pos_group = 'QB'
                elif row['position'] in ['RB', 'FB']:
                    grade = self._calculate_simple_rb_grade(row)
                    pos_group = 'RB'
                elif row['position'] in ['WR', 'TE']:
                    grade = self._calculate_simple_wr_te_grade(row)
                    pos_group = 'WR_TE'
                else:
                    continue
                
                grades_list.append({
                    'player_id': row['player_id'],
                    'player_name': row.get('player_name', row.get('player_display_name')),
                    'team': row.get('recent_team'),
                    'position': row['position'],
                    'position_group': pos_group,
                    'player_type': 'OFFENSE',
                    'season': row['season'],
                    'week': row['week'],
                    'numeric_grade': grade
                })
            except Exception as e:
                continue
        
        # Add defensive players with simplified grading
        defensive_grades = self._calculate_simple_defensive_grades()
        grades_list.extend(defensive_grades)
        
        # Convert to DataFrame
        self.player_grades = pd.DataFrame(grades_list)
        
        # Filter to players with minimum games (3+)
        if not self.player_grades.empty:
            game_counts = self.player_grades.groupby('player_id').size()
            qualified_players = game_counts[game_counts >= 3].index
            self.player_grades = self.player_grades[
                self.player_grades['player_id'].isin(qualified_players)
            ]
        
        print(f"Player grades calculated for {self.player_grades['player_id'].nunique()} players")
        
        # Debug: Show grade distribution and position breakdown
        if not self.player_grades.empty:
            print(f"Grade distribution:")
            print(f"  Mean: {self.player_grades['numeric_grade'].mean():.1f}")
            print(f"  Median: {self.player_grades['numeric_grade'].median():.1f}")
            print(f"  Min: {self.player_grades['numeric_grade'].min():.1f}")
            print(f"  Max: {self.player_grades['numeric_grade'].max():.1f}")
            
            print(f"Position breakdown:")
            pos_counts = self.player_grades['position_group'].value_counts()
            for pos, count in pos_counts.items():
                avg_grade = self.player_grades[self.player_grades['position_group'] == pos]['numeric_grade'].mean()
                print(f"  {pos}: {count} players (avg: {avg_grade:.1f})")
            
            print(f"Team breakdown (top 10):")
            team_counts = self.player_grades['team'].value_counts().head(10)
            for team, count in team_counts.items():
                print(f"  {team}: {count} player-games")
    
    def _calculate_simple_qb_grade(self, row):
        """Simplified QB grading that produces reasonable 50-90 range"""
        base_score = 50
        
        # Passing yards (0-15 points)
        yards_score = min(row['passing_yards'] / 20, 15)
        
        # Completion percentage (0-15 points)
        if row['attempts'] > 0:
            comp_pct = row['completions'] / row['attempts']
            comp_score = max(0, (comp_pct - 0.5) * 30)
        else:
            comp_score = 0
        
        # Touchdowns (0-15 points)
        td_score = min(row['passing_tds'] * 5, 15)
        
        # Interception penalty (0 to -10 points)
        int_penalty = min(row['interceptions'] * -3, 0)
        
        total_score = base_score + yards_score + comp_score + td_score + int_penalty
        return max(min(total_score, 95), 25)
    
    def _calculate_simple_rb_grade(self, row):
        """Simplified RB grading"""
        base_score = 50
        
        # Rushing yards (0-20 points)
        rush_score = min(row['rushing_yards'] / 8, 20)
        
        # Yards per carry bonus (0-10 points)
        if row['carries'] > 0:
            ypc = row['rushing_yards'] / row['carries']
            ypc_score = max(0, (ypc - 3.5) * 5)
        else:
            ypc_score = 0
        
        # Touchdowns (0-15 points)
        td_score = min((row['rushing_tds'] + row['receiving_tds']) * 7, 15)
        
        # Receiving contribution (0-10 points)
        rec_score = min(row['receiving_yards'] / 15 + row['receptions'], 10)
        
        total_score = base_score + rush_score + ypc_score + td_score + rec_score
        return max(min(total_score, 95), 25)
    
    def _calculate_simple_wr_te_grade(self, row):
        """Simplified WR/TE grading"""
        base_score = 50
        
        # Receiving yards (0-25 points)
        yards_score = min(row['receiving_yards'] / 6, 25)
        
        # Receptions (0-15 points)
        rec_score = min(row['receptions'] * 2.5, 15)
        
        # Touchdowns (0-15 points)
        td_score = min(row['receiving_tds'] * 8, 15)
        
        # Catch rate bonus (0-5 points)
        if row['targets'] > 0:
            catch_rate = row['receptions'] / row['targets']
            catch_score = max(0, (catch_rate - 0.6) * 12.5)
        else:
            catch_score = 0
        
        total_score = base_score + yards_score + rec_score + td_score + catch_score
        return max(min(total_score, 95), 25)
    
    def _calculate_simple_defensive_grades(self):
        """Calculate simplified defensive grades from play-by-play data"""
        if self.pbp_data is None:
            return []
        
        defensive_stats = {}
        
        # Process sacks (full and half sacks)
        sacks = self.pbp_data[self.pbp_data['sack'] == 1]
        for _, play in sacks.iterrows():
            # Full sacks
            if pd.notna(play.get('sack_player_name')):
                key = (play['sack_player_id'], play['sack_player_name'], 
                      play['season'], play['week'], play.get('defteam', 'UNK'))
                if key not in defensive_stats:
                    defensive_stats[key] = {'sacks': 0, 'tackles': 0, 'ints': 0, 'pds': 0, 'ff': 0}
                defensive_stats[key]['sacks'] += 1.0
            
            # Half sacks
            for half_sack_col in ['half_sack_1_player_name', 'half_sack_2_player_name']:
                if pd.notna(play.get(half_sack_col)):
                    id_col = half_sack_col.replace('_name', '_id')
                    key = (play[id_col], play[half_sack_col], 
                          play['season'], play['week'], play.get('defteam', 'UNK'))
                    if key not in defensive_stats:
                        defensive_stats[key] = {'sacks': 0, 'tackles': 0, 'ints': 0, 'pds': 0, 'ff': 0}
                    defensive_stats[key]['sacks'] += 0.5
        
        # Process interceptions
        ints = self.pbp_data[self.pbp_data['interception'] == 1]
        for _, play in ints.iterrows():
            if pd.notna(play.get('interception_player_name')):
                key = (play['interception_player_id'], play['interception_player_name'],
                      play['season'], play['week'], play.get('defteam', 'UNK'))
                if key not in defensive_stats:
                    defensive_stats[key] = {'sacks': 0, 'tackles': 0, 'ints': 0, 'pds': 0, 'ff': 0}
                defensive_stats[key]['ints'] += 1
        
        # Process tackles
        for _, play in self.pbp_data.iterrows():
            # Solo tackles
            if pd.notna(play.get('solo_tackle_1_player_name')):
                key = (play['solo_tackle_1_player_id'], play['solo_tackle_1_player_name'],
                      play['season'], play['week'], play.get('defteam', 'UNK'))
                if key not in defensive_stats:
                    defensive_stats[key] = {'sacks': 0, 'tackles': 0, 'ints': 0, 'pds': 0, 'ff': 0}
                defensive_stats[key]['tackles'] += 1
            
            # Assist tackles
            for assist_col in ['assist_tackle_1_player_name', 'assist_tackle_2_player_name']:
                if pd.notna(play.get(assist_col)):
                    id_col = assist_col.replace('_name', '_id')
                    key = (play[id_col], play[assist_col],
                          play['season'], play['week'], play.get('defteam', 'UNK'))
                    if key not in defensive_stats:
                        defensive_stats[key] = {'sacks': 0, 'tackles': 0, 'ints': 0, 'pds': 0, 'ff': 0}
                    defensive_stats[key]['tackles'] += 0.5
        
        # Process pass deflections
        for _, play in self.pbp_data.iterrows():
            if pd.notna(play.get('pass_defense_1_player_name')):
                key = (play['pass_defense_1_player_id'], play['pass_defense_1_player_name'],
                      play['season'], play['week'], play.get('defteam', 'UNK'))
                if key not in defensive_stats:
                    defensive_stats[key] = {'sacks': 0, 'tackles': 0, 'ints': 0, 'pds': 0, 'ff': 0}
                defensive_stats[key]['pds'] += 1
        
        # Process forced fumbles
        for _, play in self.pbp_data.iterrows():
            if pd.notna(play.get('forced_fumble_player_1_player_name')):
                key = (play['forced_fumble_player_1_player_id'], play['forced_fumble_player_1_player_name'],
                      play['season'], play['week'], play.get('defteam', 'UNK'))
                if key not in defensive_stats:
                    defensive_stats[key] = {'sacks': 0, 'tackles': 0, 'ints': 0, 'pds': 0, 'ff': 0}
                defensive_stats[key]['ff'] += 1
        
        # Convert to grades and aggregate by player-season
        weekly_grades = []
        for (player_id, player_name, season, week, team), stats in defensive_stats.items():
            # Simple defensive grading
            base_score = 55  # Base defensive participation
            
            # Scale defensive stats for weekly performance
            sack_score = stats['sacks'] * 10      # 10 points per sack
            int_score = stats['ints'] * 12        # 12 points per INT
            tackle_score = min(stats['tackles'] * 2, 15)  # Up to 15 points for tackles
            pd_score = stats['pds'] * 4           # 4 points per pass deflection
            ff_score = stats['ff'] * 8            # 8 points per forced fumble
            
            grade = base_score + sack_score + int_score + tackle_score + pd_score + ff_score
            grade = max(min(grade, 95), 30)
            
            weekly_grades.append({
                'player_id': player_id,
                'player_name': player_name,
                'team': team,
                'position': 'DEF',
                'position_group': 'DEFENSE',
                'player_type': 'DEFENSE',
                'season': season,
                'week': week,
                'numeric_grade': grade
            })
        
        print(f"Processed defensive stats for {len(defensive_stats)} player-week combinations")
        return weekly_grades
    
    def extract_coaching_info(self):
        """Extract coaching information from schedule data"""
        if self.schedule_data is None:
            return
        
        coaches = {}
        for _, game in self.schedule_data.iterrows():
            season = game['season']
            
            # Home team coach
            if pd.notna(game['home_coach']) and pd.notna(game['home_team']):
                coach_key = (game['home_coach'], season)
                if coach_key not in coaches:
                    coaches[coach_key] = {
                        'name': game['home_coach'],
                        'season': season,
                        'teams': set(),
                        'games': []
                    }
                coaches[coach_key]['teams'].add(game['home_team'])
                coaches[coach_key]['games'].append({
                    'game_id': game['game_id'],
                    'team': game['home_team'],
                    'is_home': True,
                    'week': game['week'],
                    'result': 'W' if pd.notna(game['home_score']) and pd.notna(game['away_score']) and game['home_score'] > game['away_score'] else 'L' if pd.notna(game['home_score']) and pd.notna(game['away_score']) else None
                })
            
            # Away team coach
            if pd.notna(game['away_coach']) and pd.notna(game['away_team']):
                coach_key = (game['away_coach'], season)
                if coach_key not in coaches:
                    coaches[coach_key] = {
                        'name': game['away_coach'],
                        'season': season,
                        'teams': set(),
                        'games': []
                    }
                coaches[coach_key]['teams'].add(game['away_team'])
                coaches[coach_key]['games'].append({
                    'game_id': game['game_id'],
                    'team': game['away_team'],
                    'is_home': False,
                    'week': game['week'],
                    'result': 'W' if pd.notna(game['away_score']) and pd.notna(game['home_score']) and game['away_score'] > game['home_score'] else 'L' if pd.notna(game['away_score']) and pd.notna(game['home_score']) else None
                })
        
        self.coaching_data = coaches
        print(f"Extracted data for {len(coaches)} coach-season combinations")
    
    def analyze_roster_quality(self, team, season):
        """FIXED: Analyze roster quality with proper grade scaling"""
        if self.player_grades is None or self.player_grades.empty:
            print(f"No player grades available for roster analysis")
            return None
        
        print(f"Analyzing roster quality for {team} ({season})...")
        
        # Get team's players for the season
        team_players = self.player_grades[
            (self.player_grades['team'] == team) & 
            (self.player_grades['season'] == season)
        ]
        
        if team_players.empty:
            print(f"No player data found for {team} in {season}")
            return None
        
        # Calculate player averages for the season
        player_avgs = team_players.groupby(['player_id', 'player_name', 'position_group']).agg({
            'numeric_grade': 'mean'
        }).reset_index()
        
        analysis = {}
        
        # Overall roster quality
        overall_avg = player_avgs['numeric_grade'].mean()
        analysis['overall_avg_grade'] = overall_avg
        analysis['total_players'] = len(player_avgs)
        
        # FIXED: Use realistic grade tiers
        analysis['elite_players'] = len(player_avgs[player_avgs['numeric_grade'] >= 80])
        analysis['good_players'] = len(player_avgs[
            (player_avgs['numeric_grade'] >= 70) & (player_avgs['numeric_grade'] < 80)
        ])
        analysis['average_players'] = len(player_avgs[
            (player_avgs['numeric_grade'] >= 60) & (player_avgs['numeric_grade'] < 70)
        ])
        analysis['below_avg_players'] = len(player_avgs[player_avgs['numeric_grade'] < 60])
        
        # Position-specific analysis
        for pos_group in ['QB', 'RB', 'WR_TE', 'DEFENSE']:
            pos_players = player_avgs[player_avgs['position_group'] == pos_group]
            if not pos_players.empty:
                analysis[f'{pos_group.lower()}_avg_grade'] = pos_players['numeric_grade'].mean()
                analysis[f'{pos_group.lower()}_count'] = len(pos_players)
            else:
                analysis[f'{pos_group.lower()}_avg_grade'] = None
                analysis[f'{pos_group.lower()}_count'] = 0
        
        # FIXED: Roster tier classification based on realistic grades
        if overall_avg >= 75:
            analysis['roster_tier'] = 'Elite'
        elif overall_avg >= 70:
            analysis['roster_tier'] = 'Good'
        elif overall_avg >= 65:
            analysis['roster_tier'] = 'Average'
        elif overall_avg >= 60:
            analysis['roster_tier'] = 'Below Average'
        else:
            analysis['roster_tier'] = 'Poor'
        
        # Calculate depth (standard deviation - lower is better for consistency)
        analysis['roster_depth'] = player_avgs['numeric_grade'].std()
        
        print(f"Roster analysis complete:")
        print(f"  Overall grade: {overall_avg:.1f} ({analysis['roster_tier']})")
        print(f"  Elite players (80+): {analysis['elite_players']}")
        print(f"  Good players (70-79): {analysis['good_players']}")
        print(f"  Total players analyzed: {analysis['total_players']}")
        
        return analysis
    
    def calculate_base_coaching_grade(self, coach_name, season):
        """Calculate base coaching performance before roster adjustment"""
        
        # Get team(s) for this coach
        coach_teams = []
        for (coach, seas), data in self.coaching_data.items():
            if coach == coach_name and seas == season:
                coach_teams.extend(list(data['teams']))
        
        if not coach_teams:
            return None
        
        # Simplified base grading - you can integrate your existing logic here
        team_record = self._get_team_record(coach_name, season)
        if not team_record:
            return None
        
        # Base grade from win percentage with adjustments
        win_pct = team_record.get('win_pct', 50)
        
        # Convert win percentage to grade (with some adjustments for context)
        if win_pct >= 80:  # 13+ wins
            base_grade = 90 + (win_pct - 80) / 2  # 90-100
        elif win_pct >= 70:  # 11-12 wins
            base_grade = 80 + (win_pct - 70)  # 80-90
        elif win_pct >= 60:  # 10 wins
            base_grade = 75 + (win_pct - 60) / 2  # 75-80
        elif win_pct >= 50:  # 8-9 wins
            base_grade = 65 + (win_pct - 50)  # 65-75
        elif win_pct >= 40:  # 6-7 wins
            base_grade = 55 + (win_pct - 40)  # 55-65
        else:  # 5 or fewer wins
            base_grade = 40 + win_pct / 4  # 40-55
        
        return {
            'base_grade': base_grade,
            'win_pct': win_pct,
            'record': f"{team_record['wins']}-{team_record['losses']}"
        }
    
    def calculate_roster_adjusted_grade(self, coach_name, season):
        """FIXED: Calculate roster-adjusted coaching grade"""
        
        # Get base coaching performance
        base_performance = self.calculate_base_coaching_grade(coach_name, season)
        if not base_performance:
            return None
        
        # Get team(s) for this coach
        coach_teams = []
        for (coach, seas), data in self.coaching_data.items():
            if coach == coach_name and seas == season:
                coach_teams.extend(list(data['teams']))
        
        if not coach_teams:
            return None
        
        # Analyze roster quality (use first team if multiple)
        roster_analysis = self.analyze_roster_quality(coach_teams[0], season)
        if not roster_analysis:
            return None
        
        base_grade = base_performance['base_grade']
        roster_grade = roster_analysis['overall_avg_grade']
        
        # FIXED: More reasonable adjustment logic
        # The adjustment should be proportional to how much better/worse the roster is than average
        league_avg_roster_grade = 65  # Assume league average is around 65
        
        # Calculate roster adjustment factor
        roster_difference = roster_grade - league_avg_roster_grade
        
        # FIXED: Smaller adjustment factor (max ±15 points instead of massive swings)
        max_adjustment = 15
        adjustment_factor = min(max(roster_difference / 10 * max_adjustment, -max_adjustment), max_adjustment)
        
        # Apply adjustment
        adjusted_grade = base_grade - adjustment_factor  # Subtract because better roster should require higher performance
        adjusted_grade = max(min(adjusted_grade, 100), 20)  # Cap between 20-100
        
        # Calculate efficiency (how much better/worse than expected given roster)
        expected_grade = 50 + (roster_grade - league_avg_roster_grade) * 0.5  # More modest expectations
        efficiency = base_grade - expected_grade
        
        return {
            'base_grade': base_grade,
            'adjusted_grade': adjusted_grade,
            'roster_quality': roster_analysis,
            'adjustment': adjustment_factor,
            'efficiency': efficiency,
            'record': base_performance['record']
        }
    
    def _get_team_record(self, coach_name, season):
        """Get team record for a coach"""
        for (coach, seas), data in self.coaching_data.items():
            if coach == coach_name and seas == season:
                wins = sum(1 for g in data['games'] if g['result'] == 'W')
                total_games = len([g for g in data['games'] if g['result'] is not None])
                win_pct = (wins / total_games * 100) if total_games > 0 else 0
                return {'wins': wins, 'losses': total_games - wins, 'win_pct': win_pct}
        return None
    
    def get_letter_grade(self, score):
        """Convert numerical grade to letter grade"""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "A-"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "B-"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        elif score >= 55:
            return "C-"
        elif score >= 50:
            return "D"
        else:
            return "F"
    
    def print_roster_aware_report(self, coach_name, season):
        """Print comprehensive roster-aware coaching report"""
        print(f"\n{'='*80}")
        print(f"ROSTER-AWARE COACHING REPORT: {coach_name}")
        print(f"Season: {season}")
        print(f"{'='*80}")
        
        # Get analysis
        analysis = self.calculate_roster_adjusted_grade(coach_name, season)
        if not analysis:
            print("No data available for analysis")
            return
        
        roster_quality = analysis['roster_quality']
        
        print(f"Team Record: {analysis['record']}")
        print(f"Base Coaching Grade: {analysis['base_grade']:.1f} ({self.get_letter_grade(analysis['base_grade'])})")
        
        print(f"\n{'ROSTER ANALYSIS':^80}")
        print("-" * 80)
        print(f"Overall Roster Grade: {roster_quality['overall_avg_grade']:.1f} ({roster_quality['roster_tier']})")
        print(f"Elite Players (80+): {roster_quality['elite_players']}")
        print(f"Good Players (70-79): {roster_quality['good_players']}")
        print(f"Average Players (60-69): {roster_quality['average_players']}")
        print(f"Below Average Players (<60): {roster_quality['below_avg_players']}")
        print(f"Total Players Analyzed: {roster_quality['total_players']}")
        
        # Position breakdown
        print(f"\nPosition Group Averages:")
        for pos in ['qb', 'rb', 'wr_te', 'defense']:
            avg_grade = roster_quality.get(f'{pos}_avg_grade')
            count = roster_quality.get(f'{pos}_count', 0)
            if avg_grade is not None:
                print(f"  {pos.upper()}: {avg_grade:.1f} ({count} players)")
            else:
                print(f"  {pos.upper()}: No data ({count} players)")
        
        print(f"\n{'ROSTER-ADJUSTED PERFORMANCE':^80}")
        print("-" * 80)
        print(f"Adjustment Applied: {analysis['adjustment']:+.1f} points")
        print(f"Final Adjusted Grade: {analysis['adjusted_grade']:.1f} ({self.get_letter_grade(analysis['adjusted_grade'])})")
        print(f"Coaching Efficiency: {analysis['efficiency']:+.1f} (vs expected)")
        
        # Interpretation
        if analysis['efficiency'] > 10:
            interpretation = "Exceptional coaching - significantly outperforming roster talent"
        elif analysis['efficiency'] > 5:
            interpretation = "Good coaching - performing above roster expectations"
        elif analysis['efficiency'] > -5:
            interpretation = "Average coaching - performing as expected given roster"
        elif analysis['efficiency'] > -10:
            interpretation = "Below average coaching - underperforming roster talent"
        else:
            interpretation = "Poor coaching - significantly underperforming roster potential"
        
        print(f"\nInterpretation: {interpretation}")
    
    def compare_coaches_roster_aware(self, coach_names, season):
        """Compare coaches with roster-aware adjustments"""
        print(f"\n{'ROSTER-AWARE COACH COMPARISON':^80}")
        print(f"Season: {season}")
        print("=" * 80)
        
        results = []
        for coach in coach_names:
            analysis = self.calculate_roster_adjusted_grade(coach, season)
            if analysis:
                results.append({
                    'coach': coach,
                    'base_grade': analysis['base_grade'],
                    'adjusted_grade': analysis['adjusted_grade'],
                    'roster_grade': analysis['roster_quality']['overall_avg_grade'],
                    'roster_tier': analysis['roster_quality']['roster_tier'],
                    'efficiency': analysis['efficiency'],
                    'record': analysis['record']
                })
        
        if not results:
            print("No data available for comparison")
            return
        
        # Print comparison table
        print(f"{'Coach':<20} {'Base':<10} {'Adjusted':<10} {'Roster':<10} {'Efficiency':<12} {'Record':<10}")
        print("-" * 80)
        
        for result in sorted(results, key=lambda x: x['adjusted_grade'], reverse=True):
            print(f"{result['coach'][:19]:<20} "
                  f"{result['base_grade']:.1f} ({self.get_letter_grade(result['base_grade']):<2}) "
                  f"{result['adjusted_grade']:.1f} ({self.get_letter_grade(result['adjusted_grade']):<2}) "
                  f"{result['roster_grade']:.1f} ({result['roster_tier'][:4]:<4}) "
                  f"{result['efficiency']:+.1f}{'':>6} "
                  f"{result['record']}")
    
    def get_available_coaches(self, season=None):
        """Get list of available coaches"""
        coaches = set()
        for (coach, seas), data in self.coaching_data.items():
            if season is None or seas == season:
                coaches.add(coach)
        return sorted(list(coaches))


    def find_coaching_overperformers(self, season, min_efficiency=10):
        """Find coaches significantly outperforming their roster talent"""
        coaches = self.get_available_coaches(season)
        overperformers = []
        
        for coach in coaches:
            analysis = self.calculate_roster_adjusted_grade(coach, season)
            if analysis and analysis['efficiency'] >= min_efficiency:
                overperformers.append({
                    'coach': coach,
                    'efficiency': analysis['efficiency'],
                    'base_grade': analysis['base_grade'],
                    'adjusted_grade': analysis['adjusted_grade'],
                    'roster_tier': analysis['roster_quality']['roster_tier'],
                    'record': analysis['record']
                })
        
        return sorted(overperformers, key=lambda x: x['efficiency'], reverse=True)
    
    def find_coaching_underperformers(self, season, max_efficiency=-10):
        """Find coaches underperforming relative to their roster talent"""
        coaches = self.get_available_coaches(season)
        underperformers = []
        
        for coach in coaches:
            analysis = self.calculate_roster_adjusted_grade(coach, season)
            if analysis and analysis['efficiency'] <= max_efficiency:
                underperformers.append({
                    'coach': coach,
                    'efficiency': analysis['efficiency'],
                    'base_grade': analysis['base_grade'],
                    'adjusted_grade': analysis['adjusted_grade'],
                    'roster_tier': analysis['roster_quality']['roster_tier'],
                    'record': analysis['record']
                })
        
        return sorted(underperformers, key=lambda x: x['efficiency'])
    
    def analyze_roster_vs_performance(self, season):
        """Analyze the relationship between roster quality and team performance"""
        coaches = self.get_available_coaches(season)
        data = []
        
        print(f"\n{'ROSTER QUALITY vs TEAM PERFORMANCE ANALYSIS':^80}")
        print(f"Season: {season}")
        print("=" * 80)
        
        for coach in coaches:
            analysis = self.calculate_roster_adjusted_grade(coach, season)
            if analysis:
                data.append({
                    'coach': coach,
                    'roster_grade': analysis['roster_quality']['overall_avg_grade'],
                    'base_grade': analysis['base_grade'],
                    'efficiency': analysis['efficiency'],
                    'roster_tier': analysis['roster_quality']['roster_tier'],
                    'elite_players': analysis['roster_quality']['elite_players'],
                    'good_players': analysis['roster_quality']['good_players']
                })
        
        if not data:
            print("No data available")
            return
        
        # Sort by roster quality
        data.sort(key=lambda x: x['roster_grade'], reverse=True)
        
        print(f"{'Coach':<20} {'Roster':<10} {'Performance':<12} {'Efficiency':<12} {'Elite/Good':<12}")
        print("-" * 80)
        
        for item in data:
            print(f"{item['coach'][:19]:<20} "
                  f"{item['roster_grade']:.1f} ({item['roster_tier'][:4]}) "
                  f"{item['base_grade']:.1f} ({self.get_letter_grade(item['base_grade']):<2}) "
                  f"{item['efficiency']:+.1f}{'':>6} "
                  f"{item['elite_players']}/{item['good_players']}")
        
        return data


def main():
    """Enhanced demonstration with comprehensive analysis"""
    
    # Initialize system
    analytics = RosterAwareCoachingAnalytics(years=[2023])
    
    # Load data
    analytics.load_data()
    analytics.extract_coaching_info()
    analytics.calculate_player_grades()
    
    # Get available coaches
    coaches = analytics.get_available_coaches(season=2023)
    print(f"\nAvailable coaches: {len(coaches)}")
    
    if coaches:
        # Demo with first few coaches
        demo_coaches = coaches[:5]
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ROSTER-AWARE ANALYSIS")
        print(f"{'='*80}")
        
        # Individual analysis for first coach
        analytics.print_roster_aware_report(demo_coaches[0], 2023)
        
        # Comparison of first 3 coaches
        print(f"\n{'COACHING COMPARISON':^80}")
        analytics.compare_coaches_roster_aware(demo_coaches[:3], 2023)
        
        # Find overperformers and underperformers
        print(f"\n{'COACHING EFFICIENCY ANALYSIS':^80}")
        print("=" * 80)
        
        overperformers = analytics.find_coaching_overperformers(2023, min_efficiency=15)
        if overperformers:
            print(f"\nTOP OVERPERFORMERS (Efficiency +15 or better):")
            print(f"{'Coach':<20} {'Efficiency':<12} {'Record':<10} {'Roster Tier':<15}")
            print("-" * 60)
            for coach_data in overperformers[:5]:
                print(f"{coach_data['coach'][:19]:<20} "
                      f"{coach_data['efficiency']:+.1f}{'':>6} "
                      f"{coach_data['record']:<10} "
                      f"{coach_data['roster_tier']}")
        
        underperformers = analytics.find_coaching_underperformers(2023, max_efficiency=-5)
        if underperformers:
            print(f"\nUNDERPERFORMERS (Efficiency -5 or worse):")
            print(f"{'Coach':<20} {'Efficiency':<12} {'Record':<10} {'Roster Tier':<15}")
            print("-" * 60)
            for coach_data in underperformers[:5]:
                print(f"{coach_data['coach'][:19]:<20} "
                      f"{coach_data['efficiency']:+.1f}{'':>6} "
                      f"{coach_data['record']:<10} "
                      f"{coach_data['roster_tier']}")
        
        # Full roster vs performance analysis
        analytics.analyze_roster_vs_performance(2023)
        
        print(f"\n{'='*80}")
        print("ENHANCED SYSTEM FEATURES:")
        print("- Comprehensive defensive player grading")
        print("- Coaching efficiency analysis")
        print("- Overperformer/underperformer identification")
        print("- Roster quality vs performance correlation")
        print("- Position-specific roster breakdowns")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()