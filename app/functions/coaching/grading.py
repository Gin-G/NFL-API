#!/usr/bin/env python3
"""
Enhanced NFL Coaching Analytics System with Player Grade Context
Integrates player performance grades to provide roster-adjusted coaching evaluations
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import the player grading system
# from ..players.grading import EnhancedNFLPlayerGrader
from ..players.grading import EnhancedNFLPlayerGrader

class RosterAwareCoachingAnalytics:
    """Enhanced coaching analytics that considers roster quality and player performance"""
    
    def __init__(self, years=None):
        """Initialize the enhanced coaching analytics system"""
        if years is None:
            years = [2023, 2024]
        self.years = years
        self.pbp_data = None
        self.schedule_data = None
        self.coaching_data = {}
        self.player_grader = None
        self.player_grades = None
        self.roster_data = None
        
    def load_data(self):
        """Load all necessary data including player grading system"""
        print("Loading NFL data for roster-aware coaching analytics...")
        
        # Load play-by-play data
        print("- Loading play-by-play data...")
        pbp_list = []
        for year in self.years:
            try:
                if year >= 2024:
                    year_pbp = nfl.import_pbp_data([year], include_participation=False)
                else:
                    year_pbp = nfl.import_pbp_data([year])
                pbp_list.append(year_pbp)
                print(f"  - {year}: {len(year_pbp)} plays loaded")
            except Exception as e:
                print(f"  - Error loading {year}: {e}")
        
        if pbp_list:
            self.pbp_data = pd.concat(pbp_list, ignore_index=True)
            print(f"Total plays loaded: {len(self.pbp_data)}")
        
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
        
        # Initialize and load player grading system
        print("- Initializing player grading system...")
        try:
            self.player_grader = EnhancedNFLPlayerGrader(years=self.years)
            print("  - Player grading system initialized successfully")
        except Exception as e:
            print(f"  - Error initializing player grader: {e}")
            self.player_grader = None
        
        print("Data loading complete!\n")
    
    def calculate_player_grades(self, min_games=3):
        """Calculate player grades for roster analysis"""
        if self.player_grader is None:
            print("Player grading system not available")
            return None
        
        print("Calculating player grades for roster analysis...")
        try:
            # Get all player grades
            all_grades = self.player_grader.calculate_all_grades(min_games=min_games)
            
            if all_grades.empty:
                print("No player grades calculated")
                return None
            
            # Get roster data
            self.roster_data = self.player_grader.rosters
            
            # Calculate average grades by player for the season
            player_season_grades = all_grades.groupby(['player_id', 'player_name', 'position', 'position_group', 'player_type']).agg({
                'numeric_grade': 'mean',
                'week': 'count'
            }).reset_index()
            
            player_season_grades.columns = ['player_id', 'player_name', 'position', 'position_group', 'player_type', 'avg_grade', 'games_played']
            
            # Merge with roster data to get team assignments
            if self.roster_data is not None and not self.roster_data.empty:
                roster_teams = self.roster_data.groupby(['player_id', 'season']).agg({
                    'team': 'first',
                    'week': 'count'
                }).reset_index()
                roster_teams.columns = ['player_id', 'season', 'primary_team', 'weeks_on_roster']
                
                player_season_grades = pd.merge(
                    player_season_grades,
                    roster_teams,
                    on='player_id',
                    how='left'
                )
            
            self.player_grades = player_season_grades
            print(f"Player grades calculated for {len(player_season_grades)} players")
            return player_season_grades
            
        except Exception as e:
            print(f"Error calculating player grades: {e}")
            return None
    
    def extract_coaching_info(self):
        """Extract coaching information from schedule data"""
        if self.schedule_data is None:
            print("No schedule data available")
            return
        
        print("Extracting coaching information...")
        
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
                    'opponent': game['away_team'],
                    'is_home': True,
                    'week': game['week'],
                    'score': game['home_score'],
                    'opp_score': game['away_score'],
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
                    'opponent': game['home_team'],
                    'is_home': False,
                    'week': game['week'],
                    'score': game['away_score'],
                    'opp_score': game['home_score'],
                    'result': 'W' if pd.notna(game['away_score']) and pd.notna(game['home_score']) and game['away_score'] > game['home_score'] else 'L' if pd.notna(game['away_score']) and pd.notna(game['home_score']) else None
                })
        
        self.coaching_data = coaches
        print(f"Extracted data for {len(coaches)} coach-season combinations")
        
        return coaches
    
    def get_roster_quality_metrics(self, team, season, position_group=None):
        """Calculate roster quality metrics for a team and season"""
        if self.player_grades is None:
            return None
        
        # Filter to team and season
        team_players = self.player_grades[
            (self.player_grades['primary_team'] == team) & 
            (self.player_grades['season'] == season)
        ].copy()
        
        if team_players.empty:
            return None
        
        # Filter by position group if specified
        if position_group:
            team_players = team_players[team_players['position_group'] == position_group]
            if team_players.empty:
                return None
        
        # Calculate metrics
        metrics = {
            'total_players': len(team_players),
            'avg_grade': team_players['avg_grade'].mean(),
            'median_grade': team_players['avg_grade'].median(),
            'top_players': len(team_players[team_players['avg_grade'] >= 80]),  # B- or better
            'elite_players': len(team_players[team_players['avg_grade'] >= 90]),  # A- or better
            'below_avg_players': len(team_players[team_players['avg_grade'] < 70]),  # Below C-
            'grade_std': team_players['avg_grade'].std(),
            'depth_score': self._calculate_depth_score(team_players)
        }
        
        # Position-specific metrics
        if position_group:
            metrics['position_group'] = position_group
        else:
            # Overall roster with position breakdowns
            position_breakdown = team_players.groupby('position_group').agg({
                'avg_grade': 'mean',
                'player_id': 'count'
            }).to_dict()
            metrics['position_breakdown'] = position_breakdown
        
        return metrics
    
    def _calculate_depth_score(self, team_players):
        """Calculate a depth score based on how many quality players a team has"""
        if len(team_players) == 0:
            return 0
        
        # Weight players by their grades - elite players contribute more to depth
        depth_points = 0
        
        for _, player in team_players.iterrows():
            grade = player['avg_grade']
            if grade >= 90:  # Elite (A-)
                depth_points += 4
            elif grade >= 80:  # Very Good (B-)
                depth_points += 3
            elif grade >= 70:  # Average (C-)
                depth_points += 2
            elif grade >= 60:  # Below Average (D-)
                depth_points += 1
            # Below 60 (F) adds 0 points
        
        # Normalize by expected roster size for position group
        return depth_points / len(team_players) * 10  # Scale to 0-40 range
    
    def calculate_roster_adjusted_coaching_grade(self, coach_name, season=None):
        """Calculate coaching grade adjusted for roster quality"""
        
        # First get the base coaching grade using existing methods
        base_grades = self.grade_coach_performance(coach_name, season)
        if not base_grades:
            return None
        
        # Get the team(s) this coach coached
        coach_teams = []
        for (coach, seas), data in self.coaching_data.items():
            if coach == coach_name and (season is None or seas == season):
                coach_teams.extend(list(data['teams']))
                if season is None:
                    season = seas  # Use the season from the data
        
        if not coach_teams:
            print(f"No team data found for coach {coach_name}")
            return base_grades
        
        # Calculate roster quality for each team coached
        roster_adjustments = {}
        
        for team in set(coach_teams):
            print(f"Analyzing roster quality for {team} ({season})...")
            
            # Get overall roster quality
            overall_quality = self.get_roster_quality_metrics(team, season)
            
            # Get position-specific quality
            offensive_quality = self.get_roster_quality_metrics(team, season, 'OFFENSE')
            defensive_quality = self.get_roster_quality_metrics(team, season, 'DEFENSE')
            
            if overall_quality:
                roster_adjustments[team] = {
                    'overall': overall_quality,
                    'offensive': offensive_quality,
                    'defensive': defensive_quality
                }
        
        if not roster_adjustments:
            print("No roster quality data available - using base grades")
            return base_grades
        
        # Calculate roster-adjusted grades
        adjusted_grades = base_grades.copy()
        
        # Average roster quality across teams coached (if multiple)
        avg_overall_grade = np.mean([r['overall']['avg_grade'] for r in roster_adjustments.values() if r['overall']])
        avg_offensive_grade = np.mean([r['offensive']['avg_grade'] for r in roster_adjustments.values() if r['offensive']])
        avg_defensive_grade = np.mean([r['defensive']['avg_grade'] for r in roster_adjustments.values() if r['defensive']])
        
        # Calculate adjustment factors based on roster quality
        # League average player grade is around 70, so we adjust based on deviation from that
        league_avg = 70
        
        # More sophisticated adjustment - consider both grade and depth
        overall_depth = np.mean([r['overall']['depth_score'] for r in roster_adjustments.values() if r['overall']])
        
        # Adjustment factors (more moderate than before)
        def calculate_adjustment_factor(roster_grade, depth_score, league_avg=70):
            """Calculate how much to adjust coaching grade based on roster"""
            grade_diff = roster_grade - league_avg
            depth_factor = min(depth_score / 20, 1.0)  # Normalize depth
            
            # More nuanced adjustment
            if roster_grade >= 85:  # Elite roster
                return 0.85 + (0.1 * depth_factor)  # Harder to look good with elite talent
            elif roster_grade >= 75:  # Good roster
                return 0.92 + (0.05 * depth_factor)  # Slight penalty
            elif roster_grade >= 65:  # Average roster
                return 1.0  # No adjustment
            elif roster_grade >= 55:  # Below average roster
                return 1.08 + (0.1 * (1 - depth_factor))  # Bonus for poor roster
            else:  # Poor roster
                return 1.15 + (0.15 * (1 - depth_factor))  # Significant bonus for terrible roster
        
        # Apply adjustments
        if avg_overall_grade:
            overall_adjustment = calculate_adjustment_factor(avg_overall_grade, overall_depth)
            print(f"Overall roster adjustment factor: {overall_adjustment:.3f} (avg grade: {avg_overall_grade:.1f})")
        
        if avg_offensive_grade and 'offensive_overall' in adjusted_grades:
            off_depth = np.mean([r['offensive']['depth_score'] for r in roster_adjustments.values() if r['offensive']])
            off_adjustment = calculate_adjustment_factor(avg_offensive_grade, off_depth)
            adjusted_grades['offensive_overall'] *= off_adjustment
            print(f"Offensive roster adjustment factor: {off_adjustment:.3f} (avg grade: {avg_offensive_grade:.1f})")
            
            # Apply to individual offensive categories
            if 'offensive' in adjusted_grades:
                for category in adjusted_grades['offensive']:
                    adjusted_grades['offensive'][category] *= off_adjustment
        
        if avg_defensive_grade and 'defensive_overall' in adjusted_grades:
            def_depth = np.mean([r['defensive']['depth_score'] for r in roster_adjustments.values() if r['defensive']])
            def_adjustment = calculate_adjustment_factor(avg_defensive_grade, def_depth)
            adjusted_grades['defensive_overall'] *= def_adjustment
            print(f"Defensive roster adjustment factor: {def_adjustment:.3f} (avg grade: {avg_defensive_grade:.1f})")
            
            # Apply to individual defensive categories
            if 'defensive' in adjusted_grades:
                for category in adjusted_grades['defensive']:
                    adjusted_grades['defensive'][category] *= def_adjustment
        
        # Recalculate overall grade with roster adjustments
        if 'offensive_overall' in adjusted_grades and 'defensive_overall' in adjusted_grades:
            coach_specialty = self._determine_coach_specialty(coach_name, season)
            
            if coach_specialty == 'offense':
                adjusted_grades['overall'] = (adjusted_grades['offensive_overall'] * 0.6) + (adjusted_grades['defensive_overall'] * 0.4)
            elif coach_specialty == 'defense':
                adjusted_grades['overall'] = (adjusted_grades['offensive_overall'] * 0.4) + (adjusted_grades['defensive_overall'] * 0.6)
            else:
                adjusted_grades['overall'] = (adjusted_grades['offensive_overall'] * 0.5) + (adjusted_grades['defensive_overall'] * 0.5)
        
        # Add roster context to the grades
        adjusted_grades['roster_context'] = {
            'roster_adjustments': roster_adjustments,
            'avg_roster_grade': avg_overall_grade,
            'avg_offensive_grade': avg_offensive_grade,
            'avg_defensive_grade': avg_defensive_grade,
            'roster_quality_tier': self._get_roster_tier(avg_overall_grade) if avg_overall_grade else 'Unknown'
        }
        
        # Ensure grades don't exceed realistic bounds
        for category in ['offensive_overall', 'defensive_overall', 'overall']:
            if category in adjusted_grades:
                adjusted_grades[category] = min(100, max(40, adjusted_grades[category]))
        
        return adjusted_grades
    
    def _get_roster_tier(self, avg_grade):
        """Classify roster quality into tiers"""
        if avg_grade >= 80:
            return 'Elite'
        elif avg_grade >= 75:
            return 'Very Good'
        elif avg_grade >= 70:
            return 'Average'
        elif avg_grade >= 65:
            return 'Below Average'
        else:
            return 'Poor'
    
    def print_roster_aware_coach_report(self, coach_name, season=None):
        """Print comprehensive coaching report with roster context"""
        print(f"\n{'='*80}")
        print(f"ROSTER-AWARE COACHING REPORT: {coach_name}")
        if season:
            print(f"Season: {season}")
        print(f"{'='*80}")
        
        # Get roster-adjusted grades
        adjusted_grades = self.calculate_roster_adjusted_coaching_grade(coach_name, season)
        if not adjusted_grades:
            print("No grading data available")
            return
        
        # Print basic coach info
        coach_info = []
        coach_specialty = self._determine_coach_specialty(coach_name, season)
        
        for (coach, seas), data in self.coaching_data.items():
            if coach == coach_name and (season is None or seas == season):
                teams = list(data['teams'])
                wins = sum(1 for g in data['games'] if g['result'] == 'W')
                total_games = len([g for g in data['games'] if g['result'] is not None])
                win_pct = (wins / total_games * 100) if total_games > 0 else 0
                
                coach_info.append({
                    'season': seas,
                    'teams': teams,
                    'record': f"{wins}-{total_games-wins}",
                    'win_pct': win_pct,
                    'games_coached': len(data['games'])
                })
        
        for info in coach_info:
            print(f"Season: {info['season']}")
            print(f"Team(s): {', '.join(info['teams'])}")
            print(f"Record: {info['record']} ({info['win_pct']:.1f}%)")
            print(f"Specialty: {coach_specialty.title()}")
        
        # Print roster context
        if 'roster_context' in adjusted_grades:
            context = adjusted_grades['roster_context']
            print(f"\n{'ROSTER CONTEXT':^80}")
            print("-" * 80)
            print(f"Overall Roster Quality: {context['roster_quality_tier']} (Avg Grade: {context['avg_roster_grade']:.1f})")
            
            if context['avg_offensive_grade']:
                off_tier = self._get_roster_tier(context['avg_offensive_grade'])
                print(f"Offensive Talent Level: {off_tier} (Avg Grade: {context['avg_offensive_grade']:.1f})")
            
            if context['avg_defensive_grade']:
                def_tier = self._get_roster_tier(context['avg_defensive_grade'])
                print(f"Defensive Talent Level: {def_tier} (Avg Grade: {context['avg_defensive_grade']:.1f})")
            
            # Show specific roster strengths/weaknesses
            for team, roster_data in context['roster_adjustments'].items():
                if roster_data['overall']:
                    overall = roster_data['overall']
                    print(f"\n{team} Roster Analysis:")
                    print(f"  Overall: {overall['avg_grade']:.1f} avg, {overall['elite_players']} elite players, {overall['depth_score']:.1f} depth")
        
        # Print performance grades (now adjusted for roster)
        print(f"\n{'ROSTER-ADJUSTED PERFORMANCE':^80}")
        print("-" * 80)
        
        # Show both base and adjusted grades for comparison
        base_grades = self.grade_coach_performance(coach_name, season)
        
        def print_grade_comparison(category, base_score, adjusted_score):
            base_letter = self.get_letter_grade(base_score)
            adj_letter = self.get_letter_grade(adjusted_score)
            change = adjusted_score - base_score
            change_str = f"({change:+.1f})" if abs(change) > 0.1 else ""
            
            indicator = "Outstanding" if adjusted_score >= 90 else "Very Good" if adjusted_score >= 80 else "Average" if adjusted_score >= 70 else "Below Average" if adjusted_score >= 60 else "Poor"
            
            print(f"{category:<35} {adjusted_score:>6.1f} ({adj_letter}) {indicator} {change_str}")
        
        if 'offensive_overall' in adjusted_grades and base_grades:
            print(f"\n{'OFFENSIVE PERFORMANCE (Roster-Adjusted)':^80}")
            print("-" * 80)
            print_grade_comparison("Offensive Overall", base_grades.get('offensive_overall', 0), adjusted_grades['offensive_overall'])
        
        if 'defensive_overall' in adjusted_grades and base_grades:
            print(f"\n{'DEFENSIVE PERFORMANCE (Roster-Adjusted)':^80}")
            print("-" * 80)
            print_grade_comparison("Defensive Overall", base_grades.get('defensive_overall', 0), adjusted_grades['defensive_overall'])
        
        if 'overall' in adjusted_grades:
            print(f"\n{'='*80}")
            print("FINAL ROSTER-ADJUSTED COACHING GRADE")
            print("="*80)
            
            base_overall = base_grades.get('overall', 0) if base_grades else 0
            print_grade_comparison("Overall Coaching Grade", base_overall, adjusted_grades['overall'])
            
            # Context explanation
            grade = adjusted_grades['overall']
            if grade >= 85:
                context = "Exceptional coaching - Getting maximum from available talent"
            elif grade >= 80:
                context = "Excellent coaching - Strong player development and strategy"
            elif grade >= 75:
                context = "Good coaching - Solid performance given roster"
            elif grade >= 70:
                context = "Average coaching - Meeting expectations"
            else:
                context = "Below expectations - Underperforming given available talent"
            
            print(f"\n{context}")
            print("="*80)
    
    # Include all the existing methods from the original coaching system
    def analyze_offensive_tendencies(self, coach_name=None, team=None, season=None):
        """Analyze offensive play-calling tendencies"""
        if self.pbp_data is None:
            print("No play-by-play data available")
            return None
        
        # Filter data based on parameters
        filtered_pbp = self.pbp_data.copy()
        
        # Filter by season if specified
        if season:
            filtered_pbp = filtered_pbp[filtered_pbp['season'] == season]
        
        # Filter by team if specified
        if team:
            filtered_pbp = filtered_pbp[filtered_pbp['posteam'] == team]
        
        # If coach specified, filter by their games
        if coach_name:
            coach_games = []
            for (coach, seas), data in self.coaching_data.items():
                if coach == coach_name and (season is None or seas == season):
                    coach_games.extend([g['game_id'] for g in data['games']])
            
            if coach_games:
                coach_team_games = []
                for (coach, seas), data in self.coaching_data.items():
                    if coach == coach_name and (season is None or seas == season):
                        for game in data['games']:
                            coach_team_games.append((game['game_id'], game['team']))
                
                # Filter to plays where the coach's team had possession
                coach_plays = []
                for game_id, coach_team in coach_team_games:
                    game_plays = filtered_pbp[
                        (filtered_pbp['game_id'] == game_id) & 
                        (filtered_pbp['posteam'] == coach_team)
                    ]
                    coach_plays.append(game_plays)
                
                if coach_plays:
                    filtered_pbp = pd.concat(coach_plays, ignore_index=True)
                else:
                    filtered_pbp = pd.DataFrame()
        
        if filtered_pbp.empty:
            print(f"No offensive plays found for the specified criteria")
            return None
        
        # Analyze offensive tendencies
        analysis = {}
        
        # Basic play type distribution
        play_types = filtered_pbp[filtered_pbp['play_type'].isin(['run', 'pass'])]['play_type'].value_counts()
        analysis['play_type_pct'] = (play_types / play_types.sum() * 100).to_dict()
        
        # Down and distance tendencies
        down_distance = {}
        for down in [1, 2, 3, 4]:
            down_plays = filtered_pbp[filtered_pbp['down'] == down]
            if not down_plays.empty:
                play_dist = down_plays[down_plays['play_type'].isin(['run', 'pass'])]['play_type'].value_counts()
                if not play_dist.empty:
                    down_distance[f'down_{down}'] = (play_dist / play_dist.sum() * 100).to_dict()
        
        analysis['down_tendencies'] = down_distance
        
        # Red zone tendencies (inside 20 yard line)
        red_zone = filtered_pbp[
            (filtered_pbp['yardline_100'] <= 20) & 
            (filtered_pbp['play_type'].isin(['run', 'pass']))
        ]
        
        if not red_zone.empty:
            rz_types = red_zone['play_type'].value_counts()
            analysis['red_zone_tendencies'] = (rz_types / rz_types.sum() * 100).to_dict()
        
        # Efficiency metrics
        passing_plays = filtered_pbp[filtered_pbp['play_type'] == 'pass']
        rushing_plays = filtered_pbp[filtered_pbp['play_type'] == 'run']
        
        if not passing_plays.empty:
            analysis['passing_efficiency'] = {
                'yards_per_attempt': passing_plays['passing_yards'].mean(),
                'completion_pct': (passing_plays['complete_pass'].sum() / len(passing_plays)) * 100,
                'td_pct': (passing_plays['pass_touchdown'].sum() / len(passing_plays)) * 100,
                'int_pct': (passing_plays['interception'].sum() / len(passing_plays)) * 100
            }
        
        if not rushing_plays.empty:
            analysis['rushing_efficiency'] = {
                'yards_per_carry': rushing_plays['rushing_yards'].mean(),
                'td_pct': (rushing_plays['rush_touchdown'].sum() / len(rushing_plays)) * 100
            }
        
        return analysis
    
    def analyze_defensive_performance(self, coach_name=None, team=None, season=None):
        """Analyze defensive performance against different offensive schemes"""
        if self.pbp_data is None:
            print("No play-by-play data available")
            return None
        
        # Filter data - this time we want plays where the team/coach was on DEFENSE
        filtered_pbp = self.pbp_data.copy()
        
        # Filter by season if specified
        if season:
            filtered_pbp = filtered_pbp[filtered_pbp['season'] == season]
        
        # Filter by team if specified (defensive team)
        if team:
            filtered_pbp = filtered_pbp[filtered_pbp['defteam'] == team]
        
        # If coach specified, filter by their defensive games
        if coach_name:
            coach_defensive_plays = []
            for (coach, seas), data in self.coaching_data.items():
                if coach == coach_name and (season is None or seas == season):
                    for game in data['games']:
                        game_plays = filtered_pbp[
                            (filtered_pbp['game_id'] == game['game_id']) & 
                            (filtered_pbp['defteam'] == game['team'])
                        ]
                        coach_defensive_plays.append(game_plays)
            
            if coach_defensive_plays:
                filtered_pbp = pd.concat(coach_defensive_plays, ignore_index=True)
            else:
                filtered_pbp = pd.DataFrame()
        
        if filtered_pbp.empty:
            print(f"No defensive plays found for the specified criteria")
            return None
        
        # Analyze defensive performance
        analysis = {}
        
        # Overall defensive stats
        passing_plays = filtered_pbp[filtered_pbp['play_type'] == 'pass']
        rushing_plays = filtered_pbp[filtered_pbp['play_type'] == 'run']
        
        if not passing_plays.empty:
            analysis['pass_defense'] = {
                'yards_per_attempt_allowed': passing_plays['passing_yards'].mean(),
                'completion_pct_allowed': (passing_plays['complete_pass'].sum() / len(passing_plays)) * 100,
                'td_pct_allowed': (passing_plays['pass_touchdown'].sum() / len(passing_plays)) * 100,
                'int_pct': (passing_plays['interception'].sum() / len(passing_plays)) * 100,
                'sack_pct': (passing_plays['sack'].sum() / len(passing_plays)) * 100
            }
        
        if not rushing_plays.empty:
            analysis['rush_defense'] = {
                'yards_per_carry_allowed': rushing_plays['rushing_yards'].mean(),
                'td_pct_allowed': (rushing_plays['rush_touchdown'].sum() / len(rushing_plays)) * 100,
                'stuff_rate': (rushing_plays[rushing_plays['rushing_yards'] <= 0].shape[0] / len(rushing_plays)) * 100
            }
        
        # Red zone defense
        red_zone_def = filtered_pbp[filtered_pbp['yardline_100'] <= 20]
        if not red_zone_def.empty:
            analysis['red_zone_defense'] = {
                'td_pct_allowed': (red_zone_def['touchdown'].sum() / len(red_zone_def)) * 100,
                'yards_per_play_allowed': red_zone_def['yards_gained'].mean()
            }
        
        return analysis
    
    def grade_coach_performance(self, coach_name, season=None):
        """Base coaching performance grading (before roster adjustment)"""
        
        # Get offensive and defensive analysis
        off_analysis = self.analyze_offensive_tendencies(coach_name=coach_name, season=season)
        def_analysis = self.analyze_defensive_performance(coach_name=coach_name, season=season)
        
        if not off_analysis and not def_analysis:
            return None
        
        grades = {}
        
        # Get team record for context
        team_record = self._get_team_record(coach_name, season)
        win_pct = team_record.get('win_pct', 50) if team_record else 50
        
        # Moderate context multiplier
        context_multiplier = 1.0
        if win_pct >= 75:  # Elite teams
            context_multiplier = 1.03
        elif win_pct >= 65:  # Good teams
            context_multiplier = 1.01
        elif win_pct <= 25:  # Terrible teams
            context_multiplier = 0.97
        elif win_pct <= 35:  # Poor teams
            context_multiplier = 0.99
        
        # Offensive Grading
        if off_analysis:
            off_grades = {}
            
            # Play calling balance
            if 'play_type_pct' in off_analysis:
                run_pct = off_analysis['play_type_pct'].get('run', 0)
                
                if 30 <= run_pct <= 55:
                    balance_score = 85
                elif 25 <= run_pct <= 60:
                    balance_score = 80
                else:
                    deviation = min(abs(30 - run_pct), abs(55 - run_pct))
                    balance_score = max(70, 85 - deviation)
                
                off_grades['play_balance'] = balance_score * context_multiplier
            
            # Passing efficiency
            if 'passing_efficiency' in off_analysis:
                pass_eff = off_analysis['passing_efficiency']
                
                # Yards per attempt
                ypa = pass_eff.get('yards_per_attempt', 0)
                if ypa >= 7.5:
                    ypa_grade = 90 + min(10, (ypa - 7.5) * 4)
                elif ypa >= 7.0:
                    ypa_grade = 80 + (ypa - 7.0) * 20
                elif ypa >= 6.5:
                    ypa_grade = 70 + (ypa - 6.5) * 20
                else:
                    ypa_grade = max(50, 70 * (ypa / 6.5))
                
                off_grades['passing_ypa'] = min(100, ypa_grade * context_multiplier)
                
                # Completion percentage
                comp_pct = pass_eff.get('completion_pct', 0)
                if comp_pct >= 65:
                    comp_grade = 85 + min(15, (comp_pct - 65) * 3)
                elif comp_pct >= 60:
                    comp_grade = 75 + (comp_pct - 60) * 2
                else:
                    comp_grade = max(50, 75 * (comp_pct / 60.0))
                
                off_grades['completion_pct'] = min(100, comp_grade * context_multiplier)
                
                # TD percentage
                td_pct = pass_eff.get('td_pct', 0)
                if td_pct >= 4.5:
                    td_grade = 85 + min(15, (td_pct - 4.5) * 6)
                elif td_pct >= 3.5:
                    td_grade = 75 + (td_pct - 3.5) * 10
                else:
                    td_grade = max(50, 75 * (td_pct / 3.5))
                
                off_grades['passing_tds'] = min(100, td_grade * context_multiplier)
                
                # Interception percentage (lower is better)
                int_pct = pass_eff.get('int_pct', 0)
                if int_pct <= 2.0:
                    int_grade = 85 + min(15, (2.0 - int_pct) * 10)
                elif int_pct <= 2.5:
                    int_grade = 75 + (2.5 - int_pct) * 20
                else:
                    int_grade = max(50, 75 - (int_pct - 2.5) * 10)
                
                off_grades['interceptions'] = min(100, int_grade * context_multiplier)
            
            # Rushing efficiency
            if 'rushing_efficiency' in off_analysis:
                rush_eff = off_analysis['rushing_efficiency']
                
                ypc = rush_eff.get('yards_per_carry', 0)
                if ypc >= 4.5:
                    ypc_grade = 85 + min(15, (ypc - 4.5) * 10)
                elif ypc >= 4.0:
                    ypc_grade = 75 + (ypc - 4.0) * 20
                else:
                    ypc_grade = max(50, 75 * (ypc / 4.0))
                
                off_grades['rushing_ypc'] = min(100, ypc_grade * context_multiplier)
            
            # Red zone efficiency
            if 'red_zone_tendencies' in off_analysis:
                off_grades['red_zone_usage'] = 75 * context_multiplier
            
            grades['offensive'] = off_grades
        
        # Defensive Grading
        if def_analysis:
            def_grades = {}
            
            if 'pass_defense' in def_analysis:
                pass_def = def_analysis['pass_defense']
                
                # Yards per attempt allowed
                ypa_allowed = pass_def.get('yards_per_attempt_allowed', 7.0)
                if ypa_allowed <= 6.5:
                    ypa_def_grade = 85 + min(15, (6.5 - ypa_allowed) * 6)
                elif ypa_allowed <= 7.0:
                    ypa_def_grade = 75 + (7.0 - ypa_allowed) * 20
                elif ypa_allowed <= 7.5:
                    ypa_def_grade = 65 + (7.5 - ypa_allowed) * 20
                else:
                    ypa_def_grade = max(50, 65 - (ypa_allowed - 7.5) * 8)
                
                def_grades['pass_defense_ypa'] = min(100, ypa_def_grade * context_multiplier)
                
                # Interception rate
                int_rate = pass_def.get('int_pct', 0)
                if int_rate >= 2.5:
                    int_def_grade = 85 + min(15, (int_rate - 2.5) * 10)
                elif int_rate >= 2.0:
                    int_def_grade = 75 + (int_rate - 2.0) * 20
                else:
                    int_def_grade = max(50, 75 * (int_rate / 2.0))
                
                def_grades['interceptions'] = min(100, int_def_grade * context_multiplier)
                
                # Sack rate
                sack_rate = pass_def.get('sack_pct', 0)
                if sack_rate >= 7.0:
                    sack_grade = 85 + min(15, (sack_rate - 7.0) * 5)
                elif sack_rate >= 5.5:
                    sack_grade = 75 + (sack_rate - 5.5) * 6.67
                else:
                    sack_grade = max(50, 75 * (sack_rate / 5.5))
                
                def_grades['pass_rush'] = min(100, sack_grade * context_multiplier)
            
            if 'rush_defense' in def_analysis:
                rush_def = def_analysis['rush_defense']
                
                # Yards per carry allowed
                ypc_allowed = rush_def.get('yards_per_carry_allowed', 4.3)
                if ypc_allowed <= 4.0:
                    ypc_def_grade = 85 + min(15, (4.0 - ypc_allowed) * 15)
                elif ypc_allowed <= 4.3:
                    ypc_def_grade = 75 + (4.3 - ypc_allowed) * 33
                elif ypc_allowed <= 4.6:
                    ypc_def_grade = 65 + (4.6 - ypc_allowed) * 33
                else:
                    ypc_def_grade = max(50, 65 - (ypc_allowed - 4.6) * 10)
                
                def_grades['rush_defense_ypc'] = min(100, ypc_def_grade * context_multiplier)
            
            if 'red_zone_defense' in def_analysis:
                rz_def = def_analysis['red_zone_defense']
                
                td_pct_allowed = rz_def.get('td_pct_allowed', 55)
                if td_pct_allowed <= 50:
                    rz_grade = 85 + min(15, (50 - td_pct_allowed) * 1)
                elif td_pct_allowed <= 60:
                    rz_grade = 75 + (60 - td_pct_allowed) * 1
                else:
                    rz_grade = max(50, 75 - (td_pct_allowed - 60) * 0.8)
                
                def_grades['red_zone_defense'] = min(100, rz_grade * context_multiplier)
            
            grades['defensive'] = def_grades
        
        # Calculate overall grades
        if grades.get('offensive'):
            off_scores = list(grades['offensive'].values())
            grades['offensive_overall'] = sum(off_scores) / len(off_scores)
        
        if grades.get('defensive'):
            def_scores = list(grades['defensive'].values())
            grades['defensive_overall'] = sum(def_scores) / len(def_scores)
        
        # Overall coaching grade
        coach_specialty = self._determine_coach_specialty(coach_name, season)
        
        if grades.get('offensive_overall') and grades.get('defensive_overall'):
            if coach_specialty == 'offense':
                grades['overall'] = (grades['offensive_overall'] * 0.6) + (grades['defensive_overall'] * 0.4)
            elif coach_specialty == 'defense':
                grades['overall'] = (grades['offensive_overall'] * 0.4) + (grades['defensive_overall'] * 0.6)
            else:
                grades['overall'] = (grades['offensive_overall'] * 0.5) + (grades['defensive_overall'] * 0.5)
        elif grades.get('offensive_overall'):
            grades['overall'] = grades['offensive_overall']
        elif grades.get('defensive_overall'):
            grades['overall'] = grades['defensive_overall']
        
        return grades
    
    def _get_team_record(self, coach_name, season):
        """Get team record for context"""
        for (coach, seas), data in self.coaching_data.items():
            if coach == coach_name and (season is None or seas == season):
                wins = sum(1 for g in data['games'] if g['result'] == 'W')
                total_games = len([g for g in data['games'] if g['result'] is not None])
                win_pct = (wins / total_games * 100) if total_games > 0 else 0
                return {'wins': wins, 'losses': total_games - wins, 'win_pct': win_pct}
        return None
    
    def _determine_coach_specialty(self, coach_name, season):
        """Determine coach specialty (existing method)"""
        offensive_coaches = ['Andy Reid', 'Sean McVay', 'Kyle Shanahan', 'Sean Payton', 
                           'Josh McDaniels', 'Arthur Smith', 'Matt LaFleur']
        defensive_coaches = ['Bill Belichick', 'Mike Tomlin', 'Pete Carroll', 'Vic Fangio',
                           'Brandon Staley', 'Robert Saleh', 'Dan Quinn']
        
        if coach_name in offensive_coaches:
            return 'offense'
        elif coach_name in defensive_coaches:
            return 'defense'
        else:
            return 'balanced'
    
    def get_letter_grade(self, numerical_grade):
        """Convert numerical grade to letter grade (existing method)"""
        if numerical_grade >= 97:
            return "A+"
        elif numerical_grade >= 93:
            return "A"
        elif numerical_grade >= 90:
            return "A-"
        elif numerical_grade >= 87:
            return "B+"
        elif numerical_grade >= 83:
            return "B"
        elif numerical_grade >= 80:
            return "B-"
        elif numerical_grade >= 77:
            return "C+"
        elif numerical_grade >= 73:
            return "C"
        elif numerical_grade >= 70:
            return "C-"
        elif numerical_grade >= 67:
            return "D+"
        elif numerical_grade >= 63:
            return "D"
        elif numerical_grade >= 60:
            return "D-"
        else:
            return "F"
    
    def compare_coaches_with_roster_context(self, coach_names, season=None):
        """Compare coaches with roster quality context"""
        print(f"\n{'ROSTER-AWARE COACH COMPARISON':^100}")
        if season:
            print(f"Season: {season}")
        print("=" * 100)
        
        coach_data = {}
        for coach in coach_names:
            adjusted_grades = self.calculate_roster_adjusted_coaching_grade(coach, season)
            base_grades = self.grade_coach_performance(coach, season)
            
            if adjusted_grades and base_grades:
                roster_quality = adjusted_grades.get('roster_context', {}).get('avg_roster_grade', 0)
                coach_data[coach] = {
                    'adjusted_overall': adjusted_grades.get('overall', 0),
                    'base_overall': base_grades.get('overall', 0),
                    'roster_grade': roster_quality,
                    'roster_tier': self._get_roster_tier(roster_quality) if roster_quality else 'Unknown'
                }
        
        if not coach_data:
            print("No data available for comparison")
            return
        
        # Print comparison table
        print(f"{'Coach':<20}{'Base Grade':<12}{'Adj Grade':<12}{'Difference':<12}{'Roster Tier':<15}{'Roster Grade':<12}")
        print("-" * 100)
        
        for coach, data in coach_data.items():
            base = data['base_overall']
            adj = data['adjusted_overall']
            diff = adj - base
            roster_tier = data['roster_tier']
            roster_grade = data['roster_grade']
            
            base_letter = self.get_letter_grade(base)
            adj_letter = self.get_letter_grade(adj)
            
            base_str = f"{base:.1f} ({base_letter})"
            adj_str = f"{adj:.1f} ({adj_letter})"
            diff_str = f"{diff:+.1f}"
            
            print(f"{coach[:19]:<20}{base_str:<12}{adj_str:<12}{diff_str:<12}{roster_tier:<15}{roster_grade:.1f}")
    
    def get_coaching_efficiency_leaders(self, season=None, min_games=8):
        """Find coaches who are most efficiently using their roster talent"""
        print("Calculating coaching efficiency (performance vs roster quality)...")
        
        coaches = self.get_all_coaches(season)
        efficiency_data = []
        
        for coach in coaches:
            try:
                adjusted_grades = self.calculate_roster_adjusted_coaching_grade(coach, season)
                base_grades = self.grade_coach_performance(coach, season)
                
                if adjusted_grades and base_grades and 'roster_context' in adjusted_grades:
                    roster_quality = adjusted_grades['roster_context'].get('avg_roster_grade', 0)
                    base_overall = base_grades.get('overall', 0)
                    adjusted_overall = adjusted_grades.get('overall', 0)
                    
                    # Calculate efficiency: how much better/worse than expected given roster
                    expected_performance = roster_quality  # Simplified expectation
                    efficiency = adjusted_overall - expected_performance
                    
                    efficiency_data.append({
                        'coach': coach,
                        'base_grade': base_overall,
                        'adjusted_grade': adjusted_overall,
                        'roster_grade': roster_quality,
                        'efficiency': efficiency,
                        'roster_tier': self._get_roster_tier(roster_quality)
                    })
            except Exception as e:
                continue
        
        if not efficiency_data:
            print("No efficiency data calculated")
            return None
        
        efficiency_df = pd.DataFrame(efficiency_data)
        efficiency_df = efficiency_df.sort_values('efficiency', ascending=False)
        
        print(f"\n{'COACHING EFFICIENCY LEADERS (Performance vs Roster Quality)':^100}")
        print("=" * 100)
        print(f"{'Coach':<20}{'Efficiency':<12}{'Adj Grade':<12}{'Roster Grade':<12}{'Roster Tier':<15}")
        print("-" * 100)
        
        for _, row in efficiency_df.head(10).iterrows():
            eff_str = f"{row['efficiency']:+.1f}"
            adj_str = f"{row['adjusted_grade']:.1f}"
            roster_str = f"{row['roster_grade']:.1f}"
            print(f"{row['coach'][:19]:<20}{eff_str:<12}{adj_str:<12}{roster_str:<12}{row['roster_tier']:<15}")
        
        return efficiency_df
    
    def get_all_coaches(self, season=None):
        """Get list of all coaches in the dataset"""
        coaches = set()
        for (coach, seas), data in self.coaching_data.items():
            if season is None or seas == season:
                coaches.add(coach)
        return sorted(list(coaches))


def main():
    """Main function demonstrating the roster-aware coaching system"""
    print("NFL Roster-Aware Coaching Analytics System")
    print("=" * 60)
    
    # Initialize the enhanced system
    analytics = RosterAwareCoachingAnalytics(years=[2023])
    
    # Load all data including player grades
    analytics.load_data()
    
    # Extract coaching information
    analytics.extract_coaching_info()
    
    # Calculate player grades for roster analysis
    player_grades = analytics.calculate_player_grades(min_games=3)
    
    if player_grades is None:
        print("Could not calculate player grades - falling back to basic coaching analysis")
        return
    
    # Get available coaches
    coaches = analytics.get_all_coaches(season=2023)
    print(f"\nAvailable coaches for 2023: {len(coaches)}")
    
    if not coaches:
        print("No coaches found")
        return
    
    # Demonstrate with a well-known coach
    example_coach = None
    for coach in coaches:
        if any(name in coach.lower() for name in ['reid', 'belichick', 'tomlin', 'harbaugh', 'mcvay']):
            example_coach = coach
            break
    
    if not example_coach:
        example_coach = coaches[0]
    
    print(f"\n{'='*80}")
    print(f"DEMONSTRATING ROSTER-AWARE ANALYSIS: {example_coach}")
    print(f"{'='*80}")
    
    # Show roster-aware coaching report
    analytics.print_roster_aware_coach_report(example_coach, season=2023)
    
    # Compare multiple coaches with roster context
    if len(coaches) >= 3:
        print(f"\n{'='*80}")
        print("ROSTER-AWARE COACH COMPARISON")
        print(f"{'='*80}")
        comparison_coaches = coaches[:3]
        analytics.compare_coaches_with_roster_context(comparison_coaches, season=2023)
    
    # Show coaching efficiency leaders
    print(f"\n{'='*80}")
    print("COACHING EFFICIENCY ANALYSIS")
    print(f"{'='*80}")
    efficiency_leaders = analytics.get_coaching_efficiency_leaders(season=2023)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print("Key Features of Roster-Aware System:")
    print("- Adjusts coaching grades based on player talent available")
    print("- Provides roster quality context for each team")
    print("- Identifies coaches who maximize limited talent")
    print("- Compares coaches on equal footing regardless of roster")
    print(f"{'='*80}")


def interactive_mode():
    """Interactive mode for exploring roster-aware coaching analytics"""
    analytics = RosterAwareCoachingAnalytics(years=[2023])
    analytics.load_data()
    analytics.extract_coaching_info()
    
    # Calculate player grades
    print("Calculating player grades for roster analysis...")
    player_grades = analytics.calculate_player_grades(min_games=3)
    
    if player_grades is None:
        print("Could not calculate player grades - some features will be limited")
    
    coaches = analytics.get_all_coaches(season=2023)
    
    while True:
        print(f"\n{'='*60}")
        print("ROSTER-AWARE COACHING ANALYTICS - INTERACTIVE MODE")
        print(f"{'='*60}")
        print("Available commands:")
        print("1. List all coaches")
        print("2. Analyze specific coach (roster-aware)")
        print("3. Compare coaches with roster context")
        print("4. Show coaching efficiency leaders")
        print("5. Analyze team roster quality")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            print(f"\nAvailable coaches for 2023 ({len(coaches)}):")
            for i, coach in enumerate(coaches, 1):
                print(f"{i:2d}. {coach}")
        
        elif choice == '2':
            coach_name = input("Enter coach name: ")
            if coach_name in coaches:
                try:
                    analytics.print_roster_aware_coach_report(coach_name, season=2023)
                except Exception as e:
                    print(f"Error analyzing coach: {e}")
            else:
                print(f"Coach '{coach_name}' not found. Use command 1 to see available coaches.")
        
        elif choice == '3':
            print("Enter coach names separated by commas:")
            coach_input = input("Coaches: ")
            coach_names = [name.strip() for name in coach_input.split(',')]
            valid_coaches = [name for name in coach_names if name in coaches]
            
            if valid_coaches:
                try:
                    analytics.compare_coaches_with_roster_context(valid_coaches, season=2023)
                except Exception as e:
                    print(f"Error comparing coaches: {e}")
            else:
                print("No valid coaches found.")
        
        elif choice == '4':
            try:
                analytics.get_coaching_efficiency_leaders(season=2023)
            except Exception as e:
                print(f"Error calculating efficiency leaders: {e}")
        
        elif choice == '5':
            team = input("Enter team abbreviation (e.g., KC, NE, DAL): ").upper()
            try:
                roster_quality = analytics.get_roster_quality_metrics(team, 2023)
                if roster_quality:
                    print(f"\n{team} Roster Quality Analysis (2023):")
                    print(f"Overall Grade: {roster_quality['avg_grade']:.1f}")
                    print(f"Quality Tier: {analytics._get_roster_tier(roster_quality['avg_grade'])}")
                    print(f"Elite Players (90+): {roster_quality['elite_players']}")
                    print(f"Top Players (80+): {roster_quality['top_players']}")
                    print(f"Below Average (<70): {roster_quality['below_avg_players']}")
                    print(f"Depth Score: {roster_quality['depth_score']:.1f}")
                    
                    if 'position_breakdown' in roster_quality:
                        print("\nPosition Group Breakdown:")
                        for pos_group, data in roster_quality['position_breakdown']['avg_grade'].items():
                            count = roster_quality['position_breakdown']['player_id'][pos_group]
                            print(f"  {pos_group}: {data:.1f} avg ({count} players)")
                else:
                    print(f"No roster data found for {team}")
            except Exception as e:
                print(f"Error analyzing roster: {e}")
        
        elif choice == '6':
            print("Thanks for using Roster-Aware Coaching Analytics!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        interactive_mode()
    else:
        main()