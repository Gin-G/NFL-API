#!/usr/bin/env python3
"""
NFL Coaching Analytics System - IMPROVED VERSION
Fixed grading logic and overall calculation issues
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class NFLCoachingAnalytics:
    def __init__(self, years=None):
        """Initialize the coaching analytics system"""
        if years is None:
            years = [2023, 2024]  # Default to recent years
        self.years = years
        self.pbp_data = None
        self.schedule_data = None
        self.coaching_data = {}
        
    def load_data(self):
        """Load all necessary data from nfl_data_py"""
        print("Loading NFL data...")
        
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
        
        # Load schedule data for coaching info
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
            
        print("Data loading complete!\n")
    
    def extract_coaching_info(self):
        """Extract coaching information from schedule data"""
        if self.schedule_data is None:
            print("No schedule data available")
            return
        
        print("Extracting coaching information...")
        
        # Create coaching mapping
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
        
        # Formation tendencies (if available)
        if 'shotgun' in filtered_pbp.columns:
            shotgun_pct = filtered_pbp['shotgun'].mean() * 100
            analysis['shotgun_pct'] = shotgun_pct
        
        # Pace analysis (no huddle)
        if 'no_huddle' in filtered_pbp.columns:
            no_huddle_pct = filtered_pbp['no_huddle'].mean() * 100
            analysis['no_huddle_pct'] = no_huddle_pct
        
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
        
        # Performance by down
        down_performance = {}
        for down in [1, 2, 3, 4]:
            down_plays = filtered_pbp[filtered_pbp['down'] == down]
            if not down_plays.empty:
                down_performance[f'down_{down}'] = {
                    'yards_per_play_allowed': down_plays['yards_gained'].mean(),
                    'success_rate_allowed': (down_plays['first_down'].sum() / len(down_plays)) * 100
                }
        
        analysis['down_performance'] = down_performance
        
        # Red zone defense
        red_zone_def = filtered_pbp[filtered_pbp['yardline_100'] <= 20]
        if not red_zone_def.empty:
            analysis['red_zone_defense'] = {
                'td_pct_allowed': (red_zone_def['touchdown'].sum() / len(red_zone_def)) * 100,
                'yards_per_play_allowed': red_zone_def['yards_gained'].mean()
            }
        
        return analysis
    
    def grade_coach_performance(self, coach_name, season=None):
        """IMPROVED: Grade a coach's performance with fixed logic"""
        
        # Get offensive and defensive analysis
        off_analysis = self.analyze_offensive_tendencies(coach_name=coach_name, season=season)
        def_analysis = self.analyze_defensive_performance(coach_name=coach_name, season=season)
        
        if not off_analysis and not def_analysis:
            return None
        
        grades = {}
        
        # Get team record for context - but use more moderate adjustments
        team_record = self._get_team_record(coach_name, season)
        win_pct = team_record.get('win_pct', 50) if team_record else 50
        
        # FIXED: Much more moderate context multiplier
        context_multiplier = 1.0
        if win_pct >= 75:  # Elite teams (12+ wins)
            context_multiplier = 1.05  # Small bonus for elite teams
        elif win_pct >= 65:  # Good teams (11+ wins) 
            context_multiplier = 1.02  # Tiny bonus
        elif win_pct <= 25:  # Terrible teams (4 or fewer wins)
            context_multiplier = 0.95  # Small penalty for awful teams
        elif win_pct <= 35:  # Poor teams (6 or fewer wins)
            context_multiplier = 0.98  # Tiny penalty
        
        # Offensive Grading with FIXED logic
        if off_analysis:
            off_grades = {}
            
            # Play calling balance - more reasonable standards
            if 'play_type_pct' in off_analysis:
                run_pct = off_analysis['play_type_pct'].get('run', 0)
                
                # Modern NFL is pass-heavy, so adjust expectations
                # Good range is 30-55% run, elite can be 25-60%
                if 30 <= run_pct <= 55:
                    balance_score = 85  # Good balance
                elif 25 <= run_pct <= 60:
                    balance_score = 80  # Acceptable
                else:
                    # Penalize extreme imbalances but not too harshly
                    deviation = min(abs(30 - run_pct), abs(55 - run_pct))
                    balance_score = max(70, 85 - deviation)
                
                off_grades['play_balance'] = balance_score * context_multiplier
            
            # FIXED: Passing efficiency with more reasonable thresholds
            if 'passing_efficiency' in off_analysis:
                pass_eff = off_analysis['passing_efficiency']
                
                # Yards per attempt - Fixed to be more realistic
                ypa = pass_eff.get('yards_per_attempt', 0)
                if ypa >= 7.5:
                    ypa_grade = 90 + min(10, (ypa - 7.5) * 4)  # Cap at 100
                elif ypa >= 7.0:
                    ypa_grade = 80 + (ypa - 7.0) * 20
                elif ypa >= 6.5:
                    ypa_grade = 70 + (ypa - 6.5) * 20
                else:
                    ypa_grade = max(50, 70 * (ypa / 6.5))
                
                off_grades['passing_ypa'] = min(100, ypa_grade * context_multiplier)
                
                # Completion percentage - more realistic
                comp_pct = pass_eff.get('completion_pct', 0)
                if comp_pct >= 65:
                    comp_grade = 85 + min(15, (comp_pct - 65) * 3)
                elif comp_pct >= 60:
                    comp_grade = 75 + (comp_pct - 60) * 2
                else:
                    comp_grade = max(50, 75 * (comp_pct / 60.0))
                
                off_grades['completion_pct'] = min(100, comp_grade * context_multiplier)
                
                # TD percentage - more realistic
                td_pct = pass_eff.get('td_pct', 0)
                if td_pct >= 4.5:
                    td_grade = 85 + min(15, (td_pct - 4.5) * 6)
                elif td_pct >= 3.5:
                    td_grade = 75 + (td_pct - 3.5) * 10
                else:
                    td_grade = max(50, 75 * (td_pct / 3.5))
                
                off_grades['passing_tds'] = min(100, td_grade * context_multiplier)
                
                # Interception percentage (lower is better) - more realistic
                int_pct = pass_eff.get('int_pct', 0)
                if int_pct <= 2.0:
                    int_grade = 85 + min(15, (2.0 - int_pct) * 10)
                elif int_pct <= 2.5:
                    int_grade = 75 + (2.5 - int_pct) * 20
                else:
                    int_grade = max(50, 75 - (int_pct - 2.5) * 10)
                
                off_grades['interceptions'] = min(100, int_grade * context_multiplier)
            
            # Rushing efficiency - more realistic
            if 'rushing_efficiency' in off_analysis:
                rush_eff = off_analysis['rushing_efficiency']
                
                # Yards per carry - more realistic standards
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
        
        # FIXED: Defensive Grading with more realistic standards
        if def_analysis:
            def_grades = {}
            
            if 'pass_defense' in def_analysis:
                pass_def = def_analysis['pass_defense']
                
                # FIXED: Yards per attempt allowed - more realistic standards
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
                
                # Interception rate - more realistic
                int_rate = pass_def.get('int_pct', 0)
                if int_rate >= 2.5:
                    int_def_grade = 85 + min(15, (int_rate - 2.5) * 10)
                elif int_rate >= 2.0:
                    int_def_grade = 75 + (int_rate - 2.0) * 20
                else:
                    int_def_grade = max(50, 75 * (int_rate / 2.0))
                
                def_grades['interceptions'] = min(100, int_def_grade * context_multiplier)
                
                # Sack rate - more realistic
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
                
                # FIXED: Yards per carry allowed - more realistic
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
                
                # Stuff rate - more realistic
                stuff_rate = rush_def.get('stuff_rate', 0)
                if stuff_rate >= 20:
                    stuff_grade = 85 + min(15, (stuff_rate - 20) * 1)
                elif stuff_rate >= 15:
                    stuff_grade = 75 + (stuff_rate - 15) * 2
                else:
                    stuff_grade = max(50, 75 * (stuff_rate / 15.0))
                
                def_grades['run_stopping'] = min(100, stuff_grade * context_multiplier)
            
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
        
        # FIXED: Calculate overall grades properly
        if grades.get('offensive'):
            off_scores = list(grades['offensive'].values())
            grades['offensive_overall'] = sum(off_scores) / len(off_scores)
        
        if grades.get('defensive'):
            def_scores = list(grades['defensive'].values())
            grades['defensive_overall'] = sum(def_scores) / len(def_scores)
        
        # FIXED: Overall coaching grade - simplified and more logical
        overall_components = []
        coach_specialty = self._determine_coach_specialty(coach_name, season)
        
        # Add offensive component
        if grades.get('offensive_overall'):
            # Offensive specialists get more weight, but not extreme
            weight = 0.6 if coach_specialty == 'offense' else 0.5
            overall_components.append(grades['offensive_overall'] * weight)
        
        # Add defensive component  
        if grades.get('defensive_overall'):
            # Defensive specialists get more weight, but not extreme
            weight = 0.6 if coach_specialty == 'defense' else 0.5
            overall_components.append(grades['defensive_overall'] * weight)
        
        # FIXED: Simple average instead of complex weighting
        if overall_components:
            grades['overall'] = sum(overall_components) / sum([0.6 if coach_specialty in ['offense', 'defense'] else 0.5 for _ in overall_components])
            
            # REMOVED: The problematic final team success bonus/penalty that was causing the F grades
            # Just keep the grade as calculated
        
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
        """Determine if coach is offense or defense-focused based on background"""
        # This could be expanded with a database of coach backgrounds
        # For now, use some known examples
        offensive_coaches = ['Andy Reid', 'Sean McVay', 'Kyle Shanahan', 'Sean Payton', 
                           'Josh McDaniels', 'Arthur Smith', 'Matt LaFleur']
        defensive_coaches = ['Bill Belichick', 'Mike Tomlin', 'Pete Carroll', 'Vic Fangio',
                           'Brandon Staley', 'Robert Saleh', 'Dan Quinn']
        
        if coach_name in offensive_coaches:
            return 'offense'
        elif coach_name in defensive_coaches:
            return 'defense'
        else:
            return 'balanced'  # Unknown coaches treated as balanced
    
    def get_letter_grade(self, numerical_grade):
        """Convert numerical grade to letter grade"""
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
    
    def print_coach_report(self, coach_name, season=None):
        """Print a comprehensive coaching report with context"""
        print(f"\n{'='*60}")
        print(f"COACHING REPORT: {coach_name}")
        if season:
            print(f"Season: {season}")
        print(f"{'='*60}")
        
        # Get basic info
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