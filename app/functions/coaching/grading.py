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
            context_multiplier = 1.03  # Small bonus for elite teams
        elif win_pct >= 65:  # Good teams (11+ wins) 
            context_multiplier = 1.01  # Tiny bonus
        elif win_pct <= 25:  # Terrible teams (4 or fewer wins)
            context_multiplier = 0.97  # Small penalty for awful teams
        elif win_pct <= 35:  # Poor teams (6 or fewer wins)
            context_multiplier = 0.99  # Tiny penalty
        
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
        coach_specialty = self._determine_coach_specialty(coach_name, season)
        
        # Simple weighted average based on specialty
        if grades.get('offensive_overall') and grades.get('defensive_overall'):
            if coach_specialty == 'offense':
                # 60% offense, 40% defense for offensive coaches
                grades['overall'] = (grades['offensive_overall'] * 0.6) + (grades['defensive_overall'] * 0.4)
            elif coach_specialty == 'defense':
                # 40% offense, 60% defense for defensive coaches
                grades['overall'] = (grades['offensive_overall'] * 0.4) + (grades['defensive_overall'] * 0.6)
            else:
                # 50/50 for balanced coaches
                grades['overall'] = (grades['offensive_overall'] * 0.5) + (grades['defensive_overall'] * 0.5)
        elif grades.get('offensive_overall'):
            # Only offensive data available
            grades['overall'] = grades['offensive_overall']
        elif grades.get('defensive_overall'):
            # Only defensive data available
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
        
        # Print basic info with context
        for info in coach_info:
            print(f"Season: {info['season']}")
            print(f"Team(s): {', '.join(info['teams'])}")
            print(f"Record: {info['record']} ({info['win_pct']:.1f}%)")
            print(f"Specialty: {coach_specialty.title()}")
            
            # Add context about team performance
            if info['win_pct'] >= 70:
                print("Elite team performance - Championship caliber")
            elif info['win_pct'] >= 60:
                print("Strong team performance - Playoff quality")
            elif info['win_pct'] >= 50:
                print("Average team performance")
            elif info['win_pct'] >= 40:
                print("Below average performance")
            else:
                print("Poor team performance - Rebuilding needed")
                
            # Games coached context
            if info['games_coached'] < 17:
                print(f"Limited sample size: Only {info['games_coached']} games coached")
        
        # Get grades
        grades = self.grade_coach_performance(coach_name, season)
        
        if not grades:
            print("No grading data available")
            return
        
        print(f"\n{'OFFENSIVE PERFORMANCE':^60}")
        print("-" * 60)
        
        if 'offensive' in grades:
            for category, score in grades['offensive'].items():
                letter = self.get_letter_grade(score)
                # Add performance indicators
                if score >= 90:
                    indicator = "Outstanding"
                elif score >= 80:
                    indicator = "Very Good"
                elif score >= 70:
                    indicator = "Average"
                elif score >= 60:
                    indicator = "Below Average"
                else:
                    indicator = "Poor"
                
                print(f"{category.replace('_', ' ').title():<35} {score:>6.1f} ({letter}) {indicator}")
            
            if 'offensive_overall' in grades:
                letter = self.get_letter_grade(grades['offensive_overall'])
                indicator = "Outstanding" if grades['offensive_overall'] >= 90 else "Very Good" if grades['offensive_overall'] >= 80 else "Average" if grades['offensive_overall'] >= 70 else "Below Average" if grades['offensive_overall'] >= 60 else "Poor"
                print(f"{'='*45}")
                print(f"{'OFFENSIVE OVERALL':<35} {grades['offensive_overall']:>6.1f} ({letter}) {indicator}")
        
        print(f"\n{'DEFENSIVE PERFORMANCE':^60}")
        print("-" * 60)
        
        if 'defensive' in grades:
            for category, score in grades['defensive'].items():
                letter = self.get_letter_grade(score)
                # Add performance indicators
                if score >= 90:
                    indicator = "Outstanding"
                elif score >= 80:
                    indicator = "Very Good"
                elif score >= 70:
                    indicator = "Average"
                elif score >= 60:
                    indicator = "Below Average"
                else:
                    indicator = "Poor"
                
                print(f"{category.replace('_', ' ').title():<35} {score:>6.1f} ({letter}) {indicator}")
            
            if 'defensive_overall' in grades:
                letter = self.get_letter_grade(grades['defensive_overall'])
                indicator = "Outstanding" if grades['defensive_overall'] >= 90 else "Very Good" if grades['defensive_overall'] >= 80 else "Average" if grades['defensive_overall'] >= 70 else "Below Average" if grades['defensive_overall'] >= 60 else "Poor"
                print(f"{'='*45}")
                print(f"{'DEFENSIVE OVERALL':<35} {grades['defensive_overall']:>6.1f} ({letter}) {indicator}")
        
        if 'overall' in grades:
            letter = self.get_letter_grade(grades['overall'])
            # Overall performance context
            if grades['overall'] >= 90:
                context = "Elite coaching - Championship level"
            elif grades['overall'] >= 85:
                context = "Excellent coaching - Top tier"
            elif grades['overall'] >= 80:
                context = "Very good coaching - Above average"
            elif grades['overall'] >= 75:
                context = "Solid coaching - League average"
            elif grades['overall'] >= 70:
                context = "Below average coaching"
            else:
                context = "Poor coaching - Needs improvement"
            
            print(f"\n{'='*60}")
            print(f"{'OVERALL COACHING GRADE':<35} {grades['overall']:>6.1f} ({letter})")
            print(f"{context}")
            print(f"{'='*60}")
            
            # Add coaching insights
            self._print_coaching_insights(coach_name, grades, coach_specialty)
    
    def _print_coaching_insights(self, coach_name, grades, specialty):
        """Print coaching insights and recommendations"""
        print(f"\n{'COACHING INSIGHTS':^60}")
        print("-" * 60)
        
        # Specialty-based insights
        if specialty == 'offense':
            if grades.get('offensive_overall', 0) >= 85:
                print("Offensive mastermind - Excellent play design and execution")
            elif grades.get('offensive_overall', 0) >= 75:
                print("Solid offensive coordinator - Good system implementation")
            else:
                print("Offensive struggles - May need scheme adjustments")
                
            if grades.get('defensive_overall', 0) < grades.get('offensive_overall', 0) - 15:
                print("Consider upgrading defensive coordinator for balance")
        
        elif specialty == 'defense':
            if grades.get('defensive_overall', 0) >= 85:
                print("Defensive genius - Elite scheme and player development")
            elif grades.get('defensive_overall', 0) >= 75:
                print("Strong defensive mind - Good tactical approach")
            else:
                print("Defensive concerns - Scheme may need overhaul")
                
            if grades.get('offensive_overall', 0) < grades.get('defensive_overall', 0) - 15:
                print("Consider upgrading offensive coordinator for balance")
        
        # Overall team balance insights
        if 'offensive_overall' in grades and 'defensive_overall' in grades:
            diff = abs(grades['offensive_overall'] - grades['defensive_overall'])
            if diff > 20:
                stronger_side = "offense" if grades['offensive_overall'] > grades['defensive_overall'] else "defense"
                weaker_side = "defense" if stronger_side == "offense" else "offense"
                print(f"Team imbalance: Strong {stronger_side} ({grades[f'{stronger_side}_overall']:.1f}) vs weak {weaker_side} ({grades[f'{weaker_side}_overall']:.1f})")
                print(f"Focus on improving {weaker_side} through personnel/coaching changes")
            elif diff < 10:
                print("Well-balanced team - Both sides contributing effectively")
        
        # Performance-based recommendations
        overall_grade = grades.get('overall', 0)
        if overall_grade >= 90:
            print("Championship-level coaching - Maintain current approach")
        elif overall_grade >= 80:
            print("Excellent coaching - Minor tweaks could reach elite level")
        elif overall_grade < 70:
            print("Areas for improvement identified - Consider system changes")
    
    def analyze_situational_performance(self, coach_name=None, team=None, season=None):
        """Analyze performance in clutch/critical situations"""
        if self.pbp_data is None:
            return None
        
        # Filter data
        filtered_pbp = self._filter_coach_plays(coach_name, team, season, offense=True)
        if filtered_pbp.empty:
            return None
        
        analysis = {}
        
        # Check available columns
        available_cols = set(filtered_pbp.columns)
        
        # 4th Quarter Performance
        if 'qtr' in available_cols:
            fourth_quarter = filtered_pbp[filtered_pbp['qtr'] == 4]
            if not fourth_quarter.empty:
                analysis['fourth_quarter'] = {
                    'total_plays': len(fourth_quarter),
                    'avg_yards_per_play': fourth_quarter['yards_gained'].mean() if 'yards_gained' in available_cols else 0,
                    'success_rate': (fourth_quarter['first_down'].sum() / len(fourth_quarter)) * 100 if 'first_down' in available_cols else 0,
                    'td_rate': (fourth_quarter['touchdown'].sum() / len(fourth_quarter)) * 100 if 'touchdown' in available_cols else 0
                }
        
        # Two-minute drill (last 2 minutes of each half) - simplified version
        if 'qtr' in available_cols and 'quarter_seconds_remaining' in available_cols:
            two_min_drill = filtered_pbp[
                (filtered_pbp['qtr'].isin([2, 4])) & (filtered_pbp['quarter_seconds_remaining'] <= 120)
            ]
            if not two_min_drill.empty:
                pass_plays = two_min_drill[two_min_drill['play_type'] == 'pass']
                analysis['two_minute_drill'] = {
                    'total_plays': len(two_min_drill),
                    'avg_yards_per_play': two_min_drill['yards_gained'].mean() if 'yards_gained' in available_cols else 0,
                    'completion_pct': (pass_plays['complete_pass'].sum() / len(pass_plays)) * 100 if len(pass_plays) > 0 and 'complete_pass' in available_cols else 0
                }
        
        # 3rd Down Conversion Performance
        if 'down' in available_cols:
            third_downs = filtered_pbp[filtered_pbp['down'] == 3]
            if not third_downs.empty:
                conversions = third_downs['first_down'].sum() if 'first_down' in available_cols else 0
                analysis['third_down_conversions'] = {
                    'attempts': len(third_downs),
                    'conversions': conversions,
                    'conversion_rate': (conversions / len(third_downs)) * 100,
                    'avg_distance': third_downs['ydstogo'].mean() if 'ydstogo' in available_cols else 0
                }
        
        # Goal Line Performance (inside 5 yard line)
        if 'yardline_100' in available_cols:
            goal_line = filtered_pbp[filtered_pbp['yardline_100'] <= 5]
            if not goal_line.empty:
                analysis['goal_line'] = {
                    'attempts': len(goal_line),
                    'touchdowns': goal_line['touchdown'].sum() if 'touchdown' in available_cols else 0,
                    'td_rate': (goal_line['touchdown'].sum() / len(goal_line)) * 100 if 'touchdown' in available_cols else 0,
                    'run_pct': (len(goal_line[goal_line['play_type'] == 'run']) / len(goal_line)) * 100
                }
        
        # 4th Down Aggressiveness
        if 'down' in available_cols:
            fourth_downs = filtered_pbp[filtered_pbp['down'] == 4]
            if not fourth_downs.empty:
                # Count punts, field goals, and go-for-it attempts
                punts = fourth_downs[fourth_downs['play_type'] == 'punt']
                field_goals = fourth_downs[fourth_downs['play_type'] == 'field_goal']
                go_for_it = fourth_downs[~fourth_downs['play_type'].isin(['punt', 'field_goal'])]
                
                analysis['fourth_down'] = {
                    'total_attempts': len(fourth_downs),
                    'punts': len(punts),
                    'field_goals': len(field_goals),
                    'go_for_it': len(go_for_it),
                    'go_for_it_pct': (len(go_for_it) / len(fourth_downs)) * 100,
                    'go_for_it_success': (go_for_it['first_down'].sum() / len(go_for_it)) * 100 if len(go_for_it) > 0 and 'first_down' in available_cols else 0
                }
        
        return analysis
    
    def _filter_coach_plays(self, coach_name, team, season, offense=True):
        """Helper method to filter plays by coach/team/season"""
        filtered_pbp = self.pbp_data.copy()
        
        if season:
            filtered_pbp = filtered_pbp[filtered_pbp['season'] == season]
        
        if team:
            team_col = 'posteam' if offense else 'defteam'
            filtered_pbp = filtered_pbp[filtered_pbp[team_col] == team]
        
        if coach_name:
            coach_plays = []
            for (coach, seas), data in self.coaching_data.items():
                if coach == coach_name and (season is None or seas == season):
                    for game in data['games']:
                        team_col = 'posteam' if offense else 'defteam'
                        game_plays = filtered_pbp[
                            (filtered_pbp['game_id'] == game['game_id']) & 
                            (filtered_pbp[team_col] == game['team'])
                        ]
                        coach_plays.append(game_plays)
            
            if coach_plays:
                filtered_pbp = pd.concat(coach_plays, ignore_index=True)
            else:
                filtered_pbp = pd.DataFrame()
        
        return filtered_pbp
    
    def get_coach_strengths_weaknesses(self, coach_name, season=None):
        """Identify coach's top strengths and biggest weaknesses"""
        grades = self.grade_coach_performance(coach_name, season)
        if not grades:
            return None
        
        all_grades = {}
        
        # Flatten all grades
        if 'offensive' in grades:
            for category, score in grades['offensive'].items():
                all_grades[f"OFF: {category.replace('_', ' ').title()}"] = score
        
        if 'defensive' in grades:
            for category, score in grades['defensive'].items():
                all_grades[f"DEF: {category.replace('_', ' ').title()}"] = score
        
        # Sort by grade
        sorted_grades = sorted(all_grades.items(), key=lambda x: x[1], reverse=True)
        
        strengths = sorted_grades[:3]  # Top 3
        weaknesses = sorted_grades[-3:]  # Bottom 3
        
        return {
            'strengths': [(cat, score, self.get_letter_grade(score)) for cat, score in strengths],
            'weaknesses': [(cat, score, self.get_letter_grade(score)) for cat, score in weaknesses[::-1]]  # Reverse for worst first
        }
    
    def compare_coaches(self, coach_names, season=None):
        """Compare multiple coaches side by side"""
        print(f"\n{'COACH COMPARISON':^80}")
        if season:
            print(f"Season: {season}")
        print("=" * 80)
        
        coach_grades = {}
        for coach in coach_names:
            grades = self.grade_coach_performance(coach, season)
            if grades:
                coach_grades[coach] = grades
        
        if not coach_grades:
            print("No data available for comparison")
            return
        
        # Print comparison table
        categories = ['offensive_overall', 'defensive_overall', 'overall']
        
        print(f"{'Category':<20}", end="")
        for coach in coach_names:
            print(f"{coach[:15]:<20}", end="")
        print()
        print("-" * (20 + 20 * len(coach_names)))
        
        for category in categories:
            print(f"{category.replace('_', ' ').title():<20}", end="")
            for coach in coach_names:
                if coach in coach_grades and category in coach_grades[coach]:
                    score = coach_grades[coach][category]
                    letter = self.get_letter_grade(score)
                    print(f"{score:>5.1f} ({letter})<{'':<8}", end="")
                else:
                    print(f"{'N/A':<20}", end="")
            print()
    
    def get_all_coaches(self, season=None):
        """Get list of all coaches in the dataset"""
        coaches = set()
        for (coach, seas), data in self.coaching_data.items():
            if season is None or seas == season:
                coaches.add(coach)
        return sorted(list(coaches))
    
    def debug_columns(self):
        """Show available columns in the dataset for debugging"""
        if self.pbp_data is not None:
            print(f"\nAvailable PBP columns ({len(self.pbp_data.columns)}):")
            cols = sorted(self.pbp_data.columns.tolist())
            for i, col in enumerate(cols):
                if i % 4 == 0:
                    print()
                print(f"{col:<25}", end="")
            print("\n")
        
        if self.schedule_data is not None:
            print(f"Available Schedule columns ({len(self.schedule_data.columns)}):")
            cols = sorted(self.schedule_data.columns.tolist())
            for i, col in enumerate(cols):
                if i % 4 == 0:
                    print()
                print(f"{col:<25}", end="")
            print("\n")


# Main analysis function - same as original but with improved grading
def main():
    """Main analysis function with improved grading"""
    print("NFL Coaching Analytics System - IMPROVED VERSION")
    print("=" * 60)
    
    # Initialize the system
    analytics = NFLCoachingAnalytics(years=[2023, 2024])
    
    # Load data
    analytics.load_data()
    
    # Extract coaching information
    analytics.extract_coaching_info()
    
    # Get available coaches
    coaches = analytics.get_all_coaches()
    print(f"\nAvailable coaches: {len(coaches)}")
    if coaches:
        print(f"Sample coaches: {coaches[:10]}")
    
    # Example analysis - pick a well-known coach
    if coaches:
        example_coach = None
        # Look for some well-known coaches
        for coach in coaches:
            if any(name in coach.lower() for name in ['reid', 'belichick', 'tomlin', 'harbaugh']):
                example_coach = coach
                break
        
        if not example_coach and coaches:
            example_coach = coaches[0]  # Just pick the first one
        
        if example_coach:
            print(f"\n{'='*60}")
            print(f"IMPROVED COACHING ANALYSIS: {example_coach}")
            print(f"{'='*60}")
            
            # Basic comprehensive report with improved grading
            analytics.print_coach_report(example_coach, season=2023)
            
            # Try advanced features with error handling
            try:
                # Situational Analysis
                print(f"\n--- DETAILED SITUATIONAL ANALYSIS ---")
                situational = analytics.analyze_situational_performance(coach_name=example_coach, season=2023)
                if situational:
                    for situation, data in situational.items():
                        print(f"\n{situation.replace('_', ' ').title()}:")
                        for key, value in data.items():
                            if isinstance(value, float):
                                print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                            else:
                                print(f"  {key.replace('_', ' ').title()}: {value}")
                else:
                    print("No situational data available")
            except Exception as e:
                print(f"Error in situational analysis: {e}")
            
            try:
                # Strengths and Weaknesses
                strengths_weak = analytics.get_coach_strengths_weaknesses(example_coach, season=2023)
                if strengths_weak:
                    print(f"\n{'TOP STRENGTHS':^60}")
                    print("-" * 60)
                    for category, score, letter in strengths_weak['strengths']:
                        print(f"{category:<45} {score:>6.1f} ({letter})")
                    
                    print(f"\n{'AREAS FOR IMPROVEMENT':^60}")
                    print("-" * 60)
                    for category, score, letter in strengths_weak['weaknesses']:
                        print(f"{category:<45} {score:>6.1f} ({letter})")
            except Exception as e:
                print(f"Error in strengths/weaknesses analysis: {e}")
            
            # Show some coaches for comparison if we have multiple
            if len(coaches) >= 3:
                print(f"\n{'='*60}")
                print("MULTI-COACH COMPARISON (IMPROVED GRADING)")
                print(f"{'='*60}")
                comparison_coaches = coaches[:3]
                try:
                    analytics.compare_coaches(comparison_coaches, season=2023)
                except Exception as e:
                    print(f"Error in coach comparison: {e}")
    
    print(f"\n{'='*60}")
    print("Analysis Complete! The grading should now be much more realistic.")
    print("Key improvements:")
    print("- Fixed overall grade calculation")
    print("- More realistic performance thresholds")
    print("- Proper weighted averaging")
    print(f"{'='*60}")


def debug_mode():
    """Debug mode to explore available data"""
    print("NFL Coaching Analytics - Debug Mode")
    print("=" * 50)
    
    analytics = NFLCoachingAnalytics(years=[2023, 2024])
    analytics.load_data()
    analytics.debug_columns()


def interactive_mode():
    """Interactive mode for exploring coaches"""
    analytics = NFLCoachingAnalytics(years=[2023, 2024])
    analytics.load_data()
    analytics.extract_coaching_info()
    
    coaches = analytics.get_all_coaches()
    
    while True:
        print(f"\n{'='*50}")
        print("INTERACTIVE COACHING ANALYTICS (IMPROVED)")
        print(f"{'='*50}")
        print("Available commands:")
        print("1. List all coaches")
        print("2. Analyze specific coach")
        print("3. Compare coaches")
        print("4. Find best coaches by category")
        print("5. Debug mode (show available columns)")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            print(f"\nAvailable coaches ({len(coaches)}):")
            for i, coach in enumerate(coaches, 1):
                print(f"{i:2d}. {coach}")
        
        elif choice == '2':
            coach_name = input("Enter coach name: ")
            if coach_name in coaches:
                season = input("Enter season (2023/2024 or press Enter for 2023): ")
                season = int(season) if season.isdigit() else 2023
                try:
                    analytics.print_coach_report(coach_name, season)
                    
                    # Try advanced analysis
                    print("\nTrying advanced analysis...")
                    situational = analytics.analyze_situational_performance(coach_name=coach_name, season=season)
                    if situational:
                        print("Situational Performance:")
                        for situation, data in situational.items():
                            print(f"  {situation}: {data}")
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
                season = input("Enter season (2023/2024 or press Enter for 2023): ")
                season = int(season) if season.isdigit() else 2023
                try:
                    analytics.compare_coaches(valid_coaches, season)
                except Exception as e:
                    print(f"Error comparing coaches: {e}")
            else:
                print("No valid coaches found.")
        
        elif choice == '4':
            print("Finding best coaches by category...")
            try:
                all_grades = {}
                for coach in coaches:
                    grades = analytics.grade_coach_performance(coach, season=2023)
                    if grades:
                        all_grades[coach] = grades
                
                if all_grades:
                    categories = ['offensive_overall', 'defensive_overall', 'overall']
                    for category in categories:
                        best = max(all_grades.items(), 
                                 key=lambda x: x[1].get(category, 0))
                        worst = min(all_grades.items(), 
                                  key=lambda x: x[1].get(category, 100))
                        
                        print(f"\n{category.replace('_', ' ').title()}:")
                        print(f"  Best: {best[0]} ({best[1].get(category, 0):.1f})")
                        print(f"  Worst: {worst[0]} ({worst[1].get(category, 0):.1f})")
                else:
                    print("No grading data available")
            except Exception as e:
                print(f"Error finding best coaches: {e}")
        
        elif choice == '5':
            analytics.debug_columns()
        
        elif choice == '6':
            print("Thanks for using NFL Coaching Analytics!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()