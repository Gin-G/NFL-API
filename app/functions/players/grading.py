#!/usr/bin/env python3
"""
Enhanced NFL Player Grading System with Individual and Unit Line Grading

Integrates offensive and defensive line grading (both individual players and units)
into the existing player grading system with realistic grade scales.
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedNFLPlayerGrader:
    """Enhanced grading system including individual and unit line grading."""
    
    def __init__(self, years: List[int] = [2023]):
        """
        Initialize the enhanced grading system.
        
        Args:
            years: List of seasons to analyze
        """
        self.years = years
        self.weekly_data = None
        self.pbp_data = None
        self.rosters = None
        self.snap_counts = None
        
        # Line positions
        self.oline_positions = ['C', 'G', 'LG', 'RG', 'T', 'LT', 'RT', 'OL']
        self.dline_positions = ['DE', 'DT', 'NT', 'EDGE', 'DL']
        
        # Grade scale definitions
        self.grade_scale = {
            'A+': (95, 100), 'A': (90, 94.9), 'A-': (85, 89.9),
            'B+': (80, 84.9), 'B': (75, 79.9), 'B-': (70, 74.9),
            'C+': (65, 69.9), 'C': (55, 64.9), 'C-': (50, 54.9),
            'D+': (45, 49.9), 'D': (40, 44.9), 'D-': (35, 39.9),
            'F': (0, 34.9)
        }
        
        print(f"Initializing Enhanced NFL Player Grading System with Line Grading for years: {years}")
        self._load_data()
    
    def _load_data(self):
        """Load all necessary data."""
        try:
            print("Loading NFL data...")
            
            # Load standard data
            self.weekly_data = nfl.import_weekly_data(self.years)
            self.pbp_data = nfl.import_pbp_data(self.years)
            self.rosters = nfl.import_weekly_rosters(self.years)
            
            # Load snap counts for line analysis
            try:
                self.snap_counts = nfl.import_snap_counts(self.years)
                print(f"- Snap counts: {len(self.snap_counts)} records")
            except Exception as e:
                print(f"- Snap counts: Error loading ({e}), will use PBP data only")
                self.snap_counts = pd.DataFrame()
            
            try:
                self._prepare_data()
            except Exception as e:
                print(f"Error in data preparation: {e}")
                print("Continuing with available data...")
                # Set empty DataFrames for line data if preparation fails
                self.oline_snap_counts = pd.DataFrame()
                self.dline_snap_counts = pd.DataFrame()
            
            print(f"Enhanced data loaded successfully!")
            print(f"- Weekly data: {len(self.weekly_data)} records")
            print(f"- Play-by-play: {len(self.pbp_data)} records")
            
        except Exception as e:
            print(f"Critical error loading data: {e}")
            print("This may be due to data structure changes in nfl_data_py")
            raise
    
    def _prepare_data(self):
        """Clean and prepare all datasets."""
        print("Preparing enhanced player data...")
        
        # Prepare offensive data (existing logic)
        self._prepare_offensive_data()
        
        # Prepare defensive data (existing logic) 
        self._prepare_defensive_data()
        
        # Prepare line-specific data
        self._prepare_line_data()
    
    def _prepare_offensive_data(self):
        """Prepare offensive player data from weekly stats."""
        if 'position' in self.rosters.columns:
            roster_info = self.rosters.groupby('player_id').agg({
                'position': 'first',
                'team': 'first'
            }).reset_index()
            
            self.weekly_data = pd.merge(
                self.weekly_data,
                roster_info,
                on='player_id',
                how='left',
                suffixes=('', '_roster')
            )
        
        # Fill missing numeric columns
        numeric_cols = self.weekly_data.select_dtypes(include=[np.number]).columns
        self.weekly_data[numeric_cols] = self.weekly_data[numeric_cols].fillna(value=0)
        
        # Filter to players with meaningful offensive stats
        self.weekly_data = self.weekly_data[
            (self.weekly_data['attempts'] > 0) |
            (self.weekly_data['carries'] > 0) |
            (self.weekly_data['targets'] > 0)
        ]
        
        print(f"Offensive data prepared: {len(self.weekly_data)} records")
    
    def _prepare_defensive_data(self):
        """Prepare defensive player data from play-by-play."""
        print("Extracting defensive statistics from play-by-play data...")
        self.defensive_weekly = self._create_defensive_weekly_stats()
        print(f"Defensive data prepared: {len(self.defensive_weekly)} records")
    
    def _prepare_line_data(self):
        """Prepare offensive and defensive line specific data."""
        print("Preparing line-specific data...")
        
        # Create line-specific play-by-play metrics
        self._add_line_metrics_to_pbp()
        
        # Prepare snap count data for line players if available
        if not self.snap_counts.empty:
            print("Available snap count columns:", list(self.snap_counts.columns))
            
            # Check if we have player_id column, if not, try to use player name
            if 'player_id' in self.snap_counts.columns:
                merge_on = ['player_id', 'team', 'week', 'season']
                roster_cols = ['player_id', 'position', 'team', 'week', 'season']
            elif 'player' in self.snap_counts.columns:
                # Use player name for merging
                merge_on = ['player', 'team', 'week', 'season']
                # Create a roster mapping by player name
                roster_by_name = self.rosters.groupby(['player_name', 'team', 'week', 'season']).agg({
                    'position': 'first'
                }).reset_index()
                roster_by_name = roster_by_name.rename(columns={'player_name': 'player'})
                roster_cols = ['player', 'position', 'team', 'week', 'season']
                
                self.line_snap_counts = self.snap_counts.merge(
                    roster_by_name[roster_cols],
                    on=merge_on,
                    how='left'
                )
            else:
                print("Warning: No suitable player identifier found in snap counts")
                self.oline_snap_counts = pd.DataFrame()
                self.dline_snap_counts = pd.DataFrame()
                return
            
            if 'player_id' in self.snap_counts.columns:
                self.line_snap_counts = self.snap_counts.merge(
                    self.rosters[roster_cols],
                    on=merge_on,
                    how='left'
                )
            
            # Filter to line positions
            if not self.line_snap_counts.empty:
                self.oline_snap_counts = self.line_snap_counts[
                    self.line_snap_counts['position'].isin(self.oline_positions)
                ].copy()
                
                self.dline_snap_counts = self.line_snap_counts[
                    self.line_snap_counts['position'].isin(self.dline_positions)
                ].copy()
                
                print(f"O-Line snap data: {len(self.oline_snap_counts)} records")
                print(f"D-Line snap data: {len(self.dline_snap_counts)} records")
            else:
                print("Warning: No line snap count data after merge")
                self.oline_snap_counts = pd.DataFrame()
                self.dline_snap_counts = pd.DataFrame()
        else:
            self.oline_snap_counts = pd.DataFrame()
            self.dline_snap_counts = pd.DataFrame()
    
    def _add_line_metrics_to_pbp(self):
        """Add line-specific metrics to play-by-play data."""
        print("Adding line performance metrics to play-by-play data...")
        
        # Filter to relevant plays
        self.line_pbp = self.pbp_data[
            self.pbp_data['play_type'].isin(['pass', 'run']) &
            self.pbp_data['posteam'].notna()
        ].copy()
        
        # Offensive line metrics
        self.line_pbp['pressure_allowed'] = (
            (self.line_pbp['sack'] == 1) |
            (self.line_pbp['qb_hit'] == 1)
        ).astype(int)
        
        # More realistic success definitions
        self.line_pbp['pass_pro_success'] = np.where(
            self.line_pbp['play_type'] == 'pass',
            1 - self.line_pbp['pressure_allowed'],
            np.nan
        )
        
        # Run blocking success (realistic threshold)
        self.line_pbp['run_success'] = np.where(
            (self.line_pbp['play_type'] == 'run') & (self.line_pbp['rushing_yards'] >= 4),
            1,
            np.where(self.line_pbp['play_type'] == 'run', 0, np.nan)
        )
        
        # Defensive line metrics
        self.line_pbp['dline_pressure'] = (
            (self.line_pbp['sack'] == 1) |
            (self.line_pbp['qb_hit'] == 1)
        ).astype(int)
        
        self.line_pbp['run_stuff'] = np.where(
            (self.line_pbp['play_type'] == 'run') & (self.line_pbp['rushing_yards'] <= 2),
            1,
            np.where(self.line_pbp['play_type'] == 'run', 0, np.nan)
        )
        
        # Negative plays
        self.line_pbp['negative_play_allowed'] = (
            (self.line_pbp['rushing_yards'] < 0) |
            (self.line_pbp['sack'] == 1)
        ).astype(int)
        
        self.line_pbp['negative_play_created'] = (
            (self.line_pbp['rushing_yards'] < 0) |
            (self.line_pbp['sack'] == 1)
        ).astype(int)
        
        print("Line metrics added successfully")
    
    def calculate_individual_oline_grades(self, min_games: int = 3):
        """Calculate grades for individual offensive linemen."""
        print("Calculating individual offensive line grades...")
        
        if self.oline_snap_counts.empty:
            print("No O-Line snap count data available - using alternative method")
            return self._calculate_oline_grades_from_pbp(min_games)
        
        # Get offensive line players with significant snaps
        oline_players = self.oline_snap_counts[
            self.oline_snap_counts['offense_snaps'] >= 20  # Minimum snaps per game
        ].copy()
        
        if oline_players.empty:
            print("No qualifying O-Line players found - using alternative method")
            return self._calculate_oline_grades_from_pbp(min_games)
        
        # Calculate team O-Line performance for context
        team_oline_performance = self._calculate_team_oline_performance()
        
        individual_grades = []
        
        for _, player_row in oline_players.iterrows():
            try:
                # Get team performance for this player's team/week
                team_perf = team_oline_performance[
                    (team_oline_performance['team'] == player_row['team']) &
                    (team_oline_performance['season'] == player_row['season']) &
                    (team_oline_performance['week'] == player_row['week'])
                ]
                
                if team_perf.empty:
                    continue
                
                team_perf = team_perf.iloc[0]
                
                # Calculate individual grade based on position and team performance
                position_grade = self._calculate_oline_position_grade(
                    player_row, team_perf
                )
                
                # Use the correct player identifier
                player_id = player_row.get('player_id', player_row.get('player', 'unknown'))
                player_name = player_row.get('player', player_row.get('player_name', 'Unknown'))
                
                individual_grades.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'position': player_row['position'],
                    'team': player_row['team'],
                    'season': player_row['season'],
                    'week': player_row['week'],
                    'snaps': player_row['offense_snaps'],
                    'snap_pct': player_row.get('offense_pct', 0),
                    'individual_grade': position_grade,
                    'letter_grade': self._numeric_to_letter_grade(position_grade),
                    'team_pass_pro': team_perf['pass_pro_rate'],
                    'team_run_success': team_perf['run_success_rate'],
                    'team_pressure_allowed': team_perf['pressure_rate']
                })
                
            except Exception as e:
                print(f"Error grading O-Line player {player_row.get('player', 'Unknown')}: {e}")
                continue
        
        oline_grades_df = pd.DataFrame(individual_grades)
        
        # Filter by minimum games
        if not oline_grades_df.empty:
            game_counts = oline_grades_df.groupby('player_id').size()
            qualified_players = game_counts[game_counts >= min_games].index
            oline_grades_df = oline_grades_df[oline_grades_df['player_id'].isin(qualified_players)]
        
        print(f"Calculated individual O-Line grades for {len(oline_grades_df)} records")
        return oline_grades_df
    
    def calculate_individual_dline_grades(self, min_games: int = 3):
        """Calculate grades for individual defensive linemen."""
        print("Calculating individual defensive line grades...")
        
        if self.dline_snap_counts.empty:
            print("No D-Line snap count data available - using alternative method")
            return self._calculate_dline_grades_from_pbp(min_games)
        
        # Get defensive line players with significant snaps
        dline_players = self.dline_snap_counts[
            self.dline_snap_counts['defense_snaps'] >= 15  # Minimum snaps per game
        ].copy()
        
        if dline_players.empty:
            print("No qualifying D-Line players found - using alternative method")
            return self._calculate_dline_grades_from_pbp(min_games)
        
        # Calculate team D-Line performance for context
        team_dline_performance = self._calculate_team_dline_performance()
        
        individual_grades = []
        
        for _, player_row in dline_players.iterrows():
            try:
                # Get team performance for this player's team/week
                team_perf = team_dline_performance[
                    (team_dline_performance['team'] == player_row['team']) &
                    (team_dline_performance['season'] == player_row['season']) &
                    (team_dline_performance['week'] == player_row['week'])
                ]
                
                if team_perf.empty:
                    continue
                
                team_perf = team_perf.iloc[0]
                
                # Calculate individual grade based on position and team performance
                position_grade = self._calculate_dline_position_grade(
                    player_row, team_perf
                )
                
                # Use the correct player identifier
                player_id = player_row.get('player_id', player_row.get('player', 'unknown'))
                player_name = player_row.get('player', player_row.get('player_name', 'Unknown'))
                
                individual_grades.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'position': player_row['position'],
                    'team': player_row['team'],
                    'season': player_row['season'],
                    'week': player_row['week'],
                    'snaps': player_row['defense_snaps'],
                    'snap_pct': player_row.get('defense_pct', 0),
                    'individual_grade': position_grade,
                    'letter_grade': self._numeric_to_letter_grade(position_grade),
                    'team_pressure_rate': team_perf['pressure_rate'],
                    'team_run_stuff_rate': team_perf['run_stuff_rate'],
                    'team_negative_plays': team_perf['negative_play_rate']
                })
                
            except Exception as e:
                print(f"Error grading D-Line player {player_row.get('player', 'Unknown')}: {e}")
                continue
        
        dline_grades_df = pd.DataFrame(individual_grades)
        
        # Filter by minimum games
        if not dline_grades_df.empty:
            game_counts = dline_grades_df.groupby('player_id').size()
            qualified_players = game_counts[game_counts >= min_games].index
            dline_grades_df = dline_grades_df[dline_grades_df['player_id'].isin(qualified_players)]
        
        print(f"Calculated individual D-Line grades for {len(dline_grades_df)} records")
        return dline_grades_df
    
    def _calculate_oline_grades_from_pbp(self, min_games: int = 3):
        """Alternative method to calculate O-Line grades when snap count data unavailable."""
        print("Using play-by-play data to estimate O-Line player performance...")
        
        # Create estimated O-Line grades based on team performance
        team_oline_performance = self._calculate_team_oline_performance()
        
        if team_oline_performance.empty:
            return pd.DataFrame()
        
        # Create mock individual grades for demonstration
        individual_grades = []
        
        for _, team_perf in team_oline_performance.iterrows():
            # Create estimated grades for typical O-Line positions
            positions = ['LT', 'LG', 'C', 'RG', 'RT']
            
            for i, pos in enumerate(positions):
                try:
                    # Create mock player data
                    mock_player = {
                        'position': pos,
                        'offense_pct': 85 + (i * 2),  # Slight variation
                        'team': team_perf['team'],
                        'season': team_perf['season'],
                        'week': team_perf['week']
                    }
                    
                    position_grade = self._calculate_oline_position_grade(mock_player, team_perf)
                    
                    individual_grades.append({
                        'player_id': f"{team_perf['team']}_{pos}_{team_perf['week']}",
                        'player_name': f"{team_perf['team']} {pos}",
                        'position': pos,
                        'team': team_perf['team'],
                        'season': team_perf['season'],
                        'week': team_perf['week'],
                        'snaps': 55,  # Estimate
                        'snap_pct': 85,  # Estimate
                        'individual_grade': position_grade,
                        'letter_grade': self._numeric_to_letter_grade(position_grade),
                        'team_pass_pro': team_perf['pass_pro_rate'],
                        'team_run_success': team_perf['run_success_rate'],
                        'team_pressure_allowed': team_perf['pressure_rate']
                    })
                    
                except Exception as e:
                    continue
        
        oline_grades_df = pd.DataFrame(individual_grades)
        print(f"Created estimated O-Line grades for {len(oline_grades_df)} records")
        return oline_grades_df
    
    def _calculate_dline_grades_from_pbp(self, min_games: int = 3):
        """Alternative method to calculate D-Line grades when snap count data unavailable."""
        print("Using play-by-play data to estimate D-Line player performance...")
        
        # Create estimated D-Line grades based on team performance
        team_dline_performance = self._calculate_team_dline_performance()
        
        if team_dline_performance.empty:
            return pd.DataFrame()
        
        # Create mock individual grades for demonstration
        individual_grades = []
        
        for _, team_perf in team_dline_performance.iterrows():
            # Create estimated grades for typical D-Line positions
            positions = ['DE', 'DT', 'DE', 'DT']  # Typical 4-man front
            
            for i, pos in enumerate(positions):
                try:
                    # Create mock player data
                    mock_player = {
                        'position': pos,
                        'defense_pct': 80 + (i * 3),  # Slight variation
                        'team': team_perf['team'],
                        'season': team_perf['season'],
                        'week': team_perf['week']
                    }
                    
                    position_grade = self._calculate_dline_position_grade(mock_player, team_perf)
                    
                    individual_grades.append({
                        'player_id': f"{team_perf['team']}_{pos}_{i}_{team_perf['week']}",
                        'player_name': f"{team_perf['team']} {pos} {i+1}",
                        'position': pos,
                        'team': team_perf['team'],
                        'season': team_perf['season'],
                        'week': team_perf['week'],
                        'snaps': 45,  # Estimate
                        'snap_pct': 75,  # Estimate
                        'individual_grade': position_grade,
                        'letter_grade': self._numeric_to_letter_grade(position_grade),
                        'team_pressure_rate': team_perf['pressure_rate'],
                        'team_run_stuff_rate': team_perf['run_stuff_rate'],
                        'team_negative_plays': team_perf['negative_play_rate']
                    })
                    
                except Exception as e:
                    continue
        
        dline_grades_df = pd.DataFrame(individual_grades)
        print(f"Created estimated D-Line grades for {len(dline_grades_df)} records")
        return dline_grades_df
    
    def calculate_team_oline_grades(self, min_plays: int = 40):
        """Calculate team offensive line unit grades."""
        print("Calculating team offensive line unit grades...")
        
        team_oline_performance = self._calculate_team_oline_performance()
        
        if team_oline_performance.empty:
            return pd.DataFrame()
        
        # Filter by minimum plays
        team_oline_performance = team_oline_performance[
            team_oline_performance['total_plays'] >= min_plays
        ]
        
        unit_grades = []
        
        for _, row in team_oline_performance.iterrows():
            try:
                # Calculate realistic unit grades (more conservative)
                pass_pro_grade = self._calculate_realistic_pass_pro_grade(row)
                run_block_grade = self._calculate_realistic_run_block_grade(row)
                overall_grade = (pass_pro_grade * 0.6) + (run_block_grade * 0.4)
                
                unit_grades.append({
                    'team': row['team'],
                    'season': row['season'],
                    'week': row['week'],
                    'pass_protection_grade': pass_pro_grade,
                    'run_blocking_grade': run_block_grade,
                    'overall_oline_grade': overall_grade,
                    'letter_grade': self._numeric_to_letter_grade(overall_grade),
                    'total_plays': row['total_plays'],
                    'pass_pro_success_rate': row['pass_pro_rate'],
                    'pressure_rate': row['pressure_rate'],
                    'run_success_rate': row['run_success_rate'],
                    'sacks_allowed': row['sacks_allowed']
                })
                
            except Exception as e:
                print(f"Error grading team O-Line {row['team']} Week {row['week']}: {e}")
                continue
        
        unit_grades_df = pd.DataFrame(unit_grades)
        print(f"Calculated team O-Line unit grades for {len(unit_grades_df)} team-week records")
        return unit_grades_df
    
    def calculate_team_dline_grades(self, min_plays: int = 40):
        """Calculate team defensive line unit grades."""
        print("Calculating team defensive line unit grades...")
        
        team_dline_performance = self._calculate_team_dline_performance()
        
        if team_dline_performance.empty:
            return pd.DataFrame()
        
        # Filter by minimum plays
        team_dline_performance = team_dline_performance[
            team_dline_performance['total_plays'] >= min_plays
        ]
        
        unit_grades = []
        
        for _, row in team_dline_performance.iterrows():
            try:
                # Calculate realistic unit grades
                pass_rush_grade = self._calculate_realistic_pass_rush_grade(row)
                run_defense_grade = self._calculate_realistic_run_defense_grade(row)
                overall_grade = (pass_rush_grade * 0.6) + (run_defense_grade * 0.4)
                
                unit_grades.append({
                    'team': row['team'],
                    'season': row['season'],
                    'week': row['week'],
                    'pass_rush_grade': pass_rush_grade,
                    'run_defense_grade': run_defense_grade,
                    'overall_dline_grade': overall_grade,
                    'letter_grade': self._numeric_to_letter_grade(overall_grade),
                    'total_plays': row['total_plays'],
                    'pressure_rate': row['pressure_rate'],
                    'run_stuff_rate': row['run_stuff_rate'],
                    'sacks': row['sacks'],
                    'negative_play_rate': row['negative_play_rate']
                })
                
            except Exception as e:
                print(f"Error grading team D-Line {row['team']} Week {row['week']}: {e}")
                continue
        
        unit_grades_df = pd.DataFrame(unit_grades)
        print(f"Calculated team D-Line unit grades for {len(unit_grades_df)} team-week records")
        return unit_grades_df
    
    def _calculate_team_oline_performance(self):
        """Calculate team offensive line performance metrics."""
        if self.line_pbp.empty:
            return pd.DataFrame()
        
        # Group by team, season, week for offensive performance
        team_performance = self.line_pbp.groupby(['posteam', 'season', 'week']).agg({
            'pass_pro_success': 'mean',
            'pressure_allowed': 'mean', 
            'sack': 'sum',
            'qb_hit': 'sum',
            'run_success': 'mean',
            'rushing_yards': 'mean',
            'negative_play_allowed': 'mean',
            'play_type': 'count'
        }).reset_index()
        
        team_performance.columns = [
            'team', 'season', 'week',
            'pass_pro_rate', 'pressure_rate', 'sacks_allowed', 'qb_hits_allowed',
            'run_success_rate', 'avg_rush_yards', 'negative_play_rate', 'total_plays'
        ]
        
        return team_performance
    
    def _calculate_team_dline_performance(self):
        """Calculate team defensive line performance metrics."""
        if self.line_pbp.empty:
            return pd.DataFrame()
        
        # Group by team, season, week for defensive performance (defteam)
        team_performance = self.line_pbp.groupby(['defteam', 'season', 'week']).agg({
            'dline_pressure': 'mean',
            'sack': 'sum',
            'qb_hit': 'sum', 
            'run_stuff': 'mean',
            'rushing_yards': 'mean',
            'negative_play_created': 'mean',
            'play_type': 'count'
        }).reset_index()
        
        team_performance.columns = [
            'team', 'season', 'week',
            'pressure_rate', 'sacks', 'qb_hits', 'run_stuff_rate', 
            'avg_rush_yards_allowed', 'negative_play_rate', 'total_plays'
        ]
        
        return team_performance
    
    def _calculate_realistic_pass_pro_grade(self, row):
        """Calculate realistic pass protection grade (50-90 range typically)."""
        base_grade = 60  # Start at D+ level
        
        # Pass protection success rate (realistic expectations)
        pass_pro_rate = row.get('pass_pro_rate', 0.65)
        if pass_pro_rate >= 0.75:  # Elite
            base_grade = 85 + min(10, (pass_pro_rate - 0.75) * 40)
        elif pass_pro_rate >= 0.70:  # Good  
            base_grade = 75 + (pass_pro_rate - 0.70) * 20
        elif pass_pro_rate >= 0.65:  # Average
            base_grade = 65 + (pass_pro_rate - 0.65) * 20
        else:  # Below average
            base_grade = 45 + (pass_pro_rate * 30)
        
        # Pressure rate penalty (realistic)
        pressure_rate = row.get('pressure_rate', 0.35)
        if pressure_rate <= 0.25:  # Elite
            pressure_bonus = 5
        elif pressure_rate <= 0.35:  # Average
            pressure_bonus = 0
        else:  # Poor
            pressure_bonus = -(pressure_rate - 0.35) * 40
        
        # Sacks allowed penalty
        sacks = row.get('sacks_allowed', 2)
        if sacks == 0:
            sack_bonus = 3
        elif sacks <= 1:
            sack_bonus = 1
        elif sacks <= 2:
            sack_bonus = 0
        else:
            sack_bonus = -(sacks - 2) * 4
        
        final_grade = base_grade + pressure_bonus + sack_bonus
        return max(min(final_grade, 95), 35)  # Cap between 35-95
    
    def _calculate_realistic_run_block_grade(self, row):
        """Calculate realistic run blocking grade (45-90 range typically)."""
        base_grade = 60
        
        # Run success rate (realistic expectations)
        run_success_rate = row.get('run_success_rate', 0.40)
        if run_success_rate >= 0.50:  # Elite
            base_grade = 80 + min(15, (run_success_rate - 0.50) * 30)
        elif run_success_rate >= 0.40:  # Average
            base_grade = 65 + (run_success_rate - 0.40) * 15
        else:  # Below average
            base_grade = 45 + (run_success_rate * 50)
        
        # Average rushing yards
        avg_yards = row.get('avg_rush_yards', 4.0)
        if avg_yards >= 4.5:
            yards_bonus = min(8, (avg_yards - 4.5) * 10)
        elif avg_yards >= 4.0:
            yards_bonus = (avg_yards - 4.0) * 6
        else:
            yards_bonus = -(4.0 - avg_yards) * 5
        
        # Negative play prevention
        negative_rate = row.get('negative_play_rate', 0.15)
        if negative_rate <= 0.10:
            negative_bonus = 4
        elif negative_rate <= 0.15:
            negative_bonus = 0
        else:
            negative_bonus = -(negative_rate - 0.15) * 20
        
        final_grade = base_grade + yards_bonus + negative_bonus
        return max(min(final_grade, 90), 40)  # Cap between 40-90
    
    def _calculate_realistic_pass_rush_grade(self, row):
        """Calculate realistic pass rush grade for D-Line."""
        base_grade = 60
        
        # Pressure rate (realistic expectations for D-Line)
        pressure_rate = row.get('pressure_rate', 0.25)
        if pressure_rate >= 0.35:  # Elite
            base_grade = 85 + min(10, (pressure_rate - 0.35) * 25)
        elif pressure_rate >= 0.25:  # Average
            base_grade = 70 + (pressure_rate - 0.25) * 15
        else:  # Below average
            base_grade = 50 + (pressure_rate * 80)
        
        # Sacks (big impact)
        sacks = row.get('sacks', 1)
        if sacks >= 3:
            sack_bonus = 10 + min(5, (sacks - 3) * 2)
        elif sacks >= 1:
            sack_bonus = sacks * 5
        else:
            sack_bonus = -3
        
        final_grade = base_grade + sack_bonus
        return max(min(final_grade, 95), 35)
    
    def _calculate_realistic_run_defense_grade(self, row):
        """Calculate realistic run defense grade for D-Line."""
        base_grade = 60
        
        # Run stuff rate
        stuff_rate = row.get('run_stuff_rate', 0.20)
        if stuff_rate >= 0.30:  # Elite
            base_grade = 80 + min(15, (stuff_rate - 0.30) * 30)
        elif stuff_rate >= 0.20:  # Average
            base_grade = 65 + (stuff_rate - 0.20) * 15
        else:  # Below average
            base_grade = 50 + (stuff_rate * 75)
        
        # Average yards allowed
        avg_yards = row.get('avg_rush_yards_allowed', 4.3)
        if avg_yards <= 3.5:
            yards_bonus = min(10, (3.5 - avg_yards) * 8)
        elif avg_yards <= 4.3:
            yards_bonus = 0
        else:
            yards_bonus = -(avg_yards - 4.3) * 6
        
        final_grade = base_grade + yards_bonus
        return max(min(final_grade, 90), 40)
    
    def _calculate_oline_position_grade(self, player_row, team_performance):
        """Calculate individual O-Line player grade based on team performance and snap count."""
        base_grade = 60
        
        # Snap count factor (more snaps = more responsibility)
        snap_pct = player_row.get('offense_pct', 0) / 100
        snap_factor = 0.8 + (snap_pct * 0.4)  # 0.8 to 1.2 multiplier
        
        # Team performance factor (individual contributes to team success)
        team_pass_pro = team_performance.get('pass_pro_rate', 0.65)
        team_run_success = team_performance.get('run_success_rate', 0.40)
        
        # Position-specific adjustments
        position = player_row.get('position', 'OL')
        if position in ['LT', 'RT']:  # Tackles more important for pass pro
            performance_score = (team_pass_pro * 0.7) + (team_run_success * 0.3)
        elif position == 'C':  # Center important for both
            performance_score = (team_pass_pro * 0.5) + (team_run_success * 0.5)
        else:  # Guards more important for run blocking
            performance_score = (team_pass_pro * 0.3) + (team_run_success * 0.7)
        
        # Convert performance to grade
        if performance_score >= 0.60:
            grade = 75 + (performance_score - 0.60) * 50
        elif performance_score >= 0.50:
            grade = 65 + (performance_score - 0.50) * 10
        else:
            grade = 45 + (performance_score * 40)
        
        final_grade = grade * snap_factor
        return max(min(final_grade, 90), 35)
    
    def _calculate_dline_position_grade(self, player_row, team_performance):
        """Calculate individual D-Line player grade based on team performance and snap count."""
        base_grade = 60
        
        # Snap count factor
        snap_pct = player_row.get('defense_pct', 0) / 100
        snap_factor = 0.8 + (snap_pct * 0.4)
        
        # Team performance factor
        team_pressure = team_performance.get('pressure_rate', 0.25)
        team_stuff_rate = team_performance.get('run_stuff_rate', 0.20)
        
        # Position-specific adjustments
        position = player_row.get('position', 'DL')
        if position in ['DE', 'EDGE']:  # Edge rushers more important for pass rush
            performance_score = (team_pressure * 0.8) + (team_stuff_rate * 0.2)
        elif position in ['DT', 'NT']:  # Interior more important for run defense
            performance_score = (team_pressure * 0.3) + (team_stuff_rate * 0.7)
        else:  # General DL
            performance_score = (team_pressure * 0.6) + (team_stuff_rate * 0.4)
        
        # Convert performance to grade
        if performance_score >= 0.35:
            grade = 75 + (performance_score - 0.35) * 60
        elif performance_score >= 0.25:
            grade = 65 + (performance_score - 0.25) * 10
        else:
            grade = 45 + (performance_score * 80)
        
        final_grade = grade * snap_factor
        return max(min(final_grade, 90), 35)
    
    def calculate_enhanced_qb_grades_with_oline(self, team_oline_grades, min_games: int = 3):
        """Calculate QB grades with O-Line adjustments using realistic O-Line grades."""
        print("Calculating enhanced QB grades with realistic O-Line adjustments...")
        
        qb_data = self.weekly_data[
            (self.weekly_data['position'] == 'QB') &
            (self.weekly_data['attempts'] >= 10)
        ].copy()
        
        if qb_data.empty:
            print("No QB data available")
            return pd.DataFrame()
        
        # Merge with realistic O-Line grades
        qb_with_oline = qb_data.merge(
            team_oline_grades[['team', 'season', 'week', 'pass_protection_grade', 'overall_oline_grade']],
            left_on=['recent_team', 'season', 'week'],
            right_on=['team', 'season', 'week'],
            how='left'
        )
        
        enhanced_grades = []
        
        for _, qb_row in qb_with_oline.iterrows():
            try:
                # Calculate base QB grade (existing logic)
                base_grade = self._calculate_qb_grade(qb_row, {'passing_yards': 250, 'passing_tds': 1.5, 'interceptions': 1.0, 'attempts': 30, 'completions': 20})
                
                # Apply realistic O-Line adjustment
                oline_grade = qb_row.get('pass_protection_grade', 65)  # Default average
                oline_adjustment = self._calculate_realistic_oline_adjustment(oline_grade, 'QB')
                adjusted_grade = base_grade * oline_adjustment
                adjusted_grade = max(min(adjusted_grade, 100), 0)
                
                enhanced_grades.append({
                    'player_id': qb_row['player_id'],
                    'player_name': qb_row.get('player_name', qb_row.get('player_display_name', 'Unknown')),
                    'position': 'QB',
                    'team': qb_row.get('team', qb_row.get('recent_team', 'Unknown')),
                    'season': qb_row['season'],
                    'week': qb_row['week'],
                    'base_grade': base_grade,
                    'oline_grade': oline_grade,
                    'oline_adjustment': oline_adjustment,
                    'adjusted_grade': adjusted_grade,
                    'grade_improvement': adjusted_grade - base_grade,
                    'oline_tier': self._get_realistic_oline_tier(oline_grade),
                    
                    # Performance metrics
                    'attempts': qb_row.get('attempts', 0),
                    'completions': qb_row.get('completions', 0),
                    'passing_yards': qb_row.get('passing_yards', 0),
                    'passing_tds': qb_row.get('passing_tds', 0),
                    'interceptions': qb_row.get('interceptions', 0)
                })
                
            except Exception as e:
                print(f"Error grading QB {qb_row.get('player_name', 'Unknown')}: {e}")
                continue
        
        enhanced_df = pd.DataFrame(enhanced_grades)
        
        # Filter by minimum games
        if not enhanced_df.empty:
            game_counts = enhanced_df.groupby('player_id').size()
            qualified_players = game_counts[game_counts >= min_games].index
            enhanced_df = enhanced_df[enhanced_df['player_id'].isin(qualified_players)]
        
        print(f"Calculated enhanced QB grades for {len(enhanced_df)} records")
        return enhanced_df
    
    def calculate_enhanced_rb_grades_with_oline(self, team_oline_grades, min_games: int = 3):
        """Calculate RB grades with O-Line adjustments using realistic O-Line grades."""
        print("Calculating enhanced RB grades with realistic O-Line adjustments...")
        
        rb_data = self.weekly_data[
            (self.weekly_data['position'] == 'RB') &
            (self.weekly_data['carries'] >= 5)
        ].copy()
        
        if rb_data.empty:
            print("No RB data available")
            return pd.DataFrame()
        
        # Merge with realistic O-Line grades
        rb_with_oline = rb_data.merge(
            team_oline_grades[['team', 'season', 'week', 'run_blocking_grade', 'overall_oline_grade']],
            left_on=['recent_team', 'season', 'week'],
            right_on=['team', 'season', 'week'],
            how='left'
        )
        
        enhanced_grades = []
        
        for _, rb_row in rb_with_oline.iterrows():
            try:
                # Calculate base RB grade (existing logic)
                base_grade = self._calculate_rb_grade(rb_row, {'rushing_yards': 80, 'rushing_tds': 0.5, 'carries': 15, 'receiving_yards': 20, 'receptions': 2})
                
                # Apply realistic O-Line adjustment
                oline_grade = rb_row.get('run_blocking_grade', 65)  # Default average
                oline_adjustment = self._calculate_realistic_oline_adjustment(oline_grade, 'RB')
                adjusted_grade = base_grade * oline_adjustment
                adjusted_grade = max(min(adjusted_grade, 100), 0)
                
                enhanced_grades.append({
                    'player_id': rb_row['player_id'],
                    'player_name': rb_row.get('player_name', rb_row.get('player_display_name', 'Unknown')),
                    'position': 'RB',
                    'team': rb_row.get('team', rb_row.get('recent_team', 'Unknown')),
                    'season': rb_row['season'],
                    'week': rb_row['week'],
                    'base_grade': base_grade,
                    'oline_grade': oline_grade,
                    'oline_adjustment': oline_adjustment,
                    'adjusted_grade': adjusted_grade,
                    'grade_improvement': adjusted_grade - base_grade,
                    'oline_tier': self._get_realistic_oline_tier(oline_grade),
                    
                    # Performance metrics
                    'carries': rb_row.get('carries', 0),
                    'rushing_yards': rb_row.get('rushing_yards', 0),
                    'rushing_tds': rb_row.get('rushing_tds', 0),
                    'receiving_yards': rb_row.get('receiving_yards', 0),
                    'receptions': rb_row.get('receptions', 0)
                })
                
            except Exception as e:
                print(f"Error grading RB {rb_row.get('player_name', 'Unknown')}: {e}")
                continue
        
        enhanced_df = pd.DataFrame(enhanced_grades)
        
        # Filter by minimum games
        if not enhanced_df.empty:
            game_counts = enhanced_df.groupby('player_id').size()
            qualified_players = game_counts[game_counts >= min_games].index
            enhanced_df = enhanced_df[enhanced_df['player_id'].isin(qualified_players)]
        
        print(f"Calculated enhanced RB grades for {len(enhanced_df)} records")
        return enhanced_df
    
    def _calculate_realistic_oline_adjustment(self, oline_grade, position):
        """Calculate realistic O-Line adjustment factor."""
        if pd.isna(oline_grade):
            return 1.0
        
        # More conservative adjustments (±10% max instead of ±20%)
        if oline_grade >= 85:  # Elite O-Line
            return 1.08 if position == 'RB' else 1.06
        elif oline_grade >= 75:  # Good O-Line
            return 1.05 if position == 'RB' else 1.03
        elif oline_grade >= 65:  # Average O-Line
            return 1.0
        elif oline_grade >= 55:  # Poor O-Line
            return 0.95 if position == 'RB' else 0.97
        else:  # Terrible O-Line
            return 0.90 if position == 'RB' else 0.93
    
    def _get_realistic_oline_tier(self, oline_grade):
        """Get realistic O-Line tier description."""
        if pd.isna(oline_grade):
            return 'Unknown'
        elif oline_grade >= 85:
            return 'Elite'
        elif oline_grade >= 75:
            return 'Good'
        elif oline_grade >= 65:
            return 'Average'
        elif oline_grade >= 55:
            return 'Poor'
        else:
            return 'Terrible'
    
    # Include existing methods from original grading system
    def _create_defensive_weekly_stats(self):
        """Create weekly defensive stats from play-by-play data."""
        defensive_stats_list = []
        
        # Process each week
        for week in self.pbp_data['week'].unique():
            if pd.isna(week):
                continue
                
            week_data = self.pbp_data[self.pbp_data['week'] == week].copy()
            
            # Aggregate sacks
            sack_stats = self._aggregate_weekly_sacks(week_data, week)
            defensive_stats_list.extend(sack_stats)
            
            # Aggregate tackles  
            tackle_stats = self._aggregate_weekly_tackles(week_data, week)
            defensive_stats_list.extend(tackle_stats)
            
            # Aggregate interceptions
            int_stats = self._aggregate_weekly_interceptions(week_data, week)
            defensive_stats_list.extend(int_stats)
            
            # Aggregate pass deflections
            pd_stats = self._aggregate_weekly_pass_deflections(week_data, week)
            defensive_stats_list.extend(pd_stats)
            
            # Aggregate forced fumbles
            ff_stats = self._aggregate_weekly_forced_fumbles(week_data, week)
            defensive_stats_list.extend(ff_stats)
            
            # Aggregate QB hits
            qbh_stats = self._aggregate_weekly_qb_hits(week_data, week)
            defensive_stats_list.extend(qbh_stats)
        
        # Convert to DataFrame and aggregate by player-week
        if defensive_stats_list:
            df = pd.DataFrame(defensive_stats_list)
            
            # Group by player-week and sum stats
            weekly_def = df.groupby(['player_id', 'player_name', 'week', 'season']).agg({
                'sacks': 'sum',
                'solo_tackles': 'sum',
                'assist_tackles': 'sum',
                'tackles_for_loss': 'sum',
                'interceptions': 'sum',
                'pass_deflections': 'sum',
                'forced_fumbles': 'sum',
                'qb_hits': 'sum'
            }).reset_index()
            
            # Calculate total tackles
            weekly_def['total_tackles'] = weekly_def['solo_tackles'] + weekly_def['assist_tackles']
            
            return weekly_def
        else:
            return pd.DataFrame()
    
    def _aggregate_weekly_sacks(self, week_data, week):
        """Aggregate sack stats for a week."""
        stats = []
        
        # Full sacks
        sacks = week_data[week_data['sack'] == 1]
        for _, play in sacks.iterrows():
            if pd.notna(play.get('sack_player_name')):
                stats.append({
                    'player_id': play['sack_player_id'],
                    'player_name': play['sack_player_name'],
                    'week': week,
                    'season': play['season'],
                    'sacks': 1.0,
                    'solo_tackles': 0, 'assist_tackles': 0, 'tackles_for_loss': 0,
                    'interceptions': 0, 'pass_deflections': 0, 'forced_fumbles': 0, 'qb_hits': 0
                })
        
        # Half sacks
        half_sacks = week_data[week_data['sack'] == 1]
        for _, play in half_sacks.iterrows():
            if pd.notna(play.get('half_sack_1_player_name')):
                stats.append({
                    'player_id': play['half_sack_1_player_id'],
                    'player_name': play['half_sack_1_player_name'],
                    'week': week,
                    'season': play['season'],
                    'sacks': 0.5,
                    'solo_tackles': 0, 'assist_tackles': 0, 'tackles_for_loss': 0,
                    'interceptions': 0, 'pass_deflections': 0, 'forced_fumbles': 0, 'qb_hits': 0
                })
            if pd.notna(play.get('half_sack_2_player_name')):
                stats.append({
                    'player_id': play['half_sack_2_player_id'],
                    'player_name': play['half_sack_2_player_name'],
                    'week': week,
                    'season': play['season'],
                    'sacks': 0.5,
                    'solo_tackles': 0, 'assist_tackles': 0, 'tackles_for_loss': 0,
                    'interceptions': 0, 'pass_deflections': 0, 'forced_fumbles': 0, 'qb_hits': 0
                })
        
        return stats
    
    def _aggregate_weekly_tackles(self, week_data, week):
        """Aggregate tackle stats for a week."""
        stats = []
        
        # Solo tackles
        for _, play in week_data.iterrows():
            if pd.notna(play.get('solo_tackle_1_player_name')):
                stats.append({
                    'player_id': play['solo_tackle_1_player_id'],
                    'player_name': play['solo_tackle_1_player_name'],
                    'week': week,
                    'season': play['season'],
                    'solo_tackles': 1,
                    'sacks': 0, 'assist_tackles': 0, 'tackles_for_loss': 0,
                    'interceptions': 0, 'pass_deflections': 0, 'forced_fumbles': 0, 'qb_hits': 0
                })
        
        # Assist tackles
        for _, play in week_data.iterrows():
            if pd.notna(play.get('assist_tackle_1_player_name')):
                stats.append({
                    'player_id': play['assist_tackle_1_player_id'],
                    'player_name': play['assist_tackle_1_player_name'],
                    'week': week,
                    'season': play['season'],
                    'assist_tackles': 1,
                    'sacks': 0, 'solo_tackles': 0, 'tackles_for_loss': 0,
                    'interceptions': 0, 'pass_deflections': 0, 'forced_fumbles': 0, 'qb_hits': 0
                })
            if pd.notna(play.get('assist_tackle_2_player_name')):
                stats.append({
                    'player_id': play['assist_tackle_2_player_id'],
                    'player_name': play['assist_tackle_2_player_name'],
                    'week': week,
                    'season': play['season'],
                    'assist_tackles': 1,
                    'sacks': 0, 'solo_tackles': 0, 'tackles_for_loss': 0,
                    'interceptions': 0, 'pass_deflections': 0, 'forced_fumbles': 0, 'qb_hits': 0
                })
        
        # Tackles for loss
        for _, play in week_data.iterrows():
            if pd.notna(play.get('tackle_for_loss_1_player_name')):
                stats.append({
                    'player_id': play['tackle_for_loss_1_player_id'],
                    'player_name': play['tackle_for_loss_1_player_name'],
                    'week': week,
                    'season': play['season'],
                    'tackles_for_loss': 1,
                    'sacks': 0, 'solo_tackles': 0, 'assist_tackles': 0,
                    'interceptions': 0, 'pass_deflections': 0, 'forced_fumbles': 0, 'qb_hits': 0
                })
        
        return stats
    
    def _aggregate_weekly_interceptions(self, week_data, week):
        """Aggregate interception stats for a week."""
        stats = []
        
        interceptions = week_data[week_data['interception'] == 1]
        for _, play in interceptions.iterrows():
            if pd.notna(play.get('interception_player_name')):
                stats.append({
                    'player_id': play['interception_player_id'],
                    'player_name': play['interception_player_name'],
                    'week': week,
                    'season': play['season'],
                    'interceptions': 1,
                    'sacks': 0, 'solo_tackles': 0, 'assist_tackles': 0, 'tackles_for_loss': 0,
                    'pass_deflections': 0, 'forced_fumbles': 0, 'qb_hits': 0
                })
        
        return stats
    
    def _aggregate_weekly_pass_deflections(self, week_data, week):
        """Aggregate pass deflection stats for a week."""
        stats = []
        
        for _, play in week_data.iterrows():
            if pd.notna(play.get('pass_defense_1_player_name')):
                stats.append({
                    'player_id': play['pass_defense_1_player_id'],
                    'player_name': play['pass_defense_1_player_name'],
                    'week': week,
                    'season': play['season'],
                    'pass_deflections': 1,
                    'sacks': 0, 'solo_tackles': 0, 'assist_tackles': 0, 'tackles_for_loss': 0,
                    'interceptions': 0, 'forced_fumbles': 0, 'qb_hits': 0
                })
        
        return stats
    
    def _aggregate_weekly_forced_fumbles(self, week_data, week):
        """Aggregate forced fumble stats for a week."""
        stats = []
        
        for _, play in week_data.iterrows():
            if pd.notna(play.get('forced_fumble_player_1_player_name')):
                stats.append({
                    'player_id': play['forced_fumble_player_1_player_id'],
                    'player_name': play['forced_fumble_player_1_player_name'],
                    'week': week,
                    'season': play['season'],
                    'forced_fumbles': 1,
                    'sacks': 0, 'solo_tackles': 0, 'assist_tackles': 0, 'tackles_for_loss': 0,
                    'interceptions': 0, 'pass_deflections': 0, 'qb_hits': 0
                })
        
        return stats
    
    def _aggregate_weekly_qb_hits(self, week_data, week):
        """Aggregate QB hit stats for a week."""
        stats = []
        
        for _, play in week_data.iterrows():
            if pd.notna(play.get('qb_hit_1_player_name')):
                stats.append({
                    'player_id': play['qb_hit_1_player_id'],
                    'player_name': play['qb_hit_1_player_name'],
                    'week': week,
                    'season': play['season'],
                    'qb_hits': 1,
                    'sacks': 0, 'solo_tackles': 0, 'assist_tackles': 0, 'tackles_for_loss': 0,
                    'interceptions': 0, 'pass_deflections': 0, 'forced_fumbles': 0
                })
        
        return stats
    
    # Include existing grading methods
    def _calculate_qb_grade(self, row, position_avg):
        """Calculate QB performance grade with improved scaling."""
        # Base scoring with more realistic expectations
        passing_yards_score = min((row['passing_yards'] / max(position_avg['passing_yards'], 180)) * 25, 35)
        
        # Completion percentage scoring (more generous)
        completion_pct = row.get('completions', 0) / max(row.get('attempts', 1), 1)
        completion_pct_score = max((completion_pct - 0.5) * 60, 0)  # Start scoring at 50% completion
        
        # TD scoring
        td_score = row['passing_tds'] * 12  # Increased value
        
        # INT penalty (less harsh)
        int_penalty = row['interceptions'] * -8
        
        # Efficiency bonus
        if row.get('attempts', 0) > 0:
            yards_per_attempt = row['passing_yards'] / row['attempts']
            efficiency_bonus = max((yards_per_attempt - 6.0) * 4, 0)  # Lower threshold
        else:
            efficiency_bonus = 0
        
        # Base points for attempting passes
        attempt_bonus = min(row.get('attempts', 0) * 0.3, 10)
        
        base_score = (passing_yards_score + completion_pct_score + td_score + 
                     int_penalty + efficiency_bonus + attempt_bonus)
        
        return max(min(base_score, 100), 0)
    
    def _calculate_rb_grade(self, row, position_avg):
        """Calculate RB performance grade."""
        rushing_yards_score = min((row['rushing_yards'] / max(position_avg['rushing_yards'], 80)) * 25, 35)
        
        if row['carries'] > 0:
            ypc = row['rushing_yards'] / row['carries']
            ypc_score = min((ypc / 4.2) * 20, 25)
        else:
            ypc_score = 0
        
        rushing_td_score = row['rushing_tds'] * 10
        receiving_score = (row['receiving_yards'] * 0.15) + (row['receptions'] * 2)
        receiving_td_score = row['receiving_tds'] * 8
        
        base_score = rushing_yards_score + ypc_score + rushing_td_score + receiving_score + receiving_td_score
        return max(min(base_score, 100), 0)
    
    def _numeric_to_letter_grade(self, score):
        """Convert numeric score to letter grade."""
        for letter, (min_score, max_score) in self.grade_scale.items():
            if min_score <= score <= max_score:
                return letter
        return 'F'
    
    def calculate_all_grades(self, min_games: int = 3):
        """Calculate all player grades including line players."""
        print("\n" + "="*60)
        print("CALCULATING ALL PLAYER GRADES INCLUDING LINE PLAYERS")
        print("="*60)
        
        # Calculate team line unit grades first
        team_oline_grades = self.calculate_team_oline_grades()
        team_dline_grades = self.calculate_team_dline_grades()
        
        # Calculate individual line grades
        individual_oline_grades = self.calculate_individual_oline_grades(min_games)
        individual_dline_grades = self.calculate_individual_dline_grades(min_games)
        
        # Calculate enhanced skill position grades with line adjustments
        enhanced_qb_grades = self.calculate_enhanced_qb_grades_with_oline(team_oline_grades, min_games)
        enhanced_rb_grades = self.calculate_enhanced_rb_grades_with_oline(team_oline_grades, min_games)
        
        # Calculate regular offensive grades (WR, TE, etc.)
        offensive_grades = self.calculate_offensive_grades(min_games)
        
        # Calculate defensive grades
        defensive_grades = self.calculate_defensive_grades(min_games)
        
        print(f"\nGrades calculated:")
        print(f"- Team O-Line units: {len(team_oline_grades)} records")
        print(f"- Team D-Line units: {len(team_dline_grades)} records")
        print(f"- Individual O-Line: {len(individual_oline_grades)} records")
        print(f"- Individual D-Line: {len(individual_dline_grades)} records")
        print(f"- Enhanced QBs: {len(enhanced_qb_grades)} records")
        print(f"- Enhanced RBs: {len(enhanced_rb_grades)} records")
        print(f"- Other offensive: {len(offensive_grades)} records")
        print(f"- Defensive players: {len(defensive_grades)} records")
        
        return {
            'team_oline_grades': team_oline_grades,
            'team_dline_grades': team_dline_grades,
            'individual_oline_grades': individual_oline_grades,
            'individual_dline_grades': individual_dline_grades,
            'enhanced_qb_grades': enhanced_qb_grades,
            'enhanced_rb_grades': enhanced_rb_grades,
            'offensive_grades': offensive_grades,
            'defensive_grades': defensive_grades
        }
    
    def generate_line_report(self, all_grades):
        """Generate comprehensive line performance report."""
        print("\n" + "="*80)
        print("NFL LINE PERFORMANCE REPORT")
        print("="*80)
        
        team_oline = all_grades['team_oline_grades']
        team_dline = all_grades['team_dline_grades']
        individual_oline = all_grades['individual_oline_grades']
        individual_dline = all_grades['individual_dline_grades']
        
        # Team O-Line rankings
        if not team_oline.empty:
            print("\nTOP 5 OFFENSIVE LINE UNITS:")
            print("-" * 50)
            season_oline = team_oline.groupby('team')['overall_oline_grade'].mean().nlargest(5)
            for team, grade in season_oline.items():
                print(f"{team:3} | {grade:5.1f} | {self._numeric_to_letter_grade(grade)}")
        
        # Team D-Line rankings
        if not team_dline.empty:
            print("\nTOP 5 DEFENSIVE LINE UNITS:")
            print("-" * 50)
            season_dline = team_dline.groupby('team')['overall_dline_grade'].mean().nlargest(5)
            for team, grade in season_dline.items():
                print(f"{team:3} | {grade:5.1f} | {self._numeric_to_letter_grade(grade)}")
        
        # Individual line player rankings
        if not individual_oline.empty:
            print("\nTOP 10 OFFENSIVE LINEMEN:")
            print("-" * 60)
            top_oline_players = individual_oline.groupby(['player_name', 'position']).agg({
                'individual_grade': 'mean',
                'week': 'count'
            }).reset_index().nlargest(10, 'individual_grade')
            
            for _, player in top_oline_players.iterrows():
                print(f"{player['player_name'][:25]:25} ({player['position']:2}) | "
                      f"{player['individual_grade']:5.1f} | {player['week']} games")
        
        if not individual_dline.empty:
            print("\nTOP 10 DEFENSIVE LINEMEN:")
            print("-" * 60)
            top_dline_players = individual_dline.groupby(['player_name', 'position']).agg({
                'individual_grade': 'mean',
                'week': 'count'
            }).reset_index().nlargest(10, 'individual_grade')
            
            for _, player in top_dline_players.iterrows():
                print(f"{player['player_name'][:25]:25} ({player['position']:2}) | "
                      f"{player['individual_grade']:5.1f} | {player['week']} games")
        
        # O-Line impact on skill positions
        enhanced_qb = all_grades['enhanced_qb_grades']
        enhanced_rb = all_grades['enhanced_rb_grades']
        
        if not enhanced_qb.empty:
            print("\n" + "="*60)
            print("O-LINE IMPACT ON QUARTERBACKS")
            print("="*60)
            
            qb_impact = enhanced_qb.groupby('player_name').agg({
                'base_grade': 'mean',
                'adjusted_grade': 'mean',
                'grade_improvement': 'mean',
                'oline_grade': 'mean'
            }).round(1)
            
            print("\nQBs MOST HELPED BY O-LINE:")
            helped_qbs = qb_impact.nlargest(5, 'grade_improvement')
            for player, stats in helped_qbs.iterrows():
                if stats['grade_improvement'] > 0:
                    print(f"{player[:25]:25} | {stats['adjusted_grade']:5.1f} "
                          f"(+{stats['grade_improvement']:4.1f}) | O-Line: {stats['oline_grade']:5.1f}")
            
            print("\nQBs MOST HURT BY O-LINE:")
            hurt_qbs = qb_impact.nsmallest(5, 'grade_improvement')
            for player, stats in hurt_qbs.iterrows():
                if stats['grade_improvement'] < 0:
                    print(f"{player[:25]:25} | {stats['adjusted_grade']:5.1f} "
                          f"({stats['grade_improvement']:4.1f}) | O-Line: {stats['oline_grade']:5.1f}")
        
        if not enhanced_rb.empty:
            print("\n" + "="*60)
            print("O-LINE IMPACT ON RUNNING BACKS")
            print("="*60)
            
            rb_impact = enhanced_rb.groupby('player_name').agg({
                'base_grade': 'mean',
                'adjusted_grade': 'mean',
                'grade_improvement': 'mean',
                'oline_grade': 'mean'
            }).round(1)
            
            print("\nRBs MOST HELPED BY O-LINE:")
            helped_rbs = rb_impact.nlargest(5, 'grade_improvement')
            for player, stats in helped_rbs.iterrows():
                if stats['grade_improvement'] > 0:
                    print(f"{player[:25]:25} | {stats['adjusted_grade']:5.1f} "
                          f"(+{stats['grade_improvement']:4.1f}) | O-Line: {stats['oline_grade']:5.1f}")
            
            print("\nRBs MOST HURT BY O-LINE:")
            hurt_rbs = rb_impact.nsmallest(5, 'grade_improvement')
            for player, stats in hurt_rbs.iterrows():
                if stats['grade_improvement'] < 0:
                    print(f"{player[:25]:25} | {stats['adjusted_grade']:5.1f} "
                          f"({stats['grade_improvement']:4.1f}) | O-Line: {stats['oline_grade']:5.1f}")

    # Include existing methods for defensive and offensive grading
    def calculate_offensive_grades(self, min_games: int = 3):
        """Calculate grades for offensive players using existing logic."""
        print("Calculating offensive player grades...")
        
        # Use existing offensive grading logic from original system
        wr_te_positions = ['WR', 'TE']
        
        grades_list = []
        
        # WR/TE grades (QB and RB are handled separately with O-Line adjustments)
        pos_data = self.weekly_data[self.weekly_data['position'].isin(wr_te_positions)].copy()
        if not pos_data.empty:
            # Calculate position averages
            position_avg = pos_data.groupby(['season', 'week']).agg({
                'receiving_yards': 'mean', 'receiving_tds': 'mean', 'receptions': 'mean',
                'targets': 'mean'
            }).mean()
            
            # Calculate grades for each player-game
            for idx, row in pos_data.iterrows():
                try:
                    grade = self._calculate_wr_te_grade(row, position_avg)
                    letter_grade = self._numeric_to_letter_grade(grade)
                    
                    grades_list.append({
                        'player_id': row['player_id'],
                        'player_name': row.get('player_name', row.get('player_display_name', 'Unknown')),
                        'position': row['position'],
                        'position_group': 'WR_TE',
                        'player_type': 'OFFENSE',
                        'team': row.get('team', row.get('recent_team', 'Unknown')),
                        'season': row['season'],
                        'week': row['week'],
                        'numeric_grade': grade,
                        'letter_grade': letter_grade
                    })
                    
                except Exception as e:
                    continue
        
        grades_df = pd.DataFrame(grades_list)
        
        # Filter players with minimum games
        if not grades_df.empty:
            game_counts = grades_df.groupby('player_id').size()
            qualified_players = game_counts[game_counts >= min_games].index
            grades_df = grades_df[grades_df['player_id'].isin(qualified_players)]
        
        print(f"Calculated offensive grades for {len(grades_df)} player-game records")
        return grades_df
    
    def calculate_defensive_grades(self, min_games: int = 3):
        """Calculate grades for defensive players."""
        print("Calculating defensive player grades...")
        
        if self.defensive_weekly.empty:
            print("No defensive data available")
            return pd.DataFrame()
        
        print(f"Processing {len(self.defensive_weekly)} defensive player-week records...")
        
        # Merge with roster data to get positions
        defensive_data = self.defensive_weekly.copy()
        
        if not self.rosters.empty:
            roster_info = self.rosters.groupby('player_id').agg({
                'position': 'first'
            }).reset_index()
            
            defensive_data = pd.merge(
                defensive_data,
                roster_info,
                on='player_id',
                how='left'
            )
            
            print(f"After merging positions: {len(defensive_data)} records")
        
        # For players without positions, set as generic DEF
        defensive_data['position'] = defensive_data['position'].fillna(value='DEF')
        
        # Map to position groups - exclude line positions (they're handled separately)
        defensive_data['position_group'] = defensive_data['position'].apply(self._map_defensive_position_group)
        
        # Remove line positions and unmapped positions
        defensive_data = defensive_data[
            defensive_data['position_group'].notna() &
            ~defensive_data['position'].isin(self.dline_positions)
        ]
        
        print(f"Final defensive data for grading: {len(defensive_data)} records")
        
        if defensive_data.empty:
            print("No defensive data remaining after position mapping")
            return pd.DataFrame()
        
        grades_list = []
        
        # Calculate position averages and grades
        for position_group in defensive_data['position_group'].unique():
            pos_data = defensive_data[defensive_data['position_group'] == position_group].copy()
            if pos_data.empty:
                continue
            
            print(f"Processing {len(pos_data)} records for {position_group}")
            
            # Calculate position averages
            position_avg = pos_data.agg({
                'sacks': 'mean',
                'total_tackles': 'mean',
                'tackles_for_loss': 'mean',
                'interceptions': 'mean',
                'pass_deflections': 'mean',
                'forced_fumbles': 'mean',
                'qb_hits': 'mean'
            })
            
            # Calculate grades
            for idx, row in pos_data.iterrows():
                try:
                    grade = self._calculate_defensive_grade(row, position_avg, position_group)
                    letter_grade = self._numeric_to_letter_grade(grade)
                    
                    grades_list.append({
                        'player_id': row['player_id'],
                        'player_name': row['player_name'],
                        'position': row.get('position', 'DEF'),
                        'position_group': position_group,
                        'player_type': 'DEFENSE',
                        'team': 'Unknown',
                        'season': row['season'],
                        'week': row['week'],
                        'numeric_grade': grade,
                        'letter_grade': letter_grade
                    })
                    
                except Exception as e:
                    print(f"Error grading {row['player_name']}: {e}")
                    continue
        
        if not grades_list:
            print("No defensive grades calculated")
            return pd.DataFrame()
        
        grades_df = pd.DataFrame(grades_list)
        
        # Filter players with minimum games
        game_counts = grades_df.groupby('player_id').size()
        qualified_players = game_counts[game_counts >= min_games].index
        grades_df = grades_df[grades_df['player_id'].isin(qualified_players)]
        
        print(f"Calculated defensive grades for {len(grades_df)} player-game records")
        return grades_df
    
    def _map_defensive_position_group(self, position):
        """Map defensive positions to position groups (excluding line positions)."""
        if pd.isna(position):
            return None
        
        position = str(position).upper()
        
        # Exclude line positions - they're handled separately
        if position in self.dline_positions:
            return None
        elif position in ['OLB', 'ILB', 'MLB', 'LB']:
            return 'LINEBACKER'
        elif position in ['CB', 'S', 'FS', 'SS', 'DB']:
            return 'SECONDARY'
        else:
            return None
    
    def _calculate_defensive_grade(self, row, position_avg, position_group):
        """Calculate defensive player grade based on position group."""
        base_score = 50  # Start with C grade
        
        if position_group == 'LINEBACKER':
            # Emphasize tackles and versatility
            tackle_score = (row['total_tackles'] / max(position_avg['total_tackles'], 3)) * 20
            tfl_score = (row['tackles_for_loss'] / max(position_avg['tackles_for_loss'], 0.2)) * 10
            sack_score = row['sacks'] * 12
            int_score = row['interceptions'] * 15
            ff_score = row['forced_fumbles'] * 8
            pd_score = row['pass_deflections'] * 3
            
            base_score = tackle_score + tfl_score + sack_score + int_score + ff_score + pd_score + 15
            
        elif position_group == 'SECONDARY':
            # Emphasize coverage stats
            int_score = row['interceptions'] * 20
            pd_score = (row['pass_deflections'] / max(position_avg['pass_deflections'], 0.5)) * 15
            tackle_score = (row['total_tackles'] / max(position_avg['total_tackles'], 2)) * 8
            ff_score = row['forced_fumbles'] * 12
            
            base_score = int_score + pd_score + tackle_score + ff_score + 25
        
        return max(min(base_score, 100), 0)
    
    def _calculate_wr_te_grade(self, row, position_avg):
        """Calculate WR/TE performance grade."""
        receiving_yards_score = min((row['receiving_yards'] / max(position_avg['receiving_yards'], 60)) * 30, 40)
        receptions_score = row['receptions'] * 3
        receiving_td_score = row['receiving_tds'] * 12
        
        if row['targets'] > 0:
            catch_rate = row['receptions'] / row['targets']
            catch_rate_score = catch_rate * 15
        else:
            catch_rate_score = 0
        
        base_score = receiving_yards_score + receptions_score + receiving_td_score + catch_rate_score
        return max(min(base_score, 100), 0)


def main():
    """Main function demonstrating the integrated line grading system."""
    
    try:
        print("Initializing Enhanced NFL Player Grading System with Line Integration...")
        
        # Initialize the enhanced grader
        grader = EnhancedNFLPlayerGrader(years=[2023])
        
        # Calculate all grades including line players
        all_grades = grader.calculate_all_grades(min_games=3)
        
        # Generate comprehensive line report
        grader.generate_line_report(all_grades)
        
        print(f"\n{'='*80}")
        print("ENHANCED PLAYER GRADING SYSTEM WITH LINE INTEGRATION READY")
        print(f"{'='*80}")
        print("Key Features:")
        print("- Individual O-Line and D-Line player grades")
        print("- Team O-Line and D-Line unit grades")
        print("- QB grades adjusted for pass protection quality")
        print("- RB grades adjusted for run blocking quality")
        print("- Realistic grade scales (50-90 typical range)")
        print("- Position-specific line grading (LT vs C vs G, etc.)")
        print("- D-Line grades for pass rush and run defense")
        
        return grader, all_grades
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return None, None


if __name__ == "__main__":
    grader, all_grades = main()