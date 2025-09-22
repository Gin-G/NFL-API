#!/usr/bin/env python3
"""
Enhanced NFL Player Performance Grading System with Defensive Players

Comprehensive system that grades both offensive and defensive players using
weekly stats (offense) and play-by-play data (defense).
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
    """Enhanced grading system for both offensive and defensive players."""
    
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
        
        # Grade scale definitions
        self.grade_scale = {
            'A+': (95, 100), 'A': (90, 94.9), 'A-': (85, 89.9),
            'B+': (80, 84.9), 'B': (75, 79.9), 'B-': (70, 74.9),
            'C+': (65, 69.9), 'C': (55, 64.9), 'C-': (50, 54.9),
            'D+': (45, 49.9), 'D': (40, 44.9), 'D-': (35, 39.9),
            'F': (0, 34.9)
        }
        
        print(f"Initializing Enhanced NFL Player Grading System for years: {years}")
        self._load_data()
    
    def _load_data(self):
        """Load both weekly and play-by-play data."""
        try:
            print("Loading weekly player data...")
            self.weekly_data = nfl.import_weekly_data(self.years)
            
            print("Loading play-by-play data...")
            self.pbp_data = nfl.import_pbp_data(self.years)
            
            print("Loading rosters...")
            self.rosters = nfl.import_weekly_rosters(self.years)
            
            self._prepare_data()
            
            print(f"Data loaded successfully!")
            print(f"- Weekly data: {len(self.weekly_data)} records")
            print(f"- Play-by-play data: {len(self.pbp_data)} records")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _prepare_data(self):
        """Clean and prepare both datasets."""
        print("Preparing data...")
        
        # Prepare offensive data (weekly stats)
        self._prepare_offensive_data()
        
        # Prepare defensive data (from play-by-play)
        self._prepare_defensive_data()
    
    def _prepare_offensive_data(self):
        """Prepare offensive player data from weekly stats."""
        # Merge with rosters for position info
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
        self.weekly_data[numeric_cols] = self.weekly_data[numeric_cols].fillna(0)
        
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
        
        # Create weekly defensive stats by aggregating play-by-play data
        self.defensive_weekly = self._create_defensive_weekly_stats()
        
        print(f"Defensive data prepared: {len(self.defensive_weekly)} records")
    
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
    
    def calculate_offensive_grades(self, min_games: int = 3):
        """Calculate grades for offensive players using existing logic."""
        print("Calculating offensive player grades...")
        
        # Use existing offensive grading logic from original system
        qb_positions = ['QB']
        rb_positions = ['RB', 'FB']
        wr_te_positions = ['WR', 'TE']
        
        grades_list = []
        
        for position_group, positions in [
            ('QB', qb_positions), ('RB', rb_positions), ('WR_TE', wr_te_positions)
        ]:
            pos_data = self.weekly_data[self.weekly_data['position'].isin(positions)].copy()
            if pos_data.empty:
                continue
            
            # Calculate position averages
            position_avg = pos_data.groupby(['season', 'week']).agg({
                'passing_yards': 'mean', 'passing_tds': 'mean', 'interceptions': 'mean',
                'attempts': 'mean', 'completions': 'mean',
                'rushing_yards': 'mean', 'rushing_tds': 'mean', 'carries': 'mean',
                'receiving_yards': 'mean', 'receiving_tds': 'mean', 'receptions': 'mean',
                'targets': 'mean'
            }).mean()
            
            # Calculate grades for each player-game
            for idx, row in pos_data.iterrows():
                try:
                    if position_group == 'QB':
                        grade = self._calculate_qb_grade(row, position_avg)
                    elif position_group == 'RB':
                        grade = self._calculate_rb_grade(row, position_avg)
                    elif position_group == 'WR_TE':
                        grade = self._calculate_wr_te_grade(row, position_avg)
                    
                    letter_grade = self._numeric_to_letter_grade(grade)
                    
                    grades_list.append({
                        'player_id': row['player_id'],
                        'player_name': row.get('player_name', row.get('player_display_name', 'Unknown')),
                        'position': row['position'],
                        'position_group': position_group,
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
            print(f"Positions found: {defensive_data['position'].value_counts().head()}")
        
        # For players without positions, try to infer from name patterns or set as generic DEF
        defensive_data['position'] = defensive_data['position'].fillna('DEF')
        
        # Map to position groups - be more flexible
        defensive_data['position_group'] = defensive_data['position'].apply(self._map_defensive_position_group)
        
        # For unmapped positions, assign based on stats
        unmapped = defensive_data['position_group'].isna()
        if unmapped.sum() > 0:
            print(f"Assigning position groups for {unmapped.sum()} unmapped players based on stats...")
            defensive_data.loc[unmapped, 'position_group'] = defensive_data.loc[unmapped].apply(
                self._infer_position_group_from_stats, axis=1
            )
        
        print(f"Position group distribution:")
        print(defensive_data['position_group'].value_counts())
        
        # Remove any still unmapped
        defensive_data = defensive_data[defensive_data['position_group'].notna()]
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
            
            print(f"{position_group} averages: Sacks={position_avg['sacks']:.2f}, Tackles={position_avg['total_tackles']:.2f}")
            
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
        print(f"Covering {grades_df['player_id'].nunique()} qualified defensive players")
        
        return grades_df
    
    def _map_defensive_position_group(self, position):
        """Map defensive positions to position groups."""
        if pd.isna(position):
            return None
        
        position = str(position).upper()
        
        if position in ['DE', 'OLB', 'EDGE']:
            return 'PASS_RUSHER'
        elif position in ['ILB', 'MLB', 'LB']:
            return 'LINEBACKER'
        elif position in ['CB', 'S', 'FS', 'SS', 'DB']:
            return 'SECONDARY'
        elif position in ['DT', 'NT']:
            return 'PASS_RUSHER'  # Interior pass rushers
        else:
            return None
    
    def _infer_position_group_from_stats(self, row):
        """Infer position group from statistical patterns."""
        # High sack/QB hit rate suggests pass rusher
        if row['sacks'] >= 0.5 or row['qb_hits'] >= 2:
            return 'PASS_RUSHER'
        
        # High interception/pass deflection rate suggests secondary
        if row['interceptions'] >= 0.2 or row['pass_deflections'] >= 1:
            return 'SECONDARY'
        
        # High tackle rate suggests linebacker
        if row['total_tackles'] >= 8:
            return 'LINEBACKER'
        
        # Default assignment based on primary stats
        if row['total_tackles'] >= 3:
            return 'LINEBACKER'
        else:
            return 'SECONDARY'
    
    def _calculate_defensive_grade(self, row, position_avg, position_group):
        """Calculate defensive player grade based on position group."""
        base_score = 50  # Start with C grade
        
        if position_group == 'PASS_RUSHER':
            # Emphasize pass rush stats with more realistic scoring
            sack_score = (row['sacks'] / max(position_avg['sacks'], 0.05)) * 15
            qb_hit_score = (row['qb_hits'] / max(position_avg['qb_hits'], 0.5)) * 10
            tfl_score = (row['tackles_for_loss'] / max(position_avg['tackles_for_loss'], 0.2)) * 8
            tackle_score = (row['total_tackles'] / max(position_avg['total_tackles'], 1)) * 7
            ff_score = row['forced_fumbles'] * 10
            
            base_score = sack_score + qb_hit_score + tfl_score + tackle_score + ff_score + 20
            
        elif position_group == 'LINEBACKER':
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
    
    # Include existing offensive grading methods from original system
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
    
    def _numeric_to_letter_grade(self, score):
        """Convert numeric score to letter grade."""
        for letter, (min_score, max_score) in self.grade_scale.items():
            if min_score <= score <= max_score:
                return letter
        return 'F'
    
    def calculate_all_grades(self, min_games: int = 3):
        """Calculate grades for both offensive and defensive players."""
        print("\n" + "="*60)
        print("CALCULATING ALL PLAYER GRADES")
        print("="*60)
        
        # Calculate offensive grades
        offensive_grades = self.calculate_offensive_grades(min_games)
        
        # Calculate defensive grades
        defensive_grades = self.calculate_defensive_grades(min_games)
        
        # Combine both datasets
        if not offensive_grades.empty and not defensive_grades.empty:
            all_grades = pd.concat([offensive_grades, defensive_grades], ignore_index=True)
        elif not offensive_grades.empty:
            all_grades = offensive_grades
        elif not defensive_grades.empty:
            all_grades = defensive_grades
        else:
            all_grades = pd.DataFrame()
        
        print(f"\nTotal grades calculated:")
        print(f"- Offensive players: {len(offensive_grades)} records")
        print(f"- Defensive players: {len(defensive_grades)} records")
        print(f"- Combined total: {len(all_grades)} records")
        
        if not all_grades.empty:
            print(f"- Unique players: {all_grades['player_id'].nunique()}")
            print(f"- Position breakdown:")
            for player_type in all_grades['player_type'].unique():
                type_data = all_grades[all_grades['player_type'] == player_type]
                print(f"  {player_type}: {type_data['position_group'].value_counts().to_dict()}")
        
        return all_grades
    
    def identify_performance_outliers(self, grades_df, std_threshold: float = 1.5):
        """Identify performance outliers for all players."""
        print("Identifying performance outliers...")
        
        outliers_list = []
        
        for player_id in grades_df['player_id'].unique():
            player_data = grades_df[grades_df['player_id'] == player_id].copy()
            
            if len(player_data) < 3:
                continue
            
            mean_grade = player_data['numeric_grade'].mean()
            std_grade = player_data['numeric_grade'].std()
            
            if std_grade == 0:
                continue
            
            for idx, row in player_data.iterrows():
                z_score = (row['numeric_grade'] - mean_grade) / std_grade
                
                is_outlier = abs(z_score) >= std_threshold
                performance_type = 'Normal'
                
                if is_outlier:
                    if z_score > 0:
                        performance_type = 'Over-Performance'
                    else:
                        performance_type = 'Under-Performance'
                
                outliers_list.append({
                    'player_id': row['player_id'],
                    'player_name': row['player_name'],
                    'position': row['position'],
                    'position_group': row['position_group'],
                    'player_type': row['player_type'],
                    'team': row['team'],
                    'season': row['season'],
                    'week': row['week'],
                    'numeric_grade': row['numeric_grade'],
                    'letter_grade': row['letter_grade'],
                    'player_avg_grade': mean_grade,
                    'z_score': z_score,
                    'performance_type': performance_type,
                    'is_outlier': is_outlier
                })
        
        outliers_df = pd.DataFrame(outliers_list)
        
        if not outliers_df.empty:
            print(f"Identified {len(outliers_df[outliers_df['is_outlier']])} outlier performances")
            print(f"Over-performances: {len(outliers_df[outliers_df['performance_type'] == 'Over-Performance'])}")
            print(f"Under-performances: {len(outliers_df[outliers_df['performance_type'] == 'Under-Performance'])}")
        
        return outliers_df
    
    def get_top_performers(self, outliers_df, player_type=None, position_group=None, 
                          metric='avg_grade', n=10):
        """Get top performing players by various metrics."""
        data = outliers_df.copy()
        
        if player_type:
            data = data[data['player_type'] == player_type]
        
        if position_group:
            data = data[data['position_group'] == position_group]
        
        # Group by player
        player_stats = data.groupby(['player_name', 'position', 'position_group', 'player_type']).agg({
            'numeric_grade': ['mean', 'std', 'count'],
            'performance_type': lambda x: (x == 'Over-Performance').sum(),
            'is_outlier': 'sum'
        }).reset_index()
        
        # Flatten column names
        player_stats.columns = ['player_name', 'position', 'position_group', 'player_type',
                               'avg_grade', 'grade_std', 'games_played', 
                               'over_performances', 'total_outliers']
        
        # Calculate consistency score
        player_stats['consistency'] = 100 - player_stats['grade_std'].fillna(0)
        
        # Sort by requested metric
        if metric == 'avg_grade':
            player_stats = player_stats.sort_values('avg_grade', ascending=False)
        elif metric == 'consistency':
            player_stats = player_stats.sort_values('consistency', ascending=False)
        elif metric == 'over_performances':
            player_stats = player_stats.sort_values('over_performances', ascending=False)
        
        return player_stats.head(n)
    
    def generate_defensive_report(self):
        """Generate a report on defensive statistics available."""
        if self.defensive_weekly.empty:
            print("No defensive data available")
            return
        
        print("\n" + "="*60)
        print("DEFENSIVE STATISTICS SUMMARY")
        print("="*60)
        
        # Season totals
        season_totals = self.defensive_weekly.groupby('player_name').agg({
            'sacks': 'sum',
            'total_tackles': 'sum',
            'tackles_for_loss': 'sum',
            'interceptions': 'sum',
            'pass_deflections': 'sum',
            'forced_fumbles': 'sum',
            'qb_hits': 'sum'
        }).round(1)
        
        print("\nTop 5 Sack Leaders:")
        top_sacks = season_totals.nlargest(5, 'sacks')[['sacks', 'qb_hits']]
        print(top_sacks)
        
        print("\nTop 5 Tackle Leaders:")
        top_tackles = season_totals.nlargest(5, 'total_tackles')[['total_tackles', 'tackles_for_loss']]
        print(top_tackles)
        
        print("\nTop 5 Interception Leaders:")
        top_ints = season_totals.nlargest(5, 'interceptions')[['interceptions', 'pass_deflections']]
        print(top_ints)
        
        return season_totals


def main():
    """Main function demonstrating the enhanced grading system."""
    
    try:
        # Initialize the enhanced grading system
        grader = EnhancedNFLPlayerGrader(years=[2023])
        
        # Generate defensive statistics report
        grader.generate_defensive_report()
        
        # Calculate all grades
        all_grades = grader.calculate_all_grades(min_games=3)
        
        if all_grades.empty:
            print("No grades calculated")
            return None, None, None
        
        # Identify outliers
        outliers_df = grader.identify_performance_outliers(all_grades, std_threshold=1.5)
        
        print("\n" + "="*60)
        print("SAMPLE REPORTS")
        print("="*60)
        
        # Top offensive players
        print("\nTop 5 Offensive Players (All Positions):")
        top_offense = grader.get_top_performers(outliers_df, player_type='OFFENSE', n=5)
        print(top_offense[['player_name', 'position_group', 'avg_grade', 'games_played']].to_string(index=False))
        
        # Top defensive players
        print("\nTop 5 Defensive Players (All Positions):")
        top_defense = grader.get_top_performers(outliers_df, player_type='DEFENSE', n=5)
        if not top_defense.empty:
            print(top_defense[['player_name', 'position_group', 'avg_grade', 'games_played']].to_string(index=False))
        else:
            print("No defensive players with sufficient data")
        
        # Position-specific breakdowns
        for pos_group in ['QB', 'RB', 'WR_TE']:
            top_pos = grader.get_top_performers(outliers_df, position_group=pos_group, n=3)
            if not top_pos.empty:
                print(f"\nTop 3 {pos_group}s:")
                print(top_pos[['player_name', 'avg_grade', 'consistency']].to_string(index=False))
        
        # Defensive position breakdowns
        for pos_group in ['PASS_RUSHER', 'LINEBACKER', 'SECONDARY']:
            top_pos = grader.get_top_performers(outliers_df, position_group=pos_group, n=3)
            if not top_pos.empty:
                print(f"\nTop 3 {pos_group}s:")
                print(top_pos[['player_name', 'avg_grade', 'consistency']].to_string(index=False))
        
        return grader, all_grades, outliers_df
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return None, None, None


if __name__ == "__main__":
    grader, all_grades, outliers = main()
    
    print("\n" + "="*60)
    print("ENHANCED SYSTEM READY")
    print("="*60)
    print("Available functions:")
    print("- grader.get_top_performers(outliers, player_type='OFFENSE')")
    print("- grader.get_top_performers(outliers, player_type='DEFENSE')")
    print("- grader.get_top_performers(outliers, position_group='PASS_RUSHER')")
    print("- grader.generate_defensive_report()")
    print("\nExample: grader.get_top_performers(outliers, position_group='QB', n=10)")