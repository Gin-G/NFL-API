#!/usr/bin/env python3
"""
Enhanced NFL Player Grading System
Updated to use nflreadpy
"""

import nflreadpy as nfl
import pandas as pd
import numpy as np
from typing import List
import warnings
warnings.filterwarnings('ignore')

class EnhancedNFLPlayerGrader:
    """Enhanced grading system including line grading."""
    
    def __init__(self, years: List[int] = [2023]):
        self.years = years
        self.weekly_data = None
        self.pbp_data = None
        self.rosters = None
        self.snap_counts = None
        
        self.oline_positions = ['C', 'G', 'LG', 'RG', 'T', 'LT', 'RT', 'OL']
        self.dline_positions = ['DE', 'DT', 'NT', 'EDGE', 'DL']
        
        self.grade_scale = {
            'A+': (95, 100), 'A': (90, 94.9), 'A-': (85, 89.9),
            'B+': (80, 84.9), 'B': (75, 79.9), 'B-': (70, 74.9),
            'C+': (65, 69.9), 'C': (55, 64.9), 'C-': (50, 54.9),
            'D+': (45, 49.9), 'D': (40, 44.9), 'D-': (35, 39.9),
            'F': (0, 34.9)
        }
        
        print(f"Initializing NFL Player Grading System for years: {years}")
        self._load_data()
    
    def _load_data(self):
        """Load all necessary data using nflreadpy."""
        try:
            print("Loading NFL data using nflreadpy...")
            
            # Convert Polars to Pandas
            self.weekly_data = nfl.load_player_stats(seasons=self.years).to_pandas()
            self.pbp_data = nfl.load_pbp(seasons=self.years).to_pandas()
            self.rosters = nfl.load_rosters_weekly(seasons=self.years).to_pandas()
            
            try:
                self.snap_counts = nfl.load_snap_counts(seasons=self.years).to_pandas()
                print(f"- Snap counts: {len(self.snap_counts)} records")
            except Exception as e:
                print(f"- Snap counts: Error ({e}), will use PBP only")
                self.snap_counts = pd.DataFrame()
            
            self._prepare_data()
            
            print(f"Data loaded successfully!")
            print(f"- Weekly data: {len(self.weekly_data)} records")
            print(f"- Play-by-play: {len(self.pbp_data)} records")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _prepare_data(self):
        """Prepare datasets."""
        print("Preparing data...")
        
        # Debug: Check what columns we actually have
        print(f"Roster columns: {list(self.rosters.columns)[:10]}")
        print(f"Weekly data columns: {list(self.weekly_data.columns)[:10]}")
        
        # Add position info to weekly data
        # Check for the actual player identifier column
        player_col = None
        if 'player_id' in self.rosters.columns:
            player_col = 'player_id'
        elif 'gsis_id' in self.rosters.columns:
            player_col = 'gsis_id'
        elif 'gsis_it_id' in self.rosters.columns:
            player_col = 'gsis_it_id'
        
        if player_col and 'position' in self.rosters.columns:
            roster_info = self.rosters.groupby(player_col).agg({
                'position': 'first',
                'team': 'first'
            }).reset_index()
            
            # Find matching column in weekly_data
            weekly_player_col = None
            if player_col in self.weekly_data.columns:
                weekly_player_col = player_col
            elif 'player_id' in self.weekly_data.columns:
                weekly_player_col = 'player_id'
            
            if weekly_player_col:
                self.weekly_data = pd.merge(
                    self.weekly_data,
                    roster_info,
                    left_on=weekly_player_col,
                    right_on=player_col,
                    how='left',
                    suffixes=('', '_roster')
                )
                print(f"Merged rosters using: {weekly_player_col} <-> {player_col}")
            else:
                print("Warning: Could not find matching player column for merge")
        else:
            print(f"Warning: Missing required columns (player_col={player_col}, position={'position' in self.rosters.columns})")
        
        # Fill missing values
        numeric_cols = self.weekly_data.select_dtypes(include=[np.number]).columns
        self.weekly_data[numeric_cols] = self.weekly_data[numeric_cols].fillna(0)
        
        # Filter to meaningful stats
        self.weekly_data = self.weekly_data[
            (self.weekly_data['attempts'] > 0) |
            (self.weekly_data['carries'] > 0) |
            (self.weekly_data['targets'] > 0)
        ]
        
        # Prepare line metrics
        self._add_line_metrics_to_pbp()
        
        print(f"Data prepared: {len(self.weekly_data)} player-game records")
    
    def _add_line_metrics_to_pbp(self):
        """Add line metrics to play-by-play data."""
        self.line_pbp = self.pbp_data[
            self.pbp_data['play_type'].isin(['pass', 'run']) &
            self.pbp_data['posteam'].notna()
        ].copy()
        
        self.line_pbp['pressure_allowed'] = (
            (self.line_pbp['sack'] == 1) | (self.line_pbp['qb_hit'] == 1)
        ).astype(int)
        
        self.line_pbp['pass_pro_success'] = np.where(
            self.line_pbp['play_type'] == 'pass',
            1 - self.line_pbp['pressure_allowed'],
            np.nan
        )
        
        self.line_pbp['run_success'] = np.where(
            (self.line_pbp['play_type'] == 'run') & (self.line_pbp['rushing_yards'] >= 4),
            1,
            np.where(self.line_pbp['play_type'] == 'run', 0, np.nan)
        )
    
    def calculate_team_oline_grades(self, min_plays: int = 40):
        """Calculate team O-Line grades."""
        print("Calculating team O-Line grades...")
        
        if self.line_pbp.empty:
            return pd.DataFrame()
        
        team_perf = self.line_pbp.groupby(['posteam', 'season', 'week']).agg({
            'pass_pro_success': 'mean',
            'pressure_allowed': 'mean',
            'sack': 'sum',
            'qb_hit': 'sum',
            'run_success': 'mean',
            'rushing_yards': 'mean',
            'play_type': 'count'
        }).reset_index()
        
        team_perf.columns = [
            'team', 'season', 'week',
            'pass_pro_rate', 'pressure_rate', 'sacks_allowed', 'qb_hits',
            'run_success_rate', 'avg_rush_yards', 'total_plays'
        ]
        
        team_perf = team_perf[team_perf['total_plays'] >= min_plays]
        
        grades = []
        for _, row in team_perf.iterrows():
            pass_grade = self._calc_pass_pro_grade(row)
            run_grade = self._calc_run_block_grade(row)
            overall = (pass_grade * 0.6) + (run_grade * 0.4)
            
            grades.append({
                'team': row['team'],
                'season': row['season'],
                'week': row['week'],
                'pass_protection_grade': pass_grade,
                'run_blocking_grade': run_grade,
                'overall_oline_grade': overall,
                'letter_grade': self._to_letter(overall)
            })
        
        result = pd.DataFrame(grades)
        print(f"Calculated {len(result)} team O-Line grades")
        return result
    
    def _calc_pass_pro_grade(self, row):
        """Calculate pass protection grade."""
        base = 60
        ppr = row.get('pass_pro_rate', 0.65)
        
        if ppr >= 0.75:
            base = 85 + min(10, (ppr - 0.75) * 40)
        elif ppr >= 0.70:
            base = 75 + (ppr - 0.70) * 20
        elif ppr >= 0.65:
            base = 65 + (ppr - 0.65) * 20
        else:
            base = 45 + (ppr * 30)
        
        return max(min(base, 95), 35)
    
    def _calc_run_block_grade(self, row):
        """Calculate run blocking grade."""
        base = 60
        rsr = row.get('run_success_rate', 0.40)
        
        if rsr >= 0.50:
            base = 80 + min(15, (rsr - 0.50) * 30)
        elif rsr >= 0.40:
            base = 65 + (rsr - 0.40) * 15
        else:
            base = 45 + (rsr * 50)
        
        return max(min(base, 90), 40)
    
    def _to_letter(self, score):
        """Convert to letter grade."""
        if pd.isna(score):
            return 'N/A'
        
        # Ensure proper ordering from highest to lowest
        if score >= 95:
            return 'A+'
        elif score >= 90:
            return 'A'
        elif score >= 85:
            return 'A-'
        elif score >= 80:
            return 'B+'
        elif score >= 75:
            return 'B'
        elif score >= 70:
            return 'B-'
        elif score >= 65:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        elif score >= 45:
            return 'D+'
        elif score >= 40:
            return 'D'
        elif score >= 35:
            return 'D-'
        else:
            return 'F'
    
    def calculate_team_dline_grades(self, min_plays: int = 40):
        """Calculate team D-Line grades."""
        print("Calculating team D-Line grades...")
        
        if self.line_pbp.empty:
            return pd.DataFrame()
        
        team_perf = self.line_pbp.groupby(['defteam', 'season', 'week']).agg({
            'pressure_allowed': 'mean',
            'sack': 'sum',
            'qb_hit': 'sum',
            'run_success': lambda x: 1 - x.mean(),  # Inverse for defense
            'rushing_yards': 'mean',
            'play_type': 'count'
        }).reset_index()
        
        team_perf.columns = [
            'team', 'season', 'week',
            'pressure_rate', 'sacks', 'qb_hits', 'run_stuff_rate',
            'avg_rush_yards_allowed', 'total_plays'
        ]
        
        team_perf = team_perf[team_perf['total_plays'] >= min_plays]
        
        grades = []
        for _, row in team_perf.iterrows():
            pass_grade = self._calc_pass_rush_grade(row)
            run_grade = self._calc_run_def_grade(row)
            overall = (pass_grade * 0.6) + (run_grade * 0.4)
            
            grades.append({
                'team': row['team'],
                'season': row['season'],
                'week': row['week'],
                'pass_rush_grade': pass_grade,
                'run_defense_grade': run_grade,
                'overall_dline_grade': overall,
                'letter_grade': self._to_letter(overall)
            })
        
        result = pd.DataFrame(grades)
        print(f"Calculated {len(result)} team D-Line grades")
        return result
    
    def _calc_pass_rush_grade(self, row):
        """Calculate pass rush grade."""
        base = 60
        pr = row.get('pressure_rate', 0.25)
        
        if pr >= 0.35:
            base = 85 + min(10, (pr - 0.35) * 25)
        elif pr >= 0.25:
            base = 70 + (pr - 0.25) * 15
        else:
            base = 50 + (pr * 80)
        
        sacks = row.get('sacks', 1)
        if sacks >= 3:
            base += 10
        elif sacks >= 1:
            base += sacks * 5
        
        return max(min(base, 95), 35)
    
    def _calc_run_def_grade(self, row):
        """Calculate run defense grade."""
        base = 60
        stuff = row.get('run_stuff_rate', 0.20)
        
        if stuff >= 0.30:
            base = 80 + min(15, (stuff - 0.30) * 30)
        elif stuff >= 0.20:
            base = 65 + (stuff - 0.20) * 15
        else:
            base = 50 + (stuff * 75)
        
        return max(min(base, 90), 40)
    
    def calculate_qb_grades(self, min_games: int = 3):
        """Calculate QB grades."""
        print("Calculating QB grades...")
        
        qb_data = self.weekly_data[
            (self.weekly_data['position'] == 'QB') &
            (self.weekly_data['attempts'] >= 10)
        ].copy()
        
        if qb_data.empty:
            return pd.DataFrame()
        
        # Debug: check available columns
        print(f"Available QB stat columns: {[c for c in qb_data.columns if 'pass' in c.lower() or 'int' in c.lower()]}")
        
        grades = []
        for _, row in qb_data.iterrows():
            grade = self._calc_qb_grade(row)
            
            grades.append({
                'player_id': row.get('player_id'),
                'player_name': row.get('player_name', row.get('player_display_name')),
                'position': 'QB',
                'team': row.get('recent_team', row.get('team')),
                'season': row['season'],
                'week': row['week'],
                'numeric_grade': grade,
                'letter_grade': self._to_letter(grade),
                'attempts': row.get('attempts', 0),
                'completions': row.get('completions', 0),
                'passing_yards': row.get('passing_yards', 0),
                'passing_tds': row.get('passing_tds', 0),
                'interceptions': row.get('passing_interceptions', 0)
            })
        
        result = pd.DataFrame(grades)
        
        if not result.empty:
            game_counts = result.groupby('player_id').size()
            qualified = game_counts[game_counts >= min_games].index
            result = result[result['player_id'].isin(qualified)]
        
        print(f"Calculated QB grades for {len(result)} records")
        return result
    
    def _calc_qb_grade(self, row):
        """Calculate QB grade."""
        base = 50
        yards = min(row.get('passing_yards', 0) / 20, 15)
        
        comp_pct = row.get('completions', 0) / max(row.get('attempts', 1), 1)
        comp = max((comp_pct - 0.5) * 30, 0)
        
        tds = min(row.get('passing_tds', 0) * 5, 15)
        # Use correct column name: passing_interceptions
        ints_val = row.get('passing_interceptions', 0)
        ints = min(ints_val * -3, 0)
        
        total = base + yards + comp + tds + ints
        return max(min(total, 100), 0)
    
    def calculate_rb_grades(self, min_games: int = 3):
        """Calculate RB grades."""
        print("Calculating RB grades...")
        
        rb_data = self.weekly_data[
            (self.weekly_data['position'] == 'RB') &
            (self.weekly_data['carries'] >= 5)
        ].copy()
        
        if rb_data.empty:
            return pd.DataFrame()
        
        grades = []
        for _, row in rb_data.iterrows():
            grade = self._calc_rb_grade(row)
            
            grades.append({
                'player_id': row.get('player_id'),
                'player_name': row.get('player_name', row.get('player_display_name')),
                'position': 'RB',
                'team': row.get('recent_team', row.get('team')),
                'season': row['season'],
                'week': row['week'],
                'numeric_grade': grade,
                'letter_grade': self._to_letter(grade),
                'carries': row['carries'],
                'rushing_yards': row['rushing_yards'],
                'rushing_tds': row['rushing_tds'],
                'receptions': row['receptions'],
                'receiving_yards': row['receiving_yards']
            })
        
        result = pd.DataFrame(grades)
        
        if not result.empty:
            game_counts = result.groupby('player_id').size()
            qualified = game_counts[game_counts >= min_games].index
            result = result[result['player_id'].isin(qualified)]
        
        print(f"Calculated RB grades for {len(result)} records")
        return result
    
    def _calc_rb_grade(self, row):
        """Calculate RB grade."""
        base = 50
        rush_yards = min(row['rushing_yards'] / 8, 20)
        
        if row['carries'] > 0:
            ypc = row['rushing_yards'] / row['carries']
            ypc_score = max(0, (ypc - 3.5) * 5)
        else:
            ypc_score = 0
        
        tds = min((row['rushing_tds'] + row.get('receiving_tds', 0)) * 7, 15)
        rec = min(row.get('receiving_yards', 0) / 15 + row.get('receptions', 0), 10)
        
        total = base + rush_yards + ypc_score + tds + rec
        return max(min(total, 95), 25)
    
    def calculate_wr_te_grades(self, min_games: int = 3):
        """Calculate WR/TE grades."""
        print("Calculating WR/TE grades...")
        
        wr_te_data = self.weekly_data[
            (self.weekly_data['position'].isin(['WR', 'TE'])) &
            (self.weekly_data['targets'] >= 3)
        ].copy()
        
        if wr_te_data.empty:
            return pd.DataFrame()
        
        grades = []
        for _, row in wr_te_data.iterrows():
            grade = self._calc_wr_te_grade(row)
            
            grades.append({
                'player_id': row.get('player_id'),
                'player_name': row.get('player_name', row.get('player_display_name')),
                'position': row['position'],
                'team': row.get('recent_team', row.get('team')),
                'season': row['season'],
                'week': row['week'],
                'numeric_grade': grade,
                'letter_grade': self._to_letter(grade),
                'targets': row['targets'],
                'receptions': row['receptions'],
                'receiving_yards': row['receiving_yards'],
                'receiving_tds': row['receiving_tds']
            })
        
        result = pd.DataFrame(grades)
        
        if not result.empty:
            game_counts = result.groupby('player_id').size()
            qualified = game_counts[game_counts >= min_games].index
            result = result[result['player_id'].isin(qualified)]
        
        print(f"Calculated WR/TE grades for {len(result)} records")
        return result
    
    def _calc_wr_te_grade(self, row):
        """Calculate WR/TE grade."""
        base = 50
        yards = min(row['receiving_yards'] / 6, 25)
        recs = min(row['receptions'] * 2.5, 15)
        tds = min(row['receiving_tds'] * 8, 15)
        
        if row['targets'] > 0:
            catch_rate = row['receptions'] / row['targets']
            catch = max(0, (catch_rate - 0.6) * 12.5)
        else:
            catch = 0
        
        total = base + yards + recs + tds + catch
        return max(min(total, 95), 25)
    
    def calculate_defensive_grades(self, min_games: int = 3):
        """Calculate defensive player grades from PBP."""
        print("Calculating defensive grades...")
        
        def_stats = self._extract_defensive_stats()
        
        if def_stats.empty:
            return pd.DataFrame()
        
        grades = []
        for _, row in def_stats.iterrows():
            grade = self._calc_def_grade(row)
            
            grades.append({
                'player_id': row.get('player_id'),
                'player_name': row['player_name'],
                'position': 'DEF',
                'team': row.get('team', 'UNK'),
                'season': row['season'],
                'week': row['week'],
                'numeric_grade': grade,
                'letter_grade': self._to_letter(grade),
                'sacks': row.get('sacks', 0),
                'tackles': row.get('tackles', 0),
                'interceptions': row.get('ints', 0),
                'pass_deflections': row.get('pds', 0)
            })
        
        result = pd.DataFrame(grades)
        
        if not result.empty:
            game_counts = result.groupby('player_id').size()
            qualified = game_counts[game_counts >= min_games].index
            result = result[result['player_id'].isin(qualified)]
        
        print(f"Calculated defensive grades for {len(result)} records")
        return result
    
    def _extract_defensive_stats(self):
        """Extract defensive stats from PBP."""
        if self.pbp_data.empty:
            return pd.DataFrame()
        
        stats = {}
        
        # Sacks
        sacks = self.pbp_data[self.pbp_data['sack'] == 1]
        for _, play in sacks.iterrows():
            if pd.notna(play.get('sack_player_name')):
                key = (play.get('sack_player_id'), play['sack_player_name'], 
                      play['season'], play['week'], play.get('defteam', 'UNK'))
                if key not in stats:
                    stats[key] = {'sacks': 0, 'tackles': 0, 'ints': 0, 'pds': 0, 'ff': 0}
                stats[key]['sacks'] += 1.0
        
        # Interceptions
        ints = self.pbp_data[self.pbp_data['interception'] == 1]
        for _, play in ints.iterrows():
            if pd.notna(play.get('interception_player_name')):
                key = (play.get('interception_player_id'), play['interception_player_name'],
                      play['season'], play['week'], play.get('defteam', 'UNK'))
                if key not in stats:
                    stats[key] = {'sacks': 0, 'tackles': 0, 'ints': 0, 'pds': 0, 'ff': 0}
                stats[key]['ints'] += 1
        
        # Convert to DataFrame
        records = []
        for (pid, pname, season, week, team), st in stats.items():
            records.append({
                'player_id': pid,
                'player_name': pname,
                'season': season,
                'week': week,
                'team': team,
                **st
            })
        
        return pd.DataFrame(records)
    
    def _calc_def_grade(self, row):
        """Calculate defensive grade."""
        base = 55
        
        sacks = row.get('sacks', 0) * 10
        ints = row.get('ints', 0) * 12
        tackles = min(row.get('tackles', 0) * 2, 15)
        pds = row.get('pds', 0) * 4
        ff = row.get('ff', 0) * 8
        
        total = base + sacks + ints + tackles + pds + ff
        return max(min(total, 95), 30)
    
    def calculate_all_grades(self, min_games: int = 3):
        """Calculate all grades."""
        print(f"\n{'='*60}")
        print("CALCULATING ALL PLAYER GRADES")
        print(f"{'='*60}")
        
        team_oline_grades = self.calculate_team_oline_grades()
        team_dline_grades = self.calculate_team_dline_grades()
        qb_grades = self.calculate_qb_grades(min_games)
        rb_grades = self.calculate_rb_grades(min_games)
        wr_te_grades = self.calculate_wr_te_grades(min_games)
        defensive_grades = self.calculate_defensive_grades(min_games)
        
        return {
            'team_oline_grades': team_oline_grades,
            'team_dline_grades': team_dline_grades,
            'qb_grades': qb_grades,
            'rb_grades': rb_grades,
            'wr_te_grades': wr_te_grades,
            'defensive_grades': defensive_grades
        }


def main():
    """Main function."""
    try:
        print("Initializing NFL Player Grading System...")
        print("="*60)
        
        grader = EnhancedNFLPlayerGrader(years=[2025])
        
        print("\nCalculating grades...")
        all_grades = grader.calculate_all_grades(min_games=3)
        
        print(f"\n{'='*60}")
        print("GRADING RESULTS")
        print(f"{'='*60}")
        
        for grade_type, df in all_grades.items():
            print(f"{grade_type}: {len(df)} records")
        
        # Show sample results for each category
        if not all_grades['team_oline_grades'].empty:
            print(f"\n{'='*60}")
            print("TOP 10 OFFENSIVE LINES")
            print(f"{'='*60}")
            top_oline = all_grades['team_oline_grades'].groupby('team')['overall_oline_grade'].mean().nlargest(10)
            for team, grade in top_oline.items():
                print(f"{team:3} | {grade:5.1f} | {grader._to_letter(grade)}")
        
        if not all_grades['team_dline_grades'].empty:
            print(f"\n{'='*60}")
            print("TOP 10 DEFENSIVE LINES")
            print(f"{'='*60}")
            top_dline = all_grades['team_dline_grades'].groupby('team')['overall_dline_grade'].mean().nlargest(10)
            for team, grade in top_dline.items():
                print(f"{team:3} | {grade:5.1f} | {grader._to_letter(grade)}")
        
        if not all_grades['qb_grades'].empty:
            print(f"\n{'='*60}")
            print("TOP 10 QUARTERBACKS")
            print(f"{'='*60}")
            top_qbs = all_grades['qb_grades'].groupby('player_name')['numeric_grade'].mean().nlargest(10)
            for player, grade in top_qbs.items():
                print(f"{player[:30]:30} | {grade:5.1f} | {grader._to_letter(grade)}")
        
        if not all_grades['rb_grades'].empty:
            print(f"\n{'='*60}")
            print("TOP 10 RUNNING BACKS")
            print(f"{'='*60}")
            top_rbs = all_grades['rb_grades'].groupby('player_name')['numeric_grade'].mean().nlargest(10)
            for player, grade in top_rbs.items():
                print(f"{player[:30]:30} | {grade:5.1f} | {grader._to_letter(grade)}")
        
        if not all_grades['wr_te_grades'].empty:
            print(f"\n{'='*60}")
            print("TOP 10 RECEIVERS")
            print(f"{'='*60}")
            top_wrs = all_grades['wr_te_grades'].groupby('player_name')['numeric_grade'].mean().nlargest(10)
            for player, grade in top_wrs.items():
                print(f"{player[:30]:30} | {grade:5.1f} | {grader._to_letter(grade)}")
        
        if not all_grades['defensive_grades'].empty:
            print(f"\n{'='*60}")
            print("TOP 10 DEFENSIVE PLAYERS")
            print(f"{'='*60}")
            top_def = all_grades['defensive_grades'].groupby('player_name')['numeric_grade'].mean().nlargest(10)
            for player, grade in top_def.items():
                print(f"{player[:30]:30} | {grade:5.1f} | {grader._to_letter(grade)}")
        
        print(f"\n{'='*60}")
        print("GRADING SYSTEM READY")
        print(f"{'='*60}")
        
        return grader, all_grades
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    grader, all_grades = main()