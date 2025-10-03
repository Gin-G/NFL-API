#!/usr/bin/env python3
"""
Fixed NFL Roster-Aware Coaching Analytics System
Updated to use nflreadpy instead of nfl_data_py
"""

import pandas as pd
import numpy as np
import nflreadpy as nfl
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
        """Load all necessary data using nflreadpy"""
        print("Loading NFL data with nflreadpy...")
        
        # Load play-by-play data using new API
        print("- Loading play-by-play data...")
        pbp_list = []
        for year in self.years:
            try:
                # load_pbp() replaces import_pbp_data()
                # nflreadpy returns Polars DataFrames, convert to Pandas
                year_pbp = nfl.load_pbp(seasons=[year]).to_pandas()
                pbp_list.append(year_pbp)
                print(f"  - {year}: {len(year_pbp)} plays loaded")
            except Exception as e:
                print(f"  - Error loading {year}: {e}")
        
        if pbp_list:
            self.pbp_data = pd.concat(pbp_list, ignore_index=True)
        
        # Load schedule data using new API
        print("- Loading schedule data...")
        schedule_list = []
        for year in self.years:
            try:
                # load_schedules() replaces import_schedules()
                # Convert Polars to Pandas
                year_schedule = nfl.load_schedules(seasons=[year]).to_pandas()
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
        
        # Load weekly data for offensive players using new API
        # load_player_stats() replaces import_weekly_data()
        # Convert Polars to Pandas
        weekly_data = nfl.load_player_stats(seasons=self.years).to_pandas()
        
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
        
        # Convert to grades
        weekly_grades = []
        for (player_id, player_name, season, week, team), stats in defensive_stats.items():
            base_score = 55
            
            sack_score = stats['sacks'] * 10
            int_score = stats['ints'] * 12
            tackle_score = min(stats['tackles'] * 2, 15)
            pd_score = stats['pds'] * 4
            ff_score = stats['ff'] * 8
            
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
    
    def analyze_roster_quality(self, team, season, key_contributors_only=True):
        """Analyze roster quality focusing on key contributors"""
        if self.player_grades is None or self.player_grades.empty:
            print(f"No player grades available for roster analysis")
            return None
        
        print(f"Analyzing roster quality for {team} ({season})...")
        
        team_players = self.player_grades[
            (self.player_grades['team'] == team) & 
            (self.player_grades['season'] == season)
        ]
        
        if team_players.empty:
            print(f"No player data found for {team} in {season}")
            return None
        
        player_stats = team_players.groupby(['player_id', 'player_name', 'position_group']).agg({
            'numeric_grade': ['mean', 'count'],
            'week': 'count'
        }).reset_index()
        
        player_stats.columns = ['player_id', 'player_name', 'position_group', 'avg_grade', 'grade_count', 'games_played']
        
        if key_contributors_only:
            key_players = self._identify_key_contributors(player_stats)
            player_avgs = key_players
            analysis_type = "Key Contributors"
        else:
            player_avgs = player_stats
            analysis_type = "All Players"
        
        analysis = {}
        analysis['analysis_type'] = analysis_type
        
        overall_avg = player_avgs['avg_grade'].mean()
        analysis['overall_avg_grade'] = overall_avg
        analysis['total_players'] = len(player_avgs)
        
        analysis['elite_players'] = len(player_avgs[player_avgs['avg_grade'] >= 78])
        analysis['good_players'] = len(player_avgs[
            (player_avgs['avg_grade'] >= 70) & (player_avgs['avg_grade'] < 78)
        ])
        analysis['average_players'] = len(player_avgs[
            (player_avgs['avg_grade'] >= 62) & (player_avgs['avg_grade'] < 70)
        ])
        analysis['below_avg_players'] = len(player_avgs[player_avgs['avg_grade'] < 62])
        
        for pos_group in ['QB', 'RB', 'WR_TE', 'DEFENSE']:
            pos_players = player_avgs[player_avgs['position_group'] == pos_group]
            if not pos_players.empty:
                analysis[f'{pos_group.lower()}_avg_grade'] = pos_players['avg_grade'].mean()
                analysis[f'{pos_group.lower()}_count'] = len(pos_players)
                analysis[f'{pos_group.lower()}_best_grade'] = pos_players['avg_grade'].max()
            else:
                analysis[f'{pos_group.lower()}_avg_grade'] = None
                analysis[f'{pos_group.lower()}_count'] = 0
                analysis[f'{pos_group.lower()}_best_grade'] = None
        
        if overall_avg >= 72:
            analysis['roster_tier'] = 'Elite'
        elif overall_avg >= 68:
            analysis['roster_tier'] = 'Good' 
        elif overall_avg >= 64:
            analysis['roster_tier'] = 'Average'
        elif overall_avg >= 60:
            analysis['roster_tier'] = 'Below Average'
        else:
            analysis['roster_tier'] = 'Poor'
        
        analysis['roster_depth'] = player_avgs['avg_grade'].std()
        analysis['top_players'] = player_avgs.nlargest(5, 'avg_grade')[['player_name', 'position_group', 'avg_grade']].to_dict('records')
        
        print(f"Roster analysis complete ({analysis_type}):")
        print(f"  Overall grade: {overall_avg:.1f} ({analysis['roster_tier']})")
        print(f"  Elite players (78+): {analysis['elite_players']}")
        print(f"  Good players (70-77): {analysis['good_players']}")
        
        return analysis
    
    def _identify_key_contributors(self, player_stats):
        """Identify key contributors based on games played"""
        key_contributors = []
        
        for pos_group in player_stats['position_group'].unique():
            pos_players = player_stats[player_stats['position_group'] == pos_group].copy()
            
            if pos_group == 'QB':
                top_qbs = pos_players.nlargest(2, 'games_played')
                key_contributors.extend(top_qbs.to_dict('records'))
                
            elif pos_group == 'RB':
                top_rbs = pos_players.nlargest(3, 'games_played')
                top_rbs = top_rbs[top_rbs['games_played'] >= 4]
                key_contributors.extend(top_rbs.to_dict('records'))
                
            elif pos_group == 'WR_TE':
                top_receivers = pos_players.nlargest(6, 'games_played')
                top_receivers = top_receivers[top_receivers['games_played'] >= 6]
                key_contributors.extend(top_receivers.to_dict('records'))
                
            elif pos_group == 'DEFENSE':
                top_defense = pos_players.nlargest(15, 'games_played')
                top_defense = top_defense[top_defense['games_played'] >= 8]
                key_contributors.extend(top_defense.to_dict('records'))
        
        key_df = pd.DataFrame(key_contributors)
        
        if not key_df.empty:
            key_df = key_df[key_df['games_played'] >= 3]
        
        return key_df
    
    def get_available_coaches(self, season=None):
        """Get list of available coaches"""
        coaches = set()
        for (coach, seas), data in self.coaching_data.items():
            if season is None or seas == season:
                coaches.add(coach)
        return sorted(list(coaches))
    
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


def main():
    """Main demonstration"""
    analytics = RosterAwareCoachingAnalytics(years=[2023])
    analytics.load_data()
    analytics.extract_coaching_info()
    analytics.calculate_player_grades()
    
    coaches = analytics.get_available_coaches(season=2023)
    print(f"\nAvailable coaches: {len(coaches)}")
    
    return analytics


if __name__ == "__main__":
    main()