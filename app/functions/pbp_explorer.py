#!/usr/bin/env python3
"""
NFL Play-by-Play Defensive Data Explorer

Explore what defensive statistics are available in nfl_data_py play-by-play data
and create aggregated defensive player statistics for grading.
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np

def explore_pbp_defensive_data(year=2023):
    """
    Load play-by-play data and explore defensive columns available.
    
    Args:
        year: Season to analyze
    """
    print(f"Loading play-by-play data for {year}...")
    pbp_data = nfl.import_pbp_data([year])
    
    print(f"Play-by-play data shape: {pbp_data.shape}")
    print(f"Total columns: {len(pbp_data.columns)}")
    
    # Look for defensive-related columns
    defensive_keywords = [
        'tackle', 'sack', 'interception', 'fumble', 'defense', 'def_', 
        'forced', 'recovered', 'assist', 'solo', 'hit', 'qb_hit',
        'pass_defense', 'deflection', 'safety'
    ]
    
    print("\n" + "="*60)
    print("DEFENSIVE-RELATED COLUMNS")
    print("="*60)
    
    defensive_cols = []
    for col in pbp_data.columns:
        if any(keyword in col.lower() for keyword in defensive_keywords):
            defensive_cols.append(col)
    
    for col in sorted(defensive_cols):
        non_null_count = pbp_data[col].notna().sum()
        print(f"{col:30} - Non-null values: {non_null_count:,}")
    
    # Show some example data
    print("\n" + "="*60)
    print("SAMPLE DEFENSIVE PLAYS")
    print("="*60)
    
    # Show sacks
    sack_plays = pbp_data[pbp_data['sack'] == 1].head(3)
    if not sack_plays.empty:
        print("\nSACK PLAYS:")
        for idx, play in sack_plays.iterrows():
            print(f"Week {play['week']}: {play['desc']}")
            for col in defensive_cols:
                if pd.notna(play[col]) and play[col] != 0:
                    print(f"  {col}: {play[col]}")
    
    # Show interceptions
    int_plays = pbp_data[pbp_data['interception'] == 1].head(3)
    if not int_plays.empty:
        print("\nINTERCEPTION PLAYS:")
        for idx, play in int_plays.iterrows():
            print(f"Week {play['week']}: {play['desc']}")
            for col in defensive_cols:
                if pd.notna(play[col]) and play[col] != 0:
                    print(f"  {col}: {play[col]}")
    
    return pbp_data, defensive_cols

def create_defensive_player_stats(pbp_data, defensive_cols):
    """
    Create aggregated defensive player statistics from play-by-play data.
    
    Args:
        pbp_data: Play-by-play DataFrame
        defensive_cols: List of defensive column names
        
    Returns:
        DataFrame with defensive player statistics
    """
    print("\n" + "="*60)
    print("CREATING DEFENSIVE PLAYER STATS")
    print("="*60)
    
    defensive_stats = []
    
    # Check for player-specific defensive columns
    player_defensive_cols = [col for col in defensive_cols if 'player' in col.lower()]
    print(f"Player-specific defensive columns: {len(player_defensive_cols)}")
    for col in player_defensive_cols[:10]:  # Show first 10
        print(f"  {col}")
    
    # Example: Aggregate sack statistics
    if 'sack_player_name' in pbp_data.columns:
        print("\nAggregating sack statistics...")
        sack_stats = pbp_data[pbp_data['sack'] == 1].groupby(['sack_player_name', 'sack_player_id']).agg({
            'sack': 'sum',
            'week': 'count'
        }).reset_index()
        sack_stats.columns = ['player_name', 'player_id', 'sacks', 'games_with_sacks']
        print(f"Found sack data for {len(sack_stats)} players")
        print("Top 5 sack leaders:")
        print(sack_stats.nlargest(5, 'sacks')[['player_name', 'sacks']])
    
    # Example: Half-sack players (if available)
    half_sack_cols = [col for col in pbp_data.columns if 'half_sack' in col.lower()]
    if half_sack_cols:
        print(f"\nHalf-sack columns available: {half_sack_cols}")
    
    # Example: Tackle statistics (if available)
    tackle_cols = [col for col in pbp_data.columns if 'tackle' in col.lower() and 'player' in col.lower()]
    if tackle_cols:
        print(f"\nTackle columns available: {tackle_cols}")
    
    # Example: Forced fumble statistics
    fumble_cols = [col for col in pbp_data.columns if 'fumble' in col.lower() and 'player' in col.lower()]
    if fumble_cols:
        print(f"\nFumble columns available: {fumble_cols}")
    
    # Example: Interception statistics
    if 'interception_player_name' in pbp_data.columns:
        print("\nAggregating interception statistics...")
        int_stats = pbp_data[pbp_data['interception'] == 1].groupby(['interception_player_name', 'interception_player_id']).agg({
            'interception': 'sum',
            'week': 'count'
        }).reset_index()
        int_stats.columns = ['player_name', 'player_id', 'interceptions', 'games_with_ints']
        print(f"Found interception data for {len(int_stats)} players")
        print("Top 5 interception leaders:")
        print(int_stats.nlargest(5, 'interceptions')[['player_name', 'interceptions']])
    
    return defensive_stats

def main():
    """Main function to explore defensive data."""
    
    # Explore what's available
    pbp_data, defensive_cols = explore_pbp_defensive_data(2023)
    
    # Create defensive stats
    defensive_stats = create_defensive_player_stats(pbp_data, defensive_cols)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total defensive-related columns found: {len(defensive_cols)}")
    print("This data can be used to create defensive player grades!")
    
    print("\nNext steps:")
    print("1. Aggregate defensive stats by player and week")
    print("2. Create position-specific defensive grading scales")
    print("3. Integrate with the main grading system")
    
    return pbp_data, defensive_cols

if __name__ == "__main__":
    pbp_data, defensive_cols = main()