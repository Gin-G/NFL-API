#!/usr/bin/env python3
"""
Player play-by-play analytics — batch computation per season.

Each function runs a small number of GROUP BY queries covering ALL players
in a season (not one query per player). This keeps the compute job fast:
~5 SQL queries per season instead of thousands.

Position groups and what's computed:
  QB   — dropbacks, EPA/db, comp%, air yards, deep/screen rate, play-action,
          sack rate, 3rd-down conv, red zone EPA, pass direction splits
  RB   — carries, EPA/carry, success rate, run direction, receiving splits
  WR   — targets, EPA/target, air yards, YAC, deep rate, direction splits
  TE   — same as WR
  DEF  — sacks, QB hits, tackles, INTs, PBDs, forced fumbles, TFL
         (coverage assignments not in standard PBP — counting stats only)
"""

import logging
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _f(v):
    try:
        return round(float(v), 4) if v is not None else None
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


# ── QB batch metrics ──────────────────────────────────────────────────────────

def compute_qb_metrics(db: Session, season: int) -> dict:
    """Return {player_id: metrics} for all QBs with ≥30 dropbacks in a season."""
    sql = text("""
        SELECT
            passer_player_id                                     AS player_id,
            MIN(passer_player_name)                              AS player_name,
            posteam                                              AS team,
            COUNT(*)                                             AS dropbacks,
            SUM(CASE WHEN complete_pass = 1 THEN 1 ELSE 0 END)  AS completions,
            AVG(CASE WHEN complete_pass = 1 THEN 1.0 ELSE 0.0 END) AS comp_pct,
            AVG(epa)                                             AS epa_per_dropback,
            SUM(yards_gained)                                    AS pass_yards,
            SUM(CASE WHEN touchdown   = 1 THEN 1 ELSE 0 END)    AS pass_tds,
            SUM(CASE WHEN interception= 1 THEN 1 ELSE 0 END)    AS interceptions,
            SUM(CASE WHEN sack        = 1 THEN 1 ELSE 0 END)    AS sacks_taken,
            AVG(CASE WHEN sack        = 1 THEN 1.0 ELSE 0.0 END) AS sack_rate,
            AVG(CASE WHEN qb_scramble = 1 THEN 1.0 ELSE 0.0 END) AS scramble_rate,
            AVG(air_yards)
                FILTER (WHERE air_yards IS NOT NULL)             AS avg_air_yards,
            AVG(CASE WHEN air_yards > 15 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE air_yards IS NOT NULL)             AS deep_rate,
            AVG(CASE WHEN air_yards <= 0  THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE air_yards IS NOT NULL)             AS screen_rate,
            AVG(yards_after_catch)
                FILTER (WHERE yards_after_catch IS NOT NULL AND complete_pass = 1) AS avg_yac,
            AVG(CASE WHEN third_down_converted = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE third_down_converted=1 OR third_down_failed=1) AS third_conv_rate,
            AVG(epa) FILTER (WHERE yardline_100 <= 20)           AS rz_epa,
            COUNT(*) FILTER (WHERE yardline_100 <= 20)           AS rz_dropbacks,
            COUNT(*) FILTER (WHERE pass_location = 'left')       AS dir_left,
            COUNT(*) FILTER (WHERE pass_location = 'middle')     AS dir_middle,
            COUNT(*) FILTER (WHERE pass_location = 'right')      AS dir_right,
            AVG(epa) FILTER (WHERE pass_location = 'left')       AS epa_left,
            AVG(epa) FILTER (WHERE pass_location = 'middle')     AS epa_middle,
            AVG(epa) FILTER (WHERE pass_location = 'right')      AS epa_right
        FROM play_by_play
        WHERE season = :season
          AND passer_player_id IS NOT NULL
          AND play_type IN ('pass', 'qb_scramble')
        GROUP BY passer_player_id, posteam
        HAVING COUNT(*) >= 30
    """)
    try:
        rows = db.execute(sql, {"season": season}).mappings().all()
    except Exception:
        db.rollback()
        logger.warning("QB metrics query failed for season %d", season)
        return {}

    results = {}
    for r in rows:
        pid = r["player_id"]
        if not pid:
            continue
        db_cnt = _i(r["dir_left"]) + _i(r["dir_middle"]) + _i(r["dir_right"])
        results[pid] = {
            "player_name": r["player_name"],
            "team": r["team"],
            "dropbacks": _i(r["dropbacks"]),
            "completions": _i(r["completions"]),
            "comp_pct": _f(r["comp_pct"]),
            "epa_per_dropback": _f(r["epa_per_dropback"]),
            "pass_yards": _i(r["pass_yards"]),
            "pass_tds": _i(r["pass_tds"]),
            "interceptions": _i(r["interceptions"]),
            "sacks_taken": _i(r["sacks_taken"]),
            "sack_rate": _f(r["sack_rate"]),
            "scramble_rate": _f(r["scramble_rate"]),
            "avg_air_yards": _f(r["avg_air_yards"]),
            "deep_rate": _f(r["deep_rate"]),
            "screen_rate": _f(r["screen_rate"]),
            "avg_yac": _f(r["avg_yac"]),
            "third_down_conv_rate": _f(r["third_conv_rate"]),
            "rz_epa": _f(r["rz_epa"]),
            "rz_dropbacks": _i(r["rz_dropbacks"]),
            "pass_direction": {
                "left":   {"attempts": _i(r["dir_left"]),   "epa": _f(r["epa_left"]),   "pct": round(_i(r["dir_left"])   / db_cnt, 3) if db_cnt else None},
                "middle": {"attempts": _i(r["dir_middle"]), "epa": _f(r["epa_middle"]), "pct": round(_i(r["dir_middle"]) / db_cnt, 3) if db_cnt else None},
                "right":  {"attempts": _i(r["dir_right"]),  "epa": _f(r["epa_right"]),  "pct": round(_i(r["dir_right"])  / db_cnt, 3) if db_cnt else None},
            },
        }

    # Optional: play-action rate (column may not exist in older seasons)
    try:
        pa_sql = text("""
            SELECT passer_player_id AS player_id, posteam AS team,
                AVG(CASE WHEN play_action = 1 THEN 1.0 ELSE 0.0 END) AS play_action_rate
            FROM play_by_play
            WHERE season = :season AND passer_player_id IS NOT NULL
              AND play_type IN ('pass', 'qb_scramble')
            GROUP BY passer_player_id, posteam
        """)
        pa_rows = db.execute(pa_sql, {"season": season}).mappings().all()
        for r in pa_rows:
            if r["player_id"] in results:
                results[r["player_id"]]["play_action_rate"] = _f(r["play_action_rate"])
    except Exception:
        db.rollback()

    return results


# ── Rusher batch metrics ──────────────────────────────────────────────────────

def compute_rush_metrics(db: Session, season: int) -> dict:
    """Return {player_id: metrics} for all rushers with ≥10 carries in a season."""
    sql = text("""
        SELECT
            rusher_player_id                                       AS player_id,
            MIN(rusher_player_name)                                AS player_name,
            posteam                                                AS team,
            COUNT(*)                                               AS carries,
            SUM(yards_gained)                                      AS rush_yards,
            AVG(yards_gained)                                      AS yards_per_carry,
            AVG(epa)                                               AS epa_per_carry,
            AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END)      AS success_rate,
            SUM(CASE WHEN touchdown = 1 THEN 1 ELSE 0 END)        AS rush_tds,
            COUNT(*) FILTER (WHERE yardline_100 <= 10)            AS rz_carries,
            AVG(epa) FILTER (WHERE yardline_100 <= 10)            AS rz_epa,
            COUNT(*) FILTER (WHERE run_location = 'middle')       AS runs_middle,
            COUNT(*) FILTER (WHERE run_location = 'left')         AS runs_left,
            COUNT(*) FILTER (WHERE run_location = 'right')        AS runs_right,
            AVG(epa) FILTER (WHERE run_location = 'middle')       AS epa_middle,
            AVG(epa) FILTER (WHERE run_location = 'left')         AS epa_left,
            AVG(epa) FILTER (WHERE run_location = 'right')        AS epa_right,
            COUNT(*) FILTER (WHERE run_gap IN ('guard','tackle') OR run_location = 'middle') AS inside_runs,
            COUNT(*) FILTER (WHERE run_gap = 'end' OR run_location IN ('left','right')) AS outside_runs,
            AVG(CASE WHEN yards_gained >= 10 THEN 1.0 ELSE 0.0 END) AS explosive_rate,
            AVG(CASE WHEN yards_gained < 0  THEN 1.0 ELSE 0.0 END)  AS negative_rate
        FROM play_by_play
        WHERE season = :season
          AND rusher_player_id IS NOT NULL
          AND play_type = 'run'
        GROUP BY rusher_player_id, posteam
        HAVING COUNT(*) >= 10
    """)
    try:
        rows = db.execute(sql, {"season": season}).mappings().all()
    except Exception:
        db.rollback()
        logger.warning("Rush metrics query failed for season %d", season)
        return {}

    results = {}
    for r in rows:
        pid = r["player_id"]
        if not pid:
            continue
        total_dir = _i(r["runs_left"]) + _i(r["runs_middle"]) + _i(r["runs_right"])
        total_gap = _i(r["inside_runs"]) + _i(r["outside_runs"])
        results[pid] = {
            "player_name": r["player_name"],
            "team": r["team"],
            "carries": _i(r["carries"]),
            "rush_yards": _i(r["rush_yards"]),
            "yards_per_carry": _f(r["yards_per_carry"]),
            "epa_per_carry": _f(r["epa_per_carry"]),
            "success_rate": _f(r["success_rate"]),
            "rush_tds": _i(r["rush_tds"]),
            "explosive_rate": _f(r["explosive_rate"]),
            "negative_rate": _f(r["negative_rate"]),
            "rz_carries": _i(r["rz_carries"]),
            "rz_epa": _f(r["rz_epa"]),
            "inside_rate": round(_i(r["inside_runs"]) / total_gap, 3) if total_gap else None,
            "outside_rate": round(_i(r["outside_runs"]) / total_gap, 3) if total_gap else None,
            "by_direction": {
                "left":   {"carries": _i(r["runs_left"]),   "epa": _f(r["epa_left"]),   "pct": round(_i(r["runs_left"])   / total_dir, 3) if total_dir else None},
                "middle": {"carries": _i(r["runs_middle"]), "epa": _f(r["epa_middle"]), "pct": round(_i(r["runs_middle"]) / total_dir, 3) if total_dir else None},
                "right":  {"carries": _i(r["runs_right"]),  "epa": _f(r["epa_right"]),  "pct": round(_i(r["runs_right"])  / total_dir, 3) if total_dir else None},
            },
        }
    return results


# ── Receiver batch metrics ────────────────────────────────────────────────────

def compute_receiver_metrics(db: Session, season: int) -> dict:
    """Return {player_id: metrics} for all receivers with ≥5 targets in a season."""
    sql = text("""
        SELECT
            receiver_player_id                                       AS player_id,
            MIN(receiver_player_name)                                AS player_name,
            posteam                                                  AS team,
            COUNT(*)                                                 AS targets,
            SUM(CASE WHEN complete_pass = 1 THEN 1 ELSE 0 END)      AS receptions,
            AVG(CASE WHEN complete_pass = 1 THEN 1.0 ELSE 0.0 END)  AS catch_rate,
            SUM(yards_gained)                                        AS rec_yards,
            AVG(yards_gained)                                        AS yards_per_target,
            AVG(yards_gained)
                FILTER (WHERE complete_pass = 1)                     AS yards_per_reception,
            AVG(epa)                                                 AS epa_per_target,
            AVG(air_yards)
                FILTER (WHERE air_yards IS NOT NULL)                 AS avg_air_yards,
            AVG(yards_after_catch)
                FILTER (WHERE yards_after_catch IS NOT NULL AND complete_pass = 1) AS avg_yac,
            SUM(CASE WHEN touchdown = 1 THEN 1 ELSE 0 END)          AS rec_tds,
            AVG(CASE WHEN air_yards > 15 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE air_yards IS NOT NULL)                 AS deep_target_rate,
            AVG(CASE WHEN air_yards <= 5  THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE air_yards IS NOT NULL)                 AS short_target_rate,
            COUNT(*) FILTER (WHERE yardline_100 <= 20)              AS rz_targets,
            SUM(CASE WHEN touchdown=1 AND yardline_100<=20 THEN 1 ELSE 0 END) AS rz_tds,
            COUNT(*) FILTER (WHERE pass_location = 'left')          AS dir_left,
            COUNT(*) FILTER (WHERE pass_location = 'middle')        AS dir_middle,
            COUNT(*) FILTER (WHERE pass_location = 'right')         AS dir_right,
            AVG(epa) FILTER (WHERE pass_location = 'left')          AS epa_left,
            AVG(epa) FILTER (WHERE pass_location = 'middle')        AS epa_middle,
            AVG(epa) FILTER (WHERE pass_location = 'right')         AS epa_right
        FROM play_by_play
        WHERE season = :season
          AND receiver_player_id IS NOT NULL
          AND play_type = 'pass'
        GROUP BY receiver_player_id, posteam
        HAVING COUNT(*) >= 5
    """)
    try:
        rows = db.execute(sql, {"season": season}).mappings().all()
    except Exception:
        db.rollback()
        logger.warning("Receiver metrics query failed for season %d", season)
        return {}

    results = {}
    for r in rows:
        pid = r["player_id"]
        if not pid:
            continue
        total_dir = _i(r["dir_left"]) + _i(r["dir_middle"]) + _i(r["dir_right"])
        results[pid] = {
            "player_name": r["player_name"],
            "team": r["team"],
            "targets": _i(r["targets"]),
            "receptions": _i(r["receptions"]),
            "catch_rate": _f(r["catch_rate"]),
            "rec_yards": _i(r["rec_yards"]),
            "yards_per_target": _f(r["yards_per_target"]),
            "yards_per_reception": _f(r["yards_per_reception"]),
            "epa_per_target": _f(r["epa_per_target"]),
            "avg_air_yards": _f(r["avg_air_yards"]),
            "avg_yac": _f(r["avg_yac"]),
            "rec_tds": _i(r["rec_tds"]),
            "deep_target_rate": _f(r["deep_target_rate"]),
            "short_target_rate": _f(r["short_target_rate"]),
            "rz_targets": _i(r["rz_targets"]),
            "rz_tds": _i(r["rz_tds"]),
            "by_direction": {
                "left":   {"targets": _i(r["dir_left"]),   "epa": _f(r["epa_left"]),   "pct": round(_i(r["dir_left"])   / total_dir, 3) if total_dir else None},
                "middle": {"targets": _i(r["dir_middle"]), "epa": _f(r["epa_middle"]), "pct": round(_i(r["dir_middle"]) / total_dir, 3) if total_dir else None},
                "right":  {"targets": _i(r["dir_right"]),  "epa": _f(r["epa_right"]),  "pct": round(_i(r["dir_right"])  / total_dir, 3) if total_dir else None},
            },
        }
    return results


# ── Defensive batch metrics ───────────────────────────────────────────────────

def compute_defensive_metrics(db: Session, season: int) -> dict:
    """
    Return {player_id: metrics} for all defenders with any tracked stat.

    Aggregates counting stats across all defensive player_id columns.
    Note: coverage assignments are not in standard PBP — this covers
    sacks, QB hits, tackles, INTs, pass breakups, forced fumbles, TFLs.
    """
    # Fetch raw defensive plays — single scan of the season's data
    sql = text("""
        SELECT
            defteam,
            sack_player_id,
            half_sack_1_player_id,
            half_sack_2_player_id,
            qb_hit_1_player_id,
            qb_hit_2_player_id,
            interception_player_id,
            pass_defense_1_player_id,
            pass_defense_2_player_id,
            solo_tackle_1_player_id,
            solo_tackle_2_player_id,
            assist_tackle_1_player_id,
            assist_tackle_2_player_id,
            tackle_for_loss_1_player_id,
            tackle_for_loss_2_player_id,
            forced_fumble_player_1_player_id,
            forced_fumble_player_2_player_id,
            fumble_recovery_1_player_id,
            play_type,
            epa,
            yards_gained
        FROM play_by_play
        WHERE season = :season
          AND play_type IN ('pass', 'run')
    """)
    try:
        rows = db.execute(sql, {"season": season}).mappings().all()
    except Exception:
        db.rollback()
        logger.warning("Defensive metrics query failed for season %d", season)
        return {}

    # Aggregate per player in Python
    stats: dict = {}

    def _acc(player_id, team, **kwargs):
        if not player_id:
            return
        if player_id not in stats:
            stats[player_id] = {
                "team": team,
                "sacks": 0.0, "qb_hits": 0, "solo_tackles": 0,
                "assist_tackles": 0, "interceptions": 0, "passes_defended": 0,
                "forced_fumbles": 0, "fumble_recoveries": 0, "tackles_for_loss": 0,
                "total_plays": 0, "epa_sum": 0.0,
            }
        for k, v in kwargs.items():
            stats[player_id][k] = stats[player_id].get(k, 0) + v

    for r in rows:
        team = r["defteam"] or ""
        _acc(r["sack_player_id"],              team, sacks=1.0, total_plays=1)
        _acc(r["half_sack_1_player_id"],        team, sacks=0.5, total_plays=1)
        _acc(r["half_sack_2_player_id"],        team, sacks=0.5, total_plays=1)
        _acc(r["qb_hit_1_player_id"],           team, qb_hits=1, total_plays=1)
        _acc(r["qb_hit_2_player_id"],           team, qb_hits=1, total_plays=1)
        _acc(r["interception_player_id"],       team, interceptions=1, total_plays=1)
        _acc(r["pass_defense_1_player_id"],     team, passes_defended=1, total_plays=1)
        _acc(r["pass_defense_2_player_id"],     team, passes_defended=1, total_plays=1)
        _acc(r["solo_tackle_1_player_id"],      team, solo_tackles=1, total_plays=1)
        _acc(r["solo_tackle_2_player_id"],      team, solo_tackles=1, total_plays=1)
        _acc(r["assist_tackle_1_player_id"],    team, assist_tackles=1, total_plays=1)
        _acc(r["assist_tackle_2_player_id"],    team, assist_tackles=1, total_plays=1)
        _acc(r["tackle_for_loss_1_player_id"],  team, tackles_for_loss=1, total_plays=1)
        _acc(r["tackle_for_loss_2_player_id"],  team, tackles_for_loss=1, total_plays=1)
        _acc(r["forced_fumble_player_1_player_id"], team, forced_fumbles=1, total_plays=1)
        _acc(r["forced_fumble_player_2_player_id"], team, forced_fumbles=1, total_plays=1)
        _acc(r["fumble_recovery_1_player_id"],  team, fumble_recoveries=1, total_plays=1)

    # Clean up and compute derived stats; filter to players with meaningful contribution
    results = {}
    for pid, s in stats.items():
        total = (s["sacks"] + s["qb_hits"] + s["solo_tackles"] +
                 s["interceptions"] + s["passes_defended"] + s["forced_fumbles"])
        if total < 1:
            continue
        results[pid] = {
            "team": s["team"],
            "sacks": round(s["sacks"], 1),
            "qb_hits": s["qb_hits"],
            "pressures": round(s["sacks"] + s["qb_hits"], 1),
            "solo_tackles": s["solo_tackles"],
            "assist_tackles": s["assist_tackles"],
            "total_tackles": s["solo_tackles"] + round(s["assist_tackles"] * 0.5, 1),
            "interceptions": s["interceptions"],
            "passes_defended": s["passes_defended"],
            "forced_fumbles": s["forced_fumbles"],
            "fumble_recoveries": s["fumble_recoveries"],
            "tackles_for_loss": s["tackles_for_loss"],
        }
    return results


# ── Position lookup ───────────────────────────────────────────────────────────

def get_player_positions(db: Session, season: int) -> dict:
    """Return {player_id: (player_name, position, team)} from player_stats for a season."""
    sql = text("""
        SELECT DISTINCT ON (player_id)
            player_id, player_display_name AS player_name, position, recent_team AS team
        FROM player_stats
        WHERE season = :season AND player_id IS NOT NULL
        ORDER BY player_id, week DESC
    """)
    try:
        rows = db.execute(sql, {"season": season}).mappings().all()
        return {r["player_id"]: (r["player_name"], r["position"], r["team"]) for r in rows}
    except Exception:
        db.rollback()
        return {}
