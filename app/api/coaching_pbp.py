#!/usr/bin/env python3
"""
Coaching play-by-play analytics helpers.

Computes formation, personnel, run-scheme, pass-detail, and defensive-scheme
metrics from the play_by_play table. Used by both the coaches router and the
chat tool implementations.

All column-dependent queries wrap in try/except — gracefully returns None/{}
when optional columns (offense_formation, offense_personnel, play_action,
number_of_pass_rushers, defenders_in_box) are absent from the table.
"""

import re
import logging
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── Personnel / formation string parsers ──────────────────────────────────────

def _parse_offense_personnel(s: str) -> str:
    """Map 'X RB, Y TE, Z WR' → standard personnel label."""
    if not s:
        return "unknown"
    rb_m = re.search(r"(\d+)\s+RB", s, re.I)
    te_m = re.search(r"(\d+)\s+TE", s, re.I)
    rb = int(rb_m.group(1)) if rb_m else 0
    te = int(te_m.group(1)) if te_m else 0
    key = f"{rb}{te}"
    labels = {
        "11": "11 (1RB 1TE 3WR)",
        "12": "12 (1RB 2TE 2WR)",
        "21": "21 (2RB 1TE 2WR)",
        "22": "22 (2RB 2TE 1WR)",
        "10": "10 (1RB 0TE 4WR)",
        "00": "Empty (0RB 0TE 5WR)",
        "13": "13 (1RB 3TE 1WR)",
        "20": "20 (2RB 0TE 3WR)",
        "23": "23 (2RB 3TE 0WR)",
    }
    return labels.get(key, f"{key} personnel")


def _parse_defense_personnel(s: str) -> str:
    """Map 'X DL, Y LB, Z DB' → standard formation name."""
    if not s:
        return "unknown"
    dl_m = re.search(r"(\d+)\s+DL", s, re.I)
    lb_m = re.search(r"(\d+)\s+LB", s, re.I)
    db_m = re.search(r"(\d+)\s+DB", s, re.I)
    dl = int(dl_m.group(1)) if dl_m else 0
    lb = int(lb_m.group(1)) if lb_m else 0
    db = int(db_m.group(1)) if db_m else 0
    table = {
        (4, 3, 4): "Base 4-3",
        (3, 4, 4): "Base 3-4",
        (4, 2, 5): "Nickel 4-2-5",
        (3, 3, 5): "Nickel 3-3-5",
        (4, 1, 6): "Dime 4-1-6",
        (3, 2, 6): "Dime 3-2-6",
        (2, 3, 6): "Dime 2-3-6",
    }
    if db >= 7:
        return f"Quarter ({dl}-{lb}-{db})"
    return table.get((dl, lb, db), f"{dl}-{lb}-{db}")


# ── Scalar helpers ─────────────────────────────────────────────────────────────

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


# ── Analytics computations ────────────────────────────────────────────────────

def compute_formation_breakdown(db: Session, team: str, season: int) -> dict:
    """Shotgun/no-huddle/play-action rates and per-formation efficiency."""
    misc_sql = text("""
        SELECT
            COUNT(*) FILTER (WHERE play_type IN ('pass','run')) AS total_plays,
            AVG(CASE WHEN shotgun = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE play_type IN ('pass','run')) AS shotgun_rate,
            AVG(CASE WHEN no_huddle = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE play_type IN ('pass','run')) AS no_huddle_rate
        FROM play_by_play
        WHERE posteam = :team AND season = :season
    """)
    misc = db.execute(misc_sql, {"team": team, "season": season}).mappings().first()

    # play_action column may not exist in all data sources
    pa_rate = None
    try:
        pa_sql = text("""
            SELECT AVG(CASE WHEN play_action = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE play_type IN ('pass','run')) AS play_action_rate
            FROM play_by_play
            WHERE posteam = :team AND season = :season
        """)
        pa_row = db.execute(pa_sql, {"team": team, "season": season}).mappings().first()
        pa_rate = _f(pa_row["play_action_rate"]) if pa_row else None
    except Exception:
        pass

    # offense_formation column may not exist
    formations: dict = {}
    try:
        form_sql = text("""
            SELECT offense_formation,
                COUNT(*) AS plays,
                SUM(CASE WHEN play_type = 'pass' THEN 1 ELSE 0 END) AS pass_plays,
                AVG(epa) AS avg_epa,
                AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) AS success_rate
            FROM play_by_play
            WHERE posteam = :team AND season = :season
              AND play_type IN ('pass','run')
              AND offense_formation IS NOT NULL
            GROUP BY offense_formation
            ORDER BY plays DESC
        """)
        form_rows = db.execute(form_sql, {"team": team, "season": season}).mappings().all()
        total = sum(_i(r["plays"]) for r in form_rows)
        for r in form_rows:
            plays = _i(r["plays"])
            pass_plays = _i(r["pass_plays"])
            formations[r["offense_formation"]] = {
                "plays": plays,
                "pct_of_plays": round(plays / total, 3) if total > 0 else 0,
                "pass_rate": round(pass_plays / plays, 3) if plays > 0 else None,
                "avg_epa": _f(r["avg_epa"]),
                "success_rate": _f(r["success_rate"]),
            }
    except Exception:
        pass

    return {
        "shotgun_rate": _f(misc["shotgun_rate"]) if misc else None,
        "no_huddle_rate": _f(misc["no_huddle_rate"]) if misc else None,
        "play_action_rate": pa_rate,
        "formation_breakdown": formations,
    }


def compute_personnel_breakdown(db: Session, team: str, season: int) -> dict:
    """Personnel grouping (11/12/21/22 etc.) frequency and efficiency."""
    try:
        sql = text("""
            SELECT offense_personnel, play_type, yards_gained, epa, success
            FROM play_by_play
            WHERE posteam = :team AND season = :season
              AND play_type IN ('pass','run')
              AND offense_personnel IS NOT NULL
        """)
        rows = db.execute(sql, {"team": team, "season": season}).mappings().all()
    except Exception:
        return {}

    groups: dict = {}
    for r in rows:
        key = _parse_offense_personnel(r["offense_personnel"] or "")
        if key not in groups:
            groups[key] = {"plays": 0, "pass_plays": 0,
                           "epa_sum": 0.0, "success_sum": 0, "yards_sum": 0.0}
        g = groups[key]
        g["plays"] += 1
        if r["play_type"] == "pass":
            g["pass_plays"] += 1
        for field, dest in [("epa", "epa_sum"), ("yards_gained", "yards_sum")]:
            if r[field] is not None:
                try:
                    g[dest] += float(r[field])
                except (TypeError, ValueError):
                    pass
        if r["success"] is not None:
            try:
                g["success_sum"] += int(r["success"])
            except (TypeError, ValueError):
                pass

    total = sum(g["plays"] for g in groups.values())
    return {
        key: {
            "plays": g["plays"],
            "pct_of_plays": round(g["plays"] / total, 3) if total > 0 else 0,
            "pass_rate": round(g["pass_plays"] / g["plays"], 3) if g["plays"] > 0 else None,
            "avg_epa": round(g["epa_sum"] / g["plays"], 4) if g["plays"] > 0 else None,
            "success_rate": round(g["success_sum"] / g["plays"], 3) if g["plays"] > 0 else None,
            "avg_yards": round(g["yards_sum"] / g["plays"], 2) if g["plays"] > 0 else None,
        }
        for key, g in sorted(groups.items(), key=lambda x: -x[1]["plays"])
    }


def compute_run_scheme(db: Session, team: str, season: int) -> dict:
    """Run direction (L/M/R) and gap (end/guard/tackle) breakdown."""
    sql = text("""
        SELECT run_location, run_gap,
            COUNT(*) AS plays,
            AVG(yards_gained) AS avg_yards,
            AVG(epa) AS avg_epa,
            AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) AS success_rate
        FROM play_by_play
        WHERE posteam = :team AND season = :season
          AND play_type = 'run'
          AND run_location IS NOT NULL
        GROUP BY run_location, run_gap
        ORDER BY plays DESC
    """)
    rows = db.execute(sql, {"team": team, "season": season}).mappings().all()

    by_location: dict = {}
    inside_plays = 0
    outside_plays = 0
    total_runs = 0

    for r in rows:
        loc = r["run_location"] or "unknown"
        gap = r["run_gap"]
        plays = _i(r["plays"])
        total_runs += plays

        if loc not in by_location:
            by_location[loc] = {"plays": 0, "_epa": 0.0, "_yards": 0.0, "_suc": 0.0}
        bl = by_location[loc]
        bl["plays"] += plays
        if r["avg_epa"] is not None:
            bl["_epa"] += float(r["avg_epa"]) * plays
        if r["avg_yards"] is not None:
            bl["_yards"] += float(r["avg_yards"]) * plays
        if r["success_rate"] is not None:
            bl["_suc"] += float(r["success_rate"]) * plays

        # Classify inside (guard/middle) vs outside (end/perimeter)
        if gap == "guard" or loc == "middle":
            inside_plays += plays
        elif gap == "end" or loc in ("left", "right"):
            outside_plays += plays

    for loc, bl in by_location.items():
        p = bl["plays"]
        bl["avg_yards"] = round(bl["_yards"] / p, 2) if p > 0 else None
        bl["avg_epa"] = round(bl["_epa"] / p, 4) if p > 0 else None
        bl["success_rate"] = round(bl["_suc"] / p, 3) if p > 0 else None
        for k in ("_epa", "_yards", "_suc"):
            del bl[k]

    total_classified = inside_plays + outside_plays
    return {
        "total_runs": total_runs,
        "inside_rate": round(inside_plays / total_classified, 3) if total_classified else None,
        "outside_rate": round(outside_plays / total_classified, 3) if total_classified else None,
        "by_location": by_location,
    }


def compute_pass_detail(db: Session, team: str, season: int) -> dict:
    """Pass depth (air yards), direction, YAC, scrambles, and dropback EPA."""
    sql = text("""
        SELECT
            AVG(air_yards)
                FILTER (WHERE play_type = 'pass' AND air_yards IS NOT NULL) AS avg_air_yards,
            COUNT(*) FILTER (WHERE play_type = 'pass' AND air_yards > 15) AS deep_passes,
            COUNT(*) FILTER (WHERE play_type = 'pass' AND air_yards <= 0) AS screen_passes,
            COUNT(*) FILTER (WHERE play_type = 'pass' AND air_yards > 0 AND air_yards <= 15) AS intermediate_passes,
            COUNT(*) FILTER (WHERE play_type = 'pass') AS total_passes,
            AVG(CASE WHEN qb_scramble = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE qb_dropback = 1) AS scramble_rate,
            AVG(epa) FILTER (WHERE qb_dropback = 1) AS dropback_epa,
            AVG(yards_after_catch)
                FILTER (WHERE play_type = 'pass' AND yards_after_catch IS NOT NULL) AS avg_yac
        FROM play_by_play
        WHERE posteam = :team AND season = :season
          AND play_type IN ('pass','run')
    """)
    row = db.execute(sql, {"team": team, "season": season}).mappings().first()

    dir_sql = text("""
        SELECT pass_location,
            COUNT(*) AS plays,
            AVG(epa) AS avg_epa,
            AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) AS success_rate,
            AVG(air_yards) AS avg_air_yards
        FROM play_by_play
        WHERE posteam = :team AND season = :season
          AND play_type = 'pass'
          AND pass_location IS NOT NULL
        GROUP BY pass_location
        ORDER BY plays DESC
    """)
    dir_rows = db.execute(dir_sql, {"team": team, "season": season}).mappings().all()

    by_direction = {
        dr["pass_location"]: {
            "plays": _i(dr["plays"]),
            "avg_epa": _f(dr["avg_epa"]),
            "success_rate": _f(dr["success_rate"]),
            "avg_air_yards": _f(dr["avg_air_yards"]),
        }
        for dr in dir_rows
    }

    total_passes = _i(row["total_passes"]) if row else 0
    deep = _i(row["deep_passes"]) if row else 0
    screen = _i(row["screen_passes"]) if row else 0
    intermediate = _i(row["intermediate_passes"]) if row else 0

    return {
        "avg_air_yards": _f(row["avg_air_yards"]) if row else None,
        "deep_pass_rate": round(deep / total_passes, 3) if total_passes else None,
        "screen_rate": round(screen / total_passes, 3) if total_passes else None,
        "intermediate_rate": round(intermediate / total_passes, 3) if total_passes else None,
        "scramble_rate": _f(row["scramble_rate"]) if row else None,
        "dropback_epa": _f(row["dropback_epa"]) if row else None,
        "avg_yac": _f(row["avg_yac"]) if row else None,
        "by_direction": by_direction,
    }


def compute_defense_scheme(db: Session, team: str, season: int) -> dict:
    """Defensive personnel groupings, blitz rate, box density, and pressure."""
    # QB pressure columns (may not exist in all data sources)
    blitz_rate = blitz_epa = non_blitz_epa = avg_dib = None
    try:
        press_sql = text("""
            SELECT
                AVG(CASE WHEN number_of_pass_rushers >= 5 THEN 1.0 ELSE 0.0 END)
                    FILTER (WHERE play_type = 'pass' AND number_of_pass_rushers IS NOT NULL) AS blitz_rate,
                AVG(epa)
                    FILTER (WHERE number_of_pass_rushers >= 5 AND play_type = 'pass') AS blitz_epa,
                AVG(epa)
                    FILTER (WHERE (number_of_pass_rushers <= 4 OR number_of_pass_rushers IS NULL)
                            AND play_type = 'pass') AS non_blitz_epa,
                AVG(defenders_in_box) FILTER (WHERE defenders_in_box IS NOT NULL) AS avg_dib
            FROM play_by_play
            WHERE defteam = :team AND season = :season
              AND play_type IN ('pass','run')
        """)
        pr = db.execute(press_sql, {"team": team, "season": season}).mappings().first()
        if pr:
            blitz_rate = _f(pr["blitz_rate"])
            blitz_epa = _f(pr["blitz_epa"])
            non_blitz_epa = _f(pr["non_blitz_epa"])
            avg_dib = _f(pr["avg_dib"])
    except Exception:
        pass

    # qb_hit and sack are standard columns
    press2_sql = text("""
        SELECT
            AVG(CASE WHEN qb_hit = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE play_type = 'pass') AS qb_hit_rate,
            AVG(CASE WHEN sack = 1 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE play_type IN ('pass','run')) AS sack_rate
        FROM play_by_play
        WHERE defteam = :team AND season = :season
          AND play_type IN ('pass','run')
    """)
    p2 = db.execute(press2_sql, {"team": team, "season": season}).mappings().first()

    # Defense personnel grouping (column may not exist)
    personnel_breakdown: dict = {}
    try:
        def_sql = text("""
            SELECT defense_personnel,
                COUNT(*) AS plays,
                AVG(epa) AS avg_epa_allowed,
                AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) AS opponent_success_rate
            FROM play_by_play
            WHERE defteam = :team AND season = :season
              AND play_type IN ('pass','run')
              AND defense_personnel IS NOT NULL
            GROUP BY defense_personnel
            ORDER BY plays DESC
            LIMIT 10
        """)
        def_rows = db.execute(def_sql, {"team": team, "season": season}).mappings().all()
        total_def = sum(_i(r["plays"]) for r in def_rows)
        for r in def_rows:
            name = _parse_defense_personnel(r["defense_personnel"] or "")
            plays = _i(r["plays"])
            if name not in personnel_breakdown:
                personnel_breakdown[name] = {
                    "plays": 0, "pct_of_plays": 0.0,
                    "avg_epa_allowed": None, "opponent_success_rate": None,
                }
            entry = personnel_breakdown[name]
            entry["plays"] += plays
            entry["avg_epa_allowed"] = _f(r["avg_epa_allowed"])
            entry["opponent_success_rate"] = _f(r["opponent_success_rate"])
        for entry in personnel_breakdown.values():
            entry["pct_of_plays"] = (
                round(entry["plays"] / total_def, 3) if total_def else 0
            )
        personnel_breakdown = dict(
            sorted(personnel_breakdown.items(), key=lambda x: -x[1]["plays"])
        )
    except Exception:
        pass

    return {
        "blitz_rate": blitz_rate,
        "avg_defenders_in_box": avg_dib,
        "blitz_epa_allowed": blitz_epa,
        "non_blitz_epa_allowed": non_blitz_epa,
        "qb_hit_rate": _f(p2["qb_hit_rate"]) if p2 else None,
        "sack_rate": _f(p2["sack_rate"]) if p2 else None,
        "personnel_breakdown": personnel_breakdown,
    }


def compute_strengths_weaknesses(
    offense: dict,
    defense: dict,
    formation: dict,
    run_scheme: dict,
    pass_detail: dict,
    defense_scheme: dict,
) -> dict:
    """Derive strengths, weaknesses, and tendency labels from aggregated metrics."""
    strengths: list = []
    weaknesses: list = []
    tendencies: list = []

    # ── Offensive efficiency ─────────────────────────────────────────────────
    epa = offense.get("avg_epa_per_play")
    if epa is not None:
        if epa > 0.10:
            strengths.append("Elite offensive efficiency (top EPA/play)")
        elif epa > 0.04:
            strengths.append("Above-average offensive efficiency")
        elif epa < -0.10:
            weaknesses.append("Poor offensive efficiency (low EPA/play)")
        elif epa < -0.02:
            weaknesses.append("Below-average offensive efficiency")

    third_conv = offense.get("third_down_conversion_rate")
    if third_conv is not None:
        if third_conv > 0.45:
            strengths.append(f"Excellent 3rd-down offense ({third_conv:.0%} conversion)")
        elif third_conv < 0.33:
            weaknesses.append(f"Struggles on 3rd down ({third_conv:.0%} conversion)")

    fourth_rate = offense.get("fourth_down_conversion_rate")
    fourth_att = offense.get("fourth_down_attempts", 0) or 0
    if fourth_att >= 5 and fourth_rate is not None:
        if fourth_rate > 0.60:
            strengths.append(f"Aggressive & successful on 4th down ({fourth_rate:.0%})")
        elif fourth_rate < 0.35:
            weaknesses.append(f"Low 4th-down conversion rate ({fourth_rate:.0%})")

    # ── Formation / scheme tendencies ────────────────────────────────────────
    sg = formation.get("shotgun_rate")
    if sg is not None:
        tendencies.append(f"Shotgun on {sg:.0%} of plays")
        if sg > 0.70:
            tendencies.append("Pass-oriented / spread offense")
        elif sg < 0.35:
            tendencies.append("Under-center / traditional offense")

    nh = formation.get("no_huddle_rate")
    if nh is not None and nh > 0.18:
        tendencies.append(f"Up-tempo / no-huddle ({nh:.0%} of plays)")

    pa = formation.get("play_action_rate")
    if pa is not None:
        if pa > 0.28:
            tendencies.append(f"Heavy play-action user ({pa:.0%})")
        elif pa < 0.12:
            tendencies.append(f"Rarely uses play action ({pa:.0%})")

    # ── Run scheme ────────────────────────────────────────────────────────────
    inside_r = run_scheme.get("inside_rate")
    outside_r = run_scheme.get("outside_rate")
    if inside_r is not None and outside_r is not None:
        if outside_r > 0.55:
            tendencies.append("Outside-zone / perimeter run game")
        elif inside_r > 0.65:
            tendencies.append("Power / inside-zone run scheme")

    run_epas = [
        loc.get("avg_epa", 0)
        for loc in run_scheme.get("by_location", {}).values()
        if loc.get("avg_epa") is not None
    ]
    if run_epas:
        avg_run_epa = sum(run_epas) / len(run_epas)
        if avg_run_epa > 0.02:
            strengths.append("Effective run game (positive EPA)")
        elif avg_run_epa < -0.15:
            weaknesses.append("Inefficient run game")

    # ── Pass depth ────────────────────────────────────────────────────────────
    deep_r = pass_detail.get("deep_pass_rate")
    if deep_r is not None:
        if deep_r > 0.25:
            tendencies.append(f"Down-field passing attack ({deep_r:.0%} deep)")
        elif deep_r < 0.10:
            tendencies.append(f"Short / check-down passing scheme ({deep_r:.0%} deep)")

    avg_ay = pass_detail.get("avg_air_yards")
    if avg_ay is not None:
        if avg_ay > 10:
            tendencies.append(f"Deep average target depth ({avg_ay:.1f} air yards)")
        elif avg_ay < 5:
            tendencies.append(f"Short average target depth ({avg_ay:.1f} air yards)")

    # ── Defensive efficiency ─────────────────────────────────────────────────
    def_epa = defense.get("avg_epa_allowed_per_play")
    if def_epa is not None:
        if def_epa < -0.08:
            strengths.append("Elite defense (suppresses opponent EPA)")
        elif def_epa < -0.02:
            strengths.append("Above-average defense")
        elif def_epa > 0.08:
            weaknesses.append("Defense allows high EPA per play")
        elif def_epa > 0.02:
            weaknesses.append("Below-average defensive efficiency")

    stop_r = defense.get("third_down_stop_rate")
    if stop_r is not None:
        if stop_r > 0.60:
            strengths.append(f"Dominant 3rd-down defense ({stop_r:.0%} stop rate)")
        elif stop_r < 0.38:
            weaknesses.append(f"3rd-down defense struggles ({stop_r:.0%} stop rate)")

    rz_td = defense.get("red_zone_td_rate_allowed")
    if rz_td is not None:
        if rz_td < 0.45:
            strengths.append(f"Strong red-zone defense ({rz_td:.0%} TD rate allowed)")
        elif rz_td > 0.65:
            weaknesses.append(f"Leaky red-zone defense ({rz_td:.0%} TD rate allowed)")

    # ── Defensive pressure ────────────────────────────────────────────────────
    sack_r = defense_scheme.get("sack_rate")
    if sack_r is not None:
        if sack_r > 0.08:
            strengths.append(f"Elite pass rush ({sack_r:.0%} sack rate)")
        elif sack_r < 0.04:
            weaknesses.append(f"Lacks pass-rush pressure ({sack_r:.0%} sack rate)")

    blitz_r = defense_scheme.get("blitz_rate")
    blitz_epa_val = defense_scheme.get("blitz_epa_allowed")
    if blitz_r is not None:
        label = f"Blitzes on {blitz_r:.0%} of passing plays"
        if blitz_r > 0.30:
            if blitz_epa_val is not None and blitz_epa_val < 0:
                strengths.append(
                    f"Effective blitz package ({blitz_r:.0%} rate, negative EPA allowed)"
                )
            tendencies.append(label + " (aggressive)")
        elif blitz_r < 0.12:
            tendencies.append(label + " (conservative coverage shell)")

    dib = defense_scheme.get("avg_defenders_in_box")
    if dib is not None:
        if dib > 7.0:
            tendencies.append(f"Stacks the box ({dib:.1f} avg defenders)")
        elif dib < 5.5:
            tendencies.append(f"Light box / coverage defense ({dib:.1f} avg defenders)")

    return {"strengths": strengths, "weaknesses": weaknesses, "tendencies": tendencies}
