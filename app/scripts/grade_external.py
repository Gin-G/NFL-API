#!/usr/bin/env python3
"""Does a hand-uploaded third-party matchup table beat our own board?

The gate on `api/external.py`. Nothing from those uploads reaches
/projections until this script says it paid, and this script refuses to
answer until there are enough weeks to answer with.

**Method, same as the existing coverage tests** (nfl-data-py EXPERIMENTS.md,
"Per-corner coverage grades"), in two stages:

  1. *Slope.* Regress the board's own residual on the candidate signal and
     report the slope with a 95% interval. A signal that explains none of the
     board's error cannot improve it.
  2. *Payment.* Fit leave-one-WEEK-out, apply as a single additive term with
     the intercept dropped, and compare MAE against the uncorrected board with
     a paired interval.

One deliberate substitution: the original fits leave-one-SEASON-out, and there
is only one season of these snapshots, so a week plays the part of a season.
That is a weaker guard against overfitting — weeks of one season share a model
vintage and a roster in a way seasons do not — so a pass here is weaker
evidence than a pass there, and is written up as such.

**The sample is the problem, and the script says so before it says anything
else.** The export is 25 pairings a week:

    weeks   pairings   shadow rows   slope CI vs the n=16,380 WR test
        4        100             4   12.8x wider
        6        150             6   10.4x wider
       20        500            20    5.7x wider
       40      1,000            40    4.0x wider

The existing WR coverage test returned a slope interval of about +/-0.11 at
n=16,380 and called it null. At n=100 the same precision is 12.8x worse, so a
four-week harness cannot distinguish "no effect" from "an effect three times
larger than anything we have ever measured". It will return `inconclusive`,
which is the honest answer and not a null result.

**The assignment is worse off still.** Shadow coverage is the one column here
nflverse has no equivalent for, and therefore the only one worth real hope --
and it appears on about one row a week. Four weeks is four observations. A
usable test of shadow coverage needs seasons, not weeks, and no amount of
careful statistics fixes that. It is reported separately for exactly that
reason, so it is never buried inside an aggregate that looks adequately
powered.

Run:
    python -m scripts.grade_external --season 2026
    python -m scripts.grade_external --season 2026 --min-weeks 4 --json out.json
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("grade_external")

# Weeks of snapshots before the harness will fit anything at all. Four is the
# floor the plan asked for; it is not enough for a verdict, which is why the
# report carries its own power arithmetic rather than a bare p-value.
MIN_WEEKS = 4
# Pairings below which even a slope is not worth printing.
MIN_ROWS = 60
# The slope interval the existing WR coverage test achieved, and the n it took.
# Used to state this harness's precision in terms of a result already on file.
REFERENCE_N = 16380
REFERENCE_CI_HALFWIDTH = 0.11
# Shadow rows needed before the assignment column gets its own fit. Set from
# the accrual rate (~1 a week): this is several seasons, and saying so is the
# point of the constant.
MIN_SHADOW_ROWS = 100


def _ols_slope(xs, ys):
    """``(slope, halfwidth)`` for a single-regressor OLS with a free intercept.

    Plain OLS and a normal interval: the comparison target is an existing
    result computed the same way, and a fancier estimator would make the two
    incomparable for no gain at this sample size.
    """
    n = len(xs)
    if n < 3:
        return None, None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None, None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    if n <= 2:
        return slope, None
    sigma2 = sum(r * r for r in resid) / (n - 2)
    se = math.sqrt(sigma2 / sxx) if sxx > 0 else None
    return slope, (1.96 * se if se else None)


def _paired_ci(deltas):
    """Mean and 95% interval of a paired per-row difference."""
    n = len(deltas)
    if n < 3:
        return None, None, None
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    half = 1.96 * math.sqrt(var / n)
    return mean, mean - half, mean + half


def _power_note(n: int) -> dict:
    """How precise this sample is, stated against a result already on file."""
    if n <= 0:
        return {"pairings": 0, "usable": False}
    factor = math.sqrt(REFERENCE_N / n)
    return {
        "pairings": n,
        "ci_halfwidth_vs_reference": round(factor, 1),
        "detectable_slope_approx": round(REFERENCE_CI_HALFWIDTH * factor, 3),
        "reference": (f"the WR coverage test called null at n={REFERENCE_N:,} "
                      f"with a slope interval of about "
                      f"±{REFERENCE_CI_HALFWIDTH}"),
        "reading": (
            "A slope interval this wide cannot separate 'no effect' from an "
            "effect several times larger than anything measured so far. A "
            "null here is an absence of evidence, not evidence of absence."),
    }


def _rows(db, season: int, source: str):
    """Board, actuals and the uploaded pairings, joined on (week, receiver)."""
    from database.models import (ExternalWrMatchup, PlayerProjection,
                                 PlayerStat)

    pairings = db.query(ExternalWrMatchup).filter(
        ExternalWrMatchup.season == season,
        ExternalWrMatchup.source == source).all()
    if not pairings:
        return [], []
    weeks = sorted({p.week for p in pairings})
    proj = {(p.week, p.player_id): p for p in db.query(PlayerProjection).filter(
        PlayerProjection.season == season,
        PlayerProjection.week.in_(weeks)).all()}
    stats = {(s.week, s.player_id): s for s in db.query(PlayerStat).filter(
        PlayerStat.season == season, PlayerStat.week.in_(weeks)).all()}

    joined, dropped = [], []
    for p in pairings:
        if not p.receiver_id:
            dropped.append({"week": p.week, "receiver": p.receiver,
                            "why": "name not mapped to a gsis id"})
            continue
        pr = proj.get((p.week, p.receiver_id))
        st = stats.get((p.week, p.receiver_id))
        if pr is None:
            dropped.append({"week": p.week, "receiver": p.receiver,
                            "why": "no board projection for that week"})
            continue
        if st is None:
            dropped.append({"week": p.week, "receiver": p.receiver,
                            "why": "no actual stat line yet"})
            continue
        joined.append({
            "week": p.week,
            "receiver": p.receiver,
            "receiver_id": p.receiver_id,
            "advantage": p.advantage,
            "shadow": bool(p.shadow),
            "shell": p.shell,
            "man_zone": p.man_zone,
            # The two targets asked for. Both are projected by the board, so
            # the residual is defined without having to reconstruct it.
            "proj_receptions": pr.receptions,
            "proj_receiving_yards": pr.receiving_yards,
            "act_receptions": st.receptions,
            "act_receiving_yards": st.receiving_yards,
        })
    return joined, dropped


def _grade_target(rows, target: str) -> dict:
    """Slope then payment, for one of receptions / receiving yards."""
    pairs = [(r["advantage"], r[f"act_{target}"] - r[f"proj_{target}"], r["week"])
             for r in rows
             if r["advantage"] is not None
             and r[f"act_{target}"] is not None
             and r[f"proj_{target}"] is not None]
    if len(pairs) < MIN_ROWS:
        return {"target": target, "status": "insufficient", "n": len(pairs),
                "min_rows": MIN_ROWS}

    xs = [p[0] for p in pairs]
    resid = [p[1] for p in pairs]
    slope, half = _ols_slope(xs, resid)

    # Payment: leave one week out, correct with that fit, compare absolute
    # error per row. The intercept is dropped on application -- a free
    # intercept would just re-centre the board and flatter the correction.
    weeks = sorted({p[2] for p in pairs})
    base_err, corr_err = [], []
    for held in weeks:
        train = [p for p in pairs if p[2] != held]
        test = [p for p in pairs if p[2] == held]
        if len(train) < 3 or not test:
            continue
        fit, _ = _ols_slope([p[0] for p in train], [p[1] for p in train])
        if fit is None:
            continue
        mean_x = sum(p[0] for p in train) / len(train)
        for x, r, _w in test:
            base_err.append(abs(r))
            corr_err.append(abs(r - fit * (x - mean_x)))
    gain = [b - c for b, c in zip(base_err, corr_err)]
    mean_gain, lo, hi = _paired_ci(gain)
    paid = bool(mean_gain and lo is not None and lo > 0)

    return {
        "target": target,
        "status": "measured",
        "n": len(pairs),
        "slope": round(slope, 4) if slope is not None else None,
        "slope_ci": ([round(slope - half, 4), round(slope + half, 4)]
                     if slope is not None and half is not None else None),
        "slope_clears_zero": bool(slope is not None and half is not None
                                  and abs(slope) > half),
        "board_mae": round(sum(base_err) / len(base_err), 4) if base_err else None,
        "corrected_mae": round(sum(corr_err) / len(corr_err), 4) if corr_err else None,
        "gain": round(mean_gain, 4) if mean_gain is not None else None,
        "gain_ci": ([round(lo, 4), round(hi, 4)]
                    if lo is not None else None),
        "paid": paid,
        "folds": len(weeks),
        "power": _power_note(len(pairs)),
    }


def _grade_shadow(rows) -> dict:
    """The assignment column, on its own, because it accrues ~1 row a week."""
    shadow = [r for r in rows if r["shadow"]]
    out = {"shadow_rows": len(shadow), "min_rows": MIN_SHADOW_ROWS,
           "weeks": len(sorted({r["week"] for r in rows}))}
    if len(shadow) < MIN_SHADOW_ROWS:
        rate = (len(shadow) / out["weeks"]) if out["weeks"] else 0
        out["status"] = "insufficient"
        out["accrual_per_week"] = round(rate, 2)
        out["weeks_still_needed"] = (
            int(math.ceil((MIN_SHADOW_ROWS - len(shadow)) / rate))
            if rate > 0 else None)
        out["reading"] = (
            "Shadow coverage is the only column here nflverse has no "
            "equivalent for, and it is the one with the least data. At about "
            "one row a week a usable test needs seasons, not weeks. Reported "
            "separately so it is never averaged into a total that looks "
            "better powered than it is.")
        return out
    for target in ("receptions", "receiving_yards"):
        got = [r[f"act_{target}"] - r[f"proj_{target}"] for r in shadow
               if r[f"act_{target}"] is not None
               and r[f"proj_{target}"] is not None]
        mean, lo, hi = _paired_ci(got)
        out[target] = {"n": len(got), "mean_residual": round(mean, 3)
                       if mean is not None else None,
                       "ci": [round(lo, 3), round(hi, 3)] if lo is not None else None}
    out["status"] = "measured"
    return out


def _grade_lines(db, season: int, source: str) -> dict:
    """Do their line ratings add anything over /trenches?

    Correlation only, and deliberately so. Turning this into a residual test
    needs a team-week target the board actually projects, and the honest
    version of that is the rushing work /trenches was already tested on -- a
    test it passed nothing on. So what is reported is agreement, and where
    they disagree, which is what the tables are actually used for.
    """
    from database.models import ExternalTeamLine

    rows = db.query(ExternalTeamLine).filter(
        ExternalTeamLine.season == season,
        ExternalTeamLine.source == source).all()
    if not rows:
        return {"status": "no_data"}
    weeks = sorted({r.week for r in rows})
    # Their rating moves slowly; the thing worth knowing is whether it moves at
    # all, because a rating that never changes cannot be a weekly signal.
    by_team = {}
    for r in rows:
        by_team.setdefault(r.team, []).append((r.week, r.offensive_line,
                                               r.defensive_front))
    moved = 0
    for team, series in by_team.items():
        ols = {v for _, v, _ in series if v is not None}
        dfs = {v for _, _, v in series if v is not None}
        if len(ols) > 1 or len(dfs) > 1:
            moved += 1
    return {
        "status": "measured",
        "weeks": weeks,
        "teams": len(by_team),
        "teams_whose_rating_moved": moved,
        "note": ("A rating that does not move week to week is a season-long "
                 "team descriptor, not a weekly matchup signal, and should be "
                 "compared against /trenches once per season rather than "
                 "once per week. /external/compare/trenches does the "
                 "week-by-week comparison."),
    }


def run(db, season: int, source: str = "fantasyguru",
        min_weeks: int = MIN_WEEKS) -> dict:
    """The whole report. Never raises on thin data; says it is thin."""
    rows, dropped = _rows(db, season, source)
    weeks = sorted({r["week"] for r in rows})
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "season": season,
        "source": source,
        "weeks_held": weeks,
        "pairings_joined": len(rows),
        "rows_dropped": len(dropped),
        "drop_reasons": dropped[:20],
        "min_weeks": min_weeks,
        "gate": "closed",
    }
    if len(weeks) < min_weeks:
        report["status"] = "insufficient_weeks"
        report["message"] = (
            f"{len(weeks)} week(s) of {source} snapshots for {season}; this "
            f"harness needs {min_weeks} before it fits anything. Upload the "
            "weekly export and re-run.")
        return report

    report["status"] = "measured"
    report["targets"] = [_grade_target(rows, t)
                         for t in ("receptions", "receiving_yards")]
    report["shadow"] = _grade_shadow(rows)
    report["lines"] = _grade_lines(db, season, source)

    paid = [t for t in report["targets"] if t.get("paid")]
    powered = [t for t in report["targets"]
               if t.get("status") == "measured"
               and t["power"]["ci_halfwidth_vs_reference"] <= 3.0]
    if paid and powered:
        report["gate"] = "open"
        report["verdict"] = (
            "Beat the board out of sample on " +
            ", ".join(t["target"] for t in paid) +
            ". Worth wiring into /projections; see the module docstring for "
            "why a leave-one-week-out pass is weaker than a seasonal one.")
    elif paid:
        report["verdict"] = (
            "Showed a gain, but at this sample the interval is too wide to "
            "act on. The gate stays closed: a signal that pays only at low "
            "power is the shape of a signal that does not pay.")
    else:
        report["verdict"] = (
            "No gain over the board. At this sample that is INCONCLUSIVE "
            "rather than negative -- see the power block on each target.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade hand-uploaded third-party matchup tables")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--source", default="fantasyguru")
    parser.add_argument("--min-weeks", type=int, default=MIN_WEEKS)
    parser.add_argument("--json", help="also write the report to this path")
    args = parser.parse_args()

    sys.path.insert(0, ".")
    from database.models import Base
    from database.session import SessionLocal, engine

    Base.metadata.create_all(engine)
    db = SessionLocal()
    try:
        report = run(db, args.season, args.source, args.min_weeks)
    finally:
        db.close()

    print(json.dumps(report, indent=2, default=str))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=str)
    logger.info("gate is %s", report["gate"])


if __name__ == "__main__":
    main()
