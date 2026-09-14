"""Aggregate prediction performance metrics from stored snapshots."""

from __future__ import annotations

from typing import Any

from app.prediction.config import MIN_SAMPLE_FOR_RATE, SIGNAL_STRENGTH_BUCKETS
from app.prediction import storage


def _rate(n: int, d: int) -> float | None:
    if d <= 0:
        return None
    return round(100.0 * n / d, 2)


def _bucket(strength: int) -> str:
    for lo, hi, label in SIGNAL_STRENGTH_BUCKETS:
        if lo <= strength <= hi:
            return label
    return "0-39"


def _streaks(outcomes: list[str]) -> tuple[int, int]:
    best_w = best_l = cur_w = cur_l = 0
    for o in outcomes:
        if o == "WIN":
            cur_w += 1
            cur_l = 0
            best_w = max(best_w, cur_w)
        elif o == "LOSS":
            cur_l += 1
            cur_w = 0
            best_l = max(best_l, cur_l)
        else:
            cur_w = cur_l = 0
    return best_w, best_l


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    evaluated = [r for r in rows if r.get("evaluated")]
    pending = [r for r in rows if not r.get("evaluated")]
    scored = [r for r in evaluated if r.get("outcome") in ("WIN", "LOSS", "TIMEOUT")]
    wins = [r for r in scored if r.get("outcome") == "WIN"]
    losses = [r for r in scored if r.get("outcome") == "LOSS"]
    timeouts = [r for r in scored if r.get("outcome") == "TIMEOUT"]
    longs = [r for r in rows if r.get("prediction") == "LONG"]
    shorts = [r for r in rows if r.get("prediction") == "SHORT"]
    neutrals = [r for r in rows if r.get("prediction") == "NEUTRAL"]

    scored_sorted = sorted(scored, key=lambda r: r.get("prediction_timestamp", ""))
    win_streak, loss_streak = _streaks([str(r.get("outcome")) for r in scored_sorted])

    def side_rate(side: str) -> dict[str, Any]:
        side_scored = [r for r in scored if r.get("prediction") == side]
        w = sum(1 for r in side_scored if r.get("outcome") == "WIN")
        l = sum(1 for r in side_scored if r.get("outcome") == "LOSS")
        t = sum(1 for r in side_scored if r.get("outcome") == "TIMEOUT")
        n = len(side_scored)
        return {
            "n": n,
            "wins": w,
            "losses": l,
            "timeouts": t,
            "win_rate": _rate(w, n) if n >= 1 else None,
            "loss_rate": _rate(l, n) if n >= 1 else None,
            "timeout_rate": _rate(t, n) if n >= 1 else None,
            "insufficient_sample": n < MIN_SAMPLE_FOR_RATE,
        }

    return {
        "total_predictions": total,
        "evaluated_predictions": len(evaluated),
        "pending_predictions": len(pending),
        "long_predictions": len(longs),
        "short_predictions": len(shorts),
        "neutral_predictions": len(neutrals),
        "wins": len(wins),
        "losses": len(losses),
        "timeouts": len(timeouts),
        "win_rate": _rate(len(wins), len(scored)),
        "loss_rate": _rate(len(losses), len(scored)),
        "timeout_rate": _rate(len(timeouts), len(scored)),
        "average_signal_strength": _mean([float(r["signal_strength"]) for r in rows if r.get("signal_strength") is not None]),
        "average_return_pct": _mean([float(r["return_pct"]) for r in scored if r.get("return_pct") is not None]),
        "average_mfe_pct": _mean([float(r["mfe_pct"]) for r in scored if r.get("mfe_pct") is not None]),
        "average_mae_pct": _mean([float(r["mae_pct"]) for r in scored if r.get("mae_pct") is not None]),
        "longest_winning_streak": win_streak,
        "longest_losing_streak": loss_streak,
        "long": side_rate("LONG"),
        "short": side_rate("SHORT"),
        "scored_n": len(scored),
        "insufficient_sample": len(scored) < MIN_SAMPLE_FOR_RATE,
        "label": "Historical prediction performance (not future probability)",
    }


def compute_stats(
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    rows = list(storage.iter_all())
    if symbol:
        rows = [r for r in rows if r.get("symbol") == symbol]
    if timeframe:
        rows = [r for r in rows if r.get("timeframe") == timeframe]

    overall = _group_metrics(rows)

    by_asset: dict[str, Any] = {}
    for sym in sorted({r.get("symbol") for r in rows if r.get("symbol")}):
        by_asset[str(sym)] = _group_metrics([r for r in rows if r.get("symbol") == sym])

    by_tf: dict[str, Any] = {}
    for tf in sorted({r.get("timeframe") for r in rows if r.get("timeframe")}):
        by_tf[str(tf)] = _group_metrics([r for r in rows if r.get("timeframe") == tf])

    by_side: dict[str, Any] = {}
    for side in ("LONG", "SHORT", "NEUTRAL"):
        by_side[side] = _group_metrics([r for r in rows if r.get("prediction") == side])

    # Signal strength vs historical win rate (LONG/SHORT scored only)
    directional = [
        r
        for r in rows
        if r.get("evaluated")
        and r.get("prediction") in ("LONG", "SHORT")
        and r.get("outcome") in ("WIN", "LOSS", "TIMEOUT")
    ]
    strength_buckets: list[dict[str, Any]] = []
    for lo, hi, label in SIGNAL_STRENGTH_BUCKETS:
        bucket_rows = [r for r in directional if lo <= int(r.get("signal_strength", 0)) <= hi]
        wins = sum(1 for r in bucket_rows if r.get("outcome") == "WIN")
        n = len(bucket_rows)
        strength_buckets.append(
            {
                "bucket": label,
                "n": n,
                "wins": wins,
                "win_rate": _rate(wins, n),
                "insufficient_sample": n < MIN_SAMPLE_FOR_RATE,
                "note": "Historical success rate for this strength band — not calibrated probability",
            }
        )

    outcome_dist = {
        "WIN": sum(1 for r in rows if r.get("outcome") == "WIN"),
        "LOSS": sum(1 for r in rows if r.get("outcome") == "LOSS"),
        "TIMEOUT": sum(1 for r in rows if r.get("outcome") == "TIMEOUT"),
        "SKIPPED": sum(1 for r in rows if r.get("outcome") == "SKIPPED"),
        "PENDING": sum(1 for r in rows if not r.get("evaluated")),
    }

    return {
        "overall": overall,
        "by_asset": by_asset,
        "by_timeframe": by_tf,
        "by_prediction": by_side,
        "by_signal_strength": strength_buckets,
        "outcome_distribution": outcome_dist,
        "min_sample_for_rate": MIN_SAMPLE_FOR_RATE,
        "disclaimer": (
            "Signal strength is an analytical score. Historical win rates are measured "
            "outcomes for past snapshots, not a guarantee of future results."
        ),
    }
