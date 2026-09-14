"""
Phase 10 — ML signal-filter research (offline only).

ML never places orders and is not connected to execution.
LIVE_TRADING_ENABLED must remain disabled.
"""

LABEL_DOC = """
Label definition (compatible with Phase-5 ATR risk model)
--------------------------------------------------------
For each candidate LONG/SHORT signal on candle i:

* Features use only bar-i (and earlier) information.
* Simulated entry is at candle i+1 open (same convention as the backtester).
* ATR for SL distance is taken from candle i (no look-ahead into i+1 indicators).
* stop = entry +/- ATR * atr_multiplier
* take = entry +/- |entry-stop| * risk_reward_ratio
* From the entry bar forward for at most ``horizon`` candles, scan high/low:
  - Same-candle SL+TP touch -> SL first (conservative, matches backtester)
  - label = 1 if TP is reached before SL
  - label = 0 if SL first, or neither hit within horizon (timeout = failure)

LONG and SHORT use the same success definition: TP before SL within horizon.

Overlap note: consecutive signals can have overlapping outcome windows; this is
documented and accepted for research. Walk-forward / chronological splits still
forbid future feature leakage.
"""
