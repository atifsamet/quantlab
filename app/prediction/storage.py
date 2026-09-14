"""
Prediction snapshot persistence (JSONL).

Historical prediction fields are immutable after creation. Evaluation fields
may be updated once. Duplicate key: symbol + timeframe + candle_timestamp.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = ROOT / "data" / "predictions"
HISTORY_PATH = PREDICTIONS_DIR / "history.jsonl"

_lock = threading.RLock()
_loaded = False
_by_id: dict[str, dict[str, Any]] = {}
_dup_index: dict[str, str] = {}  # composite key -> id


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _composite_key(symbol: str, timeframe: str, candle_timestamp: str) -> str:
    return f"{symbol}|{timeframe}|{candle_timestamp}"


def ensure_store() -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        HISTORY_PATH.touch()


def _load_unlocked() -> None:
    global _loaded
    ensure_store()
    _by_id.clear()
    _dup_index.clear()
    if HISTORY_PATH.is_file() and HISTORY_PATH.stat().st_size > 0:
        with HISTORY_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = str(rec["id"])
                _by_id[rid] = rec
                _dup_index[_composite_key(rec["symbol"], rec["timeframe"], rec["candle_timestamp"])] = rid
    _loaded = True


def reload_store() -> None:
    with _lock:
        global _loaded
        _loaded = False
        _load_unlocked()


def _ensure_loaded() -> None:
    if not _loaded:
        _load_unlocked()


def _rewrite_unlocked() -> None:
    ensure_store()
    tmp = HISTORY_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in sorted(_by_id.values(), key=lambda r: r.get("prediction_timestamp", "")):
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    tmp.replace(HISTORY_PATH)


def get_by_id(prediction_id: str) -> dict[str, Any] | None:
    with _lock:
        _ensure_loaded()
        rec = _by_id.get(prediction_id)
        return None if rec is None else dict(rec)


def find_duplicate(symbol: str, timeframe: str, candle_timestamp: str) -> dict[str, Any] | None:
    with _lock:
        _ensure_loaded()
        rid = _dup_index.get(_composite_key(symbol, timeframe, candle_timestamp))
        if rid is None:
            return None
        return dict(_by_id[rid])


def append_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Insert a new prediction snapshot.

    Returns existing record if duplicate (symbol, timeframe, candle_timestamp).
    """
    with _lock:
        _ensure_loaded()
        key = _composite_key(record["symbol"], record["timeframe"], record["candle_timestamp"])
        if key in _dup_index:
            return dict(_by_id[_dup_index[key]])
        if "id" not in record or not record["id"]:
            record = {**record, "id": str(uuid.uuid4())}
        rid = str(record["id"])
        _by_id[rid] = record
        _dup_index[key] = rid
        ensure_store()
        with HISTORY_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return dict(record)


def update_evaluation(
    prediction_id: str,
    evaluation_fields: dict[str, Any],
    *,
    flush: bool = True,
) -> dict[str, Any] | None:
    """
    Update evaluation fields only. Never overwrites original prediction fields.
    Idempotent if already evaluated.
    """
    with _lock:
        _ensure_loaded()
        rec = _by_id.get(prediction_id)
        if rec is None:
            return None
        if rec.get("evaluated"):
            return dict(rec)
        allowed = {
            "evaluated",
            "outcome",
            "evaluation_timestamp",
            "evaluation_horizon",
            "bars_to_outcome",
            "exit_price",
            "return_pct",
            "mfe_pct",
            "mae_pct",
            "directional_correct",
            "evaluation_note",
        }
        updated = dict(rec)
        for k, v in evaluation_fields.items():
            if k in allowed:
                updated[k] = v
        updated["evaluated"] = True
        if "evaluation_timestamp" not in updated or not updated["evaluation_timestamp"]:
            updated["evaluation_timestamp"] = _utc_now_iso()
        _by_id[prediction_id] = updated
        if flush:
            _rewrite_unlocked()
        return dict(updated)


def flush_store() -> None:
    """Persist in-memory records to disk (full rewrite)."""
    with _lock:
        _ensure_loaded()
        _rewrite_unlocked()


def bulk_append(records: list[dict[str, Any]]) -> dict[str, int]:
    """
    Append many new snapshots efficiently. Dedupes by composite key.
    Returns counts: created, duplicates.
    """
    created = 0
    duplicates = 0
    with _lock:
        _ensure_loaded()
        ensure_store()
        with HISTORY_PATH.open("a", encoding="utf-8") as fh:
            for record in records:
                key = _composite_key(record["symbol"], record["timeframe"], record["candle_timestamp"])
                if key in _dup_index:
                    duplicates += 1
                    continue
                if "id" not in record or not record["id"]:
                    record = {**record, "id": str(uuid.uuid4())}
                rid = str(record["id"])
                _by_id[rid] = record
                _dup_index[key] = rid
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                created += 1
    return {"created": created, "duplicates": duplicates}


def list_records(
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    prediction: str | None = None,
    outcome: str | None = None,
    evaluated: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    with _lock:
        _ensure_loaded()
        rows: list[dict[str, Any]] = list(_by_id.values())
    if symbol:
        rows = [r for r in rows if r.get("symbol") == symbol]
    if timeframe:
        rows = [r for r in rows if r.get("timeframe") == timeframe]
    if prediction:
        rows = [r for r in rows if r.get("prediction") == prediction]
    if outcome:
        rows = [r for r in rows if r.get("outcome") == outcome]
    if evaluated is not None:
        rows = [r for r in rows if bool(r.get("evaluated")) is evaluated]
    rows.sort(key=lambda r: r.get("prediction_timestamp", ""), reverse=True)
    return [dict(r) for r in rows[offset : offset + max(0, limit)]]


def iter_all() -> Iterable[dict[str, Any]]:
    with _lock:
        _ensure_loaded()
        return [dict(r) for r in _by_id.values()]


def count_records() -> int:
    with _lock:
        _ensure_loaded()
        return len(_by_id)


def clear_all_for_tests() -> None:
    """Test helper — empties in-memory store and file."""
    with _lock:
        global _loaded
        ensure_store()
        HISTORY_PATH.write_text("", encoding="utf-8")
        _by_id.clear()
        _dup_index.clear()
        _loaded = True
