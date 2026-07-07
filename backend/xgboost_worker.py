"""Standalone XGBoost training worker — run in a subprocess to isolate native crashes."""

from __future__ import annotations

import json
import sys

from xgboost_trainer import train_xgboost


def main() -> int:
    payload = json.load(sys.stdin)
    data = payload["data"]
    config = payload["config"]
    try:
        result = train_xgboost(data, config)
        json.dump({"ok": True, "result": result}, sys.stdout)
        return 0
    except MemoryError as exc:
        json.dump(
            {"ok": False, "error": f"Out of memory: {exc}", "type": "MemoryError"},
            sys.stdout,
        )
        return 1
    except Exception as exc:
        json.dump(
            {"ok": False, "error": str(exc), "type": type(exc).__name__},
            sys.stdout,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
