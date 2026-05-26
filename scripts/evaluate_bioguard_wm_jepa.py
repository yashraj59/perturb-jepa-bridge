from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Read Phase 8 v2 BioGuard-WM-JEPA probe metrics.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics_path = args.output_dir / "metrics_eval.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    print(json.dumps(json.loads(metrics_path.read_text(encoding="utf-8")), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
