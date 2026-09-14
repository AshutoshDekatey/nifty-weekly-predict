from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nifty_intraday.pipeline import refresh_and_train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh Yahoo data, retrain five-session models, and capture Monday signals."
    )
    parser.add_argument(
        "--full", action="store_true", help="Ignore cache and download full history."
    )
    args = parser.parse_args()
    bundle, _, diagnostics = refresh_and_train(force_full=args.full)
    print(
        json.dumps(
            {
                "trained_at_utc": bundle["metadata"]["trained_at_utc"],
                "targets": bundle["metadata"]["targets"],
                "data_latest_nifty": bundle["metadata"]["data_latest_nifty"],
                "refresh": diagnostics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
