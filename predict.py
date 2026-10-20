"""Command-line prediction: python predict.py --json samples/valid.json"""
from __future__ import annotations

import argparse
import json
import sys

from src.inference import ArtifactsMissingError, predict_record
from src.validation import RecordValidationError


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict yield, risk and cluster for one farm record.")
    parser.add_argument("--json", required=True, help="path to a JSON file holding one farm record")
    args = parser.parse_args()
    try:
        with open(args.json, encoding="utf-8") as fh:
            record = json.load(fh)
    except FileNotFoundError:
        print(json.dumps({"error": f"File not found: {args.json}"}), file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON: {exc}"}), file=sys.stderr)
        return 2
    try:
        result = predict_record(record)
    except RecordValidationError as exc:
        print(json.dumps({"error": "Invalid input", "details": exc.errors}, indent=2), file=sys.stderr)
        return 2
    except ArtifactsMissingError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
