"""Single entry point: recreate data, validate, train all tasks and regenerate every artifact.

Example:
    python run_all.py --student-id ug2423862@ines.ac.rw --seed 3862
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.train import run_training


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-id", required=True, help="institutional username, e.g. ug2423862@ines.ac.rw")
    parser.add_argument("--seed", type=int, required=True, help="last four registration digits (LLLL)")
    args = parser.parse_args()
    try:
        result = run_training(args.student_id, args.seed)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    root = result["paths"].root
    print("\nGenerated artifacts:")
    for p in sorted(Path(root, "artifacts").rglob("*")) + sorted(Path(root, "models").glob("*")):
        if p.is_file():
            print("  ", p.relative_to(root))
    print(f"\nDone. Verification code {result['verification_code']}  |  SHA-256 {result['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
