"""Generate Suite A (Jin-scale multi-device instances); deterministic.

    python scripts/generate_suite_a.py [--instances 4]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_tools.suite_a import write_suite  # noqa: E402
from dataset_tools.validate import validate_record  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", type=int, default=4, help="per configuration; i0 is the test instance")
    args = ap.parse_args()
    import json
    manifest = write_suite(ROOT / "datasets", args.instances)
    bad = 0
    for name, m in manifest.items():
        rec = json.loads((ROOT / "datasets" / m["path"]).read_text())
        errors, _ = validate_record(rec)
        bad += bool(errors)
        print(f"{'OK  ' if not errors else 'FAIL'} {name:16s} tasks={m['num_tasks']:4d} edges={m['num_edges']:4d} "
              f"devices={m['num_devices']:3d} edge servers={m['edge_servers']:2d} {m['role']}"
              + (f"  {errors[:2]}" if errors else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
