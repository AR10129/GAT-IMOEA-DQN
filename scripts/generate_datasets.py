"""Generate the interval DAG workflow datasets for GAT-IMOEA-DQN.

    python scripts/generate_datasets.py                    # everything in generation.yaml
    python scripts/generate_datasets.py --families sipht --sizes 100 200

Writes datasets/workflows/<family>/<family>_<size>.json and datasets/manifest.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import zlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_tools import pegasus, wfcommons_source  # noqa: E402
from dataset_tools.builder import build_record, dump_record  # noqa: E402
from dataset_tools.validate import validate_record  # noqa: E402


def make_seed(base: int, family: str, size: int, instance: int) -> int:
    return (base + zlib.crc32(f"{family}:{size}:{instance}".encode())) % (2 ** 31)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "datasets" / "generation.yaml"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "workflows"))
    ap.add_argument("--families", nargs="*")
    ap.add_argument("--sizes", nargs="*", type=int)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    families = args.families or list(cfg["families"])
    sizes = args.sizes or cfg["sizes"]
    conv = cfg["conversion"]
    wmin, wmax = conv["interval_half_width"]
    out = pathlib.Path(args.out)
    manifest_path = out.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"workflows": {}}

    failures = 0
    for fam in families:
        source = cfg["families"][fam]
        (out / fam).mkdir(parents=True, exist_ok=True)
        for size in sizes:
            for inst in range(cfg.get("instances_per_size", 1)):
                seed = make_seed(cfg["base_seed"], fam, size, inst)
                name = f"{fam}_{size}" + (f"_i{inst}" if cfg.get("instances_per_size", 1) > 1 else "")
                t0 = time.time()
                try:
                    if source == "pegasus":
                        raw = pegasus.GENERATORS[fam](size, seed)
                    else:
                        raw = wfcommons_source.generate(fam, size, seed)
                    rec = build_record(raw, name, seed, conv["reference_ghz"], wmin, wmax)
                    rec["requested_tasks"] = size
                    errors, warnings = validate_record(rec)
                except Exception as exc:  # keep going, report at the end
                    print(f"FAIL {name}: {exc}")
                    failures += 1
                    continue
                if errors:
                    print(f"INVALID {name}: {errors[:3]}")
                    failures += 1
                    continue
                path = out / fam / f"{name}.json"
                dump_record(rec, path)
                s = rec["stats"]
                manifest["workflows"][name] = {
                    "path": str(path.relative_to(out.parent)).replace("\\", "/"),
                    "family": fam, "source": raw.generator, "seed": seed,
                    "requested_tasks": size, "num_tasks": rec["num_tasks"], "num_edges": rec["num_edges"],
                    "depth": s["depth"], "components": s["components"],
                    "entries": len(rec["entry_tasks"]), "exits": len(rec["exit_tasks"]),
                }
                warn = f"  warnings: {warnings}" if warnings else ""
                print(f"ok   {name:20s} K={rec['num_tasks']:5d} E={rec['num_edges']:5d} depth={s['depth']:2d} "
                      f"comp={s['components']:2d} entry={len(rec['entry_tasks']):4d} exit={len(rec['exit_tasks']):3d} "
                      f"({time.time() - t0:.1f}s){warn}")

    manifest["workflows"] = dict(sorted(manifest["workflows"].items()))
    manifest["generation_config"] = cfg
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest['workflows'])} workflows in manifest, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
