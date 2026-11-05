"""Score the 1000-task investigation runs on the Suite B scale (normalisation from all Suite B policies)."""
import json, glob, os, re, sys
import numpy as np
sys.path.insert(0, r"C:\Users\mikasa\Downloads\gat_imoea_dqn")
from gat_imoea_dqn.metrics import pooled_reference, igd, hv_jin

E = r"C:\Users\mikasa\Downloads\gat_imoea_dqn\results\experiments"
CASES = ["cybershake_1000@x1", "epigenomics_1000@x20", "sipht_1000@x20"]


def load(exp, case, tag=None, smax=10):
    out = {}
    for f in glob.glob(os.path.join(E, exp, "runs", case + "__*.json")):
        n = os.path.basename(f)
        m = re.match(r"^.+?__(?P<p>[a-z0-9_]+?)(?:__t\d+)?__s(?P<s>\d+)\.json$", n)
        if int(m["s"]) >= smax:
            continue
        r = json.load(open(f))
        key = m["p"] + (tag or "")
        out.setdefault(key, []).append((np.asarray(r["front_lo"]), np.asarray(r["front_hi"]), r["seconds"]))
    return out


tot = {}
for c in CASES:
    base = load("suite_b", c)                                   # every Suite B policy, 10 seeds: fixes the scale
    ideal = np.min([lo.min(0) for v in base.values() for lo, _, _ in v], 0)
    nadir = np.max([hi.max(0) for v in base.values() for _, hi, _ in v], 0)
    span = nadir - ideal
    extra = {k: v for k, v in load("inv_act", c, smax=5).items() if k.startswith("final_")}
    extra.update(load("inv_g2000", c, "@G2000", smax=5))
    allp = {**base, **extra}
    norm = {k: [((lo - ideal) / span, (hi - ideal) / span, t) for lo, hi, t in v] for k, v in allp.items()}
    ref = pooled_reference([(lo + hi) / 2 for v in norm.values() for lo, hi, _ in v])
    show = ["final", "imoead_orig", "final_p09", "final_a23", "final_a12", "final@G2000", "imoead_orig@G2000"]
    print(f"\n{c}  (seeds 0-4; Suite B normalisation; reference pooled over all)")
    print(f"{'policy':20s} {'IGD':>8s} {'HV_jin':>8s} {'min cost':>9s} {'sec':>6s}")
    for k in show:
        v = norm[k][:5] if k in base else norm[k]
        I = [igd((lo + hi) / 2, ref) for lo, hi, _ in v]
        H = [hv_jin(lo, hi, [1.1] * 3) for lo, hi, _ in v]
        C = [((lo + hi) / 2)[:, 2].min() for lo, hi, _ in v]
        T = [t for _, _, t in v]
        tot.setdefault(k, []).append((np.mean(I), np.mean(H)))
        print(f"{k:20s} {np.mean(I):8.4f} {np.mean(H):8.4f} {np.mean(C):9.3f} {np.mean(T):6.0f}")
print("\nmean over the 3 cases (IGD, HV_jin):")
for k, v in tot.items():
    print(f"  {k:20s} {np.mean([a for a, _ in v]):.4f}  {np.mean([b for _, b in v]):.4f}")
