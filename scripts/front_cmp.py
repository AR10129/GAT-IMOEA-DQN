"""final vs imoead_orig on Suite B: where on the front does each win?"""
import json, glob, re, sys, os
import numpy as np
sys.path.insert(0, r"C:\Users\mikasa\Downloads\gat_imoea_dqn")
from gat_imoea_dqn.metrics import pooled_reference, nondominated, hv_jin

D = r"C:\Users\mikasa\Downloads\gat_imoea_dqn\results\experiments\suite_b\runs"
size = sys.argv[1] if len(sys.argv) > 1 else "1000"
A, B = "final", "imoead_orig"
files = glob.glob(os.path.join(D, "*.json"))
wfs = sorted({os.path.basename(f).split("__")[0] for f in files if f"_{size}@" in os.path.basename(f)})
print(f"{'case':22s} {'refshare A/B':>12s} | ext lo A vs B (per obj, norm)        | hi-end | HV A,B | A dom by B, B dom by A")
for wf in wfs:
    rs = []
    for f in files:
        n = os.path.basename(f)
        if not n.startswith(wf + "__"):
            continue
        r = json.load(open(f))
        pol = n.split("__")[1]
        s = int(re.search(r"__s(\d+)\.json", n).group(1))
        if s >= 10:
            continue
        rs.append((pol, np.asarray(r["front_lo"]), np.asarray(r["front_hi"])))
    ideal = np.min([lo.min(0) for _, lo, _ in rs], 0)
    nadir = np.max([hi.max(0) for _, _, hi in rs], 0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1)
    nm = [(p, (lo - ideal) / span, (hi - ideal) / span) for p, lo, hi in rs]
    ref = pooled_reference([(lo + hi) / 2 for _, lo, hi in nm])
    out = {}
    for P in (A, B):
        mids = np.vstack([(lo + hi) / 2 for p, lo, hi in nm if p == P])
        hvs = [hv_jin(lo, hi, [1.1] * 3) for p, lo, hi in nm if p == P]
        out[P] = dict(mid=mids, hv=np.mean(hvs))
    # share of the reference front each policy's points hit (within 1e-9)
    def share(M):
        d = np.min(np.linalg.norm(ref[:, None, :] - M[None, :, :], axis=2), axis=1)
        return np.mean(d < 1e-9)
    ua = nondominated(out[A]["mid"]); ub = nondominated(out[B]["mid"])
    def dom_frac(X, Y):  # fraction of X dominated by some point of Y
        le = np.all(Y[None, :, :] <= X[:, None, :], axis=2) & np.any(Y[None, :, :] < X[:, None, :], axis=2)
        return np.mean(le.any(axis=1))
    mnA, mnB = out[A]["mid"].min(0), out[B]["mid"].min(0)
    mxA, mxB = ua.max(0), ub.max(0)
    print(f"{wf:22s} {share(out[A]['mid']):.2f}/{share(out[B]['mid']):.2f} | min A {np.round(mnA,3)} B {np.round(mnB,3)} | max A {np.round(mxA,2)} B {np.round(mxB,2)} | "
          f"{out[A]['hv']:.4f},{out[B]['hv']:.4f} | {dom_frac(ua, ub):.2f}, {dom_frac(ub, ua):.2f}")
