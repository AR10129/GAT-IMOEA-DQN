"""R1: build every paper table (LaTeX) and collect every paper figure, in one command.

  python scripts/build_paper.py              # score what is not cached yet, then write paper/
  python scripts/build_paper.py --rescore    # rescore every table from the run files

Scoring. Each scored table lists its experiments and policies. Per instance (per instance and
level for E6), one normalisation (ideal / nadir over the final interval fronts of the table's
own policies) and one IGD reference front (non-dominated midpoints of the same pool), exactly as
`scripts/stats.py` does for E3. Agent runs with several training seeds are averaged per seed.
`extra` policies (the heuristics) are scored on the pool's normalisation and reference without
being part of the pool, so they do not stretch it. Scores are cached in
results/stats/paper_<table>_runs.csv.

Statistics. Wilcoxon rank-sum against `final`, two-sided, Holm over the other policies of the
table per instance and metric, alpha 0.05: + better than final, - worse, = no significant
difference. Friedman over instances with the Nemenyi critical difference (Demsar 2006).

Output: paper/tables/*.tex (booktabs; \\ours is the method's name), paper/figures/*.pdf and
paper/preview.tex, a standalone document that inputs every table and figure. The figures are
made by the scripts named in FIGURES; this script copies them.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import pathlib
import re
import shutil
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

EXP = ROOT / "results" / "experiments"
STATS = ROOT / "results" / "stats"
PAPER_DIR = ROOT / "paper"
RUN_NAME = re.compile(r"^(?P<wf>.+?)__(?P<pol>[A-Za-z0-9_]+?)(?:__t(?P<t>\d+))?__s(?P<s>\d+)\.json$")
METRICS = {"IGD": True, "HV_jin": False, "Imprecision": True, "HV_hi": False}       # lower is better?

PAPER = ["final", "imoead_orig", "imoead_awn", "imea_ied", "ip_moea_hv", "nsga2_mid", "nsga2_int",
         "nsga3_mid", "fixed_fav", "imoea_dqn"]
LABEL = {"final": r"\ours{}", "imoead_orig": "IMOEA/D", "imoead_awn": "IMOEA/D-AWN", "imea_ied": "IMEA-IED",
         "ip_moea_hv": "IP-MOEA", "nsga2_mid": "NSGA-II (mid)", "nsga2_int": "NSGA-II (int)",
         "nsga3_mid": "NSGA-III", "fixed_fav": "Jin backbone", "imoea_dqn": "IMOEA-DQN",
         "trained_stable": "IMOEA-DQN (offline GAT)", "ip_moea": "IP-MOEA (mid crowding)",
         "heuristics": "Heuristics"}
SHORT = {"final": r"\ours{}", "imoead_orig": "IMOEA/D", "imoead_awn": "AWN", "imea_ied": "IED",
         "ip_moea_hv": "IP-MOEA", "nsga2_mid": "NSGA-II\\textsubscript{m}", "nsga2_int": "NSGA-II\\textsubscript{i}",
         "nsga3_mid": "NSGA-III", "fixed_fav": "Jin bb.", "imoea_dqn": "IMOEA-DQN"}

SUITE_A = ["suite_a_main", "suite_a_e2", "rerun_ipmoea_a", "rerun_imoeadqn_a"]
SUITE_B = ["suite_b", "suite_b_e2", "rerun_ipmoea_b", "rerun_imoeadqn_b"]
E4 = ["final", "imoead_orig", "nsga2_mid", "fixed_fav"]             # the E6 / scalability policy set

ABL_ROWS = [("Build-up", "jin", "Jin et al.\\ backbone ($R_{IN}$, random ties, DAEC mating)"),
            ("Build-up", "ptch", "+ Tchebycheff Rank C"),
            ("Build-up", "ptchpicd", "+ ICD tie-break"),
            ("Build-up", "full", "+ neighbourhood mating ($T=20$, $p_{local}=0.9$)"),
            ("Build-up", "final", "\\ours{} ($T=10$, $p_{local}=1.0$)"),
            ("Leave one out (from $T=20$, $p_{local}=0.9$)", "mtch", "without Tchebycheff Rank C ($R_{IN}$)"),
            ("Leave one out (from $T=20$, $p_{local}=0.9$)", "micd", "without ICD tie-break"),
            ("Leave one out (from $T=20$, $p_{local}=0.9$)", "ptchpicd", "without neighbourhood mating"),
            ("Sensitivity (from $T=20$, $p_{local}=0.9$)", "T10", "$T=10$"),
            ("Sensitivity (from $T=20$, $p_{local}=0.9$)", "T40", "$T=40$"),
            ("Sensitivity (from $T=20$, $p_{local}=0.9$)", "p05", "$p_{local}=0.5$"),
            ("Sensitivity (from $T=20$, $p_{local}=0.9$)", "p10", "$p_{local}=1.0$"),
            ("Sensitivity (from $T=20$, $p_{local}=0.9$)", "w53", "$N/2$ weight vectors"),
            ("Sensitivity (from $T=20$, $p_{local}=0.9$)", "w210", "$2N$ weight vectors"),
            ("References", "conv", "action $(1, 0, 0)$ instead of $(5/6, 1/6, 0)$"),
            ("References", "imoead_orig", "IMOEA/D")]
E6_LEVELS = [("e6_base", "Base (Jin et al.\\ scenario, data $\\times 20$)"), ("e6_energy_all", "Energy of all tiers"),
             ("e6_width05", "Interval half-widths $\\times 0.5$"), ("e6_width2", "Interval half-widths $\\times 2$"),
             ("e6_bw05", "Wireless bandwidth $\\times 0.5$"), ("e6_bw2", "Wireless bandwidth $\\times 2$"),
             ("e6_edges1", "1 edge server"), ("e6_edges6", "6 edge servers"),
             ("e6_edgequeue", "Edge servers queue tasks")]

# name -> list of (group, instance, experiments, pool policies, extra policies, rename, max seeds)
def scored_tables():
    T = {"suite_a": [("", None, SUITE_A, PAPER, ["heuristics"], {}, None)],
         "suite_b": [("", None, SUITE_B, PAPER, ["heuristics"], {}, None)],
         "ablation": [("", None, ["ablation", "suite_a_main"], list(dict.fromkeys(r[1] for r in ABL_ROWS)), [], {}, 10)],
         "sensitivity": [(lvl, None, [lvl], E4, [], {}, None) for lvl, _ in E6_LEVELS],
         "scalability": [("", None, ["scale_2000"], E4, [], {}, None)],
         "learners": [("", w, ["dqn_fix_test", "suite_a_main", "rerun_imoeadqn_a"],
                       [r[1] for r in LEARN_ROWS if r[1] != "imoea_dqn"], ["imoea_dqn"], {}, 4)   # far off; not in the pool
                      for w in LEARN_CASES],
         "gat": [(g, None, [e], [a, b], [], {}, None) for g, e, a, b, _ in GAT_PAIRS]}
    return T


# D1 / C4: learned control of (alpha, beta, gamma). The learners and their references run on the
# Tchebycheff backbone with the earlier mating default (T = 20, p_local = 0.9); final is the frozen method.
LEARN_CASES = ["small_20_i0@x1", "basic_60_i0@x20", "medium_100_i0@x1", "medium_250_i0@x20"]
LEARN_ROWS = [("Fixed actions", "tch_fixed_fav", "best fixed action $(5/6, 1/6, 0)$"),
              ("Fixed actions", "tch_conv", "action $(1, 0, 0)$"),
              ("Fixed actions", "tch_random_conv", r"random action each step, among the 10 with $\alpha \ge 1/2$"),
              ("Learned control (DQN)", "tch_trained_stable", "every generation, PIBI reward, 28 actions, GAT"),
              ("Learned control (DQN)", "tch_trained_nogat_stable", "every generation, PIBI reward, 28 actions, no GAT"),
              ("Learned control (DQN)", "tch_trained_blk", "every 50 generations, HV-gain reward, 28 actions, GAT"),
              ("Learned control (DQN)", "tch_trained_nogat_blk", "every 50 generations, HV-gain reward, 28 actions, no GAT"),
              ("Learned control (DQN)", "tch_trained_l8", "every 25 generations, log-HV reward, 10 actions, GAT"),
              ("Learned control (DQN)", "tch_trained_nogat_l8", "every 25 generations, log-HV reward, 10 actions, no GAT"),
              ("For comparison", "final", r"\ours{} ($T = 10$, $p_{local} = 1.0$, fixed action)"),
              ("For comparison", "imoea_dqn", r"IMOEA-DQN (Jin et al.\ backbone, online DQN)"),
              ("For comparison", "imoead_orig", "IMOEA/D")]
GAT_PAIRS = [("jin", "suite_a_main", "trained_stable", "trained_nogat_stable",
              r"Jin et al.\ backbone, every generation, PIBI reward"),
             ("tch", "suite_a_tch", "tch_trained_stable", "tch_trained_nogat_stable",
              "Tchebycheff backbone, every generation, PIBI reward"),
             ("blk", "dqn_fix_test", "tch_trained_blk", "tch_trained_nogat_blk",
              "Tchebycheff backbone, every 50 generations, HV-gain reward"),
             ("l8", "dqn_fix_test", "tch_trained_l8", "tch_trained_nogat_l8",
              "Tchebycheff backbone, every 25 generations, log-HV reward, 10 actions")]


FIGURES = {"e5_convergence": "scripts/convergence.py (E5)", "a3_stall": "scripts/a3_tiers.py --stall (A3)",
           "a3_tradeoff": "scripts/a3_tiers.py (A3)", "d1_reward_audit": "scripts/d1_learning.py (D1)",
           "d1_training": "scripts/d1_learning.py (D1)"}


# ---------------------------------------------------------------- scoring

def _instances(exps):
    names = set()
    for e in exps:
        for p in (EXP / e / "runs").glob("*.json"):
            m = RUN_NAME.match(p.name)
            if m:
                names.add(m["wf"])
    return sorted(names, key=natural)


def natural(w):
    return re.sub(r"\d+", "", w), [int(x) for x in re.findall(r"\d+", w)]


def _load_one(exps, inst, policies, max_seeds):
    d = {}
    for e in exps:
        for p in sorted((EXP / e / "runs").glob(f"{inst}__*.json")):
            m = RUN_NAME.match(p.name)
            if not m or m["wf"] != inst or m["pol"] not in policies:
                continue
            s = int(m["s"])
            if max_seeds is not None and s >= max_seeds:
                continue
            key = (m["pol"], s, int(m["t"]) if m["t"] is not None else None)
            if key not in d:
                r = json.loads(p.read_text())
                d[key] = (np.asarray(r["front_lo"], float), np.asarray(r["front_hi"], float))
    return d


def _score_job(args):
    table, group, inst, exps, pool_pols, extra, max_seeds = args
    from gat_imoea_dqn.metrics import hv_jin, hypervolume_3d, igd, imprecision, pooled_reference
    runs = _load_one(exps, inst, set(pool_pols) | set(extra), max_seeds)
    pool = {k: v for k, v in runs.items() if k[0] in pool_pols}
    if not pool:                    # e.g. old pilot instances in results/experiments/ablation
        return []
    ideal = np.min([lo.min(0) for lo, _ in pool.values()], 0)
    nadir = np.max([hi.max(0) for _, hi in pool.values()], 0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    norm = {k: ((lo - ideal) / span, (hi - ideal) / span) for k, (lo, hi) in runs.items()}
    ref = pooled_reference([(lo + hi) / 2 for k, (lo, hi) in norm.items() if k[0] in pool_pols])
    per = {}
    for k, (lo, hi) in norm.items():
        v = (igd((lo + hi) / 2, ref), hv_jin(lo, hi, [1.1] * 3), imprecision(lo, hi), hypervolume_3d(hi, [1.1] * 3))
        per.setdefault((k[0], k[1]), []).append(v)
    return [[table, group, inst, pol, s, *(f"{x:.6g}" for x in np.mean(v, axis=0))] for (pol, s), v in sorted(per.items())]


def score(rescore: bool, workers: int):
    jobs, todo = [], []
    for name, parts in scored_tables().items():
        out = STATS / f"paper_{name}_runs.csv"
        if out.exists() and not rescore:
            continue
        todo.append(name)
        for group, inst, exps, pols, extra, _, ms in parts:
            for w in ([inst] if inst else _instances(exps[:1])):        # instances of the first experiment
                jobs.append((name, group, w, exps, pols, extra, ms))
    if not jobs:
        return
    print(f"scoring {len(jobs)} instance groups for {', '.join(todo)}", flush=True)
    rows = {n: [] for n in todo}
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_score_job, jobs), 1):
            for r in res:
                rows[r[0]].append(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)
    for n, rr in rows.items():
        rr.sort(key=lambda r: (r[1], natural(r[2]), r[3], int(r[4])))
        with open(STATS / f"paper_{n}_runs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["table", "group", "instance", "policy", "seed", *METRICS])
            w.writerows(rr)


def read_scores(name):
    """{(group, instance): {policy: {metric: np.array over seeds}}}"""
    out = {}
    with open(STATS / f"paper_{name}_runs.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = out.setdefault((r["group"], r["instance"]), {}).setdefault(r["policy"], {m: [] for m in METRICS})
            for m in METRICS:
                d[m].append(float(r[m]))
    return {k: {p: {m: np.asarray(v) for m, v in d.items()} for p, d in pp.items()} for k, pp in out.items()}


# ---------------------------------------------------------------- statistics

def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty_like(p), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def symbols(vals, others, metric, ref="final", alpha=0.05):
    """{policy: '+', '-' or '='} against ref, Wilcoxon rank-sum, Holm over `others`."""
    from scipy.stats import mannwhitneyu
    lb = METRICS[metric]
    r = vals[ref][metric]
    ps, better = [], []
    for p in others:
        x = vals[p][metric]
        try:
            ps.append(mannwhitneyu(x, r, alternative="two-sided").pvalue)
        except ValueError:
            ps.append(1.0)
        better.append(x.mean() < r.mean() if lb else x.mean() > r.mean())
    return {p: "=" if q >= alpha else ("+" if b else "-") for p, q, b in zip(others, holm(ps), better)}


Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def friedman(S, pols, metric):
    from scipy.stats import friedmanchisquare, rankdata
    keys = sorted(S, key=lambda k: natural(k[1]))
    M = np.array([[S[k][p][metric].mean() for p in pols] for k in keys])
    R = np.array([rankdata(row if METRICS[metric] else -row) for row in M])
    chi, pv = friedmanchisquare(*M.T)
    k, n = len(pols), len(keys)
    return R.mean(0), chi, pv, Q05[k] * np.sqrt(k * (k + 1) / (6 * n))


# ---------------------------------------------------------------- LaTeX helpers

def esc(s):
    return s.replace("_", r"\_").replace("%", r"\%")


def inst_name(w):
    m = re.match(r"^(.*?)(?:_i\d+)?@x(\d+)$", w)
    base, mult = (m[1], m[2]) if m else (w, None)
    base = base.replace("_", "-")
    return f"{base} $\\times{mult}$" if mult else base


def num(x, d=3):
    return f"{x:.{d}f}"


def sym_tex(s):
    return {"+": r"$^{+}$", "-": r"$^{-}$", "=": r"$^{\approx}$"}.get(s, "")


def write(name, body):
    p = PAPER_DIR / "tables" / f"{name}.tex"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("% generated by scripts/build_paper.py; do not edit by hand\n" + body + "\n", encoding="utf-8")
    print("wrote", p.relative_to(ROOT))


def table(caption, label, colspec, head, rows, size=r"\scriptsize", sideways=False, note=None):
    env = "sidewaystable" if sideways else "table"
    L = [f"\\begin{{{env}}}[!htbp]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *head, "\\midrule", *rows,
         "\\bottomrule", "\\end{tabular}"]
    if note:
        L += ["\\par\\smallskip", f"\\parbox{{\\linewidth}}{{\\footnotesize {note}}}"]
    L.append(f"\\end{{{env}}}")
    return "\n".join(L)


SYM_NOTE = (r"$^{+}$ / $^{-}$ / $^{\approx}$: significantly better / worse than \ours{} / no significant "
            r"difference (Wilcoxon rank-sum, two-sided, Holm-corrected over the compared methods, $\alpha = 0.05$). "
            r"Best mean per row in bold.")


# ---------------------------------------------------------------- tables

def t_suite(suite, S, metric, scale, digits):
    keys = sorted(S, key=lambda k: natural(k[1]))
    others = [p for p in PAPER if p != "final"]
    lb = METRICS[metric]
    rows, cnt = [], {p: {"+": 0, "-": 0, "=": 0} for p in others}
    for k in keys:
        V = S[k]
        sy = symbols(V, others, metric)
        means = {p: V[p][metric].mean() * scale for p in PAPER}
        best = min(means.values()) if lb else max(means.values())
        cells = []
        for p in PAPER:
            m, sd = means[p], V[p][metric].std(ddof=1) * scale
            c = f"{m:.{digits}f}"
            if np.isclose(m, best):
                c = f"\\textbf{{{c}}}"
            c += f" ({sd:.{digits}f})"
            if p != "final":
                c += sym_tex(sy[p])
                cnt[p][sy[p]] += 1
            cells.append(c)
        rows.append(inst_name(k[1]) + " & " + " & ".join(cells) + r" \\")
    rows.append("\\midrule")
    rows.append("$+/-/\\approx$ & -- & " + " & ".join(f"{cnt[p]['+']}/{cnt[p]['-']}/{cnt[p]['=']}" for p in others) + r" \\")
    mname = {"IGD": "IGD", "HV_jin": "HV", "Imprecision": "Imprecision", "HV_hi": "HV of the upper bounds"}[metric]
    sc = "" if scale == 1 else f" ($\\times 10^{{-{int(np.log10(scale))}}}$)"
    n = len(keys)
    cap = (f"Suite {suite}: {mname}{sc} per instance, mean (standard deviation) over the runs "
           f"({'lower' if lb else 'higher'} is better). {n} instances; \\ours{{}}, IMOEA/D and the Jin backbone "
           f"30 runs, the other methods 10.")
    head = ["Instance & " + " & ".join(SHORT[p] for p in PAPER) + r" \\"]
    return table(cap, f"tab:suite{suite.lower()}_{metric.lower()}", "l" + "c" * len(PAPER), head, rows,
                 size=r"\tiny", sideways=True, note=SYM_NOTE + (" Imprecision is per solution and does not account for front size."
                                                               if metric == "Imprecision" else ""))


def t_friedman(SA, SB):
    rows = []
    for suite, S in (("A", SA), ("B", SB)):
        for metric in ("IGD", "HV_jin", "Imprecision"):
            R, chi, pv, cd = friedman(S, PAPER, metric)
            best = R.min()
            cells = [f"\\textbf{{{r:.2f}}}" if np.isclose(r, best) else f"{r:.2f}" for r in R]
            mn = {"IGD": "IGD", "HV_jin": "HV", "Imprecision": "Imprec."}[metric]
            rows.append(f"{suite} & {mn} & " + " & ".join(cells) + f" & {cd:.2f} & {pv:.1e}" + r" \\")
        if suite == "A":
            rows.append("\\midrule")
    head = ["Suite & Metric & " + " & ".join(SHORT[p] for p in PAPER) + r" & CD & $p$ \\"]
    cap = ("Mean Friedman ranks over the instances (lower is better), Nemenyi critical difference (CD, "
           "$\\alpha = 0.05$) and the Friedman test's $p$-value. Suite A has 12 instances, Suite B 45.")
    return table(cap, "tab:friedman", "ll" + "c" * len(PAPER) + "cc", head, rows, size=r"\scriptsize")


def t_heuristics(SA, SB):
    rows = []
    for suite, S in (("A", SA), ("B", SB)):
        groups = {}
        for k, V in S.items():
            if "heuristics" not in V:
                continue
            size = re.search(r"_(\d+)", k[1])[1]
            g = ("all" if suite == "A" else f"{size} tasks")
            for gg in ([g, "all"] if suite == "B" else [g]):
                d = groups.setdefault(gg, {"f": [], "h": [], "fh": [], "hh": [], "n": 0})
                d["f"].append(V["final"]["IGD"].mean()), d["h"].append(V["heuristics"]["IGD"].mean())
                d["fh"].append(V["final"]["HV_jin"].mean()), d["hh"].append(V["heuristics"]["HV_jin"].mean())
                d["n"] += 1
        for g in sorted(groups, key=lambda x: (x == "all", natural(x))):
            d = groups[g]
            rows.append(f"{suite} & {g} & {d['n']} & {np.mean(d['f']):.4f} & {np.mean(d['h']):.4f} & "
                        f"{np.mean(d['h']) / np.mean(d['f']):.1f} & {np.mean(d['fh']):.3f} & {np.mean(d['hh']):.3f}" + r" \\")
        if suite == "A":
            rows.append("\\midrule")
    head = [r"Suite & Instances & $n$ & IGD \ours{} & IGD heuristics & ratio & HV \ours{} & HV heuristics \\"]
    cap = ("Reference heuristics (all-local, all-edge, all-cloud and three HEFT-style list schedulers; their "
           "non-dominated set) against \\ours{}, scored on the normalisation and reference front of the main tables "
           "without being part of them. Means over instances.")
    return table(cap, "tab:heuristics", "llcccccc", head, rows, size=r"\scriptsize")


def t_robustness():
    agg = {}
    with open(STATS / "robustness.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = agg.setdefault((r["suite"], r["pool"]), {"IGD": [], "HV": [], "IGD1000": []})
            d["IGD"].append(r["IGD_sym"]), d["HV"].append(r["HV_sym"])
            if re.search(r"_1000@", r["instance"]):
                d["IGD1000"].append(r["IGD_sym"])
    names = {"paper10": "All ten methods (main tables)", "union12": "Ten methods + two superseded baselines",
             "strong5": "\\ours{}, IMOEA/D, IMOEA/D-AWN, NSGA-II (mid / int)", "pair2": "\\ours{} and IMOEA/D only"}
    c = lambda v: f"{v.count('+')}/{v.count('-')}/{v.count('=')}"                       # noqa: E731
    rows = []
    for suite in ("A", "B"):
        for pool in names:
            d = agg[(suite, pool)]
            rows.append(f"{suite} & {names[pool]} & {c(d['IGD'])} & {c(d['HV'])} & "
                        f"{c(d['IGD1000']) if d['IGD1000'] else '--'}" + r" \\")
        if suite == "A":
            rows.append("\\midrule")
    head = [r"Suite & Pool defining normalisation and reference front & IGD & HV & IGD, 1000 tasks \\"]
    cap = ("Robustness of the comparison with IMOEA/D to the normalisation: \\ours{} better / worse / not "
           "significantly different (Wilcoxon rank-sum, $\\alpha = 0.05$, 30 + 30 runs), when the per-instance "
           "normalisation and the IGD reference front are taken from different method pools.")
    return table(cap, "tab:robustness", "llccc", head, rows, size=r"\scriptsize")


def t_ablation(S):
    keys = sorted(S, key=lambda k: natural(k[1]))
    pols = list(dict.fromkeys(r[1] for r in ABL_ROWS))
    rows, last = [], None
    for group, p, lab in ABL_ROWS:
        ref = "final" if group == "Build-up" or p == "imoead_orig" else "full"
        if group != last:
            if last is not None:
                rows.append("\\midrule")
            rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{group}}}}}" + r" \\")
            last = group
        igd = np.mean([S[k][p]["IGD"].mean() for k in keys])
        hv = np.mean([S[k][p]["HV_jin"].mean() for k in keys])
        cells = []
        for m in ("IGD", "HV_jin"):
            if p == ref:
                cells.append("--")
                continue
            s = [symbols({ref: S[k][ref], p: S[k][p]}, [p], m, ref=ref)[p] for k in keys]
            cells.append(f"{s.count('+')}/{s.count('-')}/{s.count('=')}")
        refl = {"final": "\\ours{}", "full": "$T=20$"}[ref]
        rows.append(f"\\quad {lab} & {igd * 100:.2f} & {hv:.3f} & {cells[0]} & {cells[1]} & {refl}" + r" \\")
    head = [r"Variant & IGD ($\times10^{-2}$) & HV & IGD $+/-/\approx$ & HV $+/-/\approx$ & against \\"]
    cap = ("Ablation on the 12 Suite A instances, action $(5/6, 1/6, 0)$ unless stated, 10 runs each. Means over "
           "instances; $+/-/\\approx$ counts instances where the variant is significantly better / worse / not "
           "different from the reference in the last column (Wilcoxon rank-sum, $\\alpha = 0.05$). One normalisation "
           "per instance over all variants.")
    return table(cap, "tab:ablation", "lccccc", head, rows, size=r"\scriptsize")


def t_sensitivity(S):
    rows = []
    for lvl, lab in E6_LEVELS:
        keys = sorted([k for k in S if k[0] == lvl], key=lambda k: natural(k[1]))
        means = [np.mean([S[k][p]["IGD"].mean() for k in keys]) * 100 for p in E4]
        cells = [f"\\textbf{{{m:.2f}}}" if np.isclose(m, min(means)) else f"{m:.2f}" for m in means]
        rec = []
        for m in ("IGD", "HV_jin"):
            s = [symbols(S[k], ["imoead_orig", "nsga2_mid", "fixed_fav"], m)["imoead_orig"] for k in keys]
            rec.append(f"{s.count('+')}/{s.count('-')}/{s.count('=')}")
        rows.append(f"{lab} & " + " & ".join(cells) + f" & {rec[0]} & {rec[1]}" + r" \\")
    head = [r"Setting & \multicolumn{4}{c}{Mean IGD ($\times10^{-2}$)} & \multicolumn{2}{c}{IMOEA/D vs \ours{}} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
            "& " + " & ".join(SHORT[p] for p in E4) + r" & IGD & HV \\"]
    cap = ("Sensitivity on the five 400-task workflows (data $\\times 20$, Jin et al.'s scenario), 10 runs each. "
           "Each row changes one setting of the base case. The last two columns count instances where IMOEA/D is "
           "significantly better / worse / not different from \\ours{} (Wilcoxon rank-sum, Holm over the three "
           "compared methods, $\\alpha = 0.05$).")
    return table(cap, "tab:sensitivity", "lcccccc", head, rows, size=r"\scriptsize")


def t_scalability(S):
    keys = sorted(S, key=lambda k: natural(k[1]))
    others = [p for p in E4 if p != "final"]
    rows, cnt = [], {m: {p: {"+": 0, "-": 0, "=": 0} for p in others} for m in ("IGD", "HV_jin")}
    for k in keys:
        cells = []
        for m, sc, dg in (("IGD", 100, 2), ("HV_jin", 1, 3)):
            sy = symbols(S[k], others, m)
            means = {p: S[k][p][m].mean() * sc for p in E4}
            best = min(means.values()) if METRICS[m] else max(means.values())
            for p in E4:
                c = f"{means[p]:.{dg}f}"
                c = f"\\textbf{{{c}}}" if np.isclose(means[p], best) else c
                if p != "final":
                    c += sym_tex(sy[p])
                    cnt[m][p][sy[p]] += 1
                cells.append(c)
        rows.append(inst_name(k[1]) + " & " + " & ".join(cells) + r" \\")
    rows.append("\\midrule")
    rows.append("$+/-/\\approx$ & " + " & ".join(
        ["--"] + [f"{cnt['IGD'][p]['+']}/{cnt['IGD'][p]['-']}/{cnt['IGD'][p]['=']}" for p in others] +
        ["--"] + [f"{cnt['HV_jin'][p]['+']}/{cnt['HV_jin'][p]['-']}/{cnt['HV_jin'][p]['=']}" for p in others]) + r" \\")
    head = [r"Instance & \multicolumn{4}{c}{IGD ($\times10^{-2}$)} & \multicolumn{4}{c}{HV} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
            "& " + " & ".join(SHORT[p] for p in E4) + " & " + " & ".join(SHORT[p] for p in E4) + r" \\"]
    cap = "Scalability: 2000-task workflows, Jin et al.'s scenario, 10 runs each; mean over runs."
    return table(cap, "tab:scalability", "l" + "c" * 8, head, rows, size=r"\scriptsize", note=SYM_NOTE)


TIMING_POLS = ["final", "imoead_orig", "imoead_awn", "imea_ied", "ip_moea_hv", "ip_moea", "nsga2_mid", "nsga2_int",
               "nsga3_mid", "fixed_fav", "imoea_dqn", "trained_stable"]


def _seconds(exps):
    sec = {}
    for e in exps:
        for p in (EXP / e / "runs").glob("*.json"):
            m = RUN_NAME.match(p.name)
            if m:
                sec.setdefault((m["wf"], m["pol"]), []).append(json.loads(p.read_text())["seconds"])
    return sec


def t_timing():
    """B4 timing (timing_a / timing_b); the corrected baselines come from timing2_a / timing2_b, which
    also re-time `final` so machine drift between the two batches can be stated."""
    sec, sec2 = _seconds(("timing_a", "timing_b")), _seconds(("timing2_a", "timing2_b"))
    insts = sorted({w for w, _ in sec}, key=lambda w: (not w.startswith(("small", "medium")),
                                                       int(re.search(r"_(\d+)", w)[1])))
    for p in ("ip_moea_hv", "imoea_dqn"):
        for w in insts:
            if (w, p) in sec2:
                sec[(w, p)] = sec2[(w, p)]
    pols = [p for p in PAPER + ["trained_stable"] if any((w, p) in sec for w in insts)]
    missing = [p for p in ("ip_moea_hv", "imoea_dqn") if p not in pols]
    rows = []
    for p in pols:
        if p == "trained_stable":
            rows.append("\\midrule")
        rows.append(LABEL[p] + " & " + " & ".join(f"{np.median(sec[(w, p)]):.1f}" if (w, p) in sec else "--"
                                                  for w in insts) + r" \\")
    head = ["Method & " + " & ".join(inst_name(w).replace(" $\\times20$", "") for w in insts) + r" \\"]
    cap = ("Wall-clock time per run in seconds (median of 2 runs, $G = 1000$, $N = 105$, one run at a time on the same "
           "machine). Suite A instances first, then Cybershake with 100--2000 tasks (data $\\times 20$). The last row "
           "is IMOEA-DQN with the offline-trained GAT controller of the learned-control analysis.")
    notes = []
    ctl = [np.median(sec2[(w, "final")]) / np.median(sec[(w, "final")]) for w in insts if (w, "final") in sec2]
    if ctl:
        notes.append(f"IP-MOEA and IMOEA-DQN were timed in a later batch together with \\ours{{}} as a control; "
                     f"\\ours{{}}'s time in that batch was {min(ctl):.2f}--{max(ctl):.2f} times its time here.")
        print(f"timing control: final later / earlier = {min(ctl):.2f}-{max(ctl):.2f}")
    if missing:
        notes.append("TODO: timing runs for " + ", ".join(LABEL[p] for p in missing) + " are missing.")
        print("WARNING: timing table lacks", missing)
    return table(cap, "tab:timing", "l" + "c" * len(insts), head, rows, size=r"\scriptsize",
                 note=" ".join(notes) or None)


def _wlt(S, keys, p, ref, m):
    s = [symbols({ref: S[k][ref], p: S[k][p]}, [p], m, ref=ref)[p] for k in keys]
    return f"{s.count('+')}/{s.count('-')}/{s.count('=')}"


def t_learners(S):
    keys = sorted(S, key=lambda k: natural(k[1]))
    ref = "tch_fixed_fav"
    rows, last = [], None
    for group, p, lab in LEARN_ROWS:
        if group != last:
            if last is not None:
                rows.append(r"\midrule")
            rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{group}}}}}" + r" \\")
            last = group
        igd = np.mean([S[k][p]["IGD"].mean() for k in keys]) * 100
        hv = np.mean([S[k][p]["HV_jin"].mean() for k in keys])
        imp = np.mean([S[k][p]["Imprecision"].mean() for k in keys])
        w = ["--", "--"] if p == ref else [_wlt(S, keys, p, ref, m) for m in ("IGD", "HV_jin")]
        rows.append(f"\\quad {lab} & {igd:.2f} & {hv:.3f} & {imp:.1f} & {w[0]} & {w[1]}" + r" \\")
    head = [r"Control of $(\alpha, \beta, \gamma)$ & IGD ($\times10^{-2}$) & HV & Imprecision & "
            r"IGD $+/-/\approx$ & HV $+/-/\approx$ \\"]
    cap = (r"Learned control of the selection weights $(\alpha, \beta, \gamma)$ against fixed actions, on four Suite A "
           r"instances (small-20 $\times$1, basic-60 $\times$20, medium-100 $\times$1, medium-250 $\times$20), 4 runs "
           r"each; DQN agents trained on the training instances (2--3 training seeds, averaged per run). All rows above "
           r"\ours{} use the Tchebycheff backbone with $T = 20$, $p_{local} = 0.9$. Means over the instances; "
           r"$+/-/\approx$ counts instances where the row is significantly better / worse / not different from the "
           r"best fixed action (Wilcoxon rank-sum, $\alpha = 0.05$; with 4 runs few differences can reach significance). "
           r"One normalisation per instance over all rows except IMOEA-DQN, which is scored on it without joining it (it lies far outside).")
    return table(cap, "tab:learners", "lccccc", head, rows, size=r"\scriptsize")


def t_gat(S):
    rows = []
    for g, _, a, b, lab in GAT_PAIRS:
        keys = sorted([k for k in S if k[0] == g], key=lambda k: natural(k[1]))
        runs = len(S[keys[0]][a]["IGD"])
        ia, ib = (np.mean([S[k][p]["IGD"].mean() for k in keys]) * 100 for p in (a, b))
        ha, hb = (np.mean([S[k][p]["HV_jin"].mean() for k in keys]) for p in (a, b))
        rows.append(f"{lab} & {len(keys)} & {runs} & {ia:.2f} / {ib:.2f} & {_wlt(S, keys, a, b, 'IGD')} & "
                    f"{ha:.3f} / {hb:.3f} & {_wlt(S, keys, a, b, 'HV_jin')}" + r" \\")
    head = [r"Learner & Inst. & Runs & IGD ($\times10^{-2}$) GAT / no GAT & GAT $+/-/\approx$ & "
            r"HV GAT / no GAT & GAT $+/-/\approx$ \\"]
    cap = (r"Graph attention (GAT) state encoder against the same learner without it, on Suite A instances. "
           r"$+/-/\approx$ counts instances where the GAT agent is significantly better / worse / not different "
           r"(Wilcoxon rank-sum, $\alpha = 0.05$). Each comparison is normalised per instance over its two agents only.")
    return table(cap, "tab:gat", "lcccccc", head, rows, size=r"\scriptsize")


def t_montecarlo():
    data, info = {}, {}
    with open(STATS / "e7_montecarlo.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            data[(r["instance"], r["policy"])] = r
    md = (STATS / "e7_montecarlo.md").read_text(encoding="utf-8")
    for blk in md.split("\n## ")[1:]:
        inst = blk.split("\n", 1)[0].strip()
        c = re.search(r"\*\*([\d.]+)%\*\*", blk)
        t = re.search(r"Kendall tau[^:]*: ([\-\d.]+)", blk)
        info[inst] = (c[1] if c else "?", t[1] if t else "?")
    insts = sorted({i for i, _ in data}, key=lambda w: (not w.startswith("medium"), natural(w)))
    rows = []
    for p in PAPER:
        cells = []
        for w in insts:
            r = data.get((w, p))
            if r is None:
                cells += ["--", "--"]
                continue
            cells.append(f"{float(r['IGD_mean']) * 100:.2f}")
            cells.append("--" if p == "final" else r["final_better_IGD"].replace("%", r"\%"))
        rows.append(LABEL[p] + " & " + " & ".join(cells) + r" \\")
    rows.append("\\midrule")
    rows.append("Containment & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{info[w][0]}\\%}}" for w in insts) + r" \\")
    rows.append("Kendall $\\tau$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{info[w][1]}}}" for w in insts) + r" \\")
    head = ["& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{inst_name(w)}}}" for w in insts) + r" \\",
            "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(insts))),
            "Method & " + " & ".join(["IGD", "\\ours{} better"] * len(insts)) + r" \\"]
    cap = ("Monte Carlo check of the final fronts under 200 crisp scenarios (every interval input drawn uniformly). "
           "IGD ($\\times10^{-2}$) against each scenario's pooled crisp front, mean over scenarios; share of scenarios in "
           "which \\ours{}'s mean over runs is lower; containment = share of solution--scenario pairs whose crisp "
           "objectives fall inside the predicted intervals; Kendall $\\tau$ between the interval IGD ranking of the "
           "methods and the Monte Carlo ranking. 10 runs per method.")
    return table(cap, "tab:montecarlo", "l" + "cc" * len(insts), head, rows, size=r"\tiny")


# ---------------------------------------------------------------- driver

PREVIEW = r"""\documentclass[review]{elsarticle}
\usepackage{booktabs,graphicx,rotating,amsmath}
\newcommand{\ours}{DR-IMOEA}
\begin{document}
\title{Tables and figures (generated)}
\maketitle
%BODY%
\end{document}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    score(args.rescore, args.workers)

    SA = {k: v for k, v in read_scores("suite_a").items()}
    SB = {k: v for k, v in read_scores("suite_b").items()}
    tabs = {}
    for suite, S in (("A", SA), ("B", SB)):
        for m, sc, dg in (("IGD", 100, 2), ("HV_jin", 1, 3), ("Imprecision", 1, 1)):
            tabs[f"suite{suite.lower()}_{m.lower()}"] = t_suite(suite, S, m, sc, dg)
    tabs["friedman"] = t_friedman(SA, SB)
    tabs["heuristics"] = t_heuristics(SA, SB)
    tabs["robustness"] = t_robustness()
    tabs["ablation"] = t_ablation(read_scores("ablation"))
    tabs["sensitivity"] = t_sensitivity(read_scores("sensitivity"))
    tabs["scalability"] = t_scalability(read_scores("scalability"))
    tabs["timing"] = t_timing()
    tabs["montecarlo"] = t_montecarlo()
    tabs["learners"] = t_learners(read_scores("learners"))
    tabs["gat"] = t_gat(read_scores("gat"))
    for n, b in tabs.items():
        write(n, b)

    figs = PAPER_DIR / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    body = [f"\\input{{tables/{n}}}" for n in tabs]
    for f, src in FIGURES.items():
        p = ROOT / "results" / "figures" / f"{f}.pdf"
        if not p.exists():
            print(f"WARNING: {p.name} missing (made by {src})")
            continue
        shutil.copy2(p, figs / p.name)
        body.append("\\begin{figure}[!htbp]\\centering\\includegraphics[width=\\linewidth]{figures/" + p.name +
                    "}\\caption{" + esc(f) + " (from " + esc(src) + ")}\\end{figure}")
    (PAPER_DIR / "preview.tex").write_text(PREVIEW.replace("%BODY%", "\n\\clearpage\n".join(body)), encoding="utf-8")
    print("wrote paper/preview.tex with", len(tabs), "tables and", len(FIGURES), "figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
