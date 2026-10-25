"""Python port of the Pegasus synthetic workflow generator (Bharathi et al., 2008).

Source: github.com/pegasus-isi/WorkflowGenerator, bharathi/src/simulation/generator.
Ported: CyberShake, LIGO Inspiral and SIPHT, which WfCommons does not ship.
Shapes, job counts and the runtime / file-size distributions (means and
variances measured from production runs) are kept as in the Java code.

Deliberate deviations from the Java code:
  * Misc.randomLong(mean, tol) uses `nextLong() % range`, which can go
    negative; here it is a proper uniform draw in [start, end).
  * A seeded numpy Generator replaces java.util.Random, so runs are reproducible.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np

from .raw import RawWorkflow

MAX_TRIES = 100


# ------------------------------------------------------------------ Misc ---
class Misc:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def truncated_normal(self, mean: float, variance: float, tolerance: float = 0.5) -> float:
        std = math.sqrt(variance)
        for _ in range(100):
            x = self.rng.standard_normal() * std + mean
            if abs(mean - x) <= tolerance * mean:
                return float(x)
        return float(mean)

    def random_int_range(self, start: int, end: int) -> int:
        """Uniform integer in [start, end) (Java Misc.randomInt(start, end))."""
        if start == end:
            return start
        return int(start + self.rng.integers(0, end - start))

    def random_int(self, mean: int, tolerance: float) -> int:
        return self.random_int_range(int(math.ceil(mean * (1.0 - tolerance))),
                                     int(math.floor(mean * (1.0 + tolerance))))

    def random_long(self, mean: int, tolerance: float) -> int:
        start = int(math.ceil(mean * (1.0 - tolerance)))
        end = int(math.floor(mean * (1.0 + tolerance)))
        return self.random_int_range(start, end) if end > start else start

    def random_toss(self, bias: float) -> bool:
        return self.rng.random() <= bias

    def random_set(self, n: int, total: int) -> List[int]:
        temp = [int(self.rng.integers(0, total)) for _ in range(n - 1)] + [total]
        temp.sort()
        prev = temp[n - 1]
        for i in range(n - 1, 0, -1):
            temp[i] -= temp[i - 1]
        if n != 1:
            temp[0] = (total + temp[0] - prev) % total
        return temp

    def max_non_zero_random_set(self, n: int, total: int, maximum: int) -> List[int]:
        if total < n:
            raise ValueError("Cannot satisfy sum.")
        for _ in range(MAX_TRIES):
            temp = self.random_set(n, total)
            if all(0 < t <= maximum for t in temp):
                return temp
        raise RuntimeError(f"too many attempts. n, sum, max = {n}, {total}, {maximum}")

    def close_non_zero_randoms(self, n: int, total: int, tolerance: float) -> List[int]:
        """Non-zero ints close to total/n that sum to total (Java int arithmetic)."""
        if total < n:
            raise ValueError(f"Impossible to satisfy: (n, sum) = {n},{total}")
        if n == 1:
            return [total]
        avg = total // n
        for _ in range(MAX_TRIES):
            temp = [(i + 1) * total // n for i in range(n - 1)] + [total]
            for i in range(n - 1):
                factor = (2.0 * self.rng.random() - 1.0) * 0.5 * tolerance
                temp[i] = int(temp[i] + factor * avg)       # Java (int) cast truncates
            for i in range(n - 1, 0, -1):
                temp[i] -= temp[i - 1]
            if all(t > 0 for t in temp):
                return temp
        temp = [avg] * (n - 1) + [total - (n - 1) * avg]
        return temp


class Dists:
    """Named distributions, mirroring populateDistributions() in the Java apps."""

    def __init__(self, misc: Misc):
        self.misc = misc
        self.d: Dict[str, Callable[[], float]] = {}

    def const(self, name: str, value: float) -> None:
        self.d[name] = lambda v=value: float(v)

    def uniform(self, name: str, lo: float, hi: float) -> None:
        self.d[name] = lambda: float(lo + self.misc.rng.random() * (hi - lo)) if hi != lo else float(lo)

    def tnorm(self, name: str, mean: float, var: float) -> None:
        self.d[name] = lambda: self.misc.truncated_normal(mean, var)

    def double(self, name: str) -> float:
        return self.d[name]()

    def long(self, name: str) -> int:
        return int(math.ceil(self.d[name]()))


class _Builder:
    def __init__(self, family: str, seed: int):
        self.rng = np.random.default_rng(seed)
        self.misc = Misc(self.rng)
        self.dist = Dists(self.misc)
        self.wf = RawWorkflow(family=family, generator="pegasus-workflowgenerator-port")
        self._counter: Dict[str, int] = {}

    def job(self, type_: str, runtime: float = 0.0) -> int:
        c = self._counter.get(type_, 0) + 1
        self._counter[type_] = c
        return self.wf.add_task(f"{type_}_{c:05d}", type_, runtime)

    def inp(self, job: int, filename: str, size: int) -> None:
        self.wf.tasks[job].inputs.setdefault(filename, max(int(size), 0))

    def out(self, job: int, filename: str, size: int) -> None:
        self.wf.tasks[job].outputs.setdefault(filename, max(int(size), 0))

    def link(self, parent: int, child: int, filename: str, size: int) -> None:
        self.wf.add_link(parent, child, filename, size)


# ------------------------------------------------------------ CyberShake ---
def cybershake(num_jobs: int, seed: int) -> RawWorkflow:
    b = _Builder("cybershake", seed)
    m, d = b.misc, b.dist
    d.tnorm("SGT", 19958666972.0, 93654683371233792.0)
    d.tnorm("SUB_SGT", 231720131.58, 27081652820787388.00)
    d.uniform("SLIP", 0, 10000)
    d.uniform("HIPO", 0, 10000)
    d.tnorm("VARIATION", 3708598.53, 10160641539133.56)
    d.const("GRM", 24000)
    d.const("BSA", 216)
    d.const("Zip_factor", 6)
    d.tnorm("ExtractSGT", 137.45, 42681.27)
    d.tnorm("SeismogramSynthesis", 43.40, 984.36)
    d.tnorm("PeakValCalcOkaya", 1.09, 3.71)
    d.const("ZipSeis_rate", 228180.52)
    d.const("ZipPSA_rate", 2782.00)

    remaining = num_jobs - 2
    if remaining < 3:
        raise ValueError(f"CyberShake needs more than 4 jobs, got {num_jobs}")
    n_sgt = max(m.random_int(int(remaining * 0.0081), 0.5), 2)
    if (remaining - n_sgt) % 2 != 0:
        n_sgt = n_sgt - 1 if n_sgt > 1 else n_sgt + 1
    n_seis = (remaining - n_sgt) // 2
    counts = sorted(m.close_non_zero_randoms(n_sgt, n_seis, 0.25))

    zip_psa = b.job("ZipPSA")
    zip_seis = b.job("ZipSeis")
    rupture = variation = 0
    seis_count = 0
    for i in range(n_sgt):
        if m.random_toss(1.0 / 20):
            rupture, variation = rupture + 1, 0
        else:
            variation += 1
        prefix = f"FFI_{rupture}_{variation}"
        e = b.job("ExtractSGT", d.double("ExtractSGT"))
        sgt = d.long("SGT")
        b.inp(e, f"{prefix}_fx.sgt", sgt)
        b.inp(e, f"{prefix}_fy.sgt", sgt)
        last_variation = None
        for _ in range(counts[i]):
            s = b.job("SeismogramSynthesis", d.double("SeismogramSynthesis"))
            var_name = f"{prefix}_txt.variation-s{d.long('SLIP'):05d}-h{d.long('HIPO'):05d}"
            var_size = d.long("VARIATION")
            b.inp(s, var_name, var_size)
            last_variation = (var_name, var_size)
            sub = d.long("SUB_SGT")
            b.link(e, s, f"{prefix}_subfx.sgt", sub)
            b.link(e, s, f"{prefix}_subfy.sgt", sub)
            grm = f"Seismogram_{prefix}_{s}.grm"
            b.link(s, zip_seis, grm, d.long("GRM"))
            p = b.job("PeakValCalcOkaya", d.double("PeakValCalcOkaya"))
            b.link(s, p, grm, d.long("GRM"))
            b.link(p, zip_psa, grm.replace("Seismogram", "PeakVals").replace("grm", "bsa"), d.long("BSA"))
            seis_count += 1
        if last_variation:                       # ExtractSGT.finish()
            b.inp(e, *last_variation)

    zs = m.random_long(int(seis_count * d.long("GRM") / d.double("Zip_factor")), 0.25)
    b.out(zip_seis, "Cybershake_Seismograms.zip", zs)
    b.wf.tasks[zip_seis].runtime_s = zs / d.double("ZipSeis_rate")
    zp = m.random_long(int(seis_count * d.long("BSA") / d.double("Zip_factor")), 0.25)
    b.out(zip_psa, "Cybershake_PSA.zip", zp)
    b.wf.tasks[zip_psa].runtime_s = zp / d.double("ZipPSA_rate")
    return b.wf


# ------------------------------------------------------- LIGO Inspiral ---
def inspiral(num_jobs: int, seed: int) -> RawWorkflow:
    if num_jobs <= 20:
        raise ValueError("Inspiral needs more than 20 jobs.")
    if num_jobs % 2:
        raise ValueError(f"Inspiral needs an even number of jobs, got {num_jobs}")
    b = _Builder("inspiral", seed)
    m, d = b.misc, b.dist
    d.tnorm("GWF", 15595958.08, 334629862.67)
    d.tnorm("TMPLTBANK.xml", 986917.26, 359037821.49)
    d.const("FAC_DARM", 8415922)
    d.const("REF_DARM", 902334)
    d.tnorm("INSPIRAL.xml", 312935.68, 335782981511.87)
    d.const("INJECTION.xml", 2421423)
    d.tnorm("THINCA.xml", 33820.71, 1888180029.78)
    d.tnorm("TRIGBANK.xml", 12779.67, 137597136.79)
    d.tnorm("TmpltBank", 18.14, 0.18)
    d.tnorm("Inspiral", 460.21, 297397.45)
    d.tnorm("Thinca", 5.37, 0.06)
    d.tnorm("TrigBank", 5.11, 0.1)

    top_down = total_edges = bn = None
    for _ in range(MAX_TRIES):                 # outer retry over the bottleneck count
        mx = max(int(math.floor(0.05 * num_jobs)), 3)
        bn = m.random_int_range(1, mx)
        if bn == 1 and (num_jobs - 2) % 4 != 0:
            bn += 1
        remaining = num_jobs - 2 * bn
        for _ in range(MAX_TRIES):
            top_down = sorted(m.close_non_zero_randoms(2, remaining // 2, 0.1))
            total_edges = m.random_int_range(top_down[1], int(math.floor(top_down[1] * 1.05)))
            if total_edges <= bn * top_down[0]:
                break
        if total_edges <= bn * top_down[0]:
            break
    else:
        raise RuntimeError("Could not find a good Inspiral distribution.")
    t0, t1 = top_down

    tmplt = [b.job("TmpltBank", d.double("TmpltBank")) for _ in range(t0)]
    up_insp = [b.job("Inspiral", d.double("Inspiral")) for _ in range(t0)]
    up_thinca = [b.job("Thinca", d.double("Thinca")) for _ in range(bn)]
    trig = [b.job("TrigBank", d.double("TrigBank")) for _ in range(t1)]
    lo_insp = [b.job("Inspiral", d.double("Inspiral")) for _ in range(t1)]
    lo_thinca = [b.job("Thinca", d.double("Thinca")) for _ in range(bn)]

    for j in up_insp + lo_insp:
        b.inp(j, "HL-INJECTIONS.xml", d.long("INJECTION.xml"))
    for i in range(t1):
        det = "H1" if i < t0 else "H2"
        files = {f"H-{det}_RDS-{i}-a.gwf": d.long("GWF"), f"H-{det}_RDS-{i}-b.gwf": d.long("GWF"),
                 "FAC_DARM.gwf": d.long("FAC_DARM"), "REF_DARM.gwf": d.long("REF_DARM")}
        targets = [tmplt[i], up_insp[i], lo_insp[i]] if i < t0 else [lo_insp[i]]
        for j in targets:
            for f, s in files.items():
                b.inp(j, f, s)

    for i in range(t0):
        b.link(tmplt[i], up_insp[i], f"H1-TMPLTBANK-{i}.xml", d.long("TMPLTBANK.xml"))

    bn_set = sorted(m.max_non_zero_random_set(bn, total_edges, len(tmplt)), reverse=True)

    def blocks(size: int):
        prev = 0
        for k in range(bn):
            start = prev
            if start + bn_set[k] > size:
                start = size - bn_set[k]
            yield k, range(start, start + bn_set[k])
            prev = start + bn_set[k]

    for k, rng in blocks(len(up_insp)):
        for j in rng:
            b.link(up_insp[j], up_thinca[k], f"H1-INSPIRAL-{j}.xml", d.long("INSPIRAL.xml"))
    for k, rng in blocks(len(trig)):
        for j in rng:
            b.link(up_thinca[k], trig[j], f"THINCA-{k}.xml", d.long("THINCA.xml"))
    for i in range(t1):
        b.link(trig[i], lo_insp[i], f"H1-TRIGBANK-{i}.xml", d.long("TRIGBANK.xml"))
    for k, rng in blocks(len(trig)):
        for j in rng:
            b.link(lo_insp[j], lo_thinca[k], f"H1-INSPIRAL_2-{j}.xml", d.long("INSPIRAL.xml"))
    for k, t in enumerate(lo_thinca):
        b.out(t, f"THINCA_2-{k}.xml", d.long("THINCA.xml"))
    return b.wf


# ----------------------------------------------------------------- SIPHT ---
def sipht(num_jobs: int, seed: int) -> RawWorkflow:
    if num_jobs < 30:
        raise ValueError(f"SIPHT needs at least 30 jobs, got {num_jobs}")
    b = _Builder("sipht", seed)
    m, d = b.misc, b.dist
    code = "NC_0025AG05"
    d.uniform("IGR_partners", 0.9 * 936, 936)
    d.const("CODE.fna", m.truncated_normal(5248967.25, 2301068882937.69))
    d.const("CODE.ptt", m.truncated_normal(328510.50, 76625546.75))
    d.const("BLAST_CODE", m.truncated_normal(11427.25, 21904571.69))
    d.const("CODE_sRNA.out_annotated", m.truncated_normal(307952.25, 3812001448.19))
    for name, val in {"RNAfold": 591942, "findterm.out": 134, "findterm.err": 0,
                      "RNAMofficial_descriptor.txt": 17626, "rnamotif.err": 310.0,
                      "expterm.dat": 45788, "blasta": 786856, "xdformat": 380164, "time": 11944,
                      "alphabet": 16, "matrix": 170917, "patser.in": 14,
                      "access_genomes2.txt": 83722, "sRNAPredict.in": 1198,
                      "All_known_sRNAs.txt": 1028403, "Seq_known_sRNAs_IGRs.txt": 172432,
                      "mix_tied_linux.cfg": 11964, "RIBOPROB.mat": 8350, "BLOSUM62": 2122,
                      "blastn2qrnadepth.pl": 69426, "qrna2gff.pl": 16995, "eqrna": 953556,
                      "qrna.err": 610, "Flanking_ORFs_known.txt": 1419426,
                      "Blast_synteny": 33.0}.items():
        d.const(name, val)
    for name, mean, var in [
            ("CODE_term.txt", 16463275, 95617942503538.19),
            ("CODE_term_candidate_nonredund", 385620, 15568279781.19),
            ("rna.ps", 2837.50, 512.25), ("vienna_input_tmp", 3974004.00, 849579407883.00),
            ("vienna_output", 7729972.50, 23454353107374.75),
            ("vienna_index_tmp", 5827902.00, 1670001877413.50),
            ("rnamotif.out", 1053510.50, 8411320934.75),
            ("transterm.out", 440409.25, 211951229.69), ("transterm.err", 96.25, 0.19),
            ("CODE_IGR_partners.txt", 288565.20, 107942475771.75),
            ("blast.err", 548.75, 0.19), ("blast.out", 511.75, 0.19),
            ("BLAST_sorted.out", 5270722.25, 11857591385947.69),
            ("CODE_PatserOut.txt", 91780.15, 61994599624.9),
            ("CODE.gbk", 11664926.25, 5027367745576.69),
            ("srna.err", 1859, 36), ("srna.out", 331823.5, 482134362.75),
            ("Blast_Out", 2491818.25, 2780189305579.69), ("OutBlastParsed", 1002636.75, 995765999448.69),
            ("OutCandidates", 222136.25, 9508831415.85), ("OutConsIG", 422672.75, 60310327140.69),
            ("OutIG", 605995, 103944177), ("OutORF", 328412.5, 559717064.75),
            ("OutTerm_temp", 115600.75, 1330343466.19), ("OutTerms", 53733.5, 184633890.75),
            ("OutTermsIG", 18803.75, 23633877.69),
            ("CODE.ffn", 4202875, 168713989549.5), ("Seq_CODE", 84073, 12298824),
            ("CODE_parsed.ffn", 802506.75, 18062265579.69),
            ("blast_candidate.out", 2965.5, 24644.25), ("blast_candidate.err", 544.75, 0.19),
            ("qrna.out", 4809.62, 9530118.98), ("CODE_QRNA.txt", 773386.5, 201840720.75),
            ("CODE_QRNAblast.txt", 3115449.75, 433512921497.19),
            ("QRNA.gff", 175489.75, 90766297.69),
            ("CODE_synteny.txt", 1146801.25, 445588527066.69),
            ("CODE_paralogues_temp.txt", 90154.75, 6814689167.19),
            ("synteny_temp.txt", 20552, 133473984.5), ("synteny_temp2.txt", 2706.5, 2262206.25),
            ("Block0_Parsed.txt", 567813.75, 880488134611.69), ("QRNA_out", 6572.75, 12596574.19),
            ("srna_annotate.err", 1070.5, .75), ("srna_annotate.out", 565462, 98726501739.5),
            ("CODE_paralogues.txt", 690731.5, 362706549090.75),
            # runtimes
            ("Findterm", 1349.47, 635206.26), ("Transterm", 55.78, 2356.11),
            ("RNAMotif", 36.42, 78.15), ("Blast", 2350.58, 1033834.66),
            ("FFN_parse", 1.64, 0.06), ("SRNA", 361.33, 117451.46),
            ("Blast_paralogues", 4.99, 1.32), ("Blast_candidate", 5.18, 0.99),
            ("Blast_QRNA", 1412.09, 9702.26), ("Patser", 1.27, 0.11),
            ("Patser_concate", 0.08, 0.01), ("SRNA_annotate", 1.68, 0.92)]:
        d.tnorm(name, mean, var)

    count = int(round(num_jobs / (18 + 13)))
    per_sub = m.close_non_zero_randoms(count, num_jobs - count * 13, 0.2)
    for sub, n_patser in enumerate(per_sub):
        tag = f"s{sub}"
        # Files made inside a sub-workflow are namespaced by it; shared
        # external inputs (genome, BLAST binaries, ...) keep one global name.
        L = lambda p, c, f, size: b.link(p, c, f"{tag}/{f}", size)   # noqa: E731
        I = b.inp

        pc = b.job("Patser_concate", d.double("Patser_concate"))
        pc_in = 0
        for _ in range(n_patser):
            p = b.job("Patser", d.double("Patser"))
            for f in ("patser.in", "alphabet"):
                I(p, f, d.long(f))
            I(p, f"{code}.fna", d.long("CODE.fna"))
            I(p, f"matrix_{p}.txt", d.long("matrix"))
            s = d.long("CODE_PatserOut.txt")
            L(p, pc, f"PatserOut{p}.txt", s)
            pc_in += s

        findterm = b.job("Findterm", d.double("Findterm"))
        rnamotif = b.job("RNAMotif", d.double("RNAMotif"))
        transterm = b.job("Transterm", d.double("Transterm"))
        blast = b.job("Blast", d.double("Blast"))
        srna = b.job("SRNA", d.double("SRNA"))
        for f in (f"{code}.fna", "RNAfold"):
            I(findterm, f, d.long("CODE.fna" if f.endswith(".fna") else f))
        for f, key in ((f"{code}.fna", "CODE.fna"), ("RNAMofficial_descriptor.txt", None)):
            I(rnamotif, f, d.long(key or f))
        for f, key in ((f"{code}.fna", "CODE.fna"), ("expterm.dat", None), (f"{code}.ptt", "CODE.ptt")):
            I(transterm, f, d.long(key or f))
        for f in ("blasta", "xdformat", "time"):
            I(blast, f, d.long(f))
        partners = int(math.ceil(d.double("IGR_partners")))
        I(blast, "IGR_partners_all.txt", d.long("CODE_IGR_partners.txt") * (partners + 1))
        b.out(blast, f"{tag}/blast.err", d.long("blast.err"))

        for f, key in (("findterm.out", None), ("findterm.err", None), ("term.txt", "CODE_term.txt"),
                       ("term_candidates_non_redund", "CODE_term_candidate_nonredund"),
                       ("rna.ps", None), ("vienna_input_tmp", None), ("vienna_output", None),
                       ("vienna_index_tmp", None)):
            L(findterm, srna, f, d.long(key or f))
        L(rnamotif, srna, "rnamotif.out", d.long("rnamotif.out"))
        L(rnamotif, srna, "rnamotif.err", d.long("rnamotif.err"))
        L(transterm, srna, "transterm.out", d.long("transterm.out"))
        L(transterm, srna, "transterm.err", d.long("transterm.err"))
        L(blast, srna, "blast.out", d.long("blast.out"))
        L(blast, srna, "BLAST_sorted.out", d.long("BLAST_sorted.out"))

        for f, key in (("access_genomes2.txt", None), ("sRNAPredict.in", None), (f"{code}.gbk", "CODE.gbk"),
                       (f"{code}.fna", "CODE.fna"), (f"{code}.ptt", "CODE.ptt"), ("All_known_sRNAs.txt", None)):
            I(srna, f, d.long(key or f))
        for f, key in (("srna.err", None), ("Blast_Out", None), ("OutBlastParsed", None),
                       ("OutCandidates", None), ("OutCandidates2", "OutCandidates"),
                       ("OutCandidates3", "OutCandidates"), ("OutConsIG", None), ("OutIG", None),
                       ("OutORF", None), ("OutTerm_temp", None), ("OutTerms", None), ("OutTermsIG", None)):
            b.out(srna, f"{tag}/{f}", d.long(key or f))

        ffn = b.job("FFN_parse", d.double("FFN_parse"))
        I(ffn, f"{code}.ffn", d.long("CODE.ffn"))
        I(ffn, f"{code}.ptt", d.long("CODE.ptt"))
        bc = b.job("Blast_candidate", d.double("Blast_candidate"))
        bq = b.job("Blast_QRNA", d.double("Blast_QRNA"))
        bs = b.job("Blast_synteny", d.double("Blast_synteny"))
        bp = b.job("Blast_paralogues", d.double("Blast_paralogues"))
        ann = b.job("SRNA_annotate", d.double("SRNA_annotate"))

        seq = d.long("Seq_CODE")
        for child in (ffn, bc, bq, bs, bp, ann):
            L(srna, child, f"Seq_{code}", seq)
        L(srna, ann, "srna.out", d.long("srna.out"))
        L(ffn, bs, f"{code}_parsed.ffn", d.long("CODE_parsed.ffn"))

        for j in (bc, bq, bs, bp):
            for f in ("blasta", "xdformat", "time"):
                I(j, f, d.long(f))
        I(bc, "Seq_known_sRNAs_IGRs.txt", d.long("Seq_known_sRNAs_IGRs.txt"))
        for f in ("mix_tied_linux.cfg", "RIBOPROB.mat", "BLOSUM62", "blastn2qrnadepth.pl", "qrna2gff.pl", "eqrna"):
            I(bq, f, d.long(f))
        partners = int(math.ceil(d.double("IGR_partners")))
        I(bq, "IGR_partners_all.txt", d.long("CODE_IGR_partners.txt") * (partners + 1))
        for f, key in (("qrna.err", None), ("qrna.out", None), ("QRNA.txt", "CODE_QRNA.txt"),
                       ("QRNAblast.txt", "CODE_QRNAblast.txt")):
            b.out(bq, f"{tag}/{f}", d.long(key or f))
        I(bs, "Flanking_ORFs_known.txt", d.long("Flanking_ORFs_known.txt"))

        L(pc, ann, "PatserOut.txt", pc_in)
        L(bc, ann, f"BLAST_{code}", d.long("BLAST_CODE"))
        L(bc, ann, "blast_candidate.out", d.long("blast_candidate.out"))
        L(bc, ann, "blast_candidate.err", d.long("blast_candidate.err"))
        L(bq, ann, "QRNA.gff", d.long("QRNA.gff"))
        L(bs, ann, "synteny.txt", d.long("CODE_synteny.txt"))
        L(bp, ann, "paralogues.txt", d.long("CODE_paralogues.txt"))

        I(ann, "access_matrix.txt", d.long("matrix"))
        I(ann, "sRNAPredict.in", d.long("sRNAPredict.in"))
        for f in ("CODE_paralogues_temp.txt", "synteny_temp.txt", "synteny_temp2.txt",
                  "Block0_Parsed.txt", "QRNA_out", "srna_annotate.err", "srna_annotate.out",
                  "CODE_sRNA.out_annotated"):
            b.out(ann, f"{tag}/{f}", d.long(f))
    return b.wf


GENERATORS = {"cybershake": cybershake, "inspiral": inspiral, "sipht": sipht}
