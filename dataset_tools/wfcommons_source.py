"""WfCommons (WfChef recipes) source: Montage, Epigenomics and the other
recipes shipped with wfcommons. Recipes are built from real Pegasus execution
traces (WfInstances), so runtimes and file sizes come from measured runs.
"""
from __future__ import annotations

import logging
import random
import re
import warnings

import numpy as np

from .raw import RawWorkflow

RECIPES = {
    "montage": "MontageRecipe",
    "epigenomics": "EpigenomicsRecipe",
    "1000genome": "GenomeRecipe",
    "blast": "BlastRecipe",
    "bwa": "BwaRecipe",
    "cycles": "CyclesRecipe",
    "seismology": "SeismologyRecipe",
    "soykb": "SoykbRecipe",
    "srasearch": "SrasearchRecipe",
    "rnaseq": "RnaseqRecipe",
}

_ID_SUFFIX = re.compile(r"_(ID)?\d+$")


def generate(family: str, num_tasks: int, seed: int) -> RawWorkflow:
    import wfcommons
    from wfcommons import WorkflowGenerator
    import wfcommons.wfchef.recipes as recipes

    logging.getLogger("wfcommons").setLevel(logging.ERROR)
    recipe_cls = getattr(recipes, RECIPES[family])
    # WfCommons draws from the global random / numpy RNGs.
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf = WorkflowGenerator(recipe_cls.from_num_tasks(num_tasks)).build_workflow()

    raw = RawWorkflow(family=family, generator=f"wfcommons-{wfcommons.__version__}")
    index = {}
    for tid, t in wf.tasks.items():
        ttype = t.category or _ID_SUFFIX.sub("", t.name)
        i = raw.add_task(t.task_id or tid, ttype, float(t.runtime))
        raw.tasks[i].inputs = {f.file_id: int(f.size) for f in t.input_files}
        raw.tasks[i].outputs = {f.file_id: int(f.size) for f in t.output_files}
        index[tid] = i
    for p, c in wf.edges:
        pi, ci = index[p], index[c]
        shared = set(raw.tasks[pi].outputs) & set(raw.tasks[ci].inputs)
        raw.links[(pi, ci)] = {f: raw.tasks[pi].outputs[f] for f in shared}
    return raw
