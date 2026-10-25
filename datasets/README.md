# GAT-IMOEA-DQN datasets

Interval-valued DAG workflows for the GAT-IMOEA-DQN framework
(`GAT-IMOEA-DQN_Workflow.pdf`), plus the three-tier environment they run on.

```
datasets/
  workflows/<family>/<family>_<size>.json   40 workflows (5 families x 8 sizes)
  scenarios/default.yaml                    device / edge / cloud environment
  generation.yaml                           settings used to generate workflows/
  manifest.json                             index: path, seed, size, depth, ...
  legacy/                                   the original HEFT-style .txt files (not used)
```

## Families and sources

| Family | Source | Notes |
|---|---|---|
| montage | WfCommons 1.6 `MontageRecipe` | Built from real Pegasus traces. Needs ≥ 60 tasks. |
| epigenomics | WfCommons 1.6 `EpigenomicsRecipe` | Built from real Pegasus traces |
| cybershake | Pegasus WorkflowGenerator port | `dataset_tools/pegasus.py` |
| inspiral | Pegasus WorkflowGenerator port (LIGO) | Made of independent blocks, so several components |
| sipht | Pegasus WorkflowGenerator port | `round(K/31)` independent sub-workflows, so several components |

WfCommons does not ship CyberShake, LIGO or SIPHT recipes. Those three are
ported from the official generator (`pegasus-isi/WorkflowGenerator`,
Bharathi et al. 2008), keeping its shapes and its measured runtime and file-size
distributions. The two deliberate deviations are listed at the top of `pegasus.py`.

Sizes requested: 60, 100, 200, 400, 600, 800, 1000, 2000. WfCommons and SIPHT
return a close but not always exact count (e.g. Montage 1000 → 991). The real
count is `num_tasks`, and the request is `requested_tasks`.

Multi-component workflows (Inspiral, SIPHT) are what the reference generator
produces. The PDF's model handles them as they are: several entry tasks, and
makespan = max over all exit tasks.

## Workflow file (schema 1.0)

```jsonc
{
  "name": "montage_200", "family": "montage", "num_tasks": 196, "num_edges": 458,
  "requested_tasks": 200,
  "source": {"generator": "wfcommons-1.6", "seed": 123},
  "units": {"workload": "Gcycles", "data": "KB (1 KB = 1000 bytes)", ...},
  "conversion": {"reference_ghz": 2.0, "interval_half_width": [0.05, 0.30]},
  "entry_tasks": [...], "exit_tasks": [...],
  "stats": {"depth": 8, "components": 1, "data_to_compute_kb_per_gcycle": ..., ...},
  "tasks": [ {"id":0, "name":"mProject_00000001", "type":"mProject", "runtime_ref_s":12.3,
              "workload_gcycles":[22.1, 27.0], "ext_input_kb":4012.5, "ext_output_kb":0.0,
              "depth":0}, ... ],
  "edges": [ {"src":0, "dst":57, "data_kb":[3100.2, 4410.8]}, ... ]
}
```

- **Tasks** are numbered in topological order, so every edge has `src < dst`.
  This also proves the graph is acyclic.
- **`workload_gcycles`** is C̃ₖ: the reference runtime × `reference_ghz`, widened
  to `[C(1−d), C(1+d)]` with `d ~ U[0.05, 0.30]` drawn separately for each task.
- **`data_kb`** is D̃ᵢⱼ: the total size of the files the parent passes to the
  child, widened the same way.
- **`ext_input_kb` / `ext_output_kb`** are files that no other task produces or
  consumes. These are uploaded from, or returned to, the originating device; the
  PDF's terminal energy term uses the upload.
- **`depth`** is the topological depth, with entry tasks at 0. It is the 5th GAT
  node feature.

## Scenario file

`scenarios/default.yaml` covers everything the PDF's equations need beyond the DAG:

- m devices: interval CPU speed, energy per Gcycle, transmit power, idle power and position
- n edge servers: interval aggregate CPU speed, κ_E, static power, position and coverage radius
- the cloud: interval CPU speed
- the wireless uplink (B̃, v₀, path loss), backhaul W̃_C, edge-to-edge W̃_E2E, and prices η_E and η_C

The PDF does not define the following, so they are marked as assumptions in the file:

- cloud energy
- the edge → device downlink
- device ↔ cloud traffic, which is relayed through the nearest in-range edge

Add more scenarios as extra YAML files (e.g. more edge servers, or wider intervals).

## Usage

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python scripts/generate_datasets.py      # regenerate (deterministic)
.venv/Scripts/python scripts/validate_datasets.py      # check everything
```

```python
from dataset_tools.loader import load_workflow, load_scenario
wf = load_workflow("datasets/workflows/cybershake/cybershake_400.json")
sc = load_scenario("datasets/scenarios/default.yaml")

wf.C_lo, wf.C_hi, wf.edge_index, wf.D_lo, wf.D_hi    # interval arrays
wf.gat_node_features()          # (K, 5) [m(C), w(C), in_deg, out_deg, depth], log-scaled
wf.gat_node_features(None)      # same, raw values as written in the PDF
wf.gat_edge_features()          # (E, 2) [m(D), w(D)]
wf.random_topological_order(rng)  # Kahn with random ties -> priority string O

sc.allowed_targets()            # allocation values x_k reachable by the origin device
R_lo, R_hi = sc.rate_matrix_mbps()   # interval rates between nodes 0..n+1
```

Node indices follow the PDF: 0 is the origin device, 1..n are edge servers and
n+1 is the cloud.
