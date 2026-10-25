"""Generator-neutral workflow representation.

Both sources (the Pegasus generator port and WfCommons) produce a RawWorkflow:
tasks with a reference runtime and the files they read/write, plus the
parent->child links with the files each link carries. builder.py turns this
into the interval dataset format used by GAT-IMOEA-DQN.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class RawTask:
    name: str                      # unique name, e.g. "ExtractSGT_00003"
    type: str                      # task category, e.g. "ExtractSGT"
    runtime_s: float               # runtime on the generator's reference core
    inputs: Dict[str, int] = field(default_factory=dict)   # file -> bytes
    outputs: Dict[str, int] = field(default_factory=dict)  # file -> bytes


@dataclass
class RawWorkflow:
    family: str
    generator: str
    tasks: List[RawTask] = field(default_factory=list)
    # (parent index, child index) -> {file: bytes} carried along that edge
    links: Dict[Tuple[int, int], Dict[str, int]] = field(default_factory=dict)

    def add_task(self, name: str, type_: str, runtime_s: float = 0.0) -> int:
        self.tasks.append(RawTask(name=name, type=type_, runtime_s=float(runtime_s)))
        return len(self.tasks) - 1

    def add_link(self, parent: int, child: int, filename: str, size: int) -> None:
        size = max(int(size), 0)
        self.links.setdefault((parent, child), {})[filename] = size
        # Pegasus semantics: a link is an output of the parent and an input of the child.
        self.tasks[parent].outputs.setdefault(filename, size)
        self.tasks[child].inputs.setdefault(filename, size)
