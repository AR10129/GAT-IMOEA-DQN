"""Compile paper/main.tex (and paper/supplement.tex) with TinyTeX, installing missing LaTeX packages.

  python scripts/compile_paper.py            # main.tex
  python scripts/compile_paper.py supplement # supplement.tex

Output goes to paper/build/ (git-ignored). Prints the page count, unresolved references and overfull
boxes wider than 5pt, so each condensing step can be checked against the 12-15 page target.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BUILD = PAPER / "build"
TEX_BIN = pathlib.Path(os.environ["APPDATA"]) / "TinyTeX" / "bin" / "windows"


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True, errors="replace", **kw)


def pdflatex(name):
    return run([str(TEX_BIN / "pdflatex.exe"), "-interaction=nonstopmode", "-halt-on-error",
                f"-output-directory={BUILD}", f"{name}.tex"])


def install_missing(log: str) -> bool:
    m = re.search(r"File `([^']+)' not found", log)
    if not m:
        return False
    file = m[1]
    found = run([str(TEX_BIN / "tlmgr.bat"), "search", "--global", "--file", "/" + file], shell=False)
    pkgs = sorted({line.rstrip(":").strip() for line in found.stdout.splitlines()
                   if line.endswith(":") and not line.startswith(" ")})
    pkg = pkgs[0] if pkgs else pathlib.Path(file).stem
    print(f"  installing {pkg} (for {file})", flush=True)
    r = run([str(TEX_BIN / "tlmgr.bat"), "install", pkg])
    if r.returncode != 0:
        print(r.stdout[-500:], r.stderr[-500:])
        return False
    return True


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "main"
    BUILD.mkdir(exist_ok=True)
    for _ in range(40):                                   # install packages until the first pass succeeds
        r = pdflatex(name)
        log = (BUILD / f"{name}.log").read_text(errors="replace") if (BUILD / f"{name}.log").exists() else r.stdout
        if r.returncode == 0:
            break
        if not install_missing(log):
            err = [l for l in log.splitlines() if l.startswith("!")]
            print("pdflatex failed:", err[:5])
            i = log.find("\n!")
            print(log[i:i + 1200])
            return 1
    env = dict(os.environ, BIBINPUTS=str(PAPER), BSTINPUTS=str(PAPER))
    if (PAPER / "refs.bib").exists() and name == "main":
        b = subprocess.run([str(TEX_BIN / "bibtex.exe"), name], cwd=BUILD, capture_output=True, text=True, env=env)
        if b.returncode not in (0, 1):
            print("bibtex:", b.stdout[-800:])
    for _ in range(2):
        pdflatex(name)
    log = (BUILD / f"{name}.log").read_text(errors="replace")
    pages = re.search(r"Output written on .*?\((\d+) pages?", log)
    undefined = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", log)))
    cites = sorted(set(re.findall(r"Citation `([^']+)' on page \d+ undefined", log)))
    overfull = [l for l in log.splitlines() if l.startswith("Overfull \\hbox") and float(re.search(r"\(([\d.]+)pt", l)[1]) > 5]
    print(f"{name}.pdf: {pages[1] if pages else '?'} pages")
    print("undefined references:", undefined or "none")
    print("undefined citations:", cites or "none")
    print(f"overfull boxes > 5pt: {len(overfull)}")
    for l in overfull[:8]:
        print("  ", l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
