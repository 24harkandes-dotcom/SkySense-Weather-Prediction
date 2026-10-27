"""Assemble weather_rainfall_prediction.ipynb from cells_a/b/c.py (no nbformat needed). Run: python tools/build_notebook.py"""
import json, importlib.util, pathlib
here = pathlib.Path(__file__).parent
def load(n, var):
    sp = importlib.util.spec_from_file_location(n, here / f"{n}.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return getattr(m, var)
cells = load("cells_a", "CELLS_A") + load("cells_b", "CELLS_B") + load("cells_c", "CELLS_C")
def src(t): L = t.split("\n"); return [l + "\n" for l in L[:-1]] + [L[-1]]
nb = {"cells": [({"cell_type": "markdown", "metadata": {}, "source": src(t)} if k == "md" else
                 {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src(t)}) for k, t in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
out = here.parent / "weather_rainfall_prediction.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False); print("wrote", out, len(cells), "cells")
