"""Plot docs/convergence.csv (from: mc_pricer --convergence docs/convergence.csv).

usage: python scripts/plot_convergence.py [csv] [output_png]
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/convergence.csv")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/convergence.png")

with open(src) as fh:
    rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
paths = np.array([r["paths"] for r in rows])

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].loglog(paths, [r["euro_abs_error"] for r in rows], "o", label="|MC − Black–Scholes|")
ax[0].loglog(paths, [r["euro_se"] for r in rows], "-", label="standard error")
ax[0].loglog(paths, rows[0]["euro_se"] * np.sqrt(paths[0] / paths), "k:", lw=0.8, label=r"$\propto N^{-1/2}$")
ax[0].set(xlabel="paths", ylabel="price error", title="European call: error falls as 1/√N")
ax[0].legend()

plain = np.array([r["asian_plain_se"] for r in rows])
cv = np.array([r["asian_cv_se"] for r in rows])
ax[1].loglog(paths, plain, "o-", label="plain Monte Carlo")
ax[1].loglog(paths, cv, "s-", label="geometric control variate")
ax[1].set(xlabel="paths", ylabel="standard error",
          title=f"Asian call: control variate, ~{np.median((plain / cv) ** 2):.0f}× less variance")
ax[1].legend()

fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110)
print(f"wrote {out}")
