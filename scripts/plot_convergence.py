"""Plot Monte Carlo convergence from `pricer convergence results/convergence.csv`.

Usage: python3 scripts/plot_convergence.py results/convergence.csv results/convergence.png
"""
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(src: str, dst: str) -> None:
    rows = defaultdict(list)
    with open(src, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["method"]].append(
                (int(r["paths"]), float(r["mean_std_error"]), float(r["rms_error"]))
            )

    style = {
        "plain": ("#4C78A8", "Plain Monte Carlo"),
        "antithetic+cv": ("#E45756", "Antithetic + control variate"),
        "rqmc": ("#54A24B", "Randomised quasi-Monte Carlo"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    for method, pts in rows.items():
        pts.sort()
        n = [p[0] for p in pts]
        colour, label = style.get(method, ("#555555", method))
        ax.plot(n, [p[1] for p in pts], "-", color=colour, lw=2, label=f"{label}: standard error")
        ax.plot(n, [p[2] for p in pts], "o", color=colour, ms=5, alpha=0.8,
                label=f"{label}: RMS error (30 seeds)")

    # Reference slope -1/2, anchored to the plain curve's first point.
    n0, se0, _ = sorted(rows["plain"])[0]
    n_ref = [n0, sorted(rows["plain"])[-1][0]]
    ax.plot(n_ref, [se0 * (n / n0) ** -0.5 for n in n_ref], "--", color="#999999", lw=1,
            label=r"$1/\sqrt{N}$ reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of simulated paths, N")
    ax.set_ylabel("Pricing error (currency units)")
    ax.set_title("European call (S=K=100, vol 20%, T=1y): pricing error vs number of paths",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/convergence.csv",
         sys.argv[2] if len(sys.argv) > 2 else "results/convergence.png")
