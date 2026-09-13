"""Aggregate per-cell metrics into the scaling table issue #5 asked for.

Reads every runs/*/metrics_*.json, groups by (arm, n_train), and reports
mean +- 95% CI across seeds. The headline comparison is `quantum` minus `se`,
paired by seed: `se` is the parameter-matched classical control, so that
difference is the circuit's contribution. `plain` is reported too because it
answers a different question (does the gating help at all) and is NOT a matched
control.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def ci95(xs):
    """Half-width of the 95% CI of the mean; 0 for a single sample."""
    if len(xs) < 2:
        return 0.0
    # t_{0.975} for small n, so 3 seeds does not get a normal-theory interval.
    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36,
         9: 2.31, 10: 2.26}.get(len(xs), 1.96)
    return t * statistics.stdev(xs) / math.sqrt(len(xs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("/home/meow/quantum_mnist_sweep/runs"))
    ap.add_argument("--metric", default="kid_mean", choices=["kid_mean", "fid"])
    ap.add_argument("--pattern", default="metrics_ddim50_n2000.json")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    by = defaultdict(dict)          # (arm, n) -> {seed: value}
    for f in sorted(args.runs.glob(f"*/{args.pattern}")):
        d = json.loads(f.read_text())
        by[(d["arm"], d["n_train"])][d["seed"]] = d[args.metric]

    arms = sorted({a for a, _ in by})
    ns = sorted({n for _, n in by})
    if not ns:
        print(f"no results matching {args.pattern}")
        return

    print(f"{args.metric} (lower is better), mean +- 95% CI across seeds\n")
    head = f"{'N':>6} | " + " | ".join(f"{a:>18}" for a in arms)
    print(head); print("-" * len(head))
    for n in ns:
        cells = []
        for a in arms:
            vals = list(by[(a, n)].values())
            cells.append(f"{statistics.mean(vals):.4f}±{ci95(vals):.4f}({len(vals)})"
                         if vals else "  --")
        print(f"{n:>6} | " + " | ".join(f"{c:>18}" for c in cells))

    # The comparison that actually answers #3/#5, paired by seed.
    if "quantum" in arms and "se" in arms:
        print("\nquantum - se, paired by seed (negative = the circuit helps):")
        for n in ns:
            q, s = by[("quantum", n)], by[("se", n)]
            shared = sorted(set(q) & set(s))
            if not shared:
                print(f"  N={n:<6} no paired seeds yet"); continue
            diffs = [q[k] - s[k] for k in shared]
            print(f"  N={n:<6} {statistics.mean(diffs):+.4f} ± {ci95(diffs):.4f} "
                  f"(n={len(diffs)} paired seeds)")
    else:
        missing = [a for a in ("quantum", "se") if a not in arms]
        print(f"\nquantum - se comparison unavailable: no results yet for {missing}")

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write(f"arm,n_train,seed,{args.metric}\n")
            for (a, n), seeds in sorted(by.items()):
                for s, v in sorted(seeds.items()):
                    fh.write(f"{a},{n},{s},{v}\n")
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
