"""Enumerate the data-scaling grid from issue #5.

The grid is (N, arm, seed). It emits one shell command per cell rather than
running them in-process, so the cells can be fed to whatever scheduler is at hand
and so a crashed cell doesn't take the sweep with it.

    # see the grid and what it costs before committing to it
    python run_scaling_study.py --digits 3 --dry-run

    # emit commands for a queue
    python run_scaling_study.py --digits 3 | xargs -L1 -I{} gq submit -g 2G -- {}

    # or just run them one at a time
    python run_scaling_study.py --digits 3 --execute

Budget note: the step budget is fixed across N on purpose. An epoch at N=10 is a
single gradient step, so an epoch-based budget would leave the low-N arms
undertrained rather than data-limited -- which is exactly the confound the study
is trying to measure.
"""

import argparse
import itertools
import shlex
import subprocess
import sys

from bottlenecks import ARMS

DEFAULT_N = (10, 25, 50, 100, 250, 500, 1000, None)  # None = the whole class

# Measured on one CPU core of this machine, n_qubits=16: the quantum arm costs
# ~21 s/step against ~0.06 s/step for the classical arms, because it backpropagates
# through a 2^16 statevector. Sampling 1000 reverse steps adds a fixed tail.
SEC_PER_STEP = {"plain": 0.06, "se": 0.06, "se_frozen": 0.06, "quantum": 21.0}
SAMPLE_TAIL_SEC = {"plain": 30, "se": 30, "se_frozen": 30, "quantum": 3300}


def cells(digits, ns, arms, seeds):
    for digit, n, arm, seed in itertools.product(digits, ns, arms, seeds):
        yield digit, n, arm, seed


# Scripts whose budget can be given in optimizer steps. Cells at different N are
# only comparable on these: at N=10 an epoch is one gradient step, so an
# epoch-based budget leaves the low-N arms undertrained rather than data-limited,
# and the resulting curve conflates "less data" with "fewer updates".
STEP_BUDGET_SCRIPTS = (
    "quantum_difussion_mnist_v7.py",
    "quantum_diffusion_mnist_v8.py",
    "quantum_difussion_pathmnist_v7.py",
)


def supports_step_budget(script):
    return any(script.endswith(s) for s in STEP_BUDGET_SCRIPTS)


def command(script, digit, n, arm, seed, steps, save_dir, n_hidden):
    cmd = [sys.executable, script, "--arm", arm, "--digits", str(digit),
           "--seed", str(seed), "--save-dir", save_dir, "--n-hidden", str(n_hidden)]
    cmd += ["--max-steps", str(steps)] if supports_step_budget(script) else []
    if n is not None:
        cmd += ["--n-train", str(n)]
    return cmd


def estimate_hours(arms, n_cells_per_arm, steps):
    total = 0.0
    for arm in arms:
        total += n_cells_per_arm * (steps * SEC_PER_STEP[arm] + SAMPLE_TAIL_SEC[arm])
    return total / 3600.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", default="quantum_difussion_mnist_v7.py")
    ap.add_argument("--digits", type=int, nargs="+", default=[3])
    ap.add_argument("--n-train", type=int, nargs="+", default=None,
                    help=f"N values (default: {DEFAULT_N}, where the last is the full class)")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="at least 5; at small N the data draw dominates")
    ap.add_argument("--max-steps", type=int, default=2000,
                    help="fixed across N so the arms are compute-matched")
    ap.add_argument("--n-hidden", type=int, default=16)
    ap.add_argument("--save-dir", default="runs")
    ap.add_argument("--dry-run", action="store_true", help="print the grid and the cost estimate only")
    ap.add_argument("--execute", action="store_true", help="run the cells sequentially in-process")
    args = ap.parse_args()

    ns = args.n_train if args.n_train is not None else list(DEFAULT_N)
    arms = args.arms
    if "se_frozen" in arms and "_v8" in args.script:
        print("note: dropping se_frozen -- v8's circuit pools its own input, so "
              "there is no down-projection to freeze", file=sys.stderr)
        arms = [a for a in arms if a != "se_frozen"]
    if len(ns) > 1 and not supports_step_budget(args.script):
        print(f"warning: {args.script} has no --max-steps, so it trains a fixed "
              f"number of *epochs*. Cells at different N are therefore not "
              f"compute-matched and the resulting curve conflates 'less data' "
              f"with 'fewer gradient steps'. Port the step budget first.",
              file=sys.stderr)
    grid = list(cells(args.digits, ns, arms, args.seeds))
    per_arm = len(args.digits) * len(ns) * len(args.seeds)

    if args.dry_run:
        hours = estimate_hours(arms, per_arm, args.max_steps)
        print(f"{len(grid)} cells: {len(args.digits)} digit(s) x {len(ns)} N x "
              f"{len(arms)} arm(s) x {len(args.seeds)} seed(s)")
        print(f"  N values : {ns}")
        print(f"  arms     : {arms}")
        print(f"  budget   : {args.max_steps} steps per cell (fixed across N)")
        print(f"\nrough serial cost: {hours:.1f} h "
              f"({hours/24:.1f} days) on one CPU core, dominated by the quantum arm")
        if "quantum" in arms:
            q_hours = estimate_hours(["quantum"], per_arm, args.max_steps)
            print(f"  quantum arm alone: {q_hours:.1f} h of that")
            print("  -> run the classical arms first; they are ~350x cheaper per step")
        return

    for digit, n, arm, seed in grid:
        cmd = command(args.script, digit, n, arm, seed, args.max_steps,
                      args.save_dir, args.n_hidden)
        if args.execute:
            print(f"### {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
            subprocess.run(cmd, check=False)
        else:
            print(" ".join(shlex.quote(c) for c in cmd))


if __name__ == "__main__":
    main()
