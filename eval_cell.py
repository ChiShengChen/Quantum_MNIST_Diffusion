"""Generate samples from one sweep cell and score them, in one job.

Reads the cell's manifest.json so the architecture cannot drift from what was
trained -- getting --arm or --n-hidden wrong is the failure mode that silently
evaluates a different model (or, with PR #8's strict loading, fails outright).

    python eval_cell.py --run-dir runs/mnist_3_se_n100_s0 --n 2000 \
        --sampler ddim --steps 50
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

ARCH_FOR = {"v7": "v7_mnist", "v8": "v8_mnist", "pathmnist": "v7_pathmnist"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", default="best_model.pth")
    ap.add_argument("--keep-samples", action="store_true",
                    help="keep the .pt of generated samples (2000 imgs ~ 6 MB)")
    args = ap.parse_args()

    run = args.run_dir
    manifest = json.loads((run / "manifest.json").read_text())
    arm = manifest["arm"]
    n_hidden = manifest.get("n_hidden", 16)
    arch = ARCH_FOR[manifest.get("script", "v7")]
    label = manifest.get("digit", manifest.get("label"))
    dataset = "pathmnist" if arch == "v7_pathmnist" else "mnist"

    samples = run / f"eval_{args.sampler}{args.steps}_n{args.n}.pt"
    out_json = run / f"metrics_{args.sampler}{args.steps}_n{args.n}.json"

    gen = [sys.executable, str(REPO_ROOT / "generate_samples.py"),
           "--arch", arch, "--checkpoint", str(run / args.checkpoint),
           "--arm", arm, "--n-hidden", str(n_hidden),
           "--sampler", args.sampler, "--steps", str(args.steps),
           "--n", str(args.n), "--batch-size", str(args.batch_size),
           "--device", args.device, "--out", str(samples)]
    if arm == "quantum":
        gen += ["--qdevice", manifest.get("qdevice") or "lightning.qubit",
                "--diff-method", manifest.get("diff_method") or "adjoint"]
    subprocess.run(gen, check=True)

    ev = [sys.executable, str(REPO_ROOT / "evaluate.py"),
          "--generated", str(samples), "--dataset", dataset,
          "--n-real", str(args.n_real), "--device", args.device,
          "--json-out", str(out_json)]
    if label is not None:
        ev += ["--label", str(label)]
    subprocess.run(ev, check=True)

    # Fold the cell's identity into the metrics file so aggregation needs one read.
    d = json.loads(out_json.read_text())
    d.update({k: manifest.get(k) for k in
              ("arm", "digit", "label", "n_train", "seed", "total_steps",
               "n_hidden", "bottleneck_trainable_params", "best_mean_loss")})
    d["sampler"] = args.sampler
    d["sampler_steps"] = args.steps
    out_json.write_text(json.dumps(d, indent=2))
    print(f"{run.name}: KID {d['kid_mean']:.5f} +- {d['kid_std']:.5f}  FID {d['fid']:.2f}")

    if not args.keep_samples:
        samples.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
