#!/usr/bin/env bash
# Is DDIM's sampling penalty the same for the quantum arm as for the classical one?
#
# The whole arm comparison assumes it is: DDIM-50 costs +0.035 KID on classical
# arms, and that is only harmless if it is common-mode. If the quantum arm's
# penalty differs by more than the quantum-vs-se effect, the comparison is
# measuring the sampler, not the circuit.
#
# Usage: check_ddim_common_mode.sh <quantum_run_dir> <se_run_dir>
set -euo pipefail
QDIR="$1"; SDIR="$2"
for d in "$QDIR" "$SDIR"; do
  arm=$(python3 -c "import json;print(json.load(open('$d/manifest.json'))['arm'])")
  for spec in "ddim 50" "ddpm 1000"; do
    set -- $spec
    gq submit --cpu-only -c 4 -m 4 -p 10 -t 24h -n cm-$arm-$1$2 -- \
      python eval_cell.py --run-dir "$d" --n 2000 --n-real 2000 --sampler $1 --steps $2
  done
done
echo "submitted 4 jobs: {quantum,se} x {ddim50,ddpm1000}"
echo "penalty(arm) = KID(ddim50) - KID(ddpm1000); common-mode iff the two penalties agree"
