#!/bin/bash
# Submit one SLURM job per action horizon by invoking
# scripts/eval/submit_evaluation.sbatch with a single horizon at a time.
#
# Usage:
#   ./scripts/eval/batch_submit.sh
#
# Run from the repo root (same place you'd normally run `sbatch`).

set -euo pipefail

ACTION_HORIZONS=(1 2 3 4 5 6 8 10 12 15)
SBATCH_SCRIPT="scripts/eval/submit_evaluation.sbatch"

echo "Submitting ${#ACTION_HORIZONS[@]} jobs (one per action horizon)..."
for h in "${ACTION_HORIZONS[@]}"; do
    echo "  -> sbatch --job-name=evaluate_isaaclab_ah${h} ${SBATCH_SCRIPT} ${h}"
    sbatch --job-name="evaluate_isaaclab_ah${h}" "${SBATCH_SCRIPT}" "${h}"
done
