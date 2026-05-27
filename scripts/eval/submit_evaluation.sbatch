#!/bin/bash
#SBATCH --job-name=evaluate_pusht
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-titanrtx
#SBATCH --exclude=freeman-titanrtx-2,freeman-titanrtx-1,gpu20-2.drl
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage:
#   sbatch scripts/eval/submit_evaluation.sbatch <action_horizon> [action_horizon ...]
#
# SLURM runner for scripts/eval/evaluate_model_custom.py.
# Pass one or more action horizons; all checkpoints in CHECKPOINT_PATH are
# evaluated at each. Other tunables (TASK, N_ROLLOUTS, N_ENVS, N_VIDEO_TRIALS)
# are constants at the top of this script.

if [[ $# -eq 0 ]]; then
    echo "Usage: sbatch scripts/eval/submit_evaluation.sbatch <action_horizon> [action_horizon ...]"
    exit 1
fi
ACTION_HORIZONS=("$@")

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
echo "DATE: $DATE"
echo "TIME: $TIME"
echo "ACTION_HORIZONS: ${ACTION_HORIZONS[*]}"

# ============================================================================
# Evaluation parameters
# ============================================================================
# Full Isaac Lab task id (must match the checkpoints' training task).
# TASK="Isaac-Insertion-Franka-IK-Rel-Mimic-v0"
TASK="Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0"

# Directory containing one or more .ckpt files. The corresponding normalizer.pt is expected one level up.
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/isaac_sim/2_obs_gear_assembly_human_expert/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/isaac_sim/2_obs_gear_assembly_human_expert_attention_double_enc/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/isaac_sim/4_obs_gear_assembly_human_expert_attention_double_enc/checkpoints"
CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/isaac_sim/8_obs_gear_assembly_human_expert_attention_double_enc/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/isaac_sim/12_obs_gear_assembly_human_expert_attention_double_enc/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/isaac_sim/16_obs_gear_assembly_human_expert_attention_double_enc/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/isaac_sim/20_obs_gear_assembly_human_expert_attention_double_enc/checkpoints"

# Number of trials per (checkpoint, action_horizon) and parallel envs per round.
N_ROLLOUTS=500
N_ENVS=16

# Number of trials per (checkpoint, action_horizon) to save MP4 videos for.
# 0 = off, -1 = all. Anything > 0 forces video recording (a bit slower).
N_VIDEO_TRIALS=20

# Maximum rollout steps per trial (overrides the env's episode_length_s).
TASK_TIMEOUT=220
# ============================================================================

# Extract the experiment name as the directory immediately before "checkpoints".
EXPERIMENT_NAME="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "TASK:            $TASK"
echo "SLURM Job ID:    $SLURM_JOB_ID"
echo "SLURM Node:      $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# --- Environment setup ---
echo "Changing directory to $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change directory to $SLURM_SUBMIT_DIR"; exit 1; }
source /data/locomotion/michzeng/miniconda3/etc/profile.d/conda.sh
conda activate /data/locomotion/michzeng/conda_envs/IsaacLab
# conda activation can clear CUDA_VISIBLE_DEVICES; restore from SLURM's assignment.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-0}}}"
echo "CUDA_VISIBLE_DEVICES after conda activate: ${CUDA_VISIBLE_DEVICES}"

mkdir -p logs

export PYTHONUNBUFFERED=1
export HOME=/data/locomotion/michzeng
# Force NVIDIA's Vulkan ICD on the H200 nodes; system Mesa/virtio ICDs aren't
# usable and Isaac Sim's RTX cameras will silently render black frames.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

echo "CHECKPOINT_PATH:  $CHECKPOINT_PATH"
echo "ACTION_HORIZONS:  ${ACTION_HORIZONS[*]}"
echo "N_ROLLOUTS:       $N_ROLLOUTS"
echo "N_ENVS:           $N_ENVS"
echo "N_VIDEO_TRIALS:   $N_VIDEO_TRIALS"
echo "TASK_TIMEOUT:     $TASK_TIMEOUT"
echo "=========================================="

# --- Enumerate checkpoints ---
shopt -s nullglob
ALL_CHECKPOINTS=("${CHECKPOINT_PATH}"/*.ckpt)
shopt -u nullglob
CHECKPOINTS=()
for c in "${ALL_CHECKPOINTS[@]}"; do
    [[ "$(basename "$c")" == "latest.ckpt" ]] && continue
    CHECKPOINTS+=("$c")
done
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "No .ckpt files found in ${CHECKPOINT_PATH} (excluding latest.ckpt)"
    exit 1
fi
echo "Found ${#CHECKPOINTS[@]} checkpoint(s):"
for c in "${CHECKPOINTS[@]}"; do echo "  - $c"; done
echo "=========================================="

# --- Run all (checkpoint, action_horizon) pairs ---
for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_stem="$(basename "${ckpt}" .ckpt)"
    for h in "${ACTION_HORIZONS[@]}"; do
        OUTPUT_DIR="outputs/${EXPERIMENT_NAME}/${ckpt_stem}/ah${h}"
        echo ""
        echo ">>> Evaluating ${ckpt_stem} @ action_horizon=${h}"
        echo ">>> Output:    ${OUTPUT_DIR}"
        echo ""
        python scripts/eval/evaluate_model_custom.py \
            --checkpoint "${ckpt}" \
            --task "${TASK}" \
            --n-rollouts "${N_ROLLOUTS}" \
            --n-envs "${N_ENVS}" \
            --n-action-steps "${h}" \
            --n-video-trials "${N_VIDEO_TRIALS}" \
            --task-timeout "${TASK_TIMEOUT}" \
            --output-dir "${OUTPUT_DIR}" \
            --resume \
            --enable_cameras \
            --headless
    done
done

echo "=========================================="
echo "Done. Results under outputs/${EXPERIMENT_NAME}/"
