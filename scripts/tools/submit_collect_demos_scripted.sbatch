#!/bin/bash
#SBATCH --job-name=isaaclab_collect_demos_scripted
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-titanrtx
#SBATCH --exclude=freeman-titanrtx-2,freeman-titanrtx-1,gpu20-2.drl,improbablex002
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage:
#   sbatch scripts/tools/submit_collect_demos_scripted.sbatch
#
# Runs the scripted-expert data collection on the configured task and then
# converts the resulting HDF5 to zarr. Set CONVERT_ONLY=1 in the environment
# to skip the collection step (useful if you already have an HDF5 and just
# want to (re)convert it to zarr).

# ============================================================================
# Collection + conversion parameters
# ============================================================================
# Full Isaac Lab task id.
TASK_ID="Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0"

# Output HDF5 / zarr paths. The zarr path is derived from the HDF5 path.
DATASET_HDF5="./datasets/gear_assembly_scripted.hdf5"
DATASET_ZARR="${DATASET_HDF5%.hdf5}.zarr"

# Total successful demos to collect, parallel envs per rollout batch, and the
# per-demo step cap (forces stuck rollouts to recycle).
NUM_DEMOS=200
NUM_ENVS=16
MAX_STEPS_PER_DEMO=300

# If 1, delete any pre-existing DATASET_HDF5 before collecting so the run
# starts from a clean file. If 0, collect_demos_scripted.py appends to the
# existing file until --num_demos total demos are present.
FRESH_COLLECTION=1

# Set to 1 to skip collection (e.g. if DATASET_HDF5 already exists and you
# just want to (re)convert it to zarr).
SKIP_COLLECTION=0
# ============================================================================

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
echo "DATE: $DATE"
echo "TIME: $TIME"
echo "TASK_ID:            $TASK_ID"
echo "NUM_DEMOS:          $NUM_DEMOS"
echo "NUM_ENVS:           $NUM_ENVS"
echo "MAX_STEPS_PER_DEMO: $MAX_STEPS_PER_DEMO"
echo "SLURM Job ID:       $SLURM_JOB_ID"
echo "SLURM Node:         $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_SUBMIT_DIR:   $SLURM_SUBMIT_DIR"

# --- Environment setup ---
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to cd to $SLURM_SUBMIT_DIR"; exit 1; }
source /data/locomotion/michzeng/miniconda3/etc/profile.d/conda.sh
conda activate /data/locomotion/michzeng/conda_envs/IsaacLab
# Conda's activate hook can clear CUDA_VISIBLE_DEVICES; restore from SLURM.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-0}}}"
echo "CUDA_VISIBLE_DEVICES after conda activate: ${CUDA_VISIBLE_DEVICES}"

mkdir -p logs datasets

export PYTHONUNBUFFERED=1
export HOME=/data/locomotion/michzeng
# Force NVIDIA's Vulkan ICD; the system Mesa/virtio ICDs aren't usable on the
# H200 nodes and Isaac Sim's RTX cameras will silently fall back to no GPU.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

echo "DATASET_HDF5: $DATASET_HDF5"
echo "DATASET_ZARR: $DATASET_ZARR"
echo "=========================================="

# --- Step 1: Collect ---
if [[ "${CONVERT_ONLY:-0}" != "1" && "${SKIP_COLLECTION}" != "1" ]]; then
    if [[ "${FRESH_COLLECTION}" == "1" && -f "$DATASET_HDF5" ]]; then
        echo "Dataset HDF5 already exists, removing for fresh collection: $DATASET_HDF5"
        rm -f "$DATASET_HDF5"
    fi
    echo ">>> Collecting: $DATASET_HDF5 (demos=$NUM_DEMOS, envs=$NUM_ENVS)"
    python scripts/tools/collect_demos_scripted.py \
        --task "$TASK_ID" \
        --dataset_file "$DATASET_HDF5" \
        --num_envs "$NUM_ENVS" \
        --num_demos "$NUM_DEMOS" \
        --max_steps_per_demo "$MAX_STEPS_PER_DEMO" \
        --enable_cameras \
        --headless
else
    echo ">>> Skipping collection step (CONVERT_ONLY=1 or SKIP_COLLECTION=1)."
fi

# --- Step 2: Convert HDF5 -> zarr ---
if [[ ! -f "$DATASET_HDF5" ]]; then
    echo "Dataset HDF5 not found, cannot convert: $DATASET_HDF5"
    exit 1
fi
echo ">>> Converting HDF5 to zarr: $DATASET_HDF5 -> $DATASET_ZARR"
python scripts/tools/hdf5_to_zarr.py \
    "$DATASET_HDF5" \
    "$DATASET_ZARR" \
    --overwrite

echo "=========================================="
echo "Done. Outputs: $DATASET_HDF5  $DATASET_ZARR"
