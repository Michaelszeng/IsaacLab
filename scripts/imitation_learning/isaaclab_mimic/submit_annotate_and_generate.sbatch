#!/bin/bash
#SBATCH --job-name=isaaclab_mimic_annotate_and_generate
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-titanrtx
#SBATCH --exclude=freeman-titanrtx-2,freeman-titanrtx-1,gpu20-2.drl
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
#   sbatch scripts/imitation_learning/isaaclab_mimic/submit_annotate_and_generate.sbatch
#
# Runs annotation (with --auto) on the source HDF5, then runs MimicGen
# generation on the annotated file. Set GENERATION_ONLY=1 in the
# environment to skip the annotation step (useful if you've already
# produced *_annotated.hdf5 and just want to regenerate).

# ============================================================================
# Annotation + generation parameters
# ============================================================================
# Full Isaac Lab task id.
TASK_ID="Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0"
# TASK_ID="Isaac-Insertion-Franka-IK-Rel-Mimic-v0"

# Input source / annotated / generated HDF5 paths.
SOURCE_HDF5="./datasets/gear_assembly_source.hdf5"
ANNOTATED_HDF5="./datasets/gear_assembly_annotated.hdf5"
GENERATED_HDF5="./datasets/gear_assembly_generated_750.hdf5"

# Zarr output paths (derived from the HDF5 paths above).
SOURCE_ZARR="${SOURCE_HDF5%.hdf5}.zarr"
GENERATED_ZARR="${GENERATED_HDF5%.hdf5}.zarr"

# Extra flags passed only to annotate_demos.py (leave empty to omit).
# Gear-assembly source demos benefit from --grasp_action_bounds; insertion
# does not currently use this.
GRASP_BOUNDS="--grasp_action_bounds 0 80"
# GRASP_BOUNDS=""

# Set to 1 to skip the annotation step (e.g. if *_annotated.hdf5 already exists).
SKIP_ANNOTATION=1

# Total demos to generate and parallel envs per round.
NUM_DEMOS=750
NUM_ENVS=16
# ============================================================================

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
echo "DATE: $DATE"
echo "TIME: $TIME"
echo "TASK_ID:  $TASK_ID"
echo "NUM_DEMOS: $NUM_DEMOS"
echo "NUM_ENVS:  $NUM_ENVS"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node:   $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

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

echo "SOURCE_HDF5:    $SOURCE_HDF5"
echo "ANNOTATED_HDF5: $ANNOTATED_HDF5"
echo "GENERATED_HDF5: $GENERATED_HDF5"
echo "SOURCE_ZARR:    $SOURCE_ZARR"
echo "GENERATED_ZARR: $GENERATED_ZARR"
echo "=========================================="

# --- Step 1: Annotate ---
if [[ "${GENERATION_ONLY:-0}" != "1" && "${SKIP_ANNOTATION}" != "1" ]]; then
    if [[ ! -f "$SOURCE_HDF5" ]]; then
        echo "Source HDF5 not found: $SOURCE_HDF5"
        exit 1
    fi
    if [[ -f "$ANNOTATED_HDF5" ]]; then
        echo "Annotated HDF5 already exists, removing: $ANNOTATED_HDF5"
        rm -f "$ANNOTATED_HDF5"
    fi
    echo ">>> Annotating: $SOURCE_HDF5 -> $ANNOTATED_HDF5"
    python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
        --task "$TASK_ID" \
        --input_file "$SOURCE_HDF5" \
        --output_file "$ANNOTATED_HDF5" \
        --auto \
        --enable_cameras \
        --headless \
        $GRASP_BOUNDS
else
    echo ">>> Skipping annotation step (GENERATION_ONLY=1 or SKIP_ANNOTATION=1)."
fi

# --- Step 2: Generate ---
if [[ ! -f "$ANNOTATED_HDF5" ]]; then
    echo "Annotated HDF5 not found, cannot generate: $ANNOTATED_HDF5"
    exit 1
fi
if [[ -f "$GENERATED_HDF5" ]]; then
    echo "Generated HDF5 already exists, removing: $GENERATED_HDF5"
    rm -f "$GENERATED_HDF5"
fi
echo ">>> Generating: $ANNOTATED_HDF5 -> $GENERATED_HDF5 (demos=$NUM_DEMOS, envs=$NUM_ENVS)"
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task "$TASK_ID" \
    --input_file "$ANNOTATED_HDF5" \
    --output_file "$GENERATED_HDF5" \
    --generation_num_trials "$NUM_DEMOS" \
    --num_envs "$NUM_ENVS" \
    --enable_cameras \
    --headless

echo "=========================================="
echo "Done. Output: $GENERATED_HDF5"

# --- Step 3: Convert source and generated HDF5 → zarr ---
echo ">>> Converting source HDF5 to zarr: $SOURCE_HDF5 -> $SOURCE_ZARR"
python scripts/tools/hdf5_to_zarr.py \
    "$SOURCE_HDF5" \
    "$SOURCE_ZARR" \
    --overwrite

echo ">>> Converting generated HDF5 to zarr: $GENERATED_HDF5 -> $GENERATED_ZARR"
python scripts/tools/hdf5_to_zarr.py \
    "$GENERATED_HDF5" \
    "$GENERATED_ZARR" \
    --overwrite

echo "=========================================="
echo "Done. Zarr outputs: $SOURCE_ZARR  $GENERATED_ZARR"
