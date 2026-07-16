## Installation on CSAIL SLURM Cluster

Follow these instructions on the SLURM cluster to set up a conda environment for Isaac Lab.

```bash
# 1. Create Conda env with the Vulkan loader (required for RTX cameras).
conda create -n IsaacLab \
  -c conda-forge --override-channels --strict-channel-priority \
  python=3.11 pip libvulkan-loader vulkan-tools
conda activate IsaacLab

# 2. Install pytorch version matching cluster's CUDA build. Consider changing versions based on your cluster configuration
pip install --index-url https://download.pytorch.org/whl/cu129 torch==2.9.1 torchvision==0.24.1

# 3. IsaacSim 5.1 from NVIDIA's PyPI
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com

# 4. Clone IsaacLab fork and install its editable extensions.
git clone https://github.com/Michaelszeng/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install

# 5. Smoke test on a GPU node (e.g. via `srun --gres=gpu:1 --pty bash`).
./isaaclab.sh -p scripts/tutorials/00_sim/launch_app.py --headless
vulkaninfo --summary | head -20
```

If `vulkaninfo` throws an error in the last step, add `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` to `~/.bashrc` or your SLURM preamble.