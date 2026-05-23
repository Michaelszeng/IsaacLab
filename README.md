![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Michael's Notes

## Installation

```bash
pip install PyQt5 zarr numcodecs
```

For policy evaluation, clone my [diffusion policy repo](https://github.com/Michaelszeng/diffusion-policy-experiments) and install (along with required dependencies):
```bash
pip install -e /home/michzeng/diffusion-policy --no-deps
pip install dill==0.3.5.1
pip install accelerate==0.13.2
pip install numba hydra-core zarr torchvision diffusers
pip install git+https://github.com/facebookresearch/r3m.git  # R3M visual encoder used by image-based checkpoints
```

### Installation on CSAIL SLURM Cluster

```bash
# 1. Conda env with the Vulkan loader (required for RTX cameras).
source /data/locomotion/michzeng/miniconda3/etc/profile.d/conda.sh
conda create -p /data/locomotion/michzeng/conda_envs/IsaacLab \
  -c conda-forge --override-channels --strict-channel-priority \
  python=3.11 pip libvulkan-loader vulkan-tools
conda activate /data/locomotion/michzeng/conda_envs/IsaacLab

# 2. PyTorch matched to the cluster's CUDA build.
#    Tedrake H200 nodes ship driver 575.57 → CUDA 12.9 → cu129.
#    The cu129 index only has torch ≥ 2.8; pick a 2.9.x build.
#    (For older drivers: cu124 with torch 2.4-2.5, cu121 with torch 2.3-2.5.)
pip install --index-url https://download.pytorch.org/whl/cu129 torch==2.9.1 torchvision==0.24.1

# 3. IsaacSim 5.1 from NVIDIA's PyPI
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com

# 4. Clone IsaacLab fork and install its editable extensions.
git clone https://github.com/Michaelszeng/IsaacLab.git /data/locomotion/michzeng/IsaacLab
cd /data/locomotion/michzeng/IsaacLab
./isaaclab.sh --install

# 5. Smoke test on a GPU node (e.g. via `srun --gres=gpu:1 --pty bash`).
./isaaclab.sh -p scripts/tutorials/00_sim/launch_app.py --headless
vulkaninfo --summary | head -20   # should list the H200 — if not, fix Vulkan ICD below.
```

Notes:
- **Vulkan ICD fix** (only if step 5 fails to find the GPU): add `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` to `~/.bashrc` or your SLURM preamble.
- **Sync local changes** to the cluster (excluding the local conda env and datasets):
  ```bash
  rsync -av --exclude=env --exclude=datasets \
    /home/michzeng/IsaacLab/ tedrake-h200-1:/data/locomotion/michzeng/IsaacLab/
  ```
- **Scale `--num_envs` up** for generation on the cluster — H200 has 143 GB VRAM, so 32-128 parallel envs is realistic (vs the 1-env limit on a local 2080 Ti).

Batch-submit a generation job (no interactive shell needed):
```bash
sbatch --gres=gpu:1 --time=4:00:00 --wrap="
  cd /data/locomotion/michzeng/IsaacLab && \
  source /data/locomotion/michzeng/miniconda3/etc/profile.d/conda.sh && \
  conda activate /data/locomotion/michzeng/conda_envs/IsaacLab && \
  python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
    --input_file ./datasets/gear_assembly_annotated.hdf5 \
    --output_file ./datasets/gear_assembly_generated.hdf5 \
    --generation_num_trials 1000 \
    --num_envs 32 \
    --enable_cameras --headless
"
```

## Data collection pipeline

Step 1 — Record ~10 source demos (SpaceMouse):

```bash
python scripts/tools/record_demos.py \
  --task Isaac-Insertion-Franka-IK-Rel-Mimic-v0 \
  --teleop_device spacemouse \
  --dataset_file ./datasets/insertion_source.hdf5 \
  --enable_cameras \
  --num_demos 30

python scripts/tools/record_demos.py \
  --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
  --teleop_device spacemouse \
  --dataset_file ./datasets/gear_assembly_source.hdf5 \
  --enable_cameras \
  --num_demos 30
```

Visualize data utility:

```bash
python scripts/tools/visualize_dataset.py ./datasets/insertion_source.hdf5

python scripts/tools/visualize_dataset.py ./datasets/gear_assembly_source.hdf5
```
    
Step 2 — Annotate subtask boundaries:

```bash
python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Insertion-Franka-IK-Rel-Mimic-v0 \
  --input_file ./datasets/insertion_source.hdf5 \
  --output_file ./datasets/insertion_annotated.hdf5 \
  --auto \
  --enable_cameras \
  --headless

python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
  --input_file ./datasets/gear_assembly_source.hdf5 \
  --output_file ./datasets/gear_assembly_annotated.hdf5 \
  --auto \
  --enable_cameras \
  --headless \
  --grasp_action_bounds 0 80
```

`--auto` uses the peg_grasped signal to find the grasp→insert boundary automatically.

Inspect the auto-generated boundaries using:

```bash
python scripts/tools/inspect_annotations.py ./datasets/insertion_source.hdf5

python scripts/tools/inspect_annotations.py ./datasets/gear_assembly_annotated.hdf5
```

Filter out bad episodes using:

```bash
python scripts/tools/filter_demos.py \
    ./datasets/insertion_source_annotated.hdf5 \
    ./datasets/insertion_source_clean.hdf5 \
    --drop-no-signal \
    --drop-boundary-outside 10 85 \
    --dry-run

python scripts/tools/filter_demos.py \
    ./datasets/gear_assembly_annotated.hdf5 \
    ./datasets/gear_assembly_clean.hdf5 \
    --drop-no-signal \
    --drop-boundary-outside 25 65 \
    --dry-run
```

Or, filter out a specific episode using (where `N` is zero-indexed): 

```bash
python scripts/tools/filter_demos.py \
  ./datasets/insertion_source_annotated.hdf5 \
  ./datasets/insertion_source_clean.hdf5 \
  --drop demo_N

python scripts/tools/filter_demos.py \
  ./datasets/gear_assembly_annotated.hdf5 \
  ./datasets/gear_assembly_clean.hdf5 \
  --drop demo_N
```


Step 3 — Generate augmented dataset:

```bash
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Insertion-Franka-IK-Rel-Mimic-v0 \
  --input_file ./datasets/insertion_annotated.hdf5 \
  --output_file ./datasets/insertion_generated.hdf5 \
  --generation_num_trials 200 \
  --enable_cameras \
  --num_envs 16

python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
  --input_file ./datasets/gear_assembly_annotated.hdf5 \
  --output_file ./datasets/gear_assembly_generated.hdf5 \
  --generation_num_trials 300 \
  --enable_cameras \
  --num_envs 16
```

Notes:
- `--generation_num_trials` is per-invocation, not a global total.
- Completed demos are flushed to the HDF5 as each episode terminates, so killing the script mid-run is safe.


Step 4 — Convert HDF5 → zarr for diffusion policy training:

```bash
python scripts/tools/hdf5_to_zarr.py \
  ./datasets/insertion_generated.hdf5 \
  ./datasets/insertion_generated.zarr

python scripts/tools/hdf5_to_zarr.py \
  ./datasets/gear_assembly_generated.hdf5 \
  /home/michzeng/diffusion-policy/data/diffusion_experiments/isaac_sim/gear_assembly_generated.zarr
```

Produces the standard diffusion-policy layout: every per-step obs becomes a flat `data/<key>` array (cameras at source resolution uint8, scalars cast to float64), with `meta/episode_ends` giving cumulative episode boundaries. Useful flags:
- `--cameras wrist_cam scene_cam_front` — keep only specific cameras.
- `--drop-failures` — skip demos with `success=False` (relevant when `generation_keep_failed=True`).
- `--dtype float32` — half the storage cost vs the default float64 if your training pipeline accepts it.
- `--overwrite` — delete an existing output zarr first.


Step 5 — Evaluate a trained diffusion policy:

```bash
python scripts/eval/evaluate_model_custom.py \
  --checkpoint /path/to/checkpoint.ckpt \
  --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
  --n-rollouts 50 \
  --enable_cameras
```

Writes `results.csv`, `summary.txt`, `results.pkl`, and `videos/` to `outputs/<date>/<time>/` (or `--output-dir`). Use `--resume` with the same `--output-dir` to pick up an interrupted eval.


---

# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


**Isaac Lab** is a GPU-accelerated, open-source framework designed to unify and simplify robotics research workflows,
such as reinforcement learning, imitation learning, and motion planning. Built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html),
it combines fast and accurate physics and sensor simulation, making it an ideal choice for sim-to-real
transfer in robotics.

Isaac Lab provides developers with a range of essential features for accurate sensor simulation, such as RTX-based
cameras, LIDAR, or contact sensors. The framework's GPU acceleration enables users to run complex simulations and
computations faster, which is key for iterative processes like reinforcement learning and data-intensive tasks.
Moreover, Isaac Lab can run locally or be distributed across the cloud, offering flexibility for large-scale deployments.

A detailed description of Isaac Lab can be found in our [arXiv paper](https://arxiv.org/abs/2511.04831).

## Key Features

Isaac Lab offers a comprehensive set of tools and environments designed to facilitate robot learning:

- **Robots**: A diverse collection of robots, from manipulators, quadrupeds, to humanoids, with more than 16 commonly available models.
- **Environments**: Ready-to-train implementations of more than 30 environments, which can be trained with popular reinforcement learning frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines. We also support multi-agent reinforcement learning.
- **Physics**: Rigid bodies, articulated systems, deformable objects
- **Sensors**: RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, ray casters.


## Getting Started

### Documentation

Our [documentation page](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started, including
detailed tutorials and step-by-step guides. Follow these links to learn more about:

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)


## Isaac Sim Version Dependency

Isaac Lab is built on top of Isaac Sim and requires specific versions of Isaac Sim that are compatible with each
release of Isaac Lab. Below, we outline the recent Isaac Lab releases and GitHub branches and their corresponding
dependency versions for Isaac Sim.

| Isaac Lab Version             | Isaac Sim Version         |
| ----------------------------- | ------------------------- |
| `main` branch                 | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.3.X`                      | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.2.X`                      | Isaac Sim 4.5 / 5.0       |
| `v2.1.X`                      | Isaac Sim 4.5             |
| `v2.0.X`                      | Isaac Sim 4.5             |


## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Show & Tell: Share Your Inspiration

We encourage you to utilize our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell)
area in the `Discussions` section of this repository. This space is designed for you to:

* Share the tutorials you've created
* Showcase your learning content
* Present exciting projects you've developed

By sharing your work, you'll inspire others and contribute to the collective knowledge
of our community. Your contributions can spark new ideas and collaborations, fostering
innovation in robotics and simulation.

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas,
  asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of
  work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features,
  or general updates.

## Connect with the NVIDIA Omniverse Community

Do you have a project or resource you'd like to share more widely? We'd love to hear from you!
Reach out to the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to explore opportunities
to spotlight your work.

You can also join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to
connect with other developers, share your projects, and help grow a vibrant, collaborative ecosystem
where creativity and technology intersect. Your contributions can make a meaningful impact on the Isaac Lab
community and beyond!

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its
corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its
dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

Note that Isaac Lab requires Isaac Sim, which includes components under proprietary licensing terms. Please see the [Isaac Sim license](docs/licenses/dependencies/isaacsim-license.txt) for information on Isaac Sim licensing.

Note that the `isaaclab_mimic` extension requires cuRobo, which has proprietary licensing terms that can be found in [`docs/licenses/dependencies/cuRobo-license.txt`](docs/licenses/dependencies/cuRobo-license.txt).


## Citation

If you use Isaac Lab in your research, please cite the technical report:

```
@article{mittal2025isaaclab,
  title={Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author={Mayank Mittal and Pascal Roth and James Tigue and Antoine Richard and Octi Zhang and Peter Du and Antonio Serrano-Muñoz and Xinjie Yao and René Zurbrügg and Nikita Rudin and Lukasz Wawrzyniak and Milad Rakhsha and Alain Denzler and Eric Heiden and Ales Borovicka and Ossama Ahmed and Iretiayo Akinola and Abrar Anwar and Mark T. Carlson and Ji Yuan Feng and Animesh Garg and Renato Gasoto and Lionel Gulich and Yijie Guo and M. Gussert and Alex Hansen and Mihir Kulkarni and Chenran Li and Wei Liu and Viktor Makoviychuk and Grzegorz Malczyk and Hammad Mazhar and Masoud Moghani and Adithyavairavan Murali and Michael Noseworthy and Alexander Poddubny and Nathan Ratliff and Welf Rehberg and Clemens Schwarke and Ritvik Singh and James Latham Smith and Bingjie Tang and Ruchik Thaker and Matthew Trepte and Karl Van Wyk and Fangzhou Yu and Alex Millane and Vikram Ramasamy and Remo Steiner and Sangeeta Subramanian and Clemens Volk and CY Chen and Neel Jawale and Ashwin Varghese Kuruttukulam and Michael A. Lin and Ajay Mandlekar and Karsten Patzwaldt and John Welsh and Huihua Zhao and Fatima Anes and Jean-Francois Lafleche and Nicolas Moënne-Loccoz and Soowan Park and Rob Stepinski and Dirk Van Gelder and Chris Amevor and Jan Carius and Jumyung Chang and Anka He Chen and Pablo de Heras Ciechomski and Gilles Daviet and Mohammad Mohajerani and Julia von Muralt and Viktor Reutskyy and Michael Sauter and Simon Schirm and Eric L. Shi and Pierre Terdiman and Kenny Vilella and Tobias Widmer and Gordon Yeoman and Tiffany Chen and Sergey Grizan and Cathy Li and Lotus Li and Connor Smith and Rafael Wiltz and Kostas Alexis and Yan Chang and David Chu and Linxi "Jim" Fan and Farbod Farshidian and Ankur Handa and Spencer Huang and Marco Hutter and Yashraj Narang and Soha Pouya and Shiwei Sheng and Yuke Zhu and Miles Macklin and Adam Moravanszky and Philipp Reist and Yunrong Guo and David Hoeller and Gavriel State},
  journal={arXiv preprint arXiv:2511.04831},
  year={2025},
  url={https://arxiv.org/abs/2511.04831}
}
```

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework.
We gratefully acknowledge the authors of Orbit for their foundational contributions.
