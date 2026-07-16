# Isaac Lab

This repo is a fork of [Issac Lab](https://github.com/isaac-sim/IsaacLab). 

This fork contains the following additions to Isaac Lab:
 - Gear Insertion Environment and MimicGen data generation pipeline used in [Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performance Policie]().
 - Markovian scripted policy for automated data generation for Gear Insertion.

## Installation

Firstly, follow the official Isaac Lab installation instructions and [documentation page](https://isaac-sim.github.io/IsaacLab):

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)


Additional `pip` installs:

```bash
pip install PyQt5 zarr numcodecs
```

Specific instructions for installing on a SLURM cluster (such as MIT CSAIL SLURM) are in `SLURM_README.md`.


## Teleop + MimicGen Data Collection Pipeline

NOTE: while the MimicGen data generation pipeline can generate many more demonstrations, they replayed/stitched versions of the original source demos (see [MimicGen](https://mimicgen.github.io/)). Make sure to collect enough source demos so your dataset has sufficient variety.

Step 1 — Record source demos (using a 3Dconnexion SpaceMouse Wireless 3DX-700043):

```bash
python scripts/tools/record_demos.py \
  --task Isaac-Insertion-Franka-IK-Rel-Mimic-v0 \
  --teleop_device spacemouse \
  --dataset_file ./datasets/insertion_source.hdf5 \
  --enable_cameras \
  --num_demos 50

python scripts/tools/record_demos.py \
  --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
  --teleop_device spacemouse \
  --dataset_file ./datasets/gear_assembly_source.hdf5 \
  --enable_cameras \
  --num_demos 50
```

Optionally, use this utility script to visualize your collected data:

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
```

Inspect the auto-generated boundaries using:

```bash
python scripts/tools/inspect_annotations.py ./datasets/insertion_source.hdf5

python scripts/tools/inspect_annotations.py ./datasets/gear_assembly_annotated.hdf5
```

Filter out badly-annotated episodes using:

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

Or, filter out a specific episode using (where `demo_N` is zero-indexed): 

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

Produces a zarr the standard diffusion-policy layout. Useful flags:
- `--cameras wrist_cam scene_cam_front` — keep only specific cameras.
- `--drop-failures` — skip demos with `success=False`.
- `--dtype float32` — half the storage cost vs the default float64 if your training pipeline accepts it.
- `--overwrite` — overwrite an existing output zarr (otherwise, this script will throw if the output zarr exists already).

WARNING: while the instructions here encompass both the Insertion and Gear Assembly tasks, the Insertion task has not been well tested. Use at your own risk.


## Markovian Scripted Expert Data Generation Pipeline

`scripts/tools/collect_demos_scripted.py` collects demonstrations automatically using a scripted finite-state machine (FSM). This FSM is Markovian in the sense that it determines its action based purely on its current observation. Only successful episodes are written to the outputted hdf5.

```bash
python scripts/tools/collect_demos_scripted.py \
  --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
  --dataset_file ./datasets/gear_assembly_source.hdf5 \
  --num_envs 32 \
  --num_demos 200 \
  --enable_cameras \
  --headless
```

To watch a single rollout in the GUI (drop `--headless`; `--num_envs` is forced to 1):

```bash
python scripts/tools/collect_demos_scripted.py \
  --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
  --dataset_file ./datasets/gear_assembly_source.hdf5 \
  --enable_cameras
```

Key flags:
- `--num_demos N` — target *total* successful demos in the file (`0` = run indefinitely). Existing demos in the file count toward the target.
- `--num_envs N` — parallel envs (headless only)
- `--n-video-trials N` — save MP4s of the first `N` attempts to `<dataset_dir>/scripted_videos/` for sanity-checking.


## Policy Evaluation

This repo contains an evaluation script for policies trained using my [diffusion-policy-experiments](https://github.com/michaelszeng/diffusion-policy-experiments) repo. It may also be adapted for other training pipelines.

The following installations are required to run policies trained with [diffusion-policy-experiments](https://github.com/michaelszeng/diffusion-policy-experiments):
```bash
git clone git@github.com:Michaelszeng/diffusion-policy-experiments.git
pip install -e /path/to/diffusion-policy-experiments --no-deps
pip install dill==0.3.5.1 accelerate==0.13.2 diffusers==0.11.1 hydra-core==1.2.0 "zarr<3" "numcodecs>=0.11.0" "numba>=0.61.0" deprecated imageio==2.22.0 imageio-ffmpeg==0.4.7 "opencv-python>=4.5.0" einops==0.4.1
pip install "transformers>=4.40,<4.45" "huggingface-hub==0.25.2"
pip install "r3m @ https://github.com/facebookresearch/r3m/archive/b2334e726887fa0206962d7984c69c5fb09cceab.tar.gz"
# pin to NumPy 1.x. numba ≥0.61 still supports this (numpy 1.24-2.1 range).
pip install "numpy<2"
```

After training a policy in [diffusion-policy-experiments](https://github.com/michaelszeng/diffusion-policy-experiments), evaluate using:


```bash
python scripts/eval/evaluate_model_custom.py \
  --checkpoint /path/to/checkpoint.ckpt \
  --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
  --n-rollouts 500 \
  --enable_cameras
```

This scripte writes results to `outputs/<date>/<time>/` (or `--output-dir`). Use `--resume` with the same `--output-dir` to resume an interrupted eval.



## Citation

If you use this repo in your research, please cite both the accompanying work: 

```
TODO
```

And the Isaac Lab technical report:

```
@article{mittal2025isaaclab,
  title={Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author={Mayank Mittal and Pascal Roth and James Tigue and Antoine Richard and Octi Zhang and Peter Du and Antonio Serrano-Muñoz and Xinjie Yao and René Zurbrügg and Nikita Rudin and Lukasz Wawrzyniak and Milad Rakhsha and Alain Denzler and Eric Heiden and Ales Borovicka and Ossama Ahmed and Iretiayo Akinola and Abrar Anwar and Mark T. Carlson and Ji Yuan Feng and Animesh Garg and Renato Gasoto and Lionel Gulich and Yijie Guo and M. Gussert and Alex Hansen and Mihir Kulkarni and Chenran Li and Wei Liu and Viktor Makoviychuk and Grzegorz Malczyk and Hammad Mazhar and Masoud Moghani and Adithyavairavan Murali and Michael Noseworthy and Alexander Poddubny and Nathan Ratliff and Welf Rehberg and Clemens Schwarke and Ritvik Singh and James Latham Smith and Bingjie Tang and Ruchik Thaker and Matthew Trepte and Karl Van Wyk and Fangzhou Yu and Alex Millane and Vikram Ramasamy and Remo Steiner and Sangeeta Subramanian and Clemens Volk and CY Chen and Neel Jawale and Ashwin Varghese Kuruttukulam and Michael A. Lin and Ajay Mandlekar and Karsten Patzwaldt and John Welsh and Huihua Zhao and Fatima Anes and Jean-Francois Lafleche and Nicolas Moënne-Loccoz and Soowan Park and Rob Stepinski and Dirk Van Gelder and Chris Amevor and Jan Carius and Jumyung Chang and Anka He Chen and Pablo de Heras Ciechomski and Gilles Daviet and Mohammad Mohajerani and Julia von Muralt and Viktor Reutskyy and Michael Sauter and Simon Schirm and Eric L. Shi and Pierre Terdiman and Kenny Vilella and Tobias Widmer and Gordon Yeoman and Tiffany Chen and Sergey Grizan and Cathy Li and Lotus Li and Connor Smith and Rafael Wiltz and Kostas Alexis and Yan Chang and David Chu and Linxi "Jim" Fan and Farbod Farshidian and Ankur Handa and Spencer Huang and Marco Hutter and Yashraj Narang and Soha Pouya and Shiwei Sheng and Yuke Zhu and Miles Macklin and Adam Moravanszky and Philipp Reist and Yunrong Guo and David Hoeller and Gavriel State},
  journal={arXiv preprint arXiv:2511.04831},
  year={2025},
  url={https://arxiv.org/abs/2511.04831}
}
```