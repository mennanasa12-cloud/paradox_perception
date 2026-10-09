# PARADOX Perception

Computer vision module for the PARADOX Autonomous Lunar Rover.
Takes the rover camera image and produces terrain segmentation,
rock detections and crater detections.

## Tasks
1. Dataset preparation
2. Terrain segmentation
3. Rock detection
4. Crater detection
5. ROS 2 perception node

## Folder structure
- `configs/`: settings (paths, classes, hyperparameters)
- `data/`: datasets (not tracked by Git)
- `notebooks/`: experiments
- `scripts/`: train and evaluate scripts
- `src/perception/`: the module code (dataset, segmentation, detection, utils)
- `ros2_ws/`: ROS 2 workspace (Task 5)
- `docs/`: design documents
- `tests/`: tests

## Datasets
- LuSNAR (primary)
- Lunar Rocks & Craters, Zenodo (secondary)

## Design
See `docs/perception_design.md`.