# Perception Module Design (Person 1)

## Input
| Topic | Type | Description |
|---|---|---|
| /camera/image | sensor_msgs/Image | RGB camera frame from the simulator |

## Outputs (proposed, to be agreed with the team)
| Topic | Type | Description |
|---|---|---|
| /perception/segmentation | sensor_msgs/Image (mono8) | Per-pixel class ID mask |
| /perception/rocks | vision_msgs/Detection2DArray | Rock boxes, class, confidence |
| /perception/craters | vision_msgs/Detection2DArray | Crater boxes or masks, confidence |

## Segmentation class IDs
| ID | Class |
|---|---|
| 0 | regolith/soil |
| 1 | crater |
| 2 | rock |
| 3 | mountain/slope |
| 4 | sky |

## Pipeline
image -> preprocess -> segmentation model -> mask
image -> preprocess -> rock detector -> rocks
image -> preprocess -> crater detector -> craters