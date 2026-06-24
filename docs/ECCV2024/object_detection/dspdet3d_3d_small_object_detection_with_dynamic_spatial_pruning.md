---
title: >-
  [Paper Note] DSPDet3D: 3D Small Object Detection with Dynamic Spatial Pruning
description: >-
  [ECCV2024][Object Detection][3D Small Object Detection] Proposed a Dynamic Spatial Pruning (DSP) strategy to progressively remove voxel features in areas where large objects have already been detected within the decoders of multi-scale 3D detectors. This allows the detector to process scenes at extremely high spatial resolutions, significantly improving small object detection accuracy (ScanNet small object mAP@0.25 boosted from 27.5% to 44.8%) while reducing GPU memory to 1/5…
tags:
  - "ECCV2024"
  - "Object Detection"
  - "3D Small Object Detection"
  - "Dynamic Spatial Pruning"
  - "Sparse Convolution"
  - "Multi-scale Detection"
  - "Point Cloud"
date: 2026-05-08
content_hash: 5b3bf486c9bbfe00
---

# DSPDet3D: 3D Small Object Detection with Dynamic Spatial Pruning

**Conference**: ECCV2024  
**arXiv**: [2305.03716](https://arxiv.org/abs/2305.03716)  
**Code**: [https://github.com/xuxw98/DSPDet3D](https://github.com/xuxw98/DSPDet3D)  
**Area**: Object Detection  
**Keywords**: 3D Small Object Detection, Dynamic Spatial Pruning, Sparse Convolution, Multi-scale Detection, Point Cloud

## TL;DR
Proposed a Dynamic Spatial Pruning (DSP) strategy to progressively remove voxel features in areas where large objects have already been detected within the decoders of multi-scale 3D detectors. This allows the detector to process scenes at extremely high spatial resolutions, significantly improving small object detection accuracy (ScanNet small object mAP@0.25 boosted from 27.5% to 44.8%) while reducing GPU memory to 1/5 of the baseline method with the same resolution.

## Background & Motivation

**Background**: Indoor 3D object detection has achieved significant progress, and mainstream methods (VoteNet, FCAF3D, TR3D) perform well on medium-to-large objects like furniture. However, most only focus on major categories such as tables, chairs, and beds, neglecting small everyday items like cups, keyboards, and bottles.

**Limitations of Prior Work**: Small objects have extremely sparse point clouds (only tens to hundreds of points), and traditional downsampling for scene representation extraction inevitably destroys their geometric details. While simply increasing spatial resolution (finer voxels) is effective, generative upsampling in the decoder leads to a voxel numbers explosion—doubling the resolution of TR3D causes the GPU memory to surge from 1250MB to 4450MB.

**Key Challenge**: The contradiction between high resolution (essential for small objects) and computational efficiency (essential for actual deployment). Decoder layers account for the vast majority of memory and computational overhead.

**Goal**: How to leverage high-resolution features to detect small objects without increasing GPU memory?

**Key Insight**: Small objects occupy only a tiny fraction of the scene space. Once large objects are detected at coarser levels, the voxels in those occupied regions become redundant for subsequent finer-level detection and can be safely pruned.

**Core Idea**: Dynamically prune voxel features of detected regions after each scale of the multi-scale detector, allowing high-resolution layers to only process the minimal regions containing small objects.

## Method

### Overall Architecture
DSPDet3D is based on the multi-scale FCOS-style architecture of TR3D, utilizing a sparse convolutional backbone to extract four-level features. Key modifications include: (1) removing max pooling in the backbone to preserve higher resolution (with the finest voxel size of 4cm); (2) replacing the decoder with four stacked DSP (Dynamic Spatial Pruning) modules. After voxelizing the input point cloud, detection is performed progressively from coarse to fine: Level 4 detects the largest objects $\rightarrow$ pruning $\rightarrow$ upsampling to Level 3 $\rightarrow$ detecting medium objects $\rightarrow$ pruning $\rightarrow$ ... $\rightarrow$ Level 1 detects the smallest objects.

### Key Designs

1. **Theoretically Derived Pruning Strategy**:

    - Function: Generates a binary pruning mask $M_i$ after detection at level $i$ to remove redundant voxels.
    - Mechanism: To ensure that pruning does not affect subsequent levels of detection, the features in a $P \times P \times P$ neighborhood around the center of each undetected object $\mathbf{c}_j$ must remain undisturbed. By analyzing receptive field propagation in sparse convolutions, the minimal cube radius that needs to be preserved for each object at level $i$ is derived as $r = \lceil(P + aff - 2)/2\rceil$ (where $aff$ is determined by kernel sizes).
    - Design Motivation: Guarantees mathematically lossless pruning—pruned voxels do not affect the prediction of any targets through subsequent convolutions.

2. **Learnable DSP Module**:

    - Function: Predicts the preservation probability $\hat{M}_i$ for each voxel using a lightweight MLP, which is discretized into a 0/1 mask using a threshold $\tau$ to guide pruning during inference.
    - Mechanism: During training, the theoretically derived $M_i$ is used as supervision (via FocalLoss), and during inference, the predicted mask is directly applied for pruning. A "weak pruning" mode is adopted during training—pruning after upsampling instead of before, and only limiting the maximum number of voxels to prevent convergence difficulties in early training phases.
    - Design Motivation: While the theoretical strategy assumes prior knowledge of ground-truth object positions, actual inference requires learning to predict which regions still contain small objects.

3. **Partial Addition**:

    - Function: Only performs addition at voxel locations where the upsampled feature $f_i^U$ exists when fusing it with the backbone feature $f_i^B$.
    - Mechanism: After pruning, upsampled features are much sparser than backbone features. Taking the union would recover pruned voxels, which defeats the purpose of pruning.
    - Comparison with Union: Taking the union yields mAP/mAPS of 57.9/36.4, whereas Partial Addition achieves 65.1/44.1 (+7.2/+7.7).

### Loss & Training
Total loss = original TR3D classification + regression loss + 0.01 $\times$ FocalLoss (for pruning mask prediction). The positive sample assignment strategy is also modified: instead of sampling inside the bounding box, the nearest $N_{pos}=6$ voxels within a $P \times S_i$ cube around the target center are sampled as positive samples, ensuring sufficient positive samples for small objects.

## Key Experimental Results

### Main Results

**ScanNet-md40 (22 categories, containing small objects)**:

| Method | mAP@0.25 | mAP@0.5 | mAP_S@0.25 | mAP_S@0.5 | Speed(FPS) | GPU Memory(MB) |
|------|----------|---------|------------|-----------|----------|---------|
| TR3D | 61.59 | 49.98 | 27.53 | 12.91 | 10.8 | 1250 |
| TR3D-higher | 65.18 | 54.03 | 41.70 | 29.56 | 5.2 | 4450 |
| DSPDet3D(τ=0) | **65.39** | **54.59** | **44.79** | **31.55** | 4.4 | 4200 |
| DSPDet3D(τ=0.3) | 65.04 | 54.35 | 43.77 | 30.38 | **12.5** | **700** |

**TO-SCENE-down (70 categories, massive tabletop small objects)**:

| Method | mAP@0.25 | mAP_S@0.25 | Speed(FPS) | GPU Memory(MB) |
|------|----------|------------|----------|---------|
| TR3D | 55.58 | 52.72 | 9.9 | 1400 |
| TR3D-higher | 63.96 | 62.84 | 4.1 | 4600 |
| DSPDet3D(τ=0.5) | **66.12** | **65.82** | **13.9** | **800** |

### Ablation Study

| Config | mAP@0.25 | mAP_S@0.25 | Description |
|------|----------|------------|------|
| Full DSP module | 65.1 | 44.1 | Full model |
| w/o Partial Addition | 55.3 | 35.5 | Drops 9.8/8.6, most critical component |
| Replace with union | 57.9 | 36.4 | Pruning effect diluted |
| Spherical preservation mask | 63.0 | 41.1 | -2.1/-3.0, cube fits receptive field better |
| Positive sample in bbox | 62.4 | 40.7 | -2.7/-3.4, insufficient positive samples for small targets |

### Key Findings
- **Pruning threshold $\tau$ offers continuous accuracy-speed trade-off**: $\tau=0$ (no pruning) achieves peak accuracy; $\tau=0.3$ incurs negligible accuracy loss on ScanNet but accelerates inference by 3$\times$ while reducing GPU memory to 1/6.
- **Optimal $r=7$ (corresponding to $P=7$)**: $r<7$ violates theoretical constraints and disrupts features, leading to performance drops; $r>7$ is not aggressive enough, wasting computation.
- **Cross-scene generalization**: Trained only on ScanNet rooms, the model directly processes entire Matterport3D buildings (>4.5 million points), completing inference in <2 seconds, whereas FCAF3D fails to detect small objects.

## Highlights & Insights
- **Theory-driven pruning design**: Instead of crafting ad-hoc pruning heuristics, the method designs the strategy from the constraint of "not affecting object detection", mathematically deriving the optimal pruning radius and preserved regions. This "derivation-first, implementation-second" paradigm is rare in the 3D detection domain, and the results powerfully validate the theory.
- **Threshold as a tuning knob**: After a single training phase, adjusting only $\tau$ at inference allows free trade-offs between accuracy and speed without retraining. This is highly beneficial for practical deployment (e.g., embedded devices vs. servers).
- **Simplicity and effectiveness of Partial Addition**: An apparently minor design choice (limiting feature fusion to pruned sparse positions) brings a massive difference of +10 mAP, showcasing that maintaining sparsity is paramount in sparse environments.

## Limitations & Future Work
- **Evaluation limited to indoor scenes**: Small objects in outdoor autonomous driving (e.g., nuScenes, such as pedestrians and traffic cones) present different characteristics (further away, sparker), and the generalizability to these scenarios remains unexplored.
- **Reliance on ground-truth distribution for supervision**: Training the pruning masks requires knowing the object center at each scale, which makes it sensitive to annotation quality.
- **Loss of long-range context due to independent subgraph processing**: Performing sparse convolution only within local voxel neighborhoods fails to capture room-level global semantic relations.
- **Future directions**: (1) Combining DSP with Transformer-based detection heads to recover global context via attention mechanisms post-pruning; (2) Introducing adaptive $\tau$ to dynamically determine pruning intensity based on scene complexity.

## Related Work & Insights
- **vs FCAF3D**: Both are multi-scale detectors. FCAF3D also employs pruning during training but relies solely on classification scores for ranking, which cannot guarantee that small object features remain undamaged. DSPDet3D ensures safe pruning via theoretical derivation.
- **vs TR3D-higher**: Simply increasing the resolution of TR3D yields comparable accuracy improvements, but at the cost of 4450MB memory compared to only 700MB for DSPDet3D ($\tau=0.3$)—a $6\times$ reduction.
- **vs 2D Small Object Detection**: While 2D approaches frequently resort to data augmentation or super-resolution, DSPDet3D leverages the inherent sparsity of 3D point clouds for spatial pruning, presenting an efficient strategy unique to 3D.

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretically derived pruning strategy is a highlight, though the overall framework of multi-scale detection + pruning is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two datasets with detailed ablations and cross-scene generalization, but lacks verification on outdoor scenarios.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivation, though high notation density presents a slight reading barrier.
- Value: ⭐⭐⭐⭐ Provides a practical and efficient solution for 3D small object detection, but the application domain is somewhat narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] DAMSDet: Dynamic Adaptive Multispectral Detection Transformer](damsdet_dynamic_adaptive_multispectral_detection_transformer_with_competitive_qu.md)
- [\[CVPR 2025\] Efficient Test-Time Adaptive Object Detection via Sensitivity-Guided Pruning](../../CVPR2025/object_detection/efficient_test-time_adaptive_object_detection_via_sensitivity-guided_pruning.md)
- [\[CVPR 2026\] Towards Persistence: Learning Topological Constraints for Event-based Small Object Detection](../../CVPR2026/object_detection/towards_persistence_learning_topological_constraints_for_event-based_small_objec.md)
- [\[ECCV 2024\] ReGround: Improving Textual and Spatial Grounding at No Cost](reground_improving_textual_and_spatial_grounding_at_no_cost.md)
- [\[ICCV 2025\] 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection](../../ICCV2025/object_detection/3dmood_lifting_2d_to_3d_for_monocular_openset_object_detecti.md)

</div>

<!-- RELATED:END -->
