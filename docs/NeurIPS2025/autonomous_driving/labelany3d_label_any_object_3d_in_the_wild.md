---
title: >-
  [Paper Note] LabelAny3D: Label Any Object 3D in the Wild
description: >-
  [NeurIPS 2025][Autonomous Driving][3D annotation] This paper proposes LabelAny3D, an analysis-by-synthesis automatic 3D annotation pipeline that reconstructs complete 3D scenes from monocular images to obtain high-quality 3D bounding box annotations. Based on this pipeline, the authors construct the COCO3D benchmark covering 80 categories of everyday objects, achieving significant improvements in open-vocabulary monocular 3D detection.
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - 3D annotation
  - monocular 3D detection
  - open-vocabulary
  - automatic annotation
  - foundation models
date: 2026-05-08
content_hash: 969fbff3680e3933
---

# LabelAny3D: Label Any Object 3D in the Wild

**Conference**: NeurIPS 2025
**arXiv**: [2601.01676](https://arxiv.org/abs/2601.01676)
**Code**: [Project Page](https://uva-computer-vision-lab.github.io/LabelAny3D/)
**Area**: Autonomous Driving
**Keywords**: 3D annotation, monocular 3D detection, open-vocabulary, automatic annotation, foundation models

## TL;DR

This paper proposes LabelAny3D, an analysis-by-synthesis automatic 3D annotation pipeline that reconstructs complete 3D scenes from monocular images to obtain high-quality 3D bounding box annotations. Based on this pipeline, the authors construct the COCO3D benchmark covering 80 categories of everyday objects, achieving significant improvements in open-vocabulary monocular 3D detection.

## Background & Motivation

Monocular 3D object detection is a fundamental task in robotics, autonomous driving, and AR/VR. Despite progress from methods such as Cube R-CNN and OVMono3D, the core bottleneck remains **the scarcity of large-scale 3D datasets**:

**Limited scene diversity**: Omni3D (the largest public 3D dataset) primarily covers indoor and autonomous driving scenes, lacking in-the-wild everyday objects (e.g., animals, food).

**High annotation cost**: 3D bounding box annotation is substantially more expensive than 2D annotation, and in-the-wild images lack depth sensors.

**Constraints of existing automatic annotation methods**:
- OVM3D-Det relies on metric depth estimation combined with LLM object size priors, failing for objects with high intra-class variation (e.g., juvenile vs. adult elephants).
- 3D Copy-Paste inserts synthetic 3D models, suffering from the sim-to-real domain gap.

Core problem: **How to generate high-quality 3D annotations from natural images with minimal human supervision?**

## Method

### Overall Architecture

LabelAny3D adopts an **analysis-by-synthesis paradigm**: it reconstructs a complete 3D scene from a monocular image and then extracts 3D annotations from the reconstructed scene.

Key observations driving the design:
1. Relative depth estimation is more reliable and consistent than metric depth.
2. Object reconstruction based on large-scale 3D datasets (Objaverse) is sufficiently accurate.
3. 2D foundation models such as SAM and Grounding DINO generalize strongly to in-the-wild visual scenes.

### Key Designs

#### 1. Image Super-Resolution

Many objects in MS-COCO suffer from low resolution due to small scale or compression. InvSR (diffusion-based super-resolution) is applied to achieve a 4× upscaling, $I^{SR} \in \mathbb{R}^{4H \times 4W \times 3}$, recovering fine details to support subsequent 3D reconstruction.

Ablation results show that removing super-resolution causes $AP_{3D}$ to drop from 43.17 to 28.13 (−15.04), making it the most impactful component.

#### 2. 2D Instance Segmentation and Filtering

High-quality segmentation masks from the COCONut dataset are used (which corrects errors in original COCO annotations). Filtering rules:
- Exclude truncated objects (where the mask intersects image boundaries beyond a threshold).
- Exclude overly small objects (where mask area falls below a threshold).

#### 3. Amodal Completion + 3D Reconstruction

For occluded objects, a diffusion-based amodal completion model from Gen3DSR is applied to inpaint missing regions. **TRELLIS** is then used for single-view 3D reconstruction, recovering a complete 3D mesh at normalized scale.

TRELLIS outperforms DreamGaussian by 6.33 AP, producing higher-fidelity reconstructions.

#### 4. Scene Geometry Estimation

**Dual-depth strategy**:
- MoGe: affine-invariant relative depth estimation (more accurate relative layout).
- Depth Pro: metric depth estimation (provides absolute scale reference).

MoGe depth is aligned to the metric scale of Depth Pro, then back-projected to a 3D point cloud via camera intrinsics.

Removing MoGe causes AP to drop from 43.17 to 22.77 (−20.4), confirming that relative depth is central to geometric accuracy.

#### 5. Pose Estimation (2D–3D Alignment)

MASt3R is used to compute dense 2D–2D correspondences between the real image and rendered views of the object mesh. 2D matching points are back-projected to 3D using known rendering parameters, and PnP+RANSAC is applied to solve the relative pose $(R, T)$, transforming the reconstructed object into the image coordinate system.

The PnP approach outperforms ICP (43.17 vs. 24.28), as 2D matching models are more robust.

#### 6. Scale Estimation and 3D Annotation Generation

Metric scale is recovered via the depth median ratio over the masked region: $s = \text{median}(D_{real}(\Omega) / D_{render}(\Omega))$.

The final 3D bounding box is obtained by uniformly sampling surface points from the mesh, estimating principal axes via PCA, and fitting a compact 3D bounding box with center position, orientation, and dimensions.

### Loss & Training

LabelAny3D itself is an annotation pipeline and does not involve training. The downstream detector OVMono3D is trained with:

$$\mathcal{L} = \sqrt{2} \exp(-\mu) \mathcal{L}_{3D} + \mu$$

where $\mathcal{L}_{3D}$ consists of decoupled 3D attribute losses (2D center offset, depth, dimensions, rotation) plus a Chamfer holistic loss, and $\mu$ denotes an uncertainty score.

Training data: 15,869 MS-COCO training images automatically annotated by LabelAny3D, without manual refinement.

## Key Experimental Results

### Main Results

**Detection performance on the COCO3D benchmark** (OVMono3D):

| Training Data | AP₃D↑ | AR₃D↑ | AP₃D^Rel↑ | AR₃D^Rel↑ |
|---|---|---|---|---|
| Omni3D (baseline) | 5.87 | 10.51 | 20.86 | 30.06 |
| OVM3D-Det* | 2.69 | 5.25 | 7.98 | 12.25 |
| LabelAny3D | 7.78 | 15.41 | 24.66 | 34.54 |
| Omni3D + LabelAny3D | **10.92** | **20.10** | **32.02** | **43.82** |

Joint training yields a +5.05 AP₃D gain (compared to Omni3D alone at 5.87).

**Pseudo-annotation quality comparison (COCO3D)**:

| Method | AP₃D | AP₃D^15 | AP₃D^50 | AR₃D |
|---|---|---|---|---|
| OVM3D-Det | 10.03 | 16.88 | 1.44 | 17.82 |
| **LabelAny3D** | **64.17** | **82.11** | **57.34** | **73.57** |

The advantage is particularly pronounced at high IoU thresholds (AP₃D^50: 57.34 vs. 1.44), demonstrating substantially superior annotation precision over OVM3D-Det.

### Ablation Study

| Configuration | AP₃D↑ |
|---|---|
| Full LabelAny3D | **43.17** |
| w/o super-resolution | 28.13 (−15.04) |
| w/o MoGe | 22.77 (−20.40) |
| w/o amodal completion | 39.22 (−3.95) |
| TRELLIS → DreamGaussian | 36.84 (−6.33) |
| PnP → ICP | 24.28 (−18.89) |
| Gen3DSR baseline | 1.95 (−41.22) |

### Key Findings

1. **MoGe relative depth is central to accuracy**: its removal causes the largest AP drop (−20.4), far exceeding the impact of metric depth.
2. **Super-resolution is unexpectedly critical**: detail recovery for small or compressed objects is essential for 3D reconstruction quality.
3. **2D matching + PnP greatly outperforms ICP**: the geometric alignment accuracy gap is substantial (43.17 vs. 24.28).
4. On the KITTI Truck category, LabelAny3D (32.74) substantially outperforms OVM3D-Det (13.46), demonstrating robustness to high intra-class variation.
5. The AP₃D between LabelAny3D pseudo-annotations and manually refined labels is 64.17, indicating that most annotations require only minimal human correction.

## Highlights & Insights

1. **Analysis-by-synthesis paradigm**: rather than directly predicting 3D boxes, the pipeline first reconstructs the 3D scene and then extracts annotations, achieving far higher precision than direct regression approaches.
2. **Foundation model orchestration**: SAM, MoGe, Depth Pro, TRELLIS, and MASt3R are composed into a unified pipeline, leveraging the complementary strengths of each model.
3. **COCO3D benchmark fills a critical gap**: the first 3D detection benchmark covering 80 categories of everyday in-the-wild objects, advancing the evaluation of open-vocabulary 3D detection.
4. **Practical utility**: the generated pseudo-annotations can be directly used to train detectors with significant performance gains.

## Limitations & Future Work

1. Foundation models still fail under severe occlusion, textureless surfaces, or extremely small objects.
2. TRELLIS may produce meshes with depth ambiguity along the viewing direction, causing misalignment with RGBD point clouds.
3. Not all objects are exhaustively annotated (severely occluded or truncated instances are excluded), making the pipeline unsuitable for evaluating end-to-end detectors.
4. All fine-tuned models exhibit performance degradation on Omni3D base categories (catastrophic forgetting).
5. Metric depth still relies on model predictions (rather than ground truth), which may introduce systematic bias.

## Related Work & Insights

- **Automatic 3D annotation**: OVM3D-Det employs LLM size priors, whereas LabelAny3D uses 3D reconstruction, yielding greater robustness to intra-class variation.
- **Model-in-the-loop annotation**: conceptually related to Stereo4D and Cap3D, but LabelAny3D focuses specifically on monocular 3D bounding box annotation.
- **Open-vocabulary 3D detection**: OVMono3D, DetAny3D, and similar methods are limited by training data diversity; LabelAny3D addresses this from the data perspective.
- **Insight**: compositional orchestration of foundation models is a viable path for scaling annotation, and the approach can be extended to tasks such as 6D pose estimation and scene completion.

## Rating

- Novelty: ⭐⭐⭐⭐ — Analysis-by-synthesis paradigm with multi-foundation-model pipeline; the approach is conceptually clear and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — COCO3D benchmark, annotation quality evaluation, downstream detection experiments, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured presentation with detailed pipeline description.
- Value: ⭐⭐⭐⭐⭐ — Fills a critical gap in in-the-wild 3D annotation and evaluation; highly practical.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] OpenBox: Annotate Any Bounding Boxes in 3D](openbox_annotate_any_bounding_boxes_in_3d.md)
- [\[ICCV 2025\] Detect Anything 3D in the Wild](../../ICCV2025/autonomous_driving/detect_anything_3d_in_the_wild.md)
- [\[AAAI 2026\] Difficulty-Aware Label-Guided Denoising for Monocular 3D Object Detection](../../AAAI2026/autonomous_driving/difficulty-aware_label-guided_denoising_for_monocular_3d_object_detection.md)
- [\[NeurIPS 2025\] Towards Predicting Any Human Trajectory in Context](towards_predicting_any_human_trajectory_in_context.md)
- [\[CVPR 2026\] Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos](../../CVPR2026/autonomous_driving/learning_to_drive_is_a_free_gift_large-scale_label-free_autonomy_pretraining_fro.md)

<!-- RELATED:END -->
