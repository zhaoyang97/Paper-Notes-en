---
title: >-
  [Paper Note] EA3D: Online Open-World 3D Object Extraction from Streaming Videos
description: >-
  [NeurIPS 2025][3D Vision][Online 3D Reconstruction] This paper proposes EA3D (ExtractAnything3D), an online open-world 3D object extraction framework that performs simultaneous geometric reconstruction and comprehensive…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Online 3D Reconstruction"
  - "Open-World Scene Understanding"
  - "Gaussian Splatting"
  - "VLM"
  - "Feature Gaussians"
  - "Semantic Segmentation"
date: 2026-05-08
content_hash: fc5654c820df9163
---

# EA3D: Online Open-World 3D Object Extraction from Streaming Videos

**Conference**: NeurIPS 2025
**arXiv**: [2510.25146](https://arxiv.org/abs/2510.25146)  
**Code**: [VDIGPKU/EA3D](https://github.com/VDIGPKU/EA3D)  
**Area**: 3D Vision
**Keywords**: Online 3D Reconstruction, Open-World Scene Understanding, Gaussian Splatting, VLM, Feature Gaussians, Semantic Segmentation

## TL;DR

This paper proposes EA3D (ExtractAnything3D), an online open-world 3D object extraction framework that performs simultaneous geometric reconstruction and comprehensive scene understanding from streaming videos via knowledge-integrated feature maps, online visual odometry, and recurrent joint optimization.

## Background & Motivation

Autonomous systems (e.g., robots) operating in unknown environments require "on-the-fly understanding"—performing online 3D reconstruction and semantic understanding simultaneously. Existing methods suffer from three key bottlenecks:

1. **Offline constraint**: NeRF/3DGS methods require complete multi-view image sets and lengthy optimization, precluding online processing.
2. **Geometric prior dependency**: Many 3D scene understanding methods require pre-built point clouds, depth maps, or meshes.
3. **2D–3D inconsistency**: VLMs excel in 2D but suffer from view inconsistency and poor occlusion handling when directly lifted to 3D.

**Mechanism**: Analogous to human perception, the framework begins reconstruction and understanding from the very first frame of a streaming video, using historical observations to guide current understanding while allowing new observations to refine past knowledge.

## Core Problem

How to perform simultaneous online 3D geometric reconstruction and open-world semantic understanding from streaming video, without known geometry, poses, or semantic annotations?

## Method

### Knowledge-Integrated Feature Maps

**Open-world VLM interpretation**: A VLM identifies all potential objects and their semantics in each frame, dynamically maintaining an online semantic cache $\Omega$. Semantic embeddings are continuous vectors $T \in \mathbb{R}^{1 \times V}$ (CLIP text encoder).

**Semantic feature maps**: Per-pixel semantic features are obtained via the CLIP visual encoder and Grounded-SAM. Category similarity scores are computed to generate binary masks, and features are aggregated and normalized: $\mathbf{S} = T \times \mathbf{f}_{sem}$.

**Physical attributes**: Extended text prompts extract object-level and part-level physical attributes from the VLM, encoded as learnable vectors $\mathbf{Y}$.

**Feature map embedding**: Knowledge-integrated features are attached to each Gaussian primitive. The integrated feature map is defined as:

$$\mathbf{F}_t = \sum_{i,j} \mathbf{X}^{\text{self}}_{i,j} \cdot \mathbf{S}_{i,j}(\mathbf{T}_k; \mathbf{Y}_{i,j}) \cdot \mathbf{C}_t$$

Matching distributions between adjacent frames propagate Gaussian features across frames:

$$\mathbf{M}_{t,t-1} = \text{Softmax}\left(\frac{\mathbf{F}_t \mathbf{F}_{t-1}^\top}{\|\mathbf{F}_t\| \|\mathbf{F}_{t-1}^\top\|}\right)$$

### Online 3D Object Extraction

**Online visual odometry**: Cut3R combined with depth estimation initializes per-frame poses, point maps, and confidence maps. An online keypoint map is maintained, and local BA optimization corrects accumulated pose drift.

**Online Gaussian update**: Feature Gaussians are incrementally added per frame, refining existing geometry and extracting new objects. New Gaussians $\mu_i$ are initialized from depth back-projection in newly observed regions; Gaussians in co-visible regions share translation and rotation to reduce redundancy. A single-step splitting strategy adaptively grows Gaussians based on gradients.

### Recurrent Joint Optimization

**Semantics-aware adaptive Gaussian regularization**:

$$\mathcal{L}_\delta = \sum |\delta_i - \bar{\delta}| F_{sem}^q$$

This encourages Gaussians belonging to the same category to exhibit similar scales.

**Joint semantic–geometric optimization**:

$$\mathcal{L} = \sum_{t=0}^{t_{\text{now}}} \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_d + \lambda_3 \mathcal{L}_{kw} + \mathcal{L}_\delta$$

where $\mathcal{L}_1$ is the photometric loss, $\mathcal{L}_d$ is the depth loss, and $\mathcal{L}_{kw}$ is the feature map L2 distance loss. Parameters are set as $\lambda_1=0.25$, $\lambda_2=0.1$, $\lambda_3=0.15$.

Final rendered features are accumulated via alpha-blending: $\hat{F} = \sum_{i} F_i \cdot \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$.

## Key Experimental Results

### Multi-Task Evaluation on ScanNet

| Method | Online | Pose-Free | PSNR↑ | SSIM↑ | mIoU↑ | mAcc↑ | AP↑ | mAP↑ |
|--------|--------|-----------|-------|-------|-------|-------|-----|------|
| LangSplat | ✗ | ✗ | 18.4 | 0.69 | 27.5 | 51.3 | - | - |
| GaussianGrouping | ✗ | ✗ | 19.6 | 0.74 | 32.6 | 56.9 | 43.6 | 24.5 |
| FeatureGS | ✗ | ✗ | 23.9 | 0.84 | 41.1 | 66.0 | 51.4 | 32.7 |
| OpenScene | ✗ | ✗ | - | - | 42.8 | 68.6 | 55.7 | 34.8 |
| EmbodiedSAM | ✗ | ✗ | - | - | 44.2 | 71.4 | 58.1 | 39.5 |
| **EA3D** | **✓** | **✓** | **25.8** | **0.89** | **46.3** | **71.8** | **57.9** | **39.9** |

Under online and pose-free conditions, EA3D surpasses offline methods in both reconstruction and understanding quality.

### Sparse-View and Online Stability on LERF

| Method | Online | 10 views PSNR | 70 views PSNR | 10 views mIoU | 70 views mIoU |
|--------|--------|--------------|--------------|---------------|---------------|
| FeatureGS | ✗ | 15.2 | 22.4 | 29.4 | 53.6 |
| OpenGaussian | ✗ | 14.9 | 22.7 | 30.1 | 55.8 |
| **EA3D** | **✓** | **21.9** | **23.2** | **53.8** | **57.4** |

With only 10 frames, EA3D substantially outperforms offline methods, demonstrating strong robustness under sparse-view conditions. Processing speed is 0.235 FPS.

### Ablation Study

Removing any individual component—semantics-aware regularization, online odometry, or joint optimization—leads to performance degradation, validating the necessity of each module.

## Highlights & Insights

1. **Unified online framework**: The first 3DGS framework to simultaneously perform online reconstruction and open-world understanding without pre-built geometry or poses.
2. **Multi-task capability**: A single framework supports rendering, semantic/instance segmentation, 3D bounding boxes, semantic occupancy, mesh generation, and other downstream tasks.
3. **Knowledge propagation mechanism**: Cross-frame feature propagation via matching distributions ensures temporal continuity of knowledge.
4. High-quality results are maintained under sparse views, achieving PSNR of 21.9 with only 10 frames.

## Limitations & Future Work

- Processing speed of only 0.235 FPS remains far from real-time online deployment.
- Requires an A100 80 GB GPU, imposing high deployment requirements.
- Open-world interpretation quality of the VLM directly affects final results and may be insufficient for small objects.
- Online accumulated pose drift may degrade performance on long sequences.
- Validation is limited to indoor datasets (ScanNet, LERF); outdoor scene evaluation is absent.

## Related Work & Insights

- **vs. FeatureGS/OpenGaussian**: These methods require offline training with complete viewpoints and known poses; EA3D processes streaming video online without poses.
- **vs. MonoGS+VFM**: SLAM-based methods with VFM post-processing rely on sparse keyframe tracking and require additional post-optimization; EA3D performs integrated joint optimization.
- **vs. EmbodiedOcc**: Designed specifically for occupancy prediction and incapable of photorealistic rendering; EA3D unifies multiple tasks.
- **vs. HiCoM**: Streaming 3DGS requiring pre-computed poses and multi-view input; EA3D autonomously estimates poses from monocular input.

The triple constraint of "online + pose-free + open-world" is both highly challenging and practically valuable. The knowledge-integrated feature map design—unifying VLM semantics, VFM visual features, and physical attributes into Gaussian primitives—provides an effective paradigm for comprehensive 3D scene understanding. The recurrent joint optimization principle of "geometry guiding understanding, understanding refining geometry" is broadly applicable.

## Rating

- ⭐ Novelty: 9/10 — First online pose-free open-world 3D object extraction framework; system-level innovation.
- ⭐ Experimental Thoroughness: 8/10 — Multi-dataset multi-task evaluation with augmented baselines and ablations; lacks outdoor and large-scale scene evaluation.
- ⭐ Writing Quality: 8/10 — Framework diagrams are clear and motivation is well-articulated; some formulas use dense notation.
- ⭐ Value: 9/10 — High practical value directly targeting robotic autonomous exploration; multi-task unification is impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards 3D Objectness Learning in an Open World](towards_3d_objectness_learning_in_an_open_world.md)
- [\[NeurIPS 2025\] OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects](onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)
- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)
- [\[NeurIPS 2025\] 4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos](4dgt_learning_a_4d_gaussian_transformer_using_realworld_mono.md)
- [\[NeurIPS 2025\] Online Segment Any 3D Thing as Instance Tracking](online_segment_any_3d_thing_as_instance_tracking.md)

</div>

<!-- RELATED:END -->
