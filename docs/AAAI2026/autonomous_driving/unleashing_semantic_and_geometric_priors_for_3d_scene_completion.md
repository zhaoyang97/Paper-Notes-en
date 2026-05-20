---
title: >-
  [Paper Note] Unleashing Semantic and Geometric Priors for 3D Scene Completion
description: >-
  [AAAI2026][Autonomous Driving][3D scene completion] This paper proposes FoundationSSC, a framework that unleashes the semantic and geometric priors of Vision Foundation Models through a dual-level decoupling design at bo…
tags:
  - "AAAI2026"
  - "Autonomous Driving"
  - "3D scene completion"
  - "vision foundation model"
  - "semantic-geometric decoupling"
  - "stereo cost volume"
date: 2026-05-08
content_hash: 998170f10d177611
---

# Unleashing Semantic and Geometric Priors for 3D Scene Completion

**Conference**: AAAI2026
**arXiv**: [2508.13601](https://arxiv.org/abs/2508.13601)  
**Code**: [D-Robotics-AI-Lab/FoundationSSC](https://github.com/D-Robotics-AI-Lab/FoundationSSC)  
**Area**: Autonomous Driving
**Keywords**: 3D scene completion, vision foundation model, semantic-geometric decoupling, stereo cost volume, autonomous driving

## TL;DR
This paper proposes FoundationSSC, a framework that unleashes the semantic and geometric priors of Vision Foundation Models through a dual-level decoupling design at both the source level and pathway level. Combined with an Axis-Aware Fusion module for integrating complementary 3D features, the method achieves state-of-the-art performance of 19.32 mIoU / 48.12 IoU on SemanticKITTI.

## Background & Motivation
- Camera-based 3D Semantic Scene Completion (SSC) provides dense geometric and semantic perception for autonomous driving.
- Existing methods employ a single coupled encoder to simultaneously provide semantic and geometric priors, causing mutual conflict between the two and limiting overall performance.
- Prior attempts to separately incorporate external depth (stereo depth) or semantic priors (VLMs) treat these as add-ons to coupled frameworks, leaving the fundamental feature conflict unresolved.
- Vision Foundation Models (VFMs) such as DINOv2 and DepthAnything offer powerful zero-shot generalization; the key challenge lies in effectively leveraging these priors to address the coupling problem.

## Core Problem
How to fundamentally decouple the extraction and processing pathways of semantic and geometric features in SSC, and fully exploit VFM priors to simultaneously improve both semantic and geometric metrics?

## Method

### Overall Architecture
Foundation Encoder → Decoupled Semantic/Geometric Pathways → Hybrid View Transformation → Axis-Aware Fusion → Decoding Head

### Key Designs

**1. Foundation Encoder (Source-level Decoupling)**
- A frozen FoundationStereo model (inheriting DINOv2/DepthAnythingV2 lineage) is used as a unified encoder.
- Three decoupled feature outputs are produced: (a) monocular image features $\mathbf{F}^{2D}$ (semantic branch); (b) disparity cost volume $\mathbf{V}_{disp}$ (geometric branch); (c) dense depth map $\mathbf{Z}$ (auxiliary use).

**2. Geometry-Aware Context Adapter (GCA)**
- Injects 3D structure awareness into the 2D semantic features from the VFM.
- Constructs a geometric prior matrix $\mathbf{M}^g = \alpha \mathbf{M}^d + (1-\alpha)\mathbf{M}^s$, fusing 3D depth distance and 2D spatial distance.
- Geometry-modulated attention: $\text{GeoAttn}(\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{M}^g) = (\text{Softmax}(\mathbf{QK}^T) \odot \beta^{\mathbf{M}^g})\mathbf{V}$

**3. Disparity-to-Depth Volume Mapping (DDVM)**
- Addresses information loss in converting the disparity cost volume to a depth distribution.
- Conventional approach: cost volume → collapsed depth map → one-hot distribution (information bottleneck).
- DDVM: learns the nonlinear mapping $\tilde{\mathbf{V}}_{depth} = f(\tilde{\mathbf{V}}_{disp})$ directly via learnable channel-mapper blocks.
- A probabilistic depth distribution $\mathbf{D}$ is generated through 3D CNN refinement followed by softmax.

**4. Axis-Aware Fusion (AAF)**
- Fuses complementary information from the LSS volume $\mathbf{F}_{lss}$ and the Voxel Transformer volume $\mathbf{F}_{vt}$.
- Three parallel axis-specific fusion units extract directional context along the XY/XZ/YZ planes respectively.
- $\mathbf{F}_{fused} = \sum_{d \in \{XY, XZ, YZ\}} (\sigma_d \mathbf{F}_{lss} + (1-\sigma_d)\mathbf{F}_{vt})$
- Anisotropic fusion outperforms isotropic 3D channel attention.

## Key Experimental Results

**SemanticKITTI test set:**

| Method | IoU | mIoU |
|--------|-----|------|
| CGFormer | 44.41 | 16.63 |
| SOAP | 46.09 | 19.09 |
| **FoundationSSC** | **48.12** | **19.32** |

- Compared to the CGFormer baseline: +3.71 IoU, +2.69 mIoU.
- Surpasses all methods using temporal information (HTCL, SOAP) while relying only on stereo input.

**SSCBench-KITTI-360:** 48.61 IoU, 21.78 mIoU (SOTA).

**Ablation Study (SemanticKITTI val):**

| Component | IoU | mIoU |
|-----------|-----|------|
| Baseline | 45.28 | 16.53 |
| +Foundation Encoder | 46.61 | 18.59 (+2.06) |
| +FE+GCA+DDVM | 47.84 | 19.56 (+3.03) |
| +FE+GCA+DDVM+AAF | 47.91 | **20.36** (+3.83) |

- AAF vs. 3D Channel Attention: 20.36 vs. 20.08 mIoU, validating the advantage of anisotropic fusion.
- DDVM vs. Depth Refinement: 20.36 vs. 19.83 mIoU, demonstrating the value of preserving probabilistic information.

## Highlights & Insights
- **Dual-level decoupling design**: source-level (encoder decouples semantic/geometric outputs) + pathway-level (dedicated processing pathways) fundamentally resolves the semantic–geometric conflict in SSC.
- **DDVM module**: avoids information loss from cost volume → depth map collapse by directly learning the nonlinear disparity-to-depth mapping.
- **AAF anisotropic fusion**: recognizes that 3D structures in driving scenes exhibit directional variation (front–back vs. left–right vs. up–down), making axis-specific design well-motivated.
- **Deep exploitation of Foundation Models**: rather than naively replacing the backbone, the paper designs a complete utilization pipeline around VFMs.

## Limitations & Future Work
- The Foundation Encoder is used in a frozen state; its large parameter count (DepthAnythingV2-L: 335M) incurs significant deployment cost.
- Validation is limited to stereo input settings; the applicability of the framework under monocular configurations remains unknown.
- The global attention matrix $\mathbf{M}^g \in \mathbb{R}^{HW \times HW}$ in GCA is computationally expensive at high resolutions.
- No direct comparison with temporal methods under identical conditions is provided (the proposed method uses only single-frame stereo input).

## Related Work & Insights
- vs. CGFormer: the baseline of FoundationSSC; dual-level decoupling yields simultaneous gains of +2.69 mIoU and +3.71 IoU.
- vs. VLScene: also introduces VLM semantic priors, but does not resolve the coupling problem, resulting in lower IoU (45.14 vs. 48.12).
- vs. SOAP (temporal method): FoundationSSC surpasses the multi-frame SOAP using only a single stereo frame.
- vs. MonoScene/VoxFormer: classic baselines; FoundationSSC improves mIoU by 5–8 points.

**Broader Implications:**
- Foundation Models for 3D perception is a current hot topic; the "frozen VFM + lightweight adapter" paradigm is more practical than full fine-tuning.
- The semantic–geometric decoupling idea is generalizable to 3D object detection, BEV perception, and related tasks.
- Disparity cost volumes contain rich probabilistic information that should not be naively collapsed into a depth map — an insight worth adopting in other depth-estimation-dependent tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ (dual-level decoupling + DDVM + AAF as three distinct contributions)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (SOTA on two datasets + multi-dimensional ablation)
- Writing Quality: ⭐⭐⭐⭐ (clear logical structure, high-quality figures)
- Value: ⭐⭐⭐⭐ (addresses the core trade-off in SSC with a generalizable framework)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Towards 3D Object-Centric Feature Learning for Semantic Scene Completion](towards_3d_object-centric_feature_learning_for_semantic_scene_completion.md)
- [\[CVPR 2026\] OccuFly: A 3D Vision Benchmark for Semantic Scene Completion from the Aerial Perspective](../../CVPR2026/autonomous_driving/occufly_a_3d_vision_benchmark_for_semantic_scene_completion_from_the_aerial_pers.md)
- [\[CVPR 2026\] Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion](../../CVPR2026/autonomous_driving/sparsity-aware_voxel_attention_and_foreground_modulation_for_3d_semantic_scene_c.md)
- [\[AAAI 2026\] HD2-SSC: High-Dimension High-Density Semantic Scene Completion for Autonomous Driving](hd2-ssc_high-dimension_high-density_semantic_scene_completion_for_autonomous_dri.md)
- [\[NeurIPS 2025\] FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance](../../NeurIPS2025/autonomous_driving/learning_temporal_3d_semantic_scene_completion_via_optical_flow_guidance.md)

</div>

<!-- RELATED:END -->
