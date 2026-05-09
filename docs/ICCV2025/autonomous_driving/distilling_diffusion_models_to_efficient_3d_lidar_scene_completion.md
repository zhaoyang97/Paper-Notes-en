---
title: >-
  [Paper Note] Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion
description: >-
  [ICCV 2025 (Oral)][Autonomous Driving][LiDAR scene completion] This paper proposes ScoreLiDAR, a diffusion model distillation framework for 3D LiDAR scene completion. By incorporating scene-wise and point-wise structural losses to guide distillation, it reduces completion time from 30.55 seconds to 5.37 seconds (>5× speedup) while surpassing all state-of-the-art methods on SemanticKITTI.
tags:
  - ICCV 2025 (Oral)
  - Autonomous Driving
  - LiDAR scene completion
  - diffusion model distillation
  - structural loss
  - point cloud reconstruction
  - autonomous driving perception
date: 2026-05-08
content_hash: 2bb03c0f1622c10c
---

# Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion

**Conference**: ICCV 2025 (Oral)  
**arXiv**: [2412.03515](https://arxiv.org/abs/2412.03515)  
**Code**: [https://github.com/happyw1nd/ScoreLiDAR](https://github.com/happyw1nd/ScoreLiDAR)  
**Area**: Autonomous Driving / 3D Vision  
**Keywords**: LiDAR scene completion, diffusion model distillation, structural loss, point cloud reconstruction, autonomous driving perception

## TL;DR

This paper proposes ScoreLiDAR, a diffusion model distillation framework for 3D LiDAR scene completion. By incorporating scene-wise and point-wise structural losses to guide distillation, it reduces completion time from 30.55 seconds to 5.37 seconds (>5× speedup) while surpassing all state-of-the-art methods on SemanticKITTI.

## Background & Motivation

**State of the Field**: 3D LiDAR scene completion aims to recover a complete 3D scene from sparse LiDAR scans. Diffusion models have been successfully applied to this task (e.g., LiDiff) owing to their training stability and high completion quality. Specifically, diffusion models perform denoising in the latent space of 3D point clouds, progressively recovering a complete scene from random noise.

**Limitations of Prior Work**: Diffusion models require a large number of sampling steps (typically 50–1000) to generate high-quality results, leading to per-frame completion times exceeding 30 seconds. For real-time applications such as autonomous driving—where the vehicle must perceive its surroundings within milliseconds—this latency is entirely unacceptable.

**Root Cause**: The quality of diffusion models stems from their multi-step iterative denoising process, yet the computational cost scales proportionally with the number of steps. Naively reducing the step count severely degrades completion quality; in particular, the geometric structure of 3D scenes (e.g., building contours, road surfaces) tends to deform under few-step sampling.

**Paper Goals**: The paper aims to design a distillation method that enables a student model to match or exceed the completion quality of the teacher model (50+ steps) within very few sampling steps (e.g., 2–8 steps), while preserving the geometric integrity of the 3D scene.

**Starting Point**: The authors observe that 3D LiDAR scenes exhibit strong structural priors—buildings form planar surfaces, roads are approximately horizontal, and trees have characteristic shapes. By explicitly constraining the student model to preserve these structural characteristics during distillation, geometric quality can be maintained under few-step sampling.

**Core Idea**: Score distillation is combined with specially designed structural losses. The structural losses constrain the output geometry of the student model at both the global scene level and the local landmark level, enabling the distilled model to accurately reconstruct 3D structures even under few-step sampling.

## Method

### Overall Architecture

ScoreLiDAR adopts a teacher–student distillation paradigm. The teacher model is the pretrained LiDiff (a diffusion model operating on 3D point clouds) using 50-step DDIM sampling. The student model shares the teacher's network architecture but targets completion within 2–8 steps. Distillation proceeds in two stages: first, score matching distillation trains the student to learn the teacher's denoising distribution; then, structural loss fine-tuning further refines the 3D geometry of the student's outputs. The input is a sparse LiDAR point cloud and the output is a completed dense 3D scene.

### Key Designs

1. **Score Matching Distillation**:

    - Function: Compresses the multi-step denoising knowledge of the teacher model into the few-step inference of the student model.
    - Mechanism: The student model is trained so that its predicted score function at each timestep matches that of the teacher. Specifically, given a noisy point cloud $x_t$, the teacher predicts $\epsilon_\text{teacher}(x_t, t)$, and the student's loss is $\mathcal{L}_\text{score} = \|\epsilon_\text{student}(x_t, t) - \epsilon_\text{teacher}(x_t, t)\|^2$. Progressive distillation is employed to halve the number of steps incrementally: 50→25→12→8→4→2.
    - Design Motivation: Directly distilling in the output space (e.g., matching the student's 2-step output to the teacher's 50-step output) leads to unstable gradients. Score matching provides supervision at every timestep, yielding more stable training.

2. **Scene-wise Structural Loss**:

    - Function: Constrains the overall geometric structure of the student model's completion results.
    - Mechanism: The completion result is voxelized, and the discrepancy between the student's output and the ground truth on the voxel occupancy grid is computed. A voxelized variant of Chamfer Distance is used: $\mathcal{L}_\text{scene} = \text{CD}_\text{voxel}(\text{Vox}(\hat{x}_0), \text{Vox}(x_0^*))$. Voxelization makes the loss more robust to minor positional deviations of individual points and focuses on overall structure (e.g., "a wall should exist here") rather than exact point positions.
    - Design Motivation: Conventional point-wise Chamfer Distance is sensitive to outliers and fails to capture global geometric structure. The voxelized structural loss better constrains the topology and macroscopic shape of the scene.

3. **Point-wise Landmark Loss**:

    - Function: Enforces accuracy of the student model at geometrically significant locations.
    - Mechanism: Landmark points are extracted from the ground-truth scene at geometrically salient locations (e.g., corners, edge intersections, plane boundaries). Farthest Point Sampling (FPS) is applied in high-curvature regions to obtain these landmarks. The deviation of the student's output at corresponding positions is then penalized: $\mathcal{L}_\text{point} = \sum_{p \in \text{landmarks}} \min_{q \in \hat{x}_0} \|p - q\|^2$, with the relative configuration among landmarks also constrained.
    - Design Motivation: The scene-wise loss ensures overall shape fidelity but may be insufficient for fine geometric details (e.g., window edges, pillar boundaries). The landmark loss complements the scene-wise loss by ensuring that important geometric details are preserved.

### Loss & Training

The total loss is the sum of three terms: $\mathcal{L} = \mathcal{L}_\text{score} + \alpha \mathcal{L}_\text{scene} + \beta \mathcal{L}_\text{point}$. Training proceeds in two stages: in the first stage, only $\mathcal{L}_\text{score}$ is used for progressive distillation (50→2 steps); in the second stage, the structural losses are introduced for fine-tuning. Training is conducted on SemanticKITTI using the Adam optimizer with a learning rate of 1e-4. A lightweight refine_net is additionally applied for final refinement after distillation.

## Key Experimental Results

### Main Results

| Method | Steps | CD↓ (×10⁻³) | IoU↑ | Time (s/frame) |
|--------|-------|-------------|------|----------------|
| ScoreLiDAR (Ours) | 8 | **3.12** | **0.84** | **5.37** |
| LiDiff (Teacher) | 50 | 3.45 | 0.82 | 30.55 |
| S2CFormer | 1 | 4.23 | 0.78 | 0.12 |
| SCPNet | 1 | 4.56 | 0.76 | 0.09 |
| JS3C-Net | 1 | 5.12 | 0.73 | 0.15 |
| LMSCNet | 1 | 5.89 | 0.70 | 0.08 |

### Ablation Study

| Configuration | CD↓ (×10⁻³) | IoU↑ | Note |
|---------------|-------------|------|------|
| Full ScoreLiDAR (8 steps) | **3.12** | **0.84** | Full model |
| w/o scene-wise structural loss | 3.58 | 0.81 | Degraded global structure |
| w/o point-wise landmark loss | 3.41 | 0.82 | Degraded fine geometry |
| w/o all structural losses | 3.89 | 0.79 | Score distillation only; notable geometric quality drop |
| 2-step sampling | 3.67 | 0.80 | Quality drops with fewer steps but still outperforms conventional methods |
| 4-step sampling | 3.28 | 0.83 | Good quality–speed trade-off |

### Key Findings

- With 8-step inference, ScoreLiDAR is not only 5.7× faster than the teacher model (50 steps) but also achieves better completion quality (CD reduced by 9.6%), attributable to the additional geometric priors provided by the structural losses.
- Contribution of structural losses: removing the scene-wise loss increases CD by 14.7%; removing the landmark loss increases CD by 9.3%; removing all structural losses increases CD by 24.7%—demonstrating that the two levels of structural constraint are complementary and both essential.
- Even compressed to 2 steps, ScoreLiDAR still outperforms all non-diffusion methods (e.g., S2CFormer), demonstrating the strong capability of diffusion distillation.
- Accepted as ICCV 2025 Oral with 66 GitHub stars, reflecting strong community recognition.

## Highlights & Insights

- **The two-level structural loss design is particularly elegant**: the scene-wise loss (voxelized CD) captures global topology, while the point-wise loss (landmark loss) addresses local detail; the two are complementary. This "macro + micro" loss design paradigm is transferable to any distillation or compression task involving 3D structural preservation.
- **The intriguing phenomenon of the student surpassing the teacher**: the structural losses serve as additional inductive biases, enabling the student model's geometric quality to exceed that of the teacher. This suggests that distillation need not be purely compressive—it can also be a process of incorporating new knowledge.
- **Practical application-oriented design**: while 5.37 seconds is not yet real-time, applying this distillation-plus-structural-loss approach to faster backbone networks holds promise for achieving real-time completion.

## Limitations & Future Work

- At 5.37 seconds per frame, the method remains too slow for real-time autonomous driving applications (requiring <100 ms); further acceleration is needed.
- Distillation is bounded by the quality ceiling of the teacher model; any systematic biases in the teacher will be inherited by the student.
- Evaluation is currently limited to SemanticKITTI; generalization to other datasets (e.g., nuScenes, Waymo) remains to be verified.
- Landmark extraction in the structural loss relies on geometric analysis and may be unstable in extremely sparse scenes.
- Future directions include combining distillation with faster 3D backbones (e.g., sparse convolutions) and extending the approach to dynamic scene completion.

## Related Work & Insights

- **vs. LiDiff (teacher model)**: LiDiff is the first work to apply diffusion models to LiDAR scene completion, achieving high-quality results at the cost of slow inference. ScoreLiDAR achieves >5× speedup while maintaining comparable or superior quality.
- **vs. S2CFormer/SCPNet (conventional methods)**: Although these methods run in real time, their quality falls far short of diffusion-based approaches. ScoreLiDAR demonstrates that distillation can achieve a markedly better speed–quality trade-off.
- **vs. image-domain distillation (e.g., LCM, SDXL-Turbo)**: 2D image distillation does not need to account for 3D geometric structure, whereas the structural losses in ScoreLiDAR represent a core innovation specifically designed for 3D scenes.

## Rating

- Novelty: ⭐⭐⭐⭐ The scene-wise + point-wise structural loss design is novel; this is the first work to introduce distillation into 3D LiDAR completion.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive baselines, detailed ablations, and multi-step analysis; consistent with ICCV Oral quality.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described and the experimental design is well-structured.
- Value: ⭐⭐⭐⭐⭐ Practically significant for autonomous driving perception; code is publicly available and has attracted broad community attention.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] LiNeXt: Revisiting LiDAR Completion with Efficient Non-Diffusion Architectures](../../AAAI2026/autonomous_driving/linext_revisiting_lidar_completion_with_efficient_non-diffusion_architectures.md)
- [\[NeurIPS 2025\] Neurosymbolic Diffusion Models](../../NeurIPS2025/autonomous_driving/neurosymbolic_diffusion_models.md)
- [\[NeurIPS 2025\] Towards Foundational LiDAR World Models with Efficient Latent Flow Matching](../../NeurIPS2025/autonomous_driving/towards_foundational_lidar_world_models_with_efficient_latent_flow_matching.md)
- [\[ICCV 2025\] Decoupled Diffusion Sparks Adaptive Scene Generation](decoupled_diffusion_sparks_adaptive_scene_generation.md)
- [\[CVPR 2026\] CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](../../CVPR2026/autonomous_driving/colc_communication-efficient_collaborative_perception_with_lidar_completion.md)

<!-- RELATED:END -->
