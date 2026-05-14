---
title: >-
  [Paper Note] Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion
description: >-
  [NeurIPS 2025][3D Vision][Novel View Synthesis] This paper reformulates sparse-input novel view synthesis as a test-time natural video completion problem. It leverages pretrained video diffusion models to generate interm…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Novel View Synthesis"
  - "Video Diffusion Models"
  - "3D Gaussian Splatting"
  - "Sparse Input"
  - "Test-Time Inference"
date: 2026-05-08
content_hash: 96c482ae5da0c78f
---

# Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion

**Conference**: NeurIPS 2025
**arXiv**: [2511.17932](https://arxiv.org/abs/2511.17932)
**Authors**: Yan Xu, Yixing Wang, Stella X. Yu
**Code**: N/A
**Area**: 3D Vision / Novel View Synthesis
**Keywords**: Novel View Synthesis, Video Diffusion Models, 3D Gaussian Splatting, Sparse Input, Test-Time Inference

## TL;DR

This paper reformulates sparse-input novel view synthesis as a test-time natural video completion problem. It leverages pretrained video diffusion models to generate intermediate pseudo-views, and iteratively optimizes 3D Gaussian Splatting (3D-GS) via an uncertainty-aware mechanism, achieving high-fidelity scene reconstruction under extremely sparse input conditions.

## Background & Motivation

Sparse-input Novel View Synthesis is a core challenge in 3D vision: given only a small number of input views (e.g., 3–5 images), the goal is to render the scene from arbitrary novel viewpoints.

**Limitations of Prior Work**:
- NeRF/3D-GS-based methods degrade severely under extremely sparse input, as under-constrained regions lack supervision.
- Diffusion-based methods (e.g., Zero-1-to-3) typically require scene-specific fine-tuning or support only single-object settings.
- Geometry-based methods (e.g., MVS) tend to fail under wide baselines.

**Key Insight**: This paper reframes "spatial interpolation across sparse viewpoints" as "completing a natural video captured by a camera gliding through the scene," enabling direct exploitation of the strong motion priors learned by pretrained video diffusion models.

## Method

### Overall Architecture

The paper proposes a **zero-shot, generation-guided** framework consisting of an iterative loop over three core components:

1. **Video Completion Module**: A pretrained video diffusion model generates intermediate views between the given sparse keyframes.
2. **Uncertainty-Aware Filtering**: Multiple independent samples are drawn to estimate the uncertainty of generated views, filtering out low-quality pseudo-views.
3. **3D-GS Reconstruction Module**: The filtered pseudo-views and original inputs are jointly used to train 3D Gaussian Splatting.

### Key Designs

**Test-Time Video Completion**:
- Sparse input images are treated as video keyframes, arranged along a predefined camera trajectory.
- A video diffusion model (e.g., Stable Video Diffusion) generates the "missing frames" between keyframes.
- Conditional sampling ensures consistency between generated frames and known keyframes.

**Uncertainty-Aware Mechanism**:
- $K$ independent samples are drawn for each target position, yielding $K$ candidate views.
- Pixel-level variance is computed as an uncertainty indicator: $U(x) = \text{Var}_{k=1}^K [I_k(x)]$
- Regions with high uncertainty receive reduced weights during 3D-GS training, preventing erroneous pseudo-views from misleading reconstruction.

**Iterative Feedback Loop**:
- Round 1: An initial 3D-GS is trained using only the sparse input views.
- Subsequent rounds: The current 3D-GS renders intermediate viewpoints → results are fused with video completion outputs → 3D-GS is updated.
- 3D geometric constraints and 2D generative priors mutually reinforce each other, progressively improving quality.

### Loss & Training

The 3D-GS training objective is a weighted reconstruction loss:

$$\mathcal{L} = \sum_{i \in \text{real}} \mathcal{L}_1(I_i, \hat{I}_i) + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}}(I_i, \hat{I}_i) + \sum_{j \in \text{pseudo}} w_j \cdot \mathcal{L}_1(I_j, \hat{I}_j)$$

where the weight $w_j$ for each pseudo-view is inversely proportional to its uncertainty: $w_j = \exp(-\beta \cdot U_j)$

## Key Experimental Results

### Main Results

**LLFF Dataset (3 input views):**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| 3D-GS | 15.82 | 0.412 | 0.498 |
| DNGaussian | 18.95 | 0.571 | 0.342 |
| FSGS | 19.34 | 0.589 | 0.328 |
| ReconFusion | 20.12 | 0.623 | 0.285 |
| **Ours** | **21.87** | **0.672** | **0.241** |

**DTU Dataset (3 input views):**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| 3D-GS | 12.45 | 0.521 | 0.412 |
| SparseNeRF | 16.82 | 0.645 | 0.335 |
| FSGS | 17.91 | 0.672 | 0.298 |
| **Ours** | **19.56** | **0.718** | **0.252** |

### Ablation Study

| Component | PSNR | SSIM | LPIPS |
|-----------|------|------|-------|
| Full Model | 21.87 | 0.672 | 0.241 |
| w/o Uncertainty-Aware | 20.43 | 0.638 | 0.278 |
| w/o Iterative Feedback | 20.89 | 0.651 | 0.262 |
| Single-Round Completion Only | 19.75 | 0.612 | 0.301 |
| Random Pseudo-View Weights | 20.15 | 0.625 | 0.289 |

### Key Findings

1. **Uncertainty-aware filtering is critical**: Removing it causes a 1.44 dB drop in PSNR, confirming that pseudo-view quality varies substantially and reliable filtering is essential.
2. **Iterative feedback yields consistent improvement**: Convergence is typically achieved within 3 rounds, with diminishing returns thereafter.
3. **Effective on larger scenes (MipNeRF-360, DL3DV)**: Achieves 2–3 dB gains over baselines.
4. **Zero-shot capability**: Requires no training or fine-tuning on any target scene.

## Highlights & Insights

- **Elegant problem reformulation**: Recasting NVS as video completion allows the framework to leverage scene motion priors from video diffusion models, circumventing the difficulties of geometry-based methods under extreme sparsity.
- **Uncertainty-aware pseudo-label strategy**: Generalizable to other tasks that use generative model outputs as supervision signals.
- **Zero-shot generalization**: Without dataset-specific training, the method consistently outperforms specialized approaches across multiple benchmarks.
- **Closed-loop synergy between 3D and 2D**: The iterative feedback loop enables mutual improvement between reconstruction and generation, representing an elegant system design.

## Limitations & Future Work

- Inference is slow: multiple sampling passes through the video diffusion model incur significant computational overhead.
- The method depends on a predefined camera trajectory; different trajectory choices may affect results.
- Video diffusion model priors may be insufficient for large-scale scene variations (e.g., outdoor long-range scenes).
- Geometric consistency of pseudo-views remains to be improved; incorporating multi-view consistency constraints is a promising direction.

## Related Work & Insights

- **Video Diffusion Models**: Stable Video Diffusion, Sora, etc.
- **Sparse-View 3D Reconstruction**: DNGaussian, FSGS, SparseNeRF, ReconFusion.
- **Generation-Guided 3D**: DreamFusion, Score Jacobian Chaining.

This work demonstrates how large-scale video generative models can empower 3D vision tasks, offering a new paradigm for deploying foundation models in the 3D domain.

## Rating

| Dimension | Score (1–10) |
|-----------|-------------|
| Novelty | 8 |
| Theoretical Depth | 5 |
| Experimental Thoroughness | 8 |
| Writing Quality | 8 |
| Practical Value | 7 |
| Overall Recommendation | 7.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos](reconstruct_inpaint_test-time_finetune_dynamic_novel-view_synthesis_from_monocul.md)
- [\[NeurIPS 2025\] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)
- [\[NeurIPS 2025\] PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)
- [\[NeurIPS 2025\] HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis](hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](../../ICCV2025/3d_vision/self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)

</div>

<!-- RELATED:END -->
