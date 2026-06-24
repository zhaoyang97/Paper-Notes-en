---
title: >-
  [Paper Note] Identity-preserving Distillation Sampling by Fixed-Point Iterator
description: >-
  [CVPR 2025][3D Vision][score distillation] Proposed Identity-preserving Distillation Sampling (IDS), which corrects the gradient errors leading to identity loss in text-conditioned score functions through Fixed-Point Iterative Regularization (FPR). This method generates guided noise instead of random noise, achieving high structural and pose preservation in both 2D image editing and 3D NeRF editing.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "score distillation"
  - "image editing"
  - "NeRF editing"
  - "identity preservation"
  - "fixed-point iteration"
  - "DDS"
date: 2026-05-08
content_hash: d6e241995999e426
---

# Identity-preserving Distillation Sampling by Fixed-Point Iterator

**Conference**: CVPR 2025  
**arXiv**: [2502.19930](https://arxiv.org/abs/2502.19930)  
**Code**: [https://github.com/shhh0620/IDS](https://github.com/shhh0620/IDS)  
**Area**: 3D Vision  
**Keywords**: score distillation, image editing, NeRF editing, identity preservation, fixed-point iteration, DDS

## TL;DR

Proposed Identity-preserving Distillation Sampling (IDS), which corrects the gradient errors leading to identity loss in text-conditioned score functions through Fixed-Point Iterative Regularization (FPR). This method generates guided noise instead of random noise, achieving high structural and pose preservation in both 2D image editing and 3D NeRF editing.

## Background & Motivation

**Background**: Score Distillation Sampling (SDS) achieves text-driven 3D generation and image editing by distilling the score function of pre-trained diffusion models. Delta Denoising Score (DDS) mitigates the blurriness issue of SDS by subtracting the source-target score difference.

**Limitations of Prior Work**: (1) The random noise in SDS results in unstable gradient directions, leading to over-saturation and blurriness; (2) Although DDS reduces noise in non-text-aligned features, the text-conditioned score $\epsilon_\phi^{src}$ itself does not precisely point to the source image, leading to accumulated errors that discard the source identity information (such as background, pose, and structure); (3) CDS and PDS attempt to maintain consistency by maximizing mutual information, but they do not analyze the inherent errors of the score itself.

**Key Challenge**: The text-conditioned score $\epsilon_\phi^{src}$ is expected to provide a gradient direction from the noisy latent to the source image. However, in practice, a text prompt can correspond to countless different images, causing a significant discrepancy between the posterior mean $\mathbf{z}_{0|t}^{src}$ pointed to by the score and the actual source image $\mathbf{z}^{src}$ (especially at large $t$). This error accumulation leads to structural changes in the editing results.

**Key Insight**: Starting from fixed-point iteration in numerical analysis, the source latent variable is iteratively corrected to make the score function precisely align with the source image, fundamentally resolving the gradient error issue.

## Method

### Overall Architecture

1. Sample random noise $\epsilon$ and timestep $t$ to construct the source latent variable $\mathbf{z}_t^{src}$.
2. Run FPR: Iteratively update $\mathbf{z}_t^{src}$ to bring the posterior mean $\mathbf{z}_{0|t}^{src}$ close to the source image $\mathbf{z}^{src}$.
3. Extract the guided noise $\epsilon^*$ from the optimized latent $\mathbf{z}_t^{src*}$.
4. Construct the target latent $\mathbf{z}_t^{trg*}$ using $\epsilon^*$ instead of random noise, and compute the IDS update direction.

### Key Designs

**1. Fixed-Point Iterative Regularization (FPR)**
- **Function**: Iteratively updates the source latent variable $\mathbf{z}_t^{src}$ to align the posterior mean calculated by Tweedie's formula with the source image.
- **Key Formulas**:
    - Posterior mean: $\mathbf{z}_{0|t}^{src} = \frac{1}{\sqrt{\alpha_t}}(\mathbf{z}_t^{src} - \sqrt{1-\alpha_t} \epsilon_\phi^{src})$
    - FPR Loss: $\mathcal{L}_{FPR} = d(\mathbf{z}^{src}, \mathbf{z}_{0|t}^{src})$ (Euclidean distance)
    - Update: $\mathbf{z}_t^{src} \leftarrow \mathbf{z}_t^{src} - \lambda \nabla_{\mathbf{z}_t^{src}} \mathcal{L}_{FPR}$
- **Iterate N times**: Recalculate the CFG score $\epsilon_\phi^{src}$ and posterior mean after each update.
- **Design Motivation**: If the score is correctly estimated as the gradient pointing towards $\mathbf{z}^{src}$, the posterior mean should contain sufficient information about the source image. The score is "corrected" by minimizing the discrepancy between the two.
- **Why update latent variables instead of noise**: Experiments show that updating $\mathbf{z}_t^{src}$ preserves more content details (as the score functions take latent variables as input).

**2. Guided Noise in Place of Random Noise**
- **Function**: After the convergence of FPR, solve backward from the optimized $\mathbf{z}_t^{src*}$ to obtain the guided noise $\epsilon^* = \frac{1}{\sqrt{1-\alpha_t}}(\mathbf{z}_t^{src*} - \sqrt{\alpha_t}\mathbf{z}^{src})$.
- **Mechanism**: Guided noise $\epsilon^*$ is no longer random Gaussian noise, but rather "structured noise aligned with the identity of the source image". Using it to generate the target latent variable $\mathbf{z}_t^{trg*}$ embeds the identity consistency constraint into the gradient direction.
- **Design Motivation**: In DDS, the source and target share the same random noise $\epsilon$, but $\epsilon$ is unconstrained and can point in any direction. The $\epsilon^*$ corrected by FPR implicitly carries the identity information of the source image.

**3. IDS Update Rule**
- **Function**: Replace the original latent variables in DDS with the corrected ones:
  $$\nabla_\theta \mathcal{L}_{IDS} = \mathbb{E}_{t,\epsilon}[(\epsilon_\phi^\omega(\mathbf{z}_t^{trg*}, y^{trg}, t) - \epsilon_\phi^\omega(\mathbf{z}_t^{src*}, y^{src}, t)) \frac{\partial \mathbf{z}^{trg}}{\partial \theta}]$$
- **Invertibility Verification**: The image edited by IDS can be perfectly reconstructed back to the source image through inverse editing (which DDS fails to do), proving that the gradient direction is correctly modified.

### Loss & Training

- FPR: $\mathcal{L}_{FPR} = \|\mathbf{z}^{src} - \mathbf{z}_{0|t}^{src}\|_2^2$
- Editing: Update target image/NeRF parameters using IDS gradients.
- Hyperparameters: $\lambda$ controls the strength of FPR regularization, and N controls the number of iterations (typically N=3).

## Key Experimental Results

### Main Results — Image Editing

**Structure Preservation Metrics (LPIPS↓, lower is better)**:

| Method | cat→pig LPIPS↓ | cat→squirrel LPIPS↓ | IP2P LPIPS↓ | IP2P PSNR↑ |
|---|---|---|---|---|
| P2P | 0.42 | 0.46 | 0.47 | 20.88 |
| PnP | 0.52 | 0.52 | 0.39 | 23.81 |
| DDS | 0.28 | 0.30 | 0.24 | 26.02 |
| CDS | 0.25 | 0.26 | 0.21 | 27.35 |
| **IDS** | **0.22** | **0.24** | **0.19** | **29.25** |

**User Preferences and GPT Ratings**:

| Method | Text Alignment↑ | Identity Preservation↑ | Quality↑ | GPT-Text↑ | GPT-Preserve↑ |
|---|---|---|---|---|---|
| DDS | 20.30% | 10.82% | 16.23% | 7.60 | 7.51 |
| CDS | 17.02% | 16.72% | 17.08% | 8.26 | 8.00 |
| **IDS** | **43.83%** | **60.49%** | **51.67%** | **8.97** | **9.00** |

### NeRF Editing

| Method | CLIP↑ | Text Preference↑ | Preservation Preference↑ | Quality Preference↑ |
|---|---|---|---|---|
| DDS | 0.1596 | 36.88% | 28.37% | 32.62% |
| CDS | 0.1597 | 22.70% | 23.40% | 21.28% |
| **IDS** | **0.1626** | **40.42%** | **48.23%** | **46.10%** |

### Ablation Study — Computational Complexity

| Setting | LPIPS↓ | CLIP↑ | Time(s/img) | Memory(GB) |
|---|---|---|---|---|
| DDS (200 steps) | 0.240 | 0.293 | 22.45 | 6.27 |
| CDS (200 steps) | 0.210 | 0.287 | 59.31 | 8.83 |
| IDS (200 steps, FPR=3) | 0.190 | 0.277 | 107.77 | 8.63 |
| IDS (100 steps, FPR=3) | 0.165 | 0.265 | 54.04 | 8.63 |

### Key Findings

1. **Significant Boost in Identity Preservation**: IDS achieves 60.49% (vs. 10.82% for DDS) in the "identity preservation" user preference dimension, representing a hugely significant improvement.
2. **Invertibility**: IDS editing followed by inverse editing can almost perfectly reconstruct the source image, which DDS cannot achieve, demonstrating that the gradient direction is correctly corrected.
3. **Trade-off in FPR Iterations**: Only N=3 iterations are needed to obtain significant performance, with escalating iterations yielding diminishing returns alongside linearly increasing computational costs.
4. **Equally Effective for NeRF Editing**: 3D scene editing demands higher consistency, making the structural preservation advantages of IDS even more pronounced in 3D (where depth maps are also cleaner).
5. **Better Convergence**: IDS with 100 steps outperforms DDS with 200 steps, and the overall computational cost is lower than CDS.

## Highlights & Insights

- Approaches the identity preservation problem in diffusion model editing from a numerical analysis perspective (fixed-point iteration), establishing a solid theoretical foundation.
- The invertibility test is a brilliant verification method: perfect reconstruction from editing to inverse editing indicates that the gradient direction is indeed correct.
- The design intuition of replacing random noise with guided noise is straightforward: rather than adding more constraints, it fundamentally corrects the source of error.
- Thorough analysis is provided on "why the text-conditioned score is imprecise": the visualization of the discrepancy between the posterior mean and the source image widening at larger timesteps demonstrates the issue clearly.

## Limitations & Future Work

- The N iterations of FPR introduce additional computational overhead (as each iteration requires an extra forward pass through the diffusion model).
- Relies on the DDS framework, thus inheriting its dependence on text prompt quality.
- Evaluated only on Stable Diffusion v1.5, leaving its generalizability to newer models (e.g., SDXL, FLUX) unverified.
- The hyperparameter $\lambda$ is sensitive; excessively large values lead to over-preservation (no editing), while small values cause the method to degenerate into DDS.
- Lacks discussion on large-scale editing (e.g., changing object categories and scene structures).

## Related Work & Insights

- **SDS (DreamFusion)**: Pioneering work in score distillation, but suffer from blurriness due to gradient noise.
- **DDS**: Reduces noise through source-target differencing, but fails to address the fundamental issue of score error.
- **CDS/PDS**: Maintain consistency by maximizing mutual information, but incur heavy computation costs with limited improvements.
- **Insight**: The imprecision of the score function in diffusion models is an underestimated problem, and fixed-point iteration provides a general correction framework.

## Rating

⭐⭐⭐⭐ — Deep theoretical analysis, clear methodology motivation, and thorough experimental validation (images + NeRF + user studies + invertibility tests). It represents an important improvement in the SDS family, with the increased computational overhead being the primary trade-off.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Identity Preserving 3D Head Stylization with Multiview Score Distillation](../../ICCV2025/3d_vision/identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)
- [\[CVPR 2025\] MICAS: Multi-grained In-Context Adaptive Sampling for 3D Point Cloud Processing](micas_multi-grained_in-context_adaptive_sampling_for_3d_point_cloud_processing.md)
- [\[CVPR 2025\] Feature-Preserving Mesh Decimation for Normal Integration](feature-preserving_mesh_decimation_for_normal_integration.md)
- [\[CVPR 2025\] 3D Dental Model Segmentation with Geometrical Boundary Preserving](3d_dental_model_segmentation_with_geometrical_boundary_preserving.md)
- [\[CVPR 2025\] Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization](gaussian_splatting_feature_fields_for_privacy-preserving_visual_localization.md)

</div>

<!-- RELATED:END -->
