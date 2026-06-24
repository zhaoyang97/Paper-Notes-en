---
title: >-
  [Paper Note] Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models
description: >-
  [CVPR 2025][3D Vision][3D Reconstruction Enhancement] Proposes Difix3D+, which utilizes a fine-tuned single-step diffusion model (SD-Turbo) to progressively generate pseudo-training views during the training phase to feed back into the 3D representation, and serves as a real-time post-processing enhancer during inference. It is compatible with both NeRF and 3DGS, achieving an average improvement of over 2x on FID.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Reconstruction Enhancement"
  - "Diffusion Models"
  - "Artifact Removal"
  - "Novel View Synthesis"
  - "Neural Rendering"
date: 2026-05-08
content_hash: e37d6e9266eb204b
---

# Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2503.01774](https://arxiv.org/abs/2503.01774)  
**Code**: [https://research.nvidia.com/labs/toronto-ai/difix3d](https://research.nvidia.com/labs/toronto-ai/difix3d)  
**Area**: 3D Vision  
**Keywords**: 3D Reconstruction Enhancement, Diffusion Models, Artifact Removal, Novel View Synthesis, Neural Rendering

## TL;DR

Proposes Difix3D+, which utilizes a fine-tuned single-step diffusion model (SD-Turbo) to progressively generate pseudo-training views during the training phase to feed back into the 3D representation, and serves as a real-time post-processing enhancer during inference. It is compatible with both NeRF and 3DGS, achieving an average improvement of over 2x on FID.

## Background & Motivation

NeRF and 3DGS can generate high-quality images near training views, but still suffer from severe artifacts (floater geometries, missing regions) in scenarios such as sparse observations or views far from training perspectives. The Key Challenge is: (1) Per-scene optimization methods lack data priors, making it impossible to reasonably hallucinate geometry and appearance in under-constrained regions; (2) Existing methods use diffusion models as score functions for step-by-step optimization (e.g., SDS), which is computationally expensive and difficult to scale to large scenes. The Key Insight of this work is: **the distribution of rendering artifacts is highly similar to the image distribution at a specific noise level during diffusion model training**, meaning a lightweight fine-tuning of a single-step diffusion model is sufficient to transform it into a "3D artifact fixer."

## Method

### Overall Architecture

The Difix3D+ pipeline consists of three steps: (1) Difix Model: Fine-tuning SD-Turbo to remove artifacts in rendered images; (2) Difix3D: Progressively generating pseudo-training views and feeding them back to update the 3D representation; (3) Difix3D+: Using Difix additionally as a real-time post-processing enhancer during inference. The entire pipeline is representation-agnostic, enabling a single model to simultaneously fix artifacts rendered by both NeRF and 3DGS.

### Key Designs

1. **Difix: Single-Step Diffusion Artifact Fixer**:
    - Function: Transforms artifact-containing rendered images into clean images.
    - Mechanism: Based on image-to-image fine-tuning of SD-Turbo, the input is the artifact-containing rendered image $\tilde{I}$ (instead of random Gaussian noise) with a lower noise level $\tau=200$ (instead of the standard $\tau=1000$). A reference view $I_{\text{ref}}$ is introduced to provide color/texture references through a cross-view reference mixing layer (which merges the view dimension into the spatial dimension to perform self-attention).
    - Design Motivation: Experiments verify that rendering artifact images closest resemble the distribution of noisy images at $\tau=200$—a $\tau$ too high alters image content, while a $\tau$ too low fails to remove artifacts. Freezing the VAE encoder and fine-tuning the decoder with LoRA allows the process to complete on a single GPU in a few hours.

2. **Progressive 3D Updates**:
    - Function: Distills the pseudo-training views fixed by Difix back into the 3D representation to ensure multi-view consistency.
    - Mechanism: Starting from the training views, the camera pose is slightly shifted towards the target view every 1.5K iterations; then rendering $\rightarrow$ Difix fixing $\rightarrow$ adding to the training set $\rightarrow$ continuing to optimize the 3D representation. This progressively expands the spatial scope of the reconstruction, ensuring the diffusion model always receives strong conditioning signals.
    - Design Motivation: Directly performing fixups on renderings far from the training views forces the diffusion model to hallucinate heavily, introducing multi-view inconsistencies. The progressive strategy makes each correction step relatively easy, and inconsistent 3D structures are eliminated in subsequent optimizations.

3. **Data Curation**:
    - Function: Constructs large-scale paired artifact-clean image training data.
    - Mechanism: Four complementary strategies: sparse reconstruction (DL3DV skip-frame training), loop reconstruction (autonomous driving scenes, translation trajectory $\rightarrow$ train NeRF $\rightarrow$ render offset views $\rightarrow$ train NeRF again), cross-reference (removing one camera from multi-camera setups for training and using others for evaluation), and under-fitted models (reducing training epochs to 25%-75%).
    - Design Motivation: Directly skipping frames on most datasets does not yield a large enough disparity (the retained views still cover the same regions), necessitating more aggressive strategies to generate sufficiently significant artifacts.

### Loss & Training

- $\mathcal{L} = \mathcal{L}_{\text{Recon}} + \mathcal{L}_{\text{LPIPS}} + 0.5 \cdot \mathcal{L}_{\text{Gram}}$
- Gram matrix loss encourages sharper details: $\mathcal{L}_{\text{Gram}} = \frac{1}{L} \sum_{l=1}^{L} \beta_l \|G_l(\hat{I}) - G_l(I)\|_2$
- Fine-tuning takes only a few hours on a single consumer-grade GPU.
- Inference post-processing takes only 76ms/frame (A100 GPU), which is over 10 times faster than multi-step diffusion.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR↑ | LPIPS↓ | FID↓ |
|--------|------|-------|--------|------|
| Nerfbusters | Nerfacto Baseline | 17.29 | 0.4021 | 134.65 |
| Nerfbusters | Nerfbusters | 17.72 | 0.3521 | 116.83 |
| Nerfbusters | **Difix3D+ (Nerfacto)** | **18.32** | **0.2789** | **49.44** |
| DL3DV | 3DGS Baseline | 17.18 | 0.3835 | 107.23 |
| DL3DV | **Difix3D+ (3DGS)** | **17.99** | **0.2932** | **40.86** |
| RDS (Driving) | Nerfacto Baseline | 19.95 | 0.5300 | 91.38 |
| RDS (Driving) | **Difix3D+ (Nerfacto)** | **21.75** | **0.4016** | **73.08** |

### Ablation Study

| Configuration | PSNR↑ | LPIPS↓ | FID↓ | Description |
|------|-------|--------|------|------|
| Nerfacto Baseline | 17.29 | 0.4021 | 134.65 | Original |
| + (a) Difix Direct Post-processing | 17.40 | 0.2996 | 49.87 | Large FID drop but inconsistent |
| + (b) One-time 3D Update | 17.97 | 0.3424 | 75.94 | PSNR increases, FID rebounds |
| + (c) Progressive 3D Update (Difix3D) | 18.08 | 0.3277 | 63.77 | Progressive superior to one-time |
| + (d) Real-time Post-processing (Difix3D+) | **18.32** | **0.2789** | **49.44** | Final optimal |

### Key Findings

- Progressive 3D update (c) is significantly better than one-time update (b): LPIPS decreases by 0.015, FID decreases by 12.
- Difix3D+ achieves **about 2.7x FID improvement** (Nerfbusters: 134.65 $\rightarrow$ 49.44), with PSNR also improving by 1dB.
- The same Difix model can simultaneously repair artifacts in both NeRF and 3DGS, demonstrating strong generalization.
- $\tau=200$ is the optimal noise level: PSNR 17.73 vs 15.64 for $\tau=600$.

## Highlights & Insights

- **Highly intuitive key insight**: rendering artifacts $\approx$ low-noise-level noisy images, allowing the single-step diffusion model to be adapted into an artifact fixer at almost "zero cost".
- **One model for all representations**: effective for both NeRF and 3DGS artifacts, indicating that it learns a general "natural image" prior.
- Double-phase usage of Difix (training + inference): improves 3D consistency during training and repairs residual imperfections during inference—featuring a very clear design philosophy.
- High computational efficiency: fine-tunes in a few hours without needing step-by-step queries to the diffusion model (over 10x faster than methods like IN2N).

## Limitations & Future Work

- The post-processing step may introduce minor inter-frame inconsistencies (though significantly mitigated by the progressive update).
- The current data curation strategies require different designs for different types of datasets; the level of automation needs improvement.
- The reference view selection strategy (closest training view) is relatively simple; more intelligent reference selection could potentially further improve performance.
- The sensitivity of hyperparameters (such as the weight of the Gram matrix loss) is not fully discussed.

## Related Work & Insights

- Concurrent work with 3DGS-Enhancer, differing in: progressive update strategy + post-processing during inference.
- Over 10x faster compared to IN2N-like methods (which query diffusion at every step).
- The noise level insight is generalizable: other degradation types (such as low resolution, compression artifacts) may also have their own optimal $\tau$.
- Data curation strategies (loop reconstruction, under-fitted models) can be directly reused in other 3D enhancement research.

## Rating

- Novelty: ⭐⭐⭐⭐ The insight that rendering artifacts $\approx$ low-noise images is novel and convincing, and the progressive update strategy is highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, evaluated across three datasets, two 3D representations, detailed ablation studies, and driving scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ A typical high-quality paper from NVIDIA, with clear logic and beautiful illustrations.
- Value: ⭐⭐⭐⭐⭐ Highly practical—general, efficient, open-source, with direct value for any 3D reconstruction work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] BokehDiff: Neural Lens Blur with One-Step Diffusion](../../ICCV2025/3d_vision/bokehdiff_neural_lens_blur_with_one-step_diffusion.md)
- [\[CVPR 2025\] Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [\[CVPR 2025\] Improving Gaussian Splatting with Localized Points Management](improving_gaussian_splatting_with_localized_points_management.md)
- [\[CVPR 2025\] Scaling Properties of Diffusion Models for Perceptual Tasks](scaling_properties_of_diffusion_models_for_perceptual_tasks.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](../../ICCV2025/3d_vision/bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)

</div>

<!-- RELATED:END -->
