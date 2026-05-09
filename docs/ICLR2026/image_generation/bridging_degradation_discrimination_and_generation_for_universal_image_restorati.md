---
title: >-
  [Paper Note] Bridging Degradation Discrimination and Generation for Universal Image Restoration
description: >-
  [ICLR 2026][Image Generation][universal image restoration] BDG performs fine-grained degradation discrimination via multi-angle multi-scale gray-level co-occurrence matrices (MAS-GLCM), and designs a three-stage diffusion training pipeline (generation → bridging → restoration) to seamlessly integrate degradation discrimination with generative priors, achieving significant fidelity improvements on all-in-one restoration and real-world super-resolution tasks.
tags:
  - ICLR 2026
  - Image Generation
  - universal image restoration
  - GLCM degradation representation
  - diffusion model
  - three-stage training
  - all-in-one restoration
date: 2026-05-08
content_hash: 83208fdb2a99ce23
---

# Bridging Degradation Discrimination and Generation for Universal Image Restoration

**Conference**: ICLR 2026
**arXiv**: [2602.00579](https://arxiv.org/abs/2602.00579)
**Code**: N/A
**Area**: Image Generation
**Keywords**: universal image restoration, GLCM degradation representation, diffusion model, three-stage training, all-in-one restoration

## TL;DR
BDG performs fine-grained degradation discrimination via multi-angle multi-scale gray-level co-occurrence matrices (MAS-GLCM), and designs a three-stage diffusion training pipeline (generation → bridging → restoration) to seamlessly integrate degradation discrimination with generative priors, achieving significant fidelity improvements on all-in-one restoration and real-world super-resolution tasks.

## Background & Motivation
**Background**: Universal image restoration requires a single model to handle multiple degradation types, necessitating both degradation discrimination and conditional generation capabilities.

**Limitations of Prior Work**:
- **Degradation discrimination approaches** (AirNet, PromptIR, etc.): introduce auxiliary discriminative networks to identify degradation types, but L1/L2 losses produce overly smooth outputs with poor performance in real-world scenarios.
- **Generative prior approaches** (StableSR, DiffBIR, etc.): leverage pre-trained diffusion models to recover rich textures, but in all-in-one settings tend to misidentify mild degradations as severe ones, generating details inconsistent with the original image.

**Key Challenge**: Degradation discrimination and generative priors have developed independently, lacking a unified framework to organically integrate the two.

**Goal**: Inject fine-grained degradation discrimination capability into diffusion models while preserving their generative priors, enabling adaptive output adjustment based on degradation severity.

**Key Insight**: Propose a novel degradation representation (MAS-GLCM) combined with a three-stage diffusion training strategy to progressively bridge discriminative information into the generative process.

**Core Idea**: Use gray-level co-occurrence matrices as content-agnostic degradation representations, and align them with diffusion model features through three-stage training to unify discrimination and generation.

## Method

### Overall Architecture
Three-stage training pipeline: (1) **Generative pre-training** — learns VE-SDE denoising on high-quality images; (2) **Bridging stage** — introduces residual conditioning and aligns MAS-GLCM features with diffusion intermediate features via contrastive learning; (3) **Restoration fine-tuning** — enables all conditions (residual + LQ image) and applies L1 loss to enhance fidelity.

### Key Designs

1. **Multi-Angle Multi-Scale Gray-Level Co-occurrence Matrix (MAS-GLCM)**:

    - **Function**: Extracts content-agnostic degradation features from degraded images.
    - **Mechanism**: Standard GLCM computes co-occurrence frequencies of pixel pairs at specific distances and orientations; MAS-GLCM averages GLCMs computed across multiple angles $\Theta$ and scales $L$: $M_{mas} = \frac{1}{n \times m} \sum_{i,j} M_{L_i \cdot \sin(\Theta_j), L_i \cdot \cos(\Theta_j)}$
    - **Design Motivation**: GLCM computation naturally discards image content (only aggregating gray-level co-occurrences), avoiding the content-coupling problem of Sobel or frequency-based methods. Multi-angle and multi-scale coverage mitigates locality bias.
    - **Experimental Validation**: KNN accuracy of 97.13% on degradation type classification (vs. Fourier 65.80%); 74.17% on degradation level classification (vs. Fourier 30.83%).

2. **Three-Stage Diffusion Training**:

    - **Base formula**: $x_{t-1} = x_t - \alpha_t x_{res}^\theta - \frac{\beta_t^2}{\bar{\beta}_t} \epsilon^\theta + \delta_t x_{lq}$
    - **Generation stage**: $\alpha_t \equiv 0, \delta_t \equiv 0$, reducing to VE-SDE denoising for pure generative prior learning.
    - **Bridging stage**: $\delta_t \equiv 0$; residual $x_{res}$ is introduced as a condition carrying degradation information. MAS-GLCM features $F_{mas}$ are aligned with diffusion intermediate features $F_{diff}$ via bidirectional cross-entropy, while an MLP performs degradation classification to ensure $F_{mas}$ retains discriminative capacity.
    - **Restoration stage**: All parameters $\alpha_t, \beta_t, \delta_t$ are activated; $x_{lq}$ is directly injected to enhance fidelity. Degradation classification is replaced by all-negative-pair contrastive learning to accommodate real-world scenarios.

3. **Degradation-Generation Bridging Loss**:

    - **Function**: Aligns GLCM features and diffusion features during the bridging stage.
    - **Mechanism**: $\mathcal{L}_{bridge} = \frac{1}{2}[\text{H}(y^{m2d}, p^{m2d}) + \text{H}(y^{d2m}, p^{d2m})]$, bidirectional cross-entropy alignment.
    - **Total loss**: $\mathcal{L}_{bdg} = \mathcal{L}_{gen} + \lambda(\mathcal{L}_{bridge} + \mathcal{L}_{deg-cls})$, with $\lambda = 0.1$.

### Loss & Training
- **Generation stage**: Standard denoising loss (noise prediction + residual prediction).
- **Bridging stage**: Denoising + feature alignment + degradation classification.
- **Restoration stage**: L1 fidelity loss + bridging feature alignment loss + all-negative contrastive learning.
- Real-world degradation is simulated via the multi-step degradation pipeline of Real-ESRGAN, with 8 intermediate states defined as pseudo-labels for "degradation order."

## Key Experimental Results

### Main Results — All-in-One Restoration

| Method | Approach | Fidelity (PSNR↑) | Perceptual Quality (LPIPS↓) |
|--------|----------|-------------------|------------------------------|
| PromptIR | Discriminative | High | Poor (over-smoothed) |
| DiffBIR | Generative | Low (inconsistent) | Good |
| **BDG** | **Discrimination + Generation Unified** | **Significantly improved** | **Maintained** |

### Ablation Study — MAS-GLCM Degradation Classification

| Degradation Representation | Type Classification Acc (%) | Level Classification Acc (%) |
|---------------------------|-----------------------------|-----------------------------|
| LQ Image | 51.44 | 20.00 |
| Sobel (gradient) | 40.80 | 23.33 |
| Laplace (gradient) | 83.05 | 20.83 |
| Fourier (frequency) | 65.80 | 30.83 |
| **MAS-GLCM** | **97.13** | **74.17** |

### Key Findings
- **MAS-GLCM substantially outperforms existing degradation representations**: surpasses Fourier by 43 percentage points on fine-grained degradation level classification.
- **All three training stages are indispensable**: removing the bridging stage and transitioning directly from generation to restoration causes a significant fidelity drop.
- **Generative priors are successfully preserved**: restoration results exhibit texture richness comparable to purely generative models while achieving substantially higher fidelity.
- **No architectural modifications required**: BDG achieves improvements solely through training strategy and loss design, without altering the network structure.

## Highlights & Insights
- **Content-agnostic nature of MAS-GLCM** is the most prominent contribution: the GLCM computation inherently excludes content information, retaining only texture statistics — thereby decoupling degradation discrimination from image semantics.
- **Elegance of the three-stage transition design**: generation → bridging → restoration progressively introduces parameters in the diffusion formula to govern the evolution of model capabilities — from pure generation to conditional generation to restoration.
- **Reformulation of degradation classification as contrastive learning**: the bridging stage uses discrete class labels, while the restoration stage adopts all-negative contrastive learning — accommodating the practical reality that real-world degradations cannot always be explicitly categorized.

## Limitations & Future Work
- Angle and scale parameters of MAS-GLCM require manual selection.
- Three-stage training increases overall training complexity.
- "Order classification" for real-world degradation relies on approximate pseudo-labels that do not fully reflect true degradation processes.
- Validation on temporal scenarios such as video restoration has not been conducted.

## Related Work & Insights
- **vs. PromptIR/AirNet**: These methods employ auxiliary networks for degradation discrimination but produce overly smooth outputs; BDG performs discrimination internally within the diffusion model.
- **vs. DiffBIR/StableSR**: These methods exploit generative priors but lack degradation awareness, resulting in poor fidelity in all-in-one settings; BDG addresses this through MAS-GLCM bridging.
- **vs. DiffUIR**: DiffUIR predicts only residuals, sacrificing generative priors; BDG simultaneously predicts both noise and residuals, preserving generative priors.

## Rating
- Novelty: ⭐⭐⭐⭐ MAS-GLCM degradation representation is original; three-stage training design demonstrates substantial depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers degradation classification validation, all-in-one restoration, and real-world super-resolution, though quantitative comparisons with additional baselines are limited.
- Writing Quality: ⭐⭐⭐⭐ Method derivation is clear, though the dense notation requires careful tracking.
- Value: ⭐⭐⭐⭐ The unified discrimination-generation framework offers practically meaningful guidance for the image restoration community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Face2Scene: Using Facial Degradation as an Oracle for Diffusion-Based Scene Restoration](../../CVPR2026/image_generation/face2scene_using_facial_degradation_as_an_oracle_for_diffusion-based_scene_resto.md)
- [\[CVPR 2026\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](../../CVPR2026/image_generation/v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[ICLR 2026\] JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation](jointdiff_bridging_continuous_and_discrete_in_multi-agent_trajectory_generation.md)
- [\[ICLR 2026\] GenDR: Lighten Generative Detail Restoration](gendr_lighten_generative_detail_restoration.md)
- [\[ICLR 2026\] Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models](bridging_generalization_gap_of_heterogeneous_federated_clients_using_generative_.md)

</div>

<!-- RELATED:END -->
