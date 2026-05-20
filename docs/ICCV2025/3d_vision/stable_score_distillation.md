---
title: >-
  [Paper Note] Stable Score Distillation
description: >-
  [ICCV 2025][3D Vision][Score Distillation] This paper proposes Stable Score Distillation (SSD), which achieves more stable and precise text-guided 2D/3D editing through single-classifier cross-prompt guidance and cross-t…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Score Distillation"
  - "3D Scene Editing"
  - "2D Image Editing"
  - "Diffusion Models"
  - "Classifier-Free Guidance"
  - "NeRF"
  - "3DGS"
date: 2026-05-08
content_hash: 5f83d5701c5ccf29
---

# Stable Score Distillation

**Conference**: ICCV 2025
**arXiv**: [2507.09168](https://arxiv.org/abs/2507.09168)  
**Code**: [https://github.com/Alex-Zhu1/SSD](https://github.com/Alex-Zhu1/SSD)  
**Area**: 3D Vision / Text-Guided Editing
**Keywords**: Score Distillation, 3D Scene Editing, 2D Image Editing, Diffusion Models, Classifier-Free Guidance, NeRF, 3DGS

## TL;DR

This paper proposes Stable Score Distillation (SSD), which achieves more stable and precise text-guided 2D/3D editing through single-classifier cross-prompt guidance and cross-trajectory regularization via a null-text branch, improving editing alignment while preserving the structural content of the source.

## Background & Motivation

Text-guided image/3D editing relies on the prior knowledge of diffusion models, yet existing score distillation methods exhibit notable deficiencies:

**Limitations of SDS**: Score Distillation Sampling introduces global optimization interference in editing tasks, causing blurriness and artifacts in non-edited regions, as it optimizes globally with respect to the entire prompt.

**Limitations of DDS**: Delta Denoising Score eliminates model bias by introducing a source branch, but lacks explicit protection of source content structure, allowing non-edited regions to be inadvertently modified (e.g., clothing changes when editing a character's face).

**Limitations of CSD**: Classifier Score Distillation employs dual classifiers to obtain cross-prompt editing directions, but similarly lacks a source structure preservation mechanism, leading to structural distortion and artifacts.

**Insufficient Editing Strength**: DDS-based methods tend to produce insufficient editing intensity in style editing scenarios, resulting in nearly imperceptible changes.

The authors make two core observations:

- **Cross-prompt**: A single classifier suffices to provide an editing direction from the source prompt to the target prompt, without the complexity of dual classifiers.
- **Cross-trajectory**: Aligning the editing direction with the source content structure ensures a stable optimization process and prevents abrupt structural changes.

## Method

### Overall Architecture

SSD is organized around three core components, with the final loss defined as their sum:

$$L_{\text{final}} = L_{\text{ssd}} + L_{\text{align}} + L_{\text{ID}}$$

where $L_{\text{ssd}}$ is the core distillation loss, $L_{\text{align}}$ is the prompt enhancement term, and $L_{\text{ID}}$ is the source latent regularization term.

### Key Design 1: Core Formulation of Stable Score Distillation

Unlike DDS, which relies on an auxiliary source branch, SSD leverages the CFG formulation to construct a cross-prompt editing direction and introduces a null-text branch for regularization:

$$L_{\text{ssd}} = \epsilon_\phi(z_t, \hat{y}) + s(\epsilon_\phi(z_t, y) - \epsilon_\phi(z_t, \hat{y})) - \epsilon_\phi(\hat{z}_t, \varnothing)$$

This expression decomposes into two terms:

$$L_{\text{ssd}} = \underbrace{w_p(\epsilon_\phi(z_t, y) - \epsilon_\phi(z_t, \hat{y}))}_{\text{cross-prompt}} + \underbrace{w_t(\epsilon_\phi(z_t, \hat{y}) - \epsilon_\phi(\hat{z}_t, \varnothing))}_{\text{cross-trajectory}}$$

- **Cross-prompt term**: A single classifier measures the prediction difference between the current latent under the target prompt $y$ and the source prompt $\hat{y}$, providing a smooth textural transition direction.
- **Cross-trajectory term**: Measures the distance between the current latent predicted under the source prompt and the source latent predicted under null-text, constraining the structure against abrupt changes. This is the key distinction between SSD and CSD — setting $w_t=0$ degrades to CSD, which fails to preserve structure.

### Key Design 2: Prompt Enhancement Branch

DDS-based methods suffer from insufficient editing strength in style editing. SSD adds a target prompt enhancement term:

$$L_{\text{align}} = w_e(\epsilon_\phi(z_t, y) - \epsilon_\phi(z_t, \varnothing))$$

This term corresponds to the standard CFG classifier direction, directly amplifying the guidance strength of the target prompt. The coefficient $w_e$ controls enhancement intensity; excessively large values lead to oversaturation and should be tuned jointly with the cross-trajectory weight.

### Key Design 3: Source Latent Regularization

In 3DGS editing, latent-space losses may cause localized gradient explosions (manifesting as bright spots). SSD therefore introduces an identity regularization term:

$$L_{\text{ID}} = w(t) \cdot (x_t - \hat{x}_t)$$

where $w(t)$ is a weight that decreases over iterations. Unlike PDS, which uses the noise-free $\hat{x}_0$, SSD uses the noisy $\hat{x}_t$ to avoid gradient explosions.

### Connection to InstructPix2Pix

The authors find a structural correspondence between SSD and the single-step inversion formula of IP2P: the intermediate term in the IP2P formulation corresponds to cross-trajectory regularization, while the final term corresponds to the cross-prompt component. This implies that applying a DDS-style loss on an IP2P model requires only the editing branch, with no need for a source branch.

## Key Experimental Results

### Main Results 1: 3D Scene Editing

| Method | CLIP Sim ↑ | Sim Dire ↑ | User Study ↑ |
|--------|-----------|-----------|-------------|
| IN2N | 0.1676 | 0.0707 | 14.54% |
| DDS | 0.1780 | 0.0401 | 5.45% |
| GS-Editor | 0.1758 | 0.0429 | 14.54% |
| DGE | 0.1758 | 0.0563 | 23.63% |
| **SSD (Ours)** | **0.1846** | **0.0773** | **41.81%** |

- Evaluated on IN2N, LLFF, and Mip-NeRF360 datasets across 6 scenes and 10 prompts.
- User study with 55 participants: SSD achieves 41.81% preference, substantially outperforming all baselines.
- SSD attains the best CLIP Sim and Sim Dire scores.

### Main Results 2: 2D Image Editing (PIE-Bench, 700 images, 9 editing types)

| Method | Distance↓ | LPIPS↓ | MSE↓ | CLIP↑ |
|--------|----------|--------|------|-------|
| DDIM + P2P | 69.43 | 208.80 | 219.88 | 25.01 |
| DDS | 14.74 | 50.58 | 45.09 | 25.86 |
| DDS + CDS | 7.15 | 33.14 | 25.29 | 24.96 |
| **Ours** | 28.13 | 82.43 | 86.64 | **26.94** |
| **Ours + CDS** | **6.90** | **32.15** | **24.21** | 25.12 |

- SSD achieves the best CLIP Similarity (26.94), validating the effectiveness of the prompt enhancement branch.
- Combined with CDS, SSD achieves state-of-the-art performance across all structure preservation metrics (Distance 6.90, LPIPS 32.15, MSE 24.21).
- Standalone SSD trades greater structural change for substantially improved editing quality, particularly in style editing.

### Ablation Study

| Component | Effect |
|-----------|--------|
| Cross-trajectory ($w_t=0$) | Degrades to CSD; structure is not preserved; saturation and artifacts appear |
| Prompt enhancement ($w_e$) | Critical for style editing; removing it significantly reduces editing strength |
| ID regularization | Suppresses localized gradient explosions (bright spots) in 3DGS; excessive weight restricts editing attributes |
| Convergence speed | ~3000 iterations for NeRF; ~1500 iterations for 3DGS (with non-increasing timestep sampling) |

## Highlights & Insights

1. **Elegant Framework Design**: Compared to DDS's dual-branch structure and CSD's dual classifiers, SSD requires only a single classifier and a null-text branch — a simpler design that simultaneously addresses both stability and editing strength.
2. **Ingenuity of Cross-Trajectory Regularization**: By comparing the prediction of the current latent under the source prompt against the prediction of the source latent under the unconditional setting, the method implicitly constrains structural changes without explicit pixel-level reconstruction losses.
3. **Theoretical Connection to IP2P**: The structural correspondence between SSD and InstructPix2Pix provides a new perspective for understanding the working mechanism of IP2P.
4. **Plug-and-Play Compatibility**: SSD integrates directly into existing DDS-based editing pipelines (NeRF editing, 2D editing) without requiring LoRA or fine-tuning, and complements methods such as CDS for further performance gains.
5. **Dominant User Study Results**: SSD receives 41.81% of votes in the 3D editing user study, nearly double the second-ranked method DGE (23.63%).

## Limitations & Future Work

1. **Optimization Speed**: As an optimization-based method, the editing process requires thousands of iterations, remaining slower than one-step or few-step approaches (e.g., TurboEdit, SD-Turbo).
2. **Trade-off Between Structure Preservation and Editing Strength**: On PIE-Bench, standalone SSD's structure distance (28.13) is higher than DDS (14.74), indicating that strong editing inherently entails greater structural change.
3. **ID Regularization Trade-off**: Excessively large weights suppress editing attributes (e.g., affecting the spider emblem on a character's chest), requiring manual hyperparameter tuning.
4. **Multiple Hyperparameters**: The weights $w_p$, $w_t$, $w_e$, and the ID regularization schedule $w(t)$ all require scene-specific adjustment.

## Related Work & Insights

- **DDS (Delta Denoising Score)**: Eliminates bias via a source branch but does not preserve structure; SSD replaces the role of the source branch with cross-trajectory regularization.
- **CSD (Classifier Score Distillation)**: Dual classifiers provide editing directions but lack structural constraints; setting $w_t=0$ in SSD recovers CSD.
- **NFSD (Noise-Free Score Distillation)**: Decomposing the CFG score reveals that the classifier is the core driver of editing direction, inspiring SSD's single-classifier design.
- **PDS (Posterior Distillation Sampling)**: Matches stochastic latents for posterior distillation; SSD simplifies its identity preservation strategy by substituting the noisy latent for the noise-free one.
- **DreamCatalyst**: Extends PDS with decreasing timestep sampling; SSD likewise adopts non-increasing timestep sampling to accelerate convergence.
- **Insight**: The CFG formulation is a natural tool for cross-distribution guidance. Generalizing it from "conditional vs. unconditional" to "target vs. source" is an elegant abstraction worth exploring in other distillation settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The single-classifier + null-text branch design is concise and effective; cross-trajectory regularization represents a meaningful contribution
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers both 3D (NeRF/3DGS) and 2D (PIE-Bench) editing with user studies and ablations, though quantitative ablation tables are absent
- **Writing Quality**: ⭐⭐⭐ — Mathematical derivations are clear, but occasional inconsistencies in LaTeX notation and minor grammatical issues are present
- **Value**: ⭐⭐⭐⭐ — Advances the score distillation editing field in a practical manner; the framework is broadly compatible and plug-and-play

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Advancing Text-to-3D Generation with Linearized Lookahead Variational Score Distillation](advancing_text-to-3d_generation_with_linearized_lookahead_variational_score_dist.md)
- [\[ICCV 2025\] Identity Preserving 3D Head Stylization with Multiview Score Distillation](identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)
- [\[ICCV 2025\] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [\[ICCV 2025\] Generating Physically Stable and Buildable Brick Structures from Text](generating_physically_stable_and_buildable_brick_structures_from_text.md)
- [\[ICCV 2025\] G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection](g2sf_geometry-guided_score_fusion_for_multimodal_industrial_anomaly_detection.md)

</div>

<!-- RELATED:END -->
