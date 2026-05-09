---
title: >-
  [Paper Note] ChordEdit: One-Step Low-Energy Transport for Image Editing
description: >-
  [CVPR 2026][Image Generation][Image Editing] Based on dynamic optimal transport theory, a low-energy Chord control field is derived to smooth unstable naive editing fields, achieving the first training-free, inversion-free, and high-fidelity real-time image editing for distilled one-step T2I models.
tags:
  - CVPR 2026
  - Image Generation
  - Image Editing
  - Optimal Transport
  - One-step Inference
  - Diffusion Distillation Models
  - Training-free Editing
date: 2026-05-08
content_hash: 9de356a373e4843f
---

# ChordEdit: One-Step Low-Energy Transport for Image Editing

**Conference**: CVPR 2026  
**arXiv**: [2602.19083](https://arxiv.org/abs/2602.19083)  
**Code**: Available (Project Page: [https://chordedit.github.io](https://chordedit.github.io))  
**Area**: Image Generation  
**Keywords**: Image Editing, Optimal Transport, One-step Inference, Diffusion Distillation Models, Training-free Editing

## TL;DR

Based on dynamic optimal transport theory, a low-energy Chord control field is derived to smooth unstable naive editing fields, achieving the first training-free, inversion-free, and high-fidelity real-time image editing for distilled one-step T2I models.

## Background & Motivation

### 1. State of the Field

One-step text-to-image (T2I) models, such as SD-Turbo, SwiftBrush-v2, and InstaFlow, have achieved unprecedented generation speeds through distillation of large-scale diffusion models, generating high-quality images in a single forward pass. This real-time generation capability naturally raises expectations for its application in text-guided image editing—if generation takes only one step, can editing also be real-time?

### 2. Limitations of Prior Work

Existing image editing methods face a dilemma between two camps:

- **Multi-step methods** (DDIM+PnP, FlowEdit, etc.): Require 30-50 inference steps, with runtimes ranging from 7-80 seconds, making real-time interaction impossible.
- **One-step trained methods** (SwiftEdit): Require training specialized inversion networks, sacrificing model-agnosticism and relying on precise inversion.
- **Training-free differential methods** (InfEdit, FlowEdit): Work well on multi-step models but fail completely on one-step models—resulting in severe object distortion and collapse of non-edited regions.

### 3. Root Cause

The distillation process of one-step models makes the mapping from text conditions to vector fields highly non-linear and sensitive. A naive editing field (target drift - source drift) is essentially the arithmetic difference between two large-magnitude, diverging trajectories, producing an unstable high-energy control field. In multi-step models, this instability is gradually absorbed through iterative small-step integration; however, in one-step models, the entire path must be covered in a single step, where the massive integration step size sharply magnifies errors, leading to total editing failure.

### 4. Paper Goals

The goal is to enable high-fidelity text-guided image editing for one-step T2I models under the premise of being **training-free and inversion-free**, while maintaining real-time speeds.

### 5. Starting Point

The authors move beyond the approach of "simple arithmetic on drifts" and redefine the editing problem as a **dynamic optimal transport (OT) problem**—finding a low-energy transport path between the source and target distributions. Starting from the Benamou-Brenier framework, they derive a theoretically guaranteed control field estimator.

### 6. Core Idea

Use time-weighted averaging instead of naive differentiation to construct a low-energy, low-variance Chord control field, allowing one-step large-step integration to complete the editing transport stably.

## Method

### Overall Architecture

The workflow of ChordEdit is extremely concise (Algorithm 1):

1. **Input**: Source image $x_{\rm src}$, source/target text $c_{\rm src}, c_{\rm tar}$, hyperparameters $t, \delta, \lambda, t_c$.
2. **Compute Chord Control Field**: Query the model at two time points $t$ and $t-\delta$, and perform time-weighted averaging.
3. **One-step Transport**: $x^{\rm pred} = x_{\rm in} + \lambda \hat{u}$.
4. **Optional Proximal Refinement**: One additional forward pass to enhance target semantics.

The entire editing process requires only 1-2 network forward evaluations (NFE), with a runtime of 0.20-0.38 seconds.

### Key Designs

#### Module 1: Observable Model

**Function**: Defines a unified interface to query one-step models with different parameterizations.

**Mechanism**: The editing anchor is fixed as the clean source image $x_\tau = x_1$. A noisy proxy $z$ is synthesized via the forward noising kernel $K_t(\cdot|x_\tau)$, and the model is queried to obtain an observable output $Q(z, t, c)$. An observable proxy field is defined:

$$\mathbf{R}(x_\tau, t) = \mathbb{E}_{z \sim K_t}[\mathcal{B}_t \Delta Q(z, t)]$$

where $\mathcal{B}_t$ is a time-dependent linear mapping that unifies different parameterizations (noise prediction, velocity prediction, etc.) into the drift/velocity space.

**Design Motivation**: Different one-step models (e.g., SD-Turbo using noise prediction, InstaFlow using velocity prediction) need to be handled uniformly. By introducing the linear mapping $\mathcal{B}_t$, the method becomes model-agnostic.

#### Module 2: Chord Control Field (CCF)

**Function**: Smooths the unstable naive editing field $\mathbf{R}$ into a low-energy control field $\hat{u}$.

**Mechanism**: From the perspective of dynamic OT, editing is viewed as an **estimation problem**. The observable field $\mathbf{R}$ is an observation of the true field $u_t$ plus zero-mean noise $\varepsilon_t$. Within a short time window $[t-\delta, t]$, the optimal estimate is obtained by minimizing a strictly convex quadratic proxy objective (balancing the recursive energy prior with the consistency of new observations):

$$\hat{u}_t(x_\tau) = \frac{t \cdot \mathbf{R}(x_\tau, t-\delta) + \delta \cdot \mathbf{R}(x_\tau, t)}{t + \delta}$$

This is essentially a **causal one-sided kernel smoothing** operator that performs time-weighted averaging on the naive field.

**Design Motivation**:
- The naive field $\mathbf{R}$ experiences energy explosion in the one-step limit (experimentally verified: energy surges as steps $S \to 1$).
- Via Jensen's inequality, time averaging guarantees $L^2$ contractivity: $\int\|\hat{u}\|^2 \leq \int\|\mathbf{R}\|^2$.
- The $L^\infty$ norm, time derivative, and spatial gradient of the field are simultaneously contracted, reducing the consistency constant $\mathcal{C}(u)$ for explicit Euler.
- This directly tightens the global $O(h)$ error bound for $h=1$ step, ensuring the stability of one-step integration.

#### Module 3: Proximal Refinement

**Function**: Optionally enhances target semantics after the transport is completed.

**Mechanism**: Performs a forward pass on the transport result $x^{\rm pred}$ using only the target prompt $c_{\rm tar}$:

$$\operatorname{prox}(x^{\rm pred}, t_c, c_{\rm tar}) = \mathcal{B}_{t_c} Q(x^{\rm pred}, t_c, c_{\rm tar})$$

**Design Motivation**: Chord transport tends to be conservative (high PSNR, lower CLIP). By decoupling structural preservation (handled by transport) and semantic enhancement (handled by refinement) into a modular design, users can balance them as needed.

### Loss & Training

ChordEdit is a **completely training-free** method. Its core formula (Eq. 4.5) is derived analytically from optimal transport theory, requiring no learning process. The only requirements are hyperparameters during the inference stage:

- $t=0.90$: Stepping time point.
- $\delta=0.15$: Smoothing window width (trade-off between stability and semantic strength).
- $\lambda=1.00$: Step size scaling.
- $t_c=0.30$: Noise level for proximal refinement.

## Key Experimental Results

### Main Results

Comprehensive comparison with multi-step/few-step/one-step methods on PIE-bench (700 samples, 10 editing categories, 512×512):

| Type | Method | PSNR↑ | MSE↓(×10³) | LPIPS↓(×10³) | CLIP-Whole↑ | CLIP-Edited↑ | Training-free | Inversion-free | Steps | NFE | Runtime(s) | Memory(MiB) |
|------|------|-------|-----------|-------------|-------------|-------------|--------|--------|------|-----|------------|-----------|
| Multi-step | DirectInv+PnP | 21.43 | 8.10 | 106.26 | 25.48 | 22.63 | ✓ | ✗ | 50 | 150 | 28.03 | 9262 |
| Multi-step | FlowEdit (SD3) | 22.17 | 7.69 | 104.81 | **26.64** | **23.69** | ✓ | ✓ | 33 | 33 | 7.22 | 17140 |
| Few-step | InfEdit (SD1.4) | **24.14** | 6.82 | **55.69** | 24.89 | 21.88 | ✓ | ✓ | 4 | 4 | 1.41 | **6502** |
| One-step | SwiftEdit | 21.71 | 8.22 | 91.22 | 24.93 | 21.85 | ✗ | ✗ | 1 | 2 | 0.54 | 15060 |
| One-step | **Ours (SD-Turbo)** | 22.20 | 6.84 | 128.25 | 25.58 | 22.96 | ✓ | ✓ | **1** | 2 | **0.38** | 6988 |
| One-step | Ours (w/o prox) | 23.89 | 5.05 | 88.36 | 24.97 | 21.87 | ✓ | ✓ | **1** | **1** | **0.20** | 6988 |

Ablation of transport and refinement:

| Method | Naive Field PSNR↑ | Naive Field CLIP-Ed↑ | Chord Field PSNR↑ | Chord Field CLIP-Ed↑ | NFE |
|------|-------------|-------------------|-------------|---------------------|-----|
| w/o prox | 21.89 | 20.83 | **23.89** | 21.87 | 1 |
| w/ prox | 21.38 | 21.96 | 22.20 | **22.96** | 2 |

### Ablation Study

**Model-agnosticism Verification**: Tested on three different one-step T2I models, ChordEdit consistently outperforms the naive baseline:

| T2I Model | Naive PSNR | ChordEdit PSNR | Naive CLIP-Ed | ChordEdit CLIP-Ed |
|----------|----------|---------------|-------------|------------------|
| InstaFlow | 22.05 | **23.05** | 20.19 | **21.39** |
| SwiftBrush-v2 | 20.52 | **22.04** | 21.06 | **22.58** |
| SD-Turbo | 21.38 | **22.20** | 21.96 | **22.96** |

**Noise Sample Analysis**: With $n=1$, the Pareto front of ChordEdit nearly overlaps with $n=2, 3, 4$, and strictly Pareto-dominates the naive method at $n=4$. The CLIP CoV across 20 seeds is only 0.20%, and PSNR CoV is only 0.07%.

### Key Findings

1. **Energy and Stability Relationship**: When steps $S \to 1$, the energy of the naive field surges and PSNR collapses; the Chord field maintains low energy and stable PSNR.
2. **Pareto Dominance**: On the LPIPS-CLIP trade-off curve, ChordEdit ($\delta \neq 0$) strictly Pareto-dominates the naive baseline ($\delta = 0$).
3. **Significant Speed Advantage**: 19× faster than FlowEdit, 208× faster than Direct Inversion, with VRAM usage around 46% of SwiftEdit.
4. **User Study**: Participants overwhelmingly preferred ChordEdit for both editing semantics (42.5%) and background preservation (48.3%).

## Highlights & Insights

1. **Theoretical Elegance**: Derives the editing control field from dynamic optimal transport (Benamou-Brenier) rather than heuristic design. Jensen's inequality directly guarantees energy contraction, and the tightening of the consistency constant provides an error bound guarantee for one-step integration.
2. **Minimalist Implementation**: The core formula is just a one-line weighted average (Eq. 4.5), requiring no additional networks, no inversion, and no masks—true plug-and-play.
3. **Modular Decoupling**: Splits the problem into "structure-preserving low-energy transport" + "optional semantic-enhancing refinement." The functions are orthogonal, allowing flexible selection by the user.
4. **Single Noise Sufficiency**: Proves that the intrinsic low-variance property of the Chord field makes Monte Carlo averaging unnecessary, yielding seed-robust results with $n=1$.
5. **Parameterization Unification**: Through the design of the linear mapping $\mathcal{B}_t$, different parameterizations (noise prediction, velocity prediction, etc.) are unified under the same framework.

## Limitations & Future Work

1. **Higher LPIPS Scores**: Full ChordEdit (w/ prox) has an LPIPS of 128.25, higher than InfEdit's 55.69, indicating that proximal refinement sacrifices perceptual similarity while enhancing semantics.
2. **Limited Editing Intensity**: As a training-free method, it may struggle with complex structural edits (e.g., large-scale pose changes).
3. **Hyperparameter Sensitivity**: $\delta$ controls the stability vs. semantics trade-off, and $t_c$ controls refinement intensity; different editing scenarios may require different hyperparameter combinations.
4. **Text-guided Only**: Currently supports only source/target prompt pair editing, without extension to other control modalities (e.g., image references, regional masks).
5. **Future Directions**: Exploring adaptive $\delta$ selection strategies, integration with attention control methods, or extending the OT framework to video editing.

## Related Work & Insights

- **FlowEdit**: Also a training-free and inversion-free method, but relies on multi-step integration to average out instability; ChordEdit resolves the one-step problem from a theoretical level.
- **SwiftEdit**: The only one-step editing competitor, but requires training an inversion network and has high VRAM overhead (15GB vs 7GB).
- **InfEdit**: Best background preservation among few-step methods (PSNR 24.14), but still requires 4 inference steps.
- **Dynamic OT Inspiration**: Applying the Benamou-Brenier framework to generative model editing is a novel perspective that might inspire more OT-guided generative control methods.

## Rating

⭐⭐⭐⭐ — The theoretical derivation is solid and elegant, the method is minimalist and efficient, and it demonstrates strong model-agnosticism. While high LPIPS and limited editing intensity are minor drawbacks, the core contributions (theoretical guarantees for low-energy transport fields + real-time performance) are highly valuable.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

## Related Papers

- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](language-free_generative_editing_from_one_visual_example.md)
- [\[CVPR 2026\] Low-Resolution Editing is All You Need for High-Resolution Editing](low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)
- [\[CVPR 2026\] WaDi: Weight Direction-aware Distillation for One-step Image Synthesis](wadi_weight_direction-aware_distillation_for_one-step_image_synthesis.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)
- [\[CVPR 2025\] SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion](../../CVPR2025/image_generation/swiftedit_lightning_fast_text-guided_image_editing_via_one-step_diffusion.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Low-Resolution Editing is All You Need for High-Resolution Editing](low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)
- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](language-free_generative_editing_from_one_visual_example.md)
- [\[CVPR 2026\] WaDi: Weight Direction-aware Distillation for One-step Image Synthesis](wadi_weight_direction-aware_distillation_for_one-step_image_synthesis.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] Group Editing: Edit Multiple Images in One Go](group_editing_edit_multiple_images_in_one_go.md)

</div>

<!-- RELATED:END -->
