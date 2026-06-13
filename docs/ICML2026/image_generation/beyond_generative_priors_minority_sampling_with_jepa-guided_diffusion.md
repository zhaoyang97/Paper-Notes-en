---
title: >-
  [Paper Note] Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion
description: >-
  [ICML2026][Image Generation][Minority Sampling] This paper proposes JEPA Guidance, which utilizes the implicit density signals from JEPA (e.g.…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Minority Sampling"
  - "Diffusion Models"
  - "JEPA"
  - "World Model Prior"
  - "Randomized SVD"
date: 2026-05-08
content_hash: 8ec6d2bc7467d94f
---

# Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion

**Conference**: ICML2026  
**arXiv**: [2605.24631](https://arxiv.org/abs/2605.24631)  
**Code**: https://github.com/soobin-um/jepa-guidance  
**Area**: Image Generation  
**Keywords**: Minority Sampling, Diffusion Models, JEPA, World Model Prior, Randomized SVD  

## TL;DR

This paper proposes JEPA Guidance, which utilizes the implicit density signals from JEPA (e.g., DINOv2) encoders to guide the sampling of diffusion models. It shifts the definition of minority samples from "low density under the generative model prior" to "low density under a world prior," enabling the generation of semantically meaningful rare samples across unconditional, class-conditional, and text-to-image scenarios.

## Background & Motivation

**Background**: Minority sampling aims to generate instances from low-density regions of the data manifold, which is essential for fields like medical diagnosis, anomaly detection, and creative AI. Diffusion models, with their ability to model complex distributions, have become the primary framework for this task. Existing methods include classifier-guided and self-contained minority guidance.

**Limitations of Prior Work**: All existing methods define "minority samples" as low-density samples under the density $p_\theta$ learned by the generative model itself. This **generator-centric** definition is inherently tied to the training data, meaning that generated "rare samples" are only uncommon within the context of a specific model (e.g., a dog on a white background) and do not necessarily correspond to semantic rarity in the real world.

**Key Challenge**: The generator prior $p_\theta$ only captures the distribution of a specific training set and fails to reflect broader real-world semantics. When "world-level" rare samples are needed (e.g., stealth aircraft, atypical characters), generator-centric methods are ineffective.

**Goal**: To transition the definition of minority samples from generator-centric to **world-centric**—measuring rarity using a world prior independent of the generative model and achieving this within diffusion sampling.

**Key Insight**: JEPA (Joint-Embedding Predictive Architecture), such as DINOv2, is trained on large-scale data, and its representations implicitly encode data density (JEPA-SCORE), serving as a natural proxy for a world prior.

**Core Idea**: The Jacobian singular values of the JEPA encoder are used to estimate density under the world prior. This is efficiently approximated using Randomized SVD and used to guide diffusion sampling toward low-density regions via gradients, achieving world-centric minority sampling.

## Method

### Overall Architecture

Given a pre-trained diffusion model $\epsilon_\theta$ and a JEPA encoder $f_\phi$ (such as DINOv2), an approximate JEPA-SCORE of the denoised estimate $\hat{x}_{0|t}$ is intermittently calculated during the reverse diffusion sampling process. Its negative gradient is then used to guide the sampling toward low-density regions under the world prior. This process requires no training and uses pre-trained models only during inference. The sampling is modified as: $x_{t-1} = \mu_\theta(x_t, t) + \Sigma_\theta^{1/2} z - \eta_t \nabla_{x_t} \text{JS}^*(\hat{x}_{0|t})$.

### Key Designs

1. **Randomized SVD Approximation of JEPA-SCORE**:

    - **Function**: Reduces the extremely high computational cost of JEPA-SCORE to a practical level.
    - **Mechanism**: JEPA-SCORE is defined as the sum of the logarithms of all singular values of the encoder Jacobian $J_f(x) \in \mathbb{R}^{d \times n}$, i.e., $\text{JS}(x) = \sum_{i=1}^r \log(\sigma_i(J_f(x)))$. Direct SVD is too costly. The method uses Randomized SVD to construct a low-rank projection matrix $Q \in \mathbb{R}^{d \times l}$ ($l \ll d$), compressing the Jacobian into $\tilde{J}_f = Q^\top J_f \in \mathbb{R}^{l \times n}$, and approximates it using only the top $k$ singular values. The paper proves an upper bound for the approximation error $\text{JS} - \bar{\text{JS}} \leq \mathcal{E}_{\text{RSVD}} + \mathcal{E}_{\text{Trunc}}$. Experiments show that $k \approx 10$ is sufficiently effective.
    - **Design Motivation**: Original JEPA-SCORE requires SVD on a large matrix, which is infeasible during each iteration of diffusion sampling. Randomized SVD reduces the complexity from $O(dn)$ to $O(ln)$.

2. **Gradient Acceleration via Envelope Theorem**:

    - **Function**: Eliminates the need for backpropagation through the internal optimization process of Randomized SVD, significantly reducing memory and computational overhead.
    - **Mechanism**: Directly calculating the gradient of $\bar{\text{JS}}(\hat{x}_{0|t})$ with respect to $x_t$ requires backpropagation through the computation graph of $Q$, as $Q$ depends on $J_f(\hat{x}_{0|t})$ and thus indirectly on $x_t$. By utilizing the Envelope Theorem, when the inner Randomized SVD reaches optimality, the optimal projection $Q^*$ can be treated as a constant (stop-gradient), such that $\text{JS}^* = \sum_{i=1}^k \log(\tilde{\sigma}_i(\text{sg}(Q^{*\top}) J_f))$. This ensures first-order gradient correctness while avoiding backpropagation through the SVD process.
    - **Design Motivation**: A naive implementation where $Q$ retains the computation graph leads to memory explosion; the Envelope Theorem provides theoretical guarantees that allow stop-gradient without losing gradient accuracy.

3. **Deferred Guidance**:

    - **Function**: Bridges the domain gap between the JEPA encoder and intermediate diffusion states and extends the method to conditional generation.
    - **Mechanism**: JEPA guidance is deferred until after an intermediate timestep $\tau T$ (e.g., $\tau = 0.8$). In early sampling steps, $\hat{x}_{0|t}$ is too blurry/noisy, creating a large gap from the clean inputs expected by the JEPA encoder. Deferred guidance allows the conditional diffusion model to sample freely first to establish conditional structure, then guides it toward low-density regions in the later stages.
    - **Design Motivation**: Experiments show that quality and text alignment drop significantly without delay ($\tau = 1.0$). Deferred guidance also naturally solves the issue of JEPA's inability to perceive conditional information—the conditional information is already integrated into the early sampling stages.

## Key Experimental Results

### Main Results—Unconditional and Class-Conditional Generation

| Dataset | Method | cFID ↓ | sFID ↓ | Prec ↑ | Rec ↑ | JEPA-SCORE ↓ |
|--------|------|--------|--------|--------|-------|--------------|
| CelebA 64² | ADM | 12.11 | 6.35 | 0.85 | 0.57 | -221.67 |
| CelebA 64² | SGMS | 61.76 | 20.42 | 0.62 | 0.84 | -171.85 |
| CelebA 64² | **Ours** | **8.50** | **4.94** | **0.82** | 0.65 | **-300.79** |
| ImageNet 256² | ADM | 26.44 | 9.70 | 0.95 | 0.51 | -102.01 |
| ImageNet 256² | BnS | 32.01 | 10.61 | 0.92 | 0.56 | -125.77 |
| ImageNet 256² | **Ours** | **18.33** | **7.62** | 0.92 | **0.68** | **-241.62** |

### Main Results—Text-to-Image Generation

| Model | Method | CLIP ↑ | PickScore ↑ | ImageReward ↑ | JEPA-SCORE ↓ |
|------|------|--------|-------------|---------------|--------------|
| SDv1.5 | DDIM | 31.52 | 21.49 | 0.21 | -292.27 |
| SDv1.5 | MinorityPrompt | 31.56 | 21.32 | 0.24 | -322.33 |
| SDv1.5 | **Ours** | 31.46 | **21.50** | 0.22 | **-355.40** |
| SDXL-Lightning | DDIM | 31.57 | 22.68 | 0.73 | -283.04 |
| SDXL-Lightning | MinorityPrompt | 31.36 | 22.62 | 0.71 | -302.17 |
| SDXL-Lightning | **Ours** | 31.52 | 22.63 | 0.73 | **-337.88** |

### Ablation Study

| Config | CLIP ↑ | PickScore ↑ | JEPA-SCORE ↓ | Description |
|------|--------|-------------|--------------|------|
| $\tau = 1.0$ (No delay) | 31.26 | 21.33 | -356.22 | Severe quality drop |
| $\tau = 0.9$ | 31.31 | 21.42 | -356.72 | Slight improvement |
| $\tau = 0.8$ (Default) | 31.40 | 21.46 | -360.82 | Best balance of quality and rarity |
| $k = 3$ | 31.56 | 22.59 | -325.35 | Insufficient rank |
| $k = 9$ (Default) | 31.52 | 22.59 | -344.85 | Sufficiently effective |
| $k = 15$ | 31.53 | 22.58 | -335.28 | Diminishing marginal returns |

### Downstream Application—Classification Data Augmentation

| Training Data | Acc ↑ | F1 ↑ | Prec ↑ | Rec ↑ | Augment Qty |
|----------|-------|------|--------|-------|--------|
| CelebA trainset | 0.898 | 0.746 | 0.815 | 0.710 | — |
| + SGMS | 0.903 | 0.757 | 0.822 | 0.724 | 50K |
| + BnS | 0.902 | 0.755 | 0.819 | 0.723 | 50K |
| + **Ours** | **0.902** | **0.775** | **0.824** | **0.731** | **30K** |

## Highlights & Insights

- **Paradigm Shift**: Redefines minority sampling from "finding rarity under the generator's distribution" to "finding rarity under a world prior." This is conceptually more sound—generator-centric rarity might just be training set bias, whereas world-centric rarity reflects true semantics.
- **Equal Emphasis on Theory and Engineering**: Randomized SVD approximation has a strict error upper bound (Proposition 4.1), and the Envelope Theorem ensures gradient correctness, avoiding ad-hoc hacks.
- **Condition-agnostic Design**: The JEPA encoder does not need to know conditional information (text/class). It is naturally compatible with conditional generation via deferred guidance, making the design exceptionally elegant.
- **Data Efficiency**: In downstream classification, using only 30K augmented samples outperforms the 50K baseline, indicating that rare samples under the world prior are more informative.

## Limitations & Future Work

- Each guidance step requires calculating the Jacobian + Randomized SVD, introducing additional computational overhead. Amortization or more efficient approximations could be explored.
- The quality of the world prior depends on the training data and capability of the JEPA encoder. Changing the encoder will alter the definition of "rarity."
- Only encoders like DINOv2/MetaCLIP were explored; video models like V-JEPA or other modalities have not been verified.
- Reversing the guidance direction can generate high-density samples and reinforce bias, presenting dual-use risks.

## Related Work & Insights

- **Minority Sampling Series** (Sehwag et al., Um et al.): Evolution from classifier-guided → self-contained → guidance-free. This paper breaks the fundamental constraint of "defining minority only under the generator prior."
- **JEPA-SCORE** (Balestriero et al., 2025): Proves that JEPA representations implicitly encode data density; this paper upgrades it from a posterior ranking tool to an online sampling guidance signal.
- **DINOv2** (Oquab et al., 2023): ViT encoder trained on 142 million images, serving as a proxy for the world prior.
- **Insights**: This framework can be generalized to other scenarios requiring a "definition of what is rare," such as fairness, robustness testing, and creative content generation.

## Rating
- Novelty: 9/10 — Outstanding conceptual contribution with the paradigm shift from generator-centric to world-centric.
- Experimental Thoroughness: 8/10 — Covers unconditional/class-conditional/text-to-image/downstream applications with detailed ablations.
- Writing Quality: 9/10 — Clear concepts, rigorous theoretical derivation, and intuitive illustrations.
- Value: 8/10 — Opens a new direction for minority sampling, though computational overhead limits actual scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond Confidence: The Rhythms of Reasoning in Generative Models](../../ICLR2026/image_generation/beyond_confidence_the_rhythms_of_reasoning_in_generative_models.md)
- [\[ICML 2026\] Threshold-Guided Optimization for Visual Generative Models](threshold-guided_optimization_for_visual_generative_models.md)
- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[CVPR 2026\] Ani3DHuman: Photorealistic 3D Human Animation with Self-guided Stochastic Sampling](../../CVPR2026/image_generation/ani3dhuman_photorealistic_3d_human_animation_with_self-guided_stochastic_samplin.md)
- [\[ICLR 2026\] Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening](../../ICLR2026/image_generation/motion_prior_distillation_in_time_reversal_sampling_for_generative_inbetweening.md)

</div>

<!-- RELATED:END -->
