---
title: >-
  [Paper Note] 统一不同生成顺序的掩码扩散模型
description: >-
  [ICML 2026][Image Generation][Paper Note] This paper proposes a unified framework OeMDM and its learnable version LoMDM—unifying random masking, autoregressive, and block diffusion models under a single NELBO by explicitly modeling "velocity" (generation priority), enabling joint learning of generation order and the diffusion backbone from scratch.
tags:
  - ICML 2026
  - Image Generation
date: 2026-05-08
content_hash: e0d2149a87640f17
---
# Unified Masked Diffusion Models with Diverse Generation Orders

**Conference**: ICML 2026  
**arXiv**: [2602.02112](https://arxiv.org/abs/2602.02112)  
**Code**: To be confirmed  
**Area**: Diffusion Models / Text Generation / Language Modeling  
**Keywords**: Masked Diffusion Models, Generation Order, Velocity Field, Joint Learning

## TL;DR
This paper proposes a unified framework OeMDM and its learnable version LoMDM—unifying random masking, autoregressive, and block diffusion models under a single NELBO by explicitly modeling "velocity" (generation priority), enabling joint learning of generation order and the diffusion backbone from scratch.

## Background & Motivation

**Background**: Masked Diffusion Models (MDMs) are potential alternatives to Autoregressive Models (ARMs), but their generation quality heavily depends on the generation order.

**Limitations of Prior Work**: Existing solutions either use hard-coded orders (e.g., block L2R) or learn order policies for pre-trained MDMs—the latter requires extra computation and leads to sub-optimal solutions due to two-stage optimization.

**Key Challenge**: MDMs themselves are order-agnostic; unified noise scheduling results in the same denoising rate for all positions, leading to completely random generation orders. Meanwhile, ordered methods operate independently without a unified perspective.

**Goal**: (1) Unify MDM, ARM, and block diffusion under a single framework; (2) Jointly learn the generation order and the diffusion model from scratch.

**Key Insight**: Explicitly formulate the generation rate implicit in the NELBO as a "velocity" function, designing position-dependent adaptive noise scheduling.

**Core Idea**: Replace the global fixed scheduler with a position-dependent scheduler, allowing the diffusion process to "know" which positions to generate first, optimizing both the backbone and generation strategy via a velocity matching loss.

## Method

### Overall Architecture
Standard masked diffusion models apply the same noise schedule to all positions, making the diffusion process entirely unaware of which positions should be generated first. This paper's approach is to explicitly extract "how fast each position is restored," which is originally implicit in the NELBO, as a position-dependent "velocity" function: the scheduler determines the noise amount for each position individually, and positions with higher velocity are restored earlier. The fixed version, OeMDM, provides a unified NELBO that subsumes random masking, ARM, and block diffusion as special cases under different schedules. The learnable version, LoMDM, further parameterizes both forward and backward velocities with neural networks to jointly learn the generation order and diffusion backbone from scratch.

### Key Designs

**1. Velocity Field Explication: Making the generation order an optimizable quantity**

The root of MDM's order-agnosticism is that global fixed scheduling causes identical denoising rates across all positions. This work introduces a free-form scheduler $\alpha_F: I \times [0,1] \to [0,1]^L$ into the forward process, allowing different positions to receive different amounts of noise, and extracts the "velocity" $A(u,t) = -\partial_t\alpha_F(u,t) \oslash (1-\alpha_F(u,t))$—representing the denoising rate of position $i$ at time $t$. Consequently, the reverse posterior and denoising process can be unified as $\text{Cat}\big((1-A^{(i)}dt)\,m + A^{(i)}dt\cdot x^{(i)}\big)$. Positions with higher velocity have a higher probability of flipping from the mask $m$ to the ground truth $x^{(i)}$ at each step, thus the generation order is directly determined by the velocity values. Representing the order as a continuous function rather than a discrete policy allows the training signal to focus on high-priority tokens and enables a principled NELBO decomposition.

**2. Generalized NELBO Decomposition: Tying training and inference via velocity matching loss**

The objective function of OeMDM can be cleanly split into two terms: $L_{\text{main}} + L_{\text{velocity}}$. $L_{\text{main}}$ is a velocity-weighted reconstruction loss, forcing the backbone to learn predictions at each position with its priority as the weight. $L_{\text{velocity}} = A(i)\big(\log A(i) - \log \hat{A}(i)\big) - \big(A(i) - \hat{A}(i)\big) \ge 0$ measures the discrepancy between the forward velocity $A_\phi$ and the backward velocity $\hat{A}_\psi$. This term is a convex form and is non-negative, reaching zero only when the two velocities are aligned. Therefore, it forces the backward order (used during inference) to approximate the learned forward order. This specifically addresses the sub-optimality inherent in two-stage methods like GenMD4—the training order and inference order are naturally consistent, eliminating the need to learn a separate order policy post-hoc.

**3. Parameter-Efficient Joint Learning: Reusing backbone features to stabilize optimization**

Adding a large separate network for the velocity field can lead to unstable joint optimization. Instead, this paper reuses the Transformer feature extractor $f$ of the diffusion backbone $\theta$, adding only a light set of MLP+Transformer layers to parameterize the forward $\alpha_\phi(x,t)$ and backward $\hat{\alpha}_\psi(z_t,t)$. The specific form is $\alpha^{(i)}_\phi(x,t) := 1 - t^{\,c_1 + c_2\cdot[\text{NormSig}(g_\phi(f(x)))]_i}$, where the normalized Sigmoid output modulates the relative priority of each position, while $c_1$ and $c_2$ control the overall rate range. Simultaneously, a stop-gradient is used to decouple the gradient paths of the scheduler and the backbone, allowing the scheduler to optimize independently without interfering with the backbone's main task, thereby learning the order end-to-end with almost no additional parameters.

## Key Experimental Results

### Main Results

| Dataset | MDLM | BD3LM(L'=4) | GenMD4 | LoMDM | Gain |
|--------|------|------------|--------|--------|------|
| LM1B   | 27.0 | -          | 26.9   | 25.4   | -1.5 vs MDLM |
| LM1B+packed | 31.8 | 28.2 | 30.0 | 27.2 | -4.6 vs MDLM |
| OWT    | 23.2 | 20.7       | 21.8   | 20.4   | -2.8 vs MDLM |

### Zero-shot Generalization

| Dataset | MDLM | BD3LM | LoMDM | vs MDLM |
|---------|------|--------|--------|---------|
| PTB     | 95.26 | 96.81 | 80.40  | ↓14.86 |
| WikiText | 32.83 | 31.31 | 27.82 | ↓5.01 |
| Lambada | 47.52 | 50.03 | 36.32  | ↓11.20 |

### Key Findings
- LoMDM outperforms MDLM on 7/7 zero-shot datasets and leads all diffusion models on 6/7; it beats the Autoregressive Transformer on 4/7 datasets.
- Generation PPL (NFE=256): LoMDM 73.98 vs MDLM 79.43.
- Ablation (disabling inference scheduling $c_2=0$): Generation PPL increases from 48.29 to 59.34.

## Highlights & Insights
- **Unified Perspective Breakthrough**: Treats ARM, MDM, and block diffusion as special cases of OeMDM under different schedules, explained within a single NELBO framework.
- **Velocity Matching Design**: The convex form of $L_{\text{velocity}} \geq 0$ ensures optimization stability while enforcing consistency between training and inference.
- **End-to-End Joint Learning**: Compared to GenMD4's frozen backbone and learned schedule, LoMDM optimizes both simultaneously, providing the backbone with order-aware training signals.

## Limitations & Future Work
- Training Cost: Requires 3x forward passes per iteration, resulting in slightly lower absolute throughput.
- Scheduler Design: The parameterization of $\alpha_\phi(x,t)$ still relies on a manually designed functional form.
- Scalability: Experiments are limited to LM1B/OWT scales; performance on large-scale models needs further validation.

## Related Work & Insights
- **vs MDLM/SEDD**: Both use random masking without order optimization; LoMDM achieves context-aware generation paths through an explicit scheduler.
- **vs BD3LM**: Uses a hard-coded L2R block structure; LoMDM learns a more flexible order.
- **vs GenMD4**: Both learn schedulers, but GenMD4 is two-stage; LoMDM optimizes end-to-end from scratch.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  First to unify discrete diffusion and autoregression using velocity fields.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Covers 3 datasets + 3 evaluation metrics + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐  Derivations are complete and clear.
- Value: ⭐⭐⭐⭐⭐  Provides a principled framework for discrete diffusion text generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ViewMask-1-to-3: Multi-View Consistent Image Generation via Multimodal Discrete Diffusion Models](viewmask-1-to-3_multi-view_consistent_image_generation_via_multimodal_discrete_d.md)
- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)
- [\[ICML 2026\] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)
- [\[ICML 2026\] Let EEG Models Learn EEG](let_eeg_models_learn_eeg.md)
- [\[ICML 2026\] From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection](from_talking_to_singing_a_new_challenge_for_audio-visual_deepfake_detection.md)

</div>

<!-- RELATED:END -->
