---
title: >-
  [Paper Note] Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design
description: >-
  [ICCV 2025][Image Generation][Machine Unlearning] This paper proposes Water4MU, a framework that integrates digital watermarking with machine unlearning (MU) via bi-level optimization (BLO). The upper level optimizes the…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Machine Unlearning"
  - "Digital Watermarking"
  - "Bi-Level Optimization"
  - "Implicit Gradients"
  - "Diffusion Model Concept Erasure"
date: 2026-05-08
content_hash: 9ec86f6db0ff0acd
---

# Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design

**Conference**: ICCV 2025
**arXiv**: [2508.10065](https://arxiv.org/abs/2508.10065)
**Code**: None
**Area**: Image Generation
**Keywords**: Machine Unlearning, Digital Watermarking, Bi-Level Optimization, Implicit Gradients, Diffusion Model Concept Erasure

## TL;DR

This paper proposes Water4MU, a framework that integrates digital watermarking with machine unlearning (MU) via bi-level optimization (BLO). The upper level optimizes the watermark network to facilitate unlearning, while the lower level performs the unlearning optimization, thereby substantially improving unlearning effectiveness without significantly compromising model utility.

## Background & Motivation

Machine Unlearning (MU) aims to remove the influence of specific data from trained models to comply with privacy regulations such as the "right to be forgotten." Existing MU methods predominantly focus on model weight-level adjustments (e.g., gradient ascent GA, fine-tuning FT), with virtually no exploration of how data-level modifications (e.g., watermarking) affect the unlearning process.

The core motivation stems from two key observations:

**Orthogonality between watermarking and unlearning**: Experiments on the exact unlearning method Retrain show that whether training on watermarked data and evaluating on original data ($\mathcal{S}_2$), or training on original data and evaluating on watermarked data ($\mathcal{S}_1$), the unlearning performance differs minimally from the baseline scenario ($\mathcal{S}_0$). This indicates that watermarking does not undermine unlearning.

**Watermarks can be actively designed to promote unlearning**: Since watermarking does not interfere with unlearning, it motivates the converse question: can the watermarking mechanism be exploited to make unlearning more efficient by optimizing watermark network parameters and messages? This is the starting point of Water4MU.

Furthermore, existing MU methods perform poorly on "challenging forgets" (data subsets that are difficult to unlearn), where Water4MU demonstrates more pronounced advantages.

## Method

### Overall Architecture

Water4MU adopts a Bi-Level Optimization (BLO) framework. The core idea treats the watermarking process as the "leader" (upper level) and the unlearning process as the "follower" (lower level), optimizing watermark encoder parameters $\boldsymbol{\psi}$ to make lower-level unlearning more efficient.

Overall pipeline:
1. **Upper level**: Optimizes watermark encoder parameters $\boldsymbol{\psi}$ and decoder parameters $\boldsymbol{\phi}$ so that the unlearned model performs well on original (non-watermarked) data, while preserving the watermark network's own functionality.
2. **Lower level**: Executes standard unlearning optimization on the watermarked forget set $\hat{\mathcal{D}_f}$ and retain set $\hat{\mathcal{D}_r}$.

### Key Designs

1. **Lower-level unlearning optimization**: Given the watermarked dataset produced by encoder $f_{\boldsymbol{\psi}}$, the lower-level objective adopts GradDiff (gradient difference):

$$\boldsymbol{\theta}_u(\boldsymbol{\psi}) = \arg\min_{\boldsymbol{\theta}} -\ell_{CE}(\boldsymbol{\theta}; \hat{\mathcal{D}_f}) + \ell_{CE}(\boldsymbol{\theta}; \hat{\mathcal{D}_r})$$

where $\hat{\mathcal{D}_f} = \{f_{\boldsymbol{\psi}}(\mathbf{x}, \mathbf{m})\}_{\mathbf{x} \in \mathcal{D}_f}$. Unlearning is achieved via gradient ascent (negative cross-entropy) on the forget set and standard training on the retain set. Crucially, the unlearned model $\boldsymbol{\theta}_u$ is a function of the watermark parameters $\boldsymbol{\psi}$.

2. **Upper-level watermark optimization**: The upper-level objective comprises two components — (a) validating the unlearned model's MU performance on original non-watermarked data, and (b) preserving the watermark network's encoding and decoding capability:

$$\hat{\mathcal{L}}(\boldsymbol{\psi}, \boldsymbol{\phi}, \boldsymbol{\theta}_u(\boldsymbol{\psi})) = \mathcal{L}_{mu}(\boldsymbol{\theta}_u(\boldsymbol{\psi}); \mathcal{D}_f, \mathcal{D}_r) + \mathcal{L}_{wm}(\boldsymbol{\psi}, \boldsymbol{\phi}; \mathbf{m}, \mathcal{D}_f \cup \mathcal{D}_r)$$

This ensures that the watermark serves both unlearning and data provenance functions.

3. **Implicit gradient computation**: A key challenge in BLO is that upper-level gradients depend on the lower-level optimal solution. The paper applies the implicit function theorem and approximates the Hessian as a diagonal matrix $\nabla^2_{\boldsymbol{\theta}\boldsymbol{\theta}} \ell_{mu} \approx \lambda \mathbf{I}$, simplifying the upper-level gradient to:

$$\frac{d\hat{\mathcal{L}}}{d\boldsymbol{\psi}} = \nabla_{\boldsymbol{\psi}} \hat{\mathcal{L}} - \frac{\partial}{\partial \boldsymbol{\psi}}[\nabla_{\boldsymbol{\theta}} \ell_{mu}^\top \nabla_{\boldsymbol{\theta}} \hat{\mathcal{L}}]$$

This enables the entire BLO to be optimized using only first-order derivatives, avoiding the computational overhead of higher-order differentiation.

4. **Watermark message selection**: With the watermark network fixed, unlearning can be further enhanced by optimizing the watermark message $\mathbf{m} \in \{0,1\}^L$, following a similar formulation with $\mathbf{m}$ replacing $\boldsymbol{\psi}$ as the upper-level variable.

5. **Extension to image generation**: Water4MU extends to prompt-level concept erasure in diffusion models, performing unlearning optimization on watermarked data corresponding to prompts for specific concepts to be forgotten.

### Loss & Training

- **Lower level**: GradDiff objective, $\lambda_f = \lambda_r = 1$; negative cross-entropy on the forget set, standard cross-entropy on the retain set.
- **Upper level**: Unlearning validation loss + watermark network training loss.
- **Watermark backbone**: HiDDeN framework, including image reconstruction loss $\ell_{rec}$ and message decoding loss $\ell_{dec}$.
- Upper-level learning rate $10^{-4}$ (10 epochs); lower-level learning rate $10^{-2}$ (3 epochs); Hessian parameter $\lambda = 10^{-2}$.

## Key Experimental Results

### Main Results

Comparison of multiple MU methods with and without Water4MU on CIFAR-10/ResNet-18 (10% random data forgetting):

| Method | UA↑ (Original→Water4MU) | MIA↑ (Original→Water4MU) | RA↑ (Original→Water4MU) |
|--------|--------------------------|--------------------------|--------------------------|
| Retrain | 6.78→10.01 (+3.23) | 16.06→19.33 (+3.27) | 100.00→99.93 (-0.07) |
| GA | 0.80→1.92 (+1.12) | 1.89→5.67 (+3.78) | 99.42→99.18 (-0.24) |
| FT | 1.85→4.93 (+3.08) | 5.60→8.26 (+2.66) | 99.66→98.75 (-0.91) |
| Sparse | 6.11→7.50 (+1.39) | 13.08→14.70 (+1.62) | 97.76→97.22 (-0.54) |
| IU | 0.64→2.62 (+1.98) | 1.53→3.67 (+2.14) | 99.43→98.98 (-0.45) |

Gains are more pronounced in the class forgetting setting: FT's UA improves from 37.29 to 53.25 (+15.96), and MIA from 55.96 to 69.24 (+13.28).

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| CIFAR-100 random forgetting | UA gain 2.13~4.19, MIA gain 0.13~7.08, RA drop ≤0.75 |
| SVHN random forgetting | UA gain 1.30~2.65, MIA gain 1.73~6.34 |
| ImageNet class forgetting | UA gain 2.12~8.19, MIA gain 0.10~3.21 |
| Watermark message selection | Further improves MU performance with fixed watermark network |
| Challenging forgets | Water4MU shows particularly pronounced advantages on hard-to-unlearn samples |

### Key Findings

- Water4MU consistently improves unlearning performance (UA and MIA metrics) across all MU methods, with only marginal degradation in model utility (RA/TA).
- Improvements are larger in the class forgetting setting, with FT benefiting the most.
- The advantage is most pronounced in "challenging forgets" and other difficult unlearning scenarios.
- The extension to prompt-level forgetting in diffusion models is likewise effective.

## Highlights & Insights

- **Cross-domain innovation**: First work to formally connect digital watermarking and machine unlearning, establishing a principled relationship between the two.
- **Plug-and-play**: Water4MU is compatible with any existing MU method without modifying the unlearning algorithm itself.
- **Theoretical elegance**: The BLO + implicit gradient methodology rests on solid optimization-theoretic foundations while remaining implementation-friendly (requiring only first-order derivatives).
- **Practical significance**: Watermarking simultaneously serves data provenance and unlearning enhancement, achieving both objectives in a unified framework.

## Limitations & Future Work

- Requires a pre-trained watermark network (HiDDeN), introducing additional computational overhead.
- The diagonal Hessian approximation may be insufficiently accurate in certain settings.
- Scalability to large-scale models (e.g., LLMs) remains to be verified.
- The marginal drop in RA/TA may require careful consideration in safety-critical applications.
- The current scope is limited to the image domain; extension to text, audio, and other modalities has not yet been explored.

## Related Work & Insights

- **HiDDeN** [Zhu et al., 2018]: The foundational watermarking framework adopted in this paper.
- **GradDiff** [Liu et al., 2022]: The gradient difference unlearning method used at the lower level.
- **Visual Prompting** [Bahng et al., 2022]: Demonstrates that data-level modifications can influence model behavior.
- Insight: Data preprocessing (watermarking/perturbation/augmentation) can be actively designed to serve downstream task objectives.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First exploration of the interaction between watermarking and MU; BLO framework is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple datasets, methods, and scenarios (classification + generation).
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear; the derivation from observations to method is natural and well-structured.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for watermark-assisted unlearning with practical application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MUNBa: Machine Unlearning via Nash Bargaining](munba_machine_unlearning_via_nash_bargaining.md)
- [\[NeurIPS 2025\] WMCopier: Forging Invisible Image Watermarks on Arbitrary Images](../../NeurIPS2025/image_generation/wmcopier_forging_invisible_image_watermarks_on_arbitrary_images.md)
- [\[ICCV 2025\] Your Text Encoder Can Be An Object-Level Watermarking Controller](your_text_encoder_can_be_an_object-level_watermarking_controller.md)
- [\[NeurIPS 2025\] Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models](../../NeurIPS2025/image_generation/shallow_diffuse_robust_and_invisible_watermarking_through_low-dimensional_subspa.md)
- [\[ICCV 2025\] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)

</div>

<!-- RELATED:END -->
