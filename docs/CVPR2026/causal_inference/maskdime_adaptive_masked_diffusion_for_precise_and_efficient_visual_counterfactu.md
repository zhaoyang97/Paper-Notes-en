---
title: >-
  [Paper Note] MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations
description: >-
  [CVPR 2026][Causal Inference][Visual counterfactual explanations] This paper proposes MaskDiME, a training-free diffusion framework that transforms global classifier guidance into decision-driven local editing via an adaptive dual-mask mechanism, enabling precise and efficient visual counterfactual explanations. MaskDiME achieves inference speeds more than 30× faster than DiME while requiring only one-tenth the GPU memory of ACE/RCSB.
tags:
  - CVPR 2026
  - Causal Inference
  - Visual counterfactual explanations
  - diffusion models
  - adaptive masking
  - explainable AI
  - classifier guidance
date: 2026-05-08
content_hash: 985f2aeef512fce0
---

# MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations

**Conference**: CVPR 2026
**arXiv**: [2602.18792](https://arxiv.org/abs/2602.18792)
**Code**: Coming soon
**Area**: Causal Inference
**Keywords**: Visual counterfactual explanations, diffusion models, adaptive masking, explainable AI, classifier guidance

## TL;DR

This paper proposes MaskDiME, a training-free diffusion framework that transforms global classifier guidance into decision-driven local editing via an adaptive dual-mask mechanism, enabling precise and efficient visual counterfactual explanations. MaskDiME achieves inference speeds more than 30× faster than DiME while requiring only one-tenth the GPU memory of ACE/RCSB.

## Background & Motivation

Visual counterfactual explanations (VCE) address the question: "What must change in an image for the model to reach a different decision?"—a more intuitive and causally grounded approach than attribution methods such as heatmaps. Diffusion models have become the dominant paradigm for VCE owing to their superior generative quality, yet existing methods face two core challenges:

**Challenge 1: High computational cost.** DiME pioneered diffusion-based counterfactual generation but relies on nested denoising and step-wise backpropagation, resulting in $O(T^2)$ complexity with slow inference and large memory footprint. Multi-stage methods such as ACE and RCSB similarly exhibit high GPU memory consumption.

**Challenge 2: Poor spatial precision.** Most methods apply global classifier guidance or implicit conditioning, causing the guidance signal to propagate indiscriminately across the entire image, making it difficult to identify which regions explain the decision. FastDiME improves speed but uses pixel-difference masks for coarse localization; the fixed masks in ACE/RCSB cannot adapt to the dynamic semantic shifts that occur during reverse diffusion (e.g., changes in facial expression).

**Core Insight**: Enabling the model to adaptively focus on decision-relevant regions at every step of the reverse diffusion process is the key to achieving precise, semantically consistent counterfactual explanations.

## Method

### Overall Architecture

MaskDiME builds on DiME's gradient-guided diffusion paradigm with three key modifications: (1) replacing nested denoising with single-step Tweedie estimation, reducing complexity from $O(T^2)$ to $O(T)$; (2) introducing an adaptive dual-mask mechanism to constrain per-step update regions; and (3) introducing a gradient scaling factor $s$ to compensate for quality loss from single-step estimation.

Given a query image $x$, forward diffusion is applied to a predefined timestep $\tau$ (default $\tau=60$, total steps $T=200$):

$$\tilde{z}_t = \sqrt{\bar{\alpha}_t}\, x + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

At each reverse diffusion step $t$, mask-constrained denoising updates are applied:

$$z_{t-1} = M_t^z \odot \mathcal{N}\!\big(\mu_\theta(z_t) - \Sigma_\theta(z_t) \nabla z_t,\, \Sigma_\theta(z_t)\big) + (1 - M_t^z) \odot \tilde{z}_{t-1}$$

Regions inside the mask undergo gradient-guided denoising, while regions outside retain the original diffusion trajectory, enabling spatially controlled counterfactual generation.

### Gradient Guidance Design

The joint loss function follows DiME, comprising three components:

$$L(x_t; y, x) = \lambda_c L_{\text{class}}(C(y|x_t)) + \lambda_p L_{\text{perc}}(x_t, x) + \lambda_l L_{L1}(x_t, x)$$

- **Classification loss** $L_{\text{class}}$: drives the generated image toward the target class semantics.
- **Perceptual loss** $L_{\text{perc}}$: preserves structural and appearance similarity to the original image.
- **L1 loss** $L_{L1}$: provides pixel-level supervision to stabilize low-level differences and reduce artifacts.

Gradients are propagated to the noise space via the reparameterization trick, with a scaling factor $s$ controlling overall guidance strength:

$$\nabla z_t = s \cdot \frac{1}{\sqrt{\bar{\alpha}_t}} \nabla_{x_t} L(x_t; y, x)$$

Hyperparameter settings: $\lambda_c \in \{8,10,15\}$ (following DiME's iterative search), $\lambda_p=30$, $\lambda_l=0.05$; $s$ is tuned per dataset (CelebA: 8, CelebA-HQ: 10, BDD: 14, ImageNet: 6.5).

### Adaptive Dual-Mask Mechanism (Core Innovation)

At each diffusion step $t$, two binary masks $M_t^x \subseteq M_t^z$ are constructed from classifier gradients:

**Step 1: Extract spatial gradient maps.** The absolute value of the gradient of the classification loss with respect to $x_t$ is taken and averaged across channels:

$$G_t = \left|\nabla_{z_t}^{\text{class}}\right|_{\text{avg}} \in \mathbb{R}^{1 \times H \times W}$$

Unlike RCSB, which uses Integrated Gradients requiring dozens of forward/backward passes, this approach requires only a single step, enabling real-time mask generation at each sampling step.

**Step 2: Construct the noise-level mask $M_t^z$.** The top-$k\%$ gradient regions in $G_t$ are set to 1 and the remainder to 0. Here $k=0.05$ for the smile attribute and $k=0.1$ for age and other datasets.

**Step 3: Construct the clean-level mask $M_t^x$.** From $M_t^z$, the top-$\rho k\%$ strongest gradient regions are further retained. Here $\rho=0.25$ for CelebA-HQ and $\rho=0.5$ for other datasets.

**Why are two masks needed?** As denoising progresses, $x_t$ increasingly resembles the counterfactual class, causing classification gradients to diminish while perceptual and L1 loss gradients dominate. The more compact $M_t^x$ prevents non-decision regions in the clean image estimate from being inappropriately modified, ensuring edits remain strictly confined to decision-relevant areas.

Both masks are processed with morphological dilation using a $5 \times 5$ kernel to enhance spatial coherence.

### Single-Step Clean Image Estimation

The current clean image is estimated in a single step via the Tweedie formula, avoiding DiME's recursive reconstruction:

$$\hat{x}_0^{(t-1)} = \frac{z_{t-1} - \sqrt{1-\bar{\alpha}_{t-1}}\, \epsilon_\theta(z_{t-1})}{\sqrt{\bar{\alpha}_{t-1}}}$$

The clean-level mask is then applied to blend the estimate, keeping non-edited regions strictly unchanged:

$$x_{t-1} = M_t^x \odot \hat{x}_0^{(t-1)} + (1-M_t^x) \odot x$$

### Loss & Training

MaskDiME is fully training-free: it directly reuses DiME's unconditional DDPM weights and the target classifier weights without any additional training or fine-tuning. The only newly introduced parameters are $s$, $k$, and $\rho$, and $k$ and $\rho$ generalize well across datasets.

## Key Experimental Results

### Main Results: CelebA Smile Attribute (128×128)

| Method | FID↓ | sFID↓ | FVA↑ | FS↑ | MNAC↓ | CD↓ | COUT↑ | FR↑ |
|--------|------|-------|------|-----|-------|-----|-------|-----|
| DiME | 3.17 | 4.89 | 98.3 | 0.73 | 3.72 | 2.30 | 0.53 | 97.2 |
| ACE $\ell_1$ | 1.27 | 3.97 | 99.9 | 0.87 | 2.94 | 1.73 | 0.78 | 97.6 |
| FastDiME-2+ | 3.24 | 5.23 | 99.9 | 0.79 | 2.91 | 2.02 | 0.41 | 98.9 |
| RCSB | 2.98 | 4.79 | 100.0 | 0.91 | 2.24 | 2.78 | 0.87 | 99.8 |
| **MaskDiME** | **0.71** | **3.29** | **100.0** | **0.91** | 2.78 | 2.41 | **0.87** | **100.0** |

MaskDiME achieves the lowest FID (0.71) and sFID (3.29), a perfect flip rate (FR=100%), and optimal or near-optimal performance on FVA/FS/COUT.

### Ablation Study: CelebA Smile

| Configuration | FID↓ | FS↑ | MNAC↓ | CD↓ | COUT↑ | FR↑ |
|---------------|------|-----|-------|-----|-------|-----|
| DiME (baseline) | 3.17 | 0.73 | 3.72 | 2.30 | 0.53 | 97.2 |
| $s$=1 & no mask | 95.76 | 0.63 | 6.15 | 2.43 | -0.16 | 55.2 |
| $s$=8 & no mask | 15.94 | 0.77 | 5.71 | 4.20 | 0.96 | 100.0 |
| Fixed mask | 4.21 | 0.86 | 2.98 | 2.03 | 0.70 | 99.7 |
| $s$=8 & adaptive mask ($\rho$=1) | 0.71 | 0.90 | 2.66 | 2.25 | 0.81 | 100.0 |
| **MaskDiME** ($\rho$=0.5) | **0.71** | **0.91** | 2.78 | 2.41 | **0.87** | **100.0** |

The ablation clearly reveals each component's contribution: single-step estimation alone ($s$=1, no mask) causes FID to spike to 95.76 with FR of only 55.2%; increasing $s$ restores FR but introduces artifacts (FID=15.94); introducing adaptive masking reduces FID to 0.71; and the dual-mask setting with $\rho=0.5$ further improves COUT from 0.81 to 0.87.

### Key Findings

- **30× speedup**: MaskDiME is 30× faster than DiME and 2.5× faster than FastDiME, with approximately one-tenth the GPU memory of ACE/RCSB.
- **Strong cross-domain generalization**: Optimal or near-optimal results are achieved across five datasets spanning faces (CelebA/CelebA-HQ), autonomous driving (BDD100K/BDD-OIA), and general classification (ImageNet).
- **FR reaches 100%** on both BDD100K and BDD-OIA, with COUT of 0.85/0.80 and $S^3$ of 0.99.
- Heatmap visualizations demonstrate that the adaptive mask progressively focuses on decision-relevant regions—the mouth (smile) and traffic lights (driving)—over the course of diffusion.
- Gradient scaling $s$ and the masking mechanism are complementary: the former controls guidance strength while the latter ensures spatial focus and semantic consistency.
- Diversity evaluation: $\sigma_L=0.0395$, higher than ACE $\ell_1$ (0.0174) but lower than DiME (0.2139)—DiME's higher diversity stems from unconstrained background modifications rather than genuine semantic diversity.

## Highlights & Insights

- **Elegant dual-mask design**: The noise-level mask defines the denoising region, while the more compact clean-level mask counteracts the decay of classification gradients, addressing the distinct requirements of noise space and clean image space separately.
- **Highly practical**: Training-free operation, linear complexity, and low memory footprint make VCE practically deployable.
- **Rigorous visualization study**: Heatmap comparisons across pixel-difference masks, fixed masks, and adaptive masks intuitively illustrate the behavioral differences among masking strategies.
- **Insightful ablation design**: The sequential addition of $s$ → mask → $\rho$ cleanly isolates the independent contribution of each component.

## Limitations & Future Work

- Masks are derived from single-step gradients rather than Integrated Gradients; gradient noise in multi-class ImageNet scenarios leads to imprecise localization, resulting in higher FID/sFID than RCSB.
- The method supports only pixel-space DDPM and has not been extended to latent diffusion models, limiting applicability in high-resolution settings.
- The absence of ground-truth annotations for counterfactual explanations makes rigorous causal validation of results difficult.

## Rating

- Novelty: ⭐⭐⭐⭐ The adaptive dual-mask mechanism is elegantly designed, offering a novel perspective grounded in the interaction between gradient dynamics and diffusion processes.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five datasets spanning three visual domains, with complete ablation and visualization analyses.
- Writing Quality: ⭐⭐⭐⭐ Outstanding visualizations (heatmaps, efficiency scatter plots) and a clear, well-structured logical progression.
- Value: ⭐⭐⭐⭐ A 30× speedup combined with one-tenth the memory makes VCE practically deployable, representing a significant contribution to the XAI field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Retrieving Counterfactuals Improves Visual In-Context Learning](retrieving_counterfactuals_improves_visual_in-context_learning.md)
- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](../../ICLR2026/causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[CVPR 2026\] Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](fighting_hallucinations_with_counterfactuals_diffusion-guided_perturbations_for_.md)
- [\[AAAI 2026\] KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education](../../AAAI2026/causal_inference/ktcf_actionable_recourse_in_knowledge_tracing_via_counterfactual_explanations_fo.md)
- [\[ICCV 2025\] A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets](../../ICCV2025/causal_inference/a_visual_leap_in_clip_compositionality_reasoning_through_gen.md)

</div>

<!-- RELATED:END -->
