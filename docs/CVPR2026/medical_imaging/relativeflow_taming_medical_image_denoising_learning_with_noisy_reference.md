---
title: >-
  [Paper Note] RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference
description: >-
  [CVPR 2026][Medical Imaging][medical image denoising] This paper proposes RelativeFlow, a flow matching-based framework that decomposes the absolute noise-to-clean mapping into relative noisier-to-noisy mappings. By inco…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "medical image denoising"
  - "flow matching"
  - "noisy reference"
  - "CT denoising"
  - "MR denoising"
date: 2026-05-08
content_hash: 7ce49b954a1b365c
---

# RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference

**Conference**: CVPR 2026
**arXiv**: [2604.15459](https://arxiv.org/abs/2604.15459)
**Code**: [github.com/Deliver0/RelativeFlow](https://github.com/Deliver0/RelativeFlow)
**Area**: Medical Imaging
**Keywords**: medical image denoising, flow matching, noisy reference, CT denoising, MR denoising

## TL;DR

This paper proposes RelativeFlow, a flow matching-based framework that decomposes the absolute noise-to-clean mapping into relative noisier-to-noisy mappings. By incorporating a consistent transport constraint and a simulation-based velocity field, RelativeFlow learns a unified denoising flow from heterogeneous noisy references, overcoming the reference bias limitation.

## Background & Motivation

Medical image denoising (MID) lacks access to absolutely clean images for supervision — only relatively high-quality references acquired under varying protocols and scanner configurations are available, with quality varying heterogeneously across categories. Three existing paradigms share notable limitations: (1) SimSDL naively treats noisy references as clean targets, leading to suboptimal convergence or reference-biased learning; (2) SSL relies on the independent noise assumption, which is difficult to satisfy in medical imaging; (3) SimSGL similarly treats noisy references as generative targets without correction. The core problem is how to learn a unified high-quality denoising mapping from heterogeneous noisy references.

## Method

### Overall Architecture

RelativeFlow decomposes the absolute denoising flow into a composition of multiple relative denoising flows. For each noisy reference $x_t$, a degradation operator generates a noisier sample $x_{t-\Delta t}$, forming a local relative denoising step. Two key components ensure the consistency and learnability of the relative flows.

### Key Designs

1. **Consistent Transport (CoT) Displacement Mapping**: A probability path is defined via linear interpolation in exponential time-space as $p_t = \lambda p_{t_i} + (1-\lambda)p_{t_j}$, with weight $\lambda = \frac{e^{-t}-e^{-t_j}}{e^{-t_i}-e^{-t_j}}$. This guarantees the transitivity of nested interpolation — the relative flow from any quality level $t_i$ to $t_j$ constitutes a component of the absolute flow $\psi_{0 \to +\infty}$ and can be sequentially composed into it. Two properties are mathematically proven: (i) relative flows are components of the absolute flow, and (ii) consecutive relative flows can be progressively composed into the absolute flow.

2. **Simulation-based Velocity Field (SVF)**: Modality-specific degradation operators $D_{\Delta t}$ (Poisson-Gaussian noise for CT, Rician noise for MR) are employed to simulate noisier samples from noisy references, yielding the velocity field training target $u = \frac{x_t - D_{\Delta t}(x_t)}{e^{\Delta t} - 1}$. The training loss is the L2 distance between the neural network prediction and this target velocity.

3. **Progressive Step-size Curriculum**: During training, the degradation step-size range $[\Delta t_{min}, \Delta t_{max}]$ is progressively expanded (multiplied/divided by a factor $\alpha$ each epoch), enabling the model to gradually learn velocity fields from small to large degradations and cover the full quality spectrum. Multi-step Euler integration is used at inference.

### Loss & Training

Training loss: $\mathcal{L}_{RF} = \mathbb{E}_{x_t, \Delta t}\left[\left\|\mathcal{N}_\theta(D_{\Delta t}(x_t), \Delta t) - \frac{x_t - D_{\Delta t}(x_t)}{e^{\Delta t} - 1}\right\|_2^2\right]$. A progressive curriculum learning strategy is adopted throughout training.

## Key Experimental Results

### Main Results

RelativeFlow is compared against 10 methods (NID + MID) on CT and MR denoising tasks:

| Task | Metric | Prev. SOTA | Ours |
|------|--------|------------|------|
| CT Denoising | PSNR/SSIM | Runner-up | **Best** |
| MR Denoising | PSNR/SSIM | Runner-up | **Best** |

RelativeFlow achieves state-of-the-art performance across all four metrics: PSNR, SSIM, RMSE, and LPIPS.

### Ablation Study

- The CoT consistency constraint is the key component for overcoming reference bias.
- Modality-specific degradation modeling in SVF outperforms generic noise models.
- Progressive step-size curriculum outperforms fixed step-size training.

### Key Findings

- RelativeFlow unifies images of varying quality levels into consistently high-quality outputs.
- The mathematical properties of CoT (componentality and composability) provide theoretical guarantees for unified denoising.
- Modality-specific degradation operators enable the framework to generalize across different medical imaging modalities.

## Highlights & Insights

- This work is the first to formally characterize the noisy reference problem and systematically address it via flow matching.
- The mathematical analysis of CoT (proofs of componentality and composability) demonstrates exceptional theoretical depth.
- Validation across both CT and MR modalities demonstrates the generality of the proposed framework.

## Limitations & Future Work

- Designing degradation operators requires domain knowledge (Poisson-Gaussian vs. Rician noise models).
- Multi-step integration at inference is slower than discriminative approaches.
- The convergence quality of the theoretical $t \to +\infty$ limit in practice warrants further empirical validation.

## Related Work & Insights

- The idea of composing relative flows offers inspiration for other learning scenarios lacking clean labels.
- The CoT mathematical framework is generalizable to other progressive quality enhancement tasks.
- The combination of modality-specific degradation modeling with general-purpose flow matching presents a paradigm worth broader adoption.

## Rating

8/10 — Combining theoretical rigor with strong empirical performance, this work represents a significant methodological contribution to the medical image denoising literature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [\[CVPR 2026\] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graphbased_learning_of_inter_and_intraview_de.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)

</div>

<!-- RELATED:END -->
