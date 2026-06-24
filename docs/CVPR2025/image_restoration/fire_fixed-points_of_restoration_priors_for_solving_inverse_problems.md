---
title: >-
  [Paper Note] FiRe: Fixed-points of Restoration Priors for Solving Inverse Problems
description: >-
  [CVPR 2025][Image Restoration][Inverse Problems] This paper proposes the FiRe framework, which formulates explicit image priors using fixed-point theory by composing general-purpose image restoration models (e.g., deblurring, super-resolution, inpainting) with their training degradation operators. This generalizes traditional Plug-and-Play (PnP) beyond denoiser-only priors and supports the ensemble of multiple restoration models, significantly outperforming existing PnP and d…
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Inverse Problems"
  - "Plug-and-Play"
  - "Image Priors"
  - "Fixed-points"
  - "Ensemble of Restoration Models"
date: 2026-05-08
content_hash: e13de5c97ba5d852
---

# FiRe: Fixed-points of Restoration Priors for Solving Inverse Problems

**Conference**: CVPR 2025  
**arXiv**: [2411.18970](https://arxiv.org/abs/2411.18970)  
**Code**: [https://github.com/matthieutrs/fire](https://github.com/matthieutrs/fire)  
**Area**: Image Restoration  
**Keywords**: Inverse Problems, Plug-and-Play, Image Priors, Fixed-points, Ensemble of Restoration Models

## TL;DR

This paper proposes the FiRe framework, which formulates explicit image priors using fixed-point theory by composing general-purpose image restoration models (e.g., deblurring, super-resolution, inpainting) with their training degradation operators. This generalizes traditional Plug-and-Play (PnP) beyond denoiser-only priors and supports the ensemble of multiple restoration models, significantly outperforming existing PnP and diffusion-based approaches on various inverse problems.

## Background & Motivation

**Background**: Image inverse problems (e.g., deblurring, super-resolution, inpainting) are typically modeled as maximum a posteriori (MAP) estimation $\min_x \frac{1}{2}\|Ax-y\|_2^2 - \log p(x)$, where the core challenge lies in choosing an appropriate image prior $p(x)$. The most popular implicit prior framework is Plug-and-Play (PnP), which employs a pre-trained denoising network as a proxy for the prior. Denoisers are linked to the score function $\nabla_x \log p(x)$ via Tweedie's formula, which has been widely scaled to PnP and diffusion sampling.

**Limitations of Prior Work**: PnP methods suffer from two fundamental limitations. First, upon convergence during inference, the input to the denoiser is already a clean/smooth image rather than a noisy image, deviating from its training distribution and causing a distribution mismatch. Second, and more critically, clean images are not fixed points of denoising networks—repeatedly applying the denoiser leads to severe artifacts (as shown in Fig. 1(b)), with the PSNR dropping rapidly. In practice, heuristics such as decaying step sizes or early stopping are required to maintain reconstruction quality, leaving a gap between theory and practice.

**Key Challenge**: The PnP framework is strictly restricted to denoising priors, while the community already possesses numerous pre-trained general-purpose restoration models (e.g., super-resolution, deblurring, inpainting) that cannot be directly utilized as PnP priors. Simply replacing the denoiser with other restoration models leads to a severe drop in reconstruction quality, and Tweedie's formula cannot be generalized to generic restoration models.

**Goal**: (1) How to harness general-purpose image restoration models (not just denoisers) as implicit priors for solving image inverse problems? (2) How to theoretically derive their corresponding prior formulations? (3) How to enable the ensemble of multiple restoration models to achieve superior prior quality?

**Key Insight**: The authors observe that a clean image is a fixed point of the composition "degradation operator $\circ$ restoration model"—specifically, degrading a clean image and then restoring it should approximately return the original image. While directly iterating a restoration model $x_{k+1} = R(x_k)$ diverges, iterating the composed operator $x_{k+1} = R(H x_k + w_k)$ converges to meaningful natural images. This fixed-point property provides a solid theoretical foundation for constructing priors.

**Core Idea**: Compose the restoration model with its training degradation operator, and utilize the distance function to the fixed-point set of this composed operator as an explicit prior.

## Method

### Overall Architecture

The core of the FiRe framework is to define the distance function to the fixed-point set $C_n$ of multiple restoration model-degradation operator pairs $(R_n, D_n)$ as the prior, formulating the optimization problem: $x^* = \arg\min_x \lambda f(x) + \frac{\gamma}{2} \mathbb{E}_\xi[d_{C_\xi}^2(x)]$, where $f(x) = \frac{1}{2}\|Ax-y\|^2$ is the data fidelity term. From the gradient of the distance function, we obtain $\frac{1}{2}\nabla d_C^2(x) = x - R(D(x))$. This implies that each iteration simply requires degrading and then restoring the image, and subtracting the original signal yields the gradient of the prior.

### Key Designs

1. **Theoretical Derivation of the Fixed-Point Prior**:

    - **Function**: Provides an explicit prior formulation for general restoration models.
    - **Mechanism**: Assuming that the composition $T = R \circ D$ of a restoration model $R$ and a degradation operator $D$ can be viewed as a projection onto some closed prox-regular set $C$ (driven by training objectives to make restored images close to $C$), the relationship between the distance function and the projection yields $T(x) = x - \frac{1}{2}\nabla d_C^2(x)$. This directly links the composed operator to an explicit prior function $p(x) \propto \exp(-\frac{1}{2} d_C^2(x))$, thereby returning to the classical variational framework.
    - **Design Motivation**: Directly substituting the denoiser in PnP with a generic restoration model fails because clean images are not fixed points of restoration models. However, clean images are fixed points of $R \circ D$—this crucial observation makes the derivation of the prior possible.

2. **FiRe-HQS Algorithm**:

    - **Function**: An iterative optimization algorithm based on the fixed-point prior.
    - **Mechanism**: In each iteration, for each restoration-degradation pair $(R^n, D^n)$, the algorithm computes the residual $r_k^n = x_k - R^n(H_k^n x_k + w_k^n)$, aggregates them via weighted summation, and performs a proximal step for data fidelity: $x_{k+1} = \text{prox}_{\lambda f}(x_k - \sum_n \gamma_n r_k^n)$. The degradation operator can incorporate random noise, corresponding to a stochastic gradient estimation of the expected prior, converging the algorithm to stochastic proximal gradient descent.
    - **Design Motivation**: Each iteration aligns the inputs of the restoration models with their training distributions (by first degrading and then restoring), resolving the distribution mismatch inherent in traditional PnP. Furthermore, the algorithm is equipped with convergence guarantees.

3. **Ensemble of Multiple Restoration Models**:

    - **Function**: Combines multiple restoration models trained on different tasks to achieve a "better" prior.
    - **Mechanism**: The fixed-point set $C_n$ of different restoration models may contain distinct spurious fixed points (e.g., those from JPEG restoration models may exhibit blocking artifacts), but real natural images should lie near the intersection of all fixed-point sets. By minimizing the weighted sum of distances to multiple fixed-point sets $\sum_n \gamma_n d_{C_n}^2(x)$, model-specific artifacts are filtered out, leading to higher-quality reconstructions. The parameter $\gamma_n$ controls the influence of each prior.
    - **Design Motivation**: The fixed-point set of a single restoration model may be imprecise, but ensembling complementary models allows them to leverage each other's strengths. For instance, while the SCUNet JPEG prior tends to generate piecewise constant regions, deblurring models preserve more fine textures; combining them yields significantly better results.

### Loss & Training

FiRe itself requires no training and directly utilizes off-the-shelf pre-trained restoration models for inference-time optimization. Core hyperparameters: $\lambda$ controls the data fidelity strength, $\gamma_n$ controls the strength of each prior, and the number of iterations is typically set to 30. When using expected priors, the strength of the degradation acts similarly to a regularization parameter.

## Key Experimental Results

### Main Results

| Dataset | Problem | FiRe-HQS (Ensemble) | DRP | DPIR | DiffPIR |
|--------|------|---------|-----|------|---------|
| Imnet100 | Gaussian Deblurring PSNR | **25.80** | 25.48 | 25.18 | 25.32 |
| Imnet100 | Motion Deblurring PSNR | **30.49** | 25.38 | 30.39 | 29.70 |
| Imnet100 | 4× SR PSNR | **23.92** | 23.23 | 23.60 | 23.89 |
| BSD20 | Gaussian Deblurring PSNR | **27.00** | 26.18 | 26.14 | 26.67 |
| BSD20 | Motion Deblurring PSNR | **31.67** | 26.05 | 30.95 | 30.85 |
| BSD20 | 4× SR PSNR | 25.15 | 23.65 | 24.64 | **26.18** |

FiRe-HQS uses an ensemble of three models (SCUNet + SwinIR x2 + Restormer), outperforming all PnP and diffusion methods in most scenarios.

### Ablation Study (Single Prior Comparison, Set3C Dataset)

| Restoration Prior | Gaussian Deblurring | Motion Deblurring | 4x SR |
|---------|-----------|-----------|---------|
| PnP-DRUNet (Baseline) | 25.93 | 27.95 | 22.54 |
| FiRe-SCUNet JPEG | 27.19 | **31.23** | 24.01 |
| FiRe-SCUNet Blind Denoising | **28.03** | 30.84 | **24.30** |
| FiRe-Restormer Gaussian Deblurring | 27.01 | 29.83 | 24.29 |
| FiRe-SwinIR SR x2 | 28.19 | 27.17 | 24.02 |
| FiRe-LAMA Inpainting | 24.72 | 26.19 | 21.80 |

### Key Findings
- Most restoration priors within the FiRe framework can match or even exceed traditional denoising PnP baselines, except for the LAMA inpainting prior.
- The SCUNet blind denoiser performs significantly better in the FiRe framework than in the standard PnP framework (28.03 vs 23.91), proving the efficacy of the fixed-point formulation.
- Multi-prior ensembles (Fig. 4) consistently yield reconstruction qualities superior to single priors by adjusting $\gamma_n$.
- The degradation strength acts as a regularization parameter: too weak leads to insufficient prior strength, while too strong introduces excessive noise.
- Despite being trained only for binary-mask inpainting, the LAMA inpainting prior still provides meaningful priors (outperforming pseudo-inverse reconstruction), indicating that restoration models can implicitly encode image priors even when task discrepancy is large.

## Highlights & Insights
- **The theoretical insight from the fixed-point perspective** is elegant: a clean image is not a fixed point of the restoration model $R$, but it is a fixed point of $R \circ D$. This observation transforms the problem from "how to generalize Tweedie's formula" to "the gradient of the distance function equals the residual", which is clean and powerful.
- **Unifying traditional PnP into a broader framework**: When the restoration model is a denoiser and $D$ represents noise injection, FiRe degenerates to the standard PnP with noise injection strategy, providing a theoretical explanation for the latter.
- **Resolving the distribution mismatch in PnP**: The inputs to the restoration model at each iteration step undergo degradation, ensuring they remain within its training distribution.
- The proposed mechanism of composing different restoration models can be extended to any "multi-prior" scenario, holding direct value for practical applications such as medical image reconstruction.

## Limitations & Future Work
- The prox-regularity assumption is strong, and the structure of the fixed-point set in practice might not satisfy it; the authors acknowledge that $T = \text{proj}_C$ might be overly idealized for natural images.
- The poor performance of the LAMA inpainting prior in FiRe suggests that the quality of the prior degrades when there is a significant discrepancy between the training degradation of the restoration model and the target problem.
- Hyperparameter tuning for $\gamma_n$ in multi-prior ensembles still requires manual adjustment, lacking an adaptive selection strategy.
- Inference requires calling multiple restoration networks, making the computational cost scale linearly with the number of models.
- Evaluations are primarily conducted on low-to-medium-resolution images, with insufficient validation on high-resolution images or other modalities (e.g., MRI).

## Related Work & Insights
- **vs DRP/ShaRP**: DRP and ShaRP also employ non-denoising restoration models as priors, but their gradients contain an $H^\top H$ term, which restricts prior information to $\text{ker}(H)^\perp$. In contrast, FiRe's gradient $x - R(Dx)$ is free from this limitation, enabling the extraction of priors along the $\text{ker}(H)$ direction as well.
- **vs DPIR/DiffPIR**: These are mainstream PnP and diffusion PnP methods, both limited to denoising priors. FiRe can leverage arbitrary restoration priors and achieves better performance in most scenarios.
- **vs Diffusion Models**: Diffusion models inherently rely on Tweedie's formula and denoising priors. FiRe's fixed-point perspective offers a theoretically more self-consistent alternative.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The fixed-point perspective incorporates general-purpose restoration models into the PnP framework, offering prominent theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across multiple inverse problems and prior combinations, though more high-resolution experiments could be added.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivation, intuitive illustrations, and a tight logical chain spanning motivation, methodology, and experiments.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for leveraging pre-trained restoration models in image inverse problems, with far-reaching impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Variational Garrote for Sparse Inverse Problems](variational_garrote_for_sparse_inverse_problems.md)
- [\[ICML 2026\] Solving Inverse Problems with Flow-based Models via Model Predictive Control](../../ICML2026/image_restoration/solving_inverse_problems_with_flow-based_models_via_model_predictive_control.md)
- [\[CVPR 2026\] PnP-CM: Consistency Models as Plug-and-Play Priors for Inverse Problems](../../CVPR2026/image_restoration/pnp-cm_consistency_models_as_plug-and-play_priors_for_inverse_problems.md)
- [\[ICLR 2026\] Flower: A Flow-Matching Solver for Inverse Problems](../../ICLR2026/image_restoration/flower_a_flow-matching_solver_for_inverse_problems.md)
- [\[CVPR 2026\] GSNR: Graph Smooth Null-Space Representation for Inverse Problems](../../CVPR2026/image_restoration/gsnr_graph_smooth_null_space_representation_for_inverse_problems.md)

</div>

<!-- RELATED:END -->
