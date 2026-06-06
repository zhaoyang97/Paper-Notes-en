---
title: >-
  [Paper Note] Counterfactual Explanations on Robust Perceptual Geodesics
description: >-
  [ICLR 2026][Causal Inference][Counterfactual Explanations] This paper proposes PCG (Perceptual Counterfactual Geodesic), a method that generates semantically faithful counterfactual explanations by optimizing geodesics o…
tags:
  - "ICLR 2026"
  - "Causal Inference"
  - "Counterfactual Explanations"
  - "Geodesic Optimization"
  - "Perceptual Metric"
  - "Adversarial Robustness"
  - "Interpretability"
date: 2026-05-08
content_hash: 3a73eefe5307719f
---

# Counterfactual Explanations on Robust Perceptual Geodesics

**Conference**: ICLR 2026
**arXiv**: [2601.18678](https://arxiv.org/abs/2601.18678)  
**Code**: Provided (anonymous)  
**Area**: Human Understanding / Explainable AI / Image Generation
**Keywords**: Counterfactual Explanations, Geodesic Optimization, Perceptual Metric, Adversarial Robustness, Interpretability

## TL;DR
This paper proposes PCG (Perceptual Counterfactual Geodesic), a method that generates semantically faithful counterfactual explanations by optimizing geodesics on a robust perceptual manifold. A two-stage optimization ensures that the resulting path is both perceptually natural and reaches the target class. PCG achieves FID=8.3 on AFHQ, substantially outperforming RSGD (FID=12.9).

## Background & Motivation

**Background**: Counterfactual explanations ("if the image looked like this, the classifier would change its prediction") are an important tool for model interpretability. Existing methods generate counterfactuals by performing gradient descent directly in pixel space or latent space.

**Limitations of Prior Work**: Pixel-space counterfactuals tend to produce unnatural adversarial perturbations, while latent-space methods may drift off the manifold, yielding unrealistic images.

**Key Challenge**: Counterfactuals must achieve "minimal change" while remaining "semantically plausible"—two objectives that frequently conflict in Euclidean space, where the shortest path may traverse low-density, unrealistic regions.

**Goal**: How can one find the shortest path to a target class subject to perceptual naturalness constraints?

**Key Insight**: Compute geodesics on the Riemannian manifold defined by a robust perceptual metric—manifold geodesics naturally follow the "ridges" of the data distribution and avoid low-density "valleys."

**Core Idea**: Define a Riemannian manifold via the feature space of an adversarially trained robust model, then compute geodesics on this manifold as counterfactual paths.

## Method

### Overall Architecture
Two-stage optimization: Phase 1 minimizes geodesic energy (keeping the path close to the data manifold); Phase 2 incorporates a classification loss under the energy constraint (steering the path toward the target class).

### Key Designs

1. **Robust Perceptual Metric**:

    - **Function**: Defines a Riemannian metric tensor using the Jacobian of an adversarially trained model.
    - **Mechanism**: $G_R(x) = \sum_k w_k \, J(h_k(x))^\top J(h_k(x))$, where $h_k$ denotes intermediate-layer features of the robust model. Pulled back to latent space: $G_z(z) = J(g(z))^\top G_R(g(z)) J(g(z))$.
    - **Design Motivation**: Feature gradients of robust models are meaningful along semantic directions (non-adversarial), so the induced metric places semantically similar points closer together.

2. **Geodesic Optimization**:

    - Phase 1: Minimize path energy $E = \int \gamma'(t)^\top G_z \, \gamma'(t) \, dt$.
    - Phase 2: Incorporate a classification loss to guide the path toward the target class.
    - **Design Motivation**: Decoupled optimization prevents the classification loss from prematurely pulling the path off the manifold.

## Key Experimental Results

| Dataset | Method | FID | R-FID | R-LPIPS |
|---------|--------|-----|-------|---------|
| AFHQ | RSGD | 12.9 | 37.8 | 0.68 |
| AFHQ | **PCG** | **8.3** | **9.1** | **0.17** |

### Key Findings
- PCG reduces R-LPIPS (robust perceptual distance) from 0.68 to 0.17, indicating that generated counterfactuals are more perceptually natural.
- Intermediate frames along the counterfactual path are also visually plausible (smooth transitions rather than abrupt changes).
- The metric defined by a robust model outperforms that of a standard model, whose feature gradients are insufficiently semantic.

## Ablation Study

### Effect of Metric Choice on Interpolation Quality

| Metric | Semantic Coherence | On-Manifold | Adversarial Vulnerability |
|--------|--------------------|-------------|--------------------------|
| Z-linear (Euclidean) | Poor; intermediate frames are blurry | No | N/A |
| Pixel MSE pullback | Poor; attribute inconsistency | Partial | High |
| Standard ResNet-50 feature pullback | Moderate; lighting drift | Yes | High |
| **Robust ResNet-50 feature pullback** | **Good; semantic gradation** | **Yes** | **Low** |

### Quantitative Counterfactual Comparison (StyleGAN2)

| Method | AFHQ $\mathcal{L}_1$↓ | AFHQ $\mathcal{L}_{\mathcal{R}}$↓ | FFHQ $\mathcal{L}_{\mathcal{R}}$↓ | PlantVillage $\mathcal{L}_{\mathcal{R}}$↓ |
|--------|-------|-------|-------|-------|
| REVISE | **1.20** | 2.70 | 2.78 | 2.87 |
| VSGD | 1.31 | 2.90 | 2.86 | 2.83 |
| RSGD | 1.73 | 2.79 | 2.81 | 2.88 |
| RSGD-C | 1.55 | 2.62 | 2.69 | 2.67 |
| **PCG** | 1.42 | **2.21** | **2.48** | **2.43** |

- REVISE achieves the lowest pixel $\ell_1$ but high $\mathcal{L}_{\mathcal{R}}$ (robust perceptual distance), indicating that its counterfactuals are pixel-close yet perceptually adversarial.
- PCG leads across all robust metrics, demonstrating that it reaches the correct side of the semantic decision boundary.

### Necessity of Two-Stage Optimization
- Phase 1 anchors the path to the robust perceptual manifold; without it, directly executing Phase 2 collapses to off-manifold solutions as in VSGD.
- The re-anchoring step in Phase 2 progressively pulls the endpoint toward the input, ensuring that counterfactuals satisfy minimal-change requirements.

## Highlights & Insights
- The combination of **Riemannian geometry and explainability** is both mathematically elegant and empirically effective—applying differential-geometric geodesics to XAI is a highly natural formulation.
- **The concept of the "semantic divide"** was introduced by Browne & Swift (2020) without a practical crossing mechanism; PCG is the first method to reliably cross this boundary via a robust metric.
- **The two-stage optimization paradigm**—build the road, then navigate it—first ensures path quality via energy minimization, then incorporates the task objective, yielding greater stability than direct end-to-end optimization.
- A **systematic taxonomy of failure modes** identifies three categories of failure in prior methods: off-manifold traversal, local gradient traps, and exploitation of fragile metrics by the generator.

## Limitations & Future Work
- The method requires an adversarially trained robust model to define the metric; such models are not readily available in all domains (e.g., medical imaging, remote sensing).
- Geodesic optimization is computationally expensive—Jacobian-vector products make each step several times costlier than standard SGD.
- Validation is limited to image classification; counterfactual explanations for text and tabular data require different manifold definitions.
- The method relies on StyleGAN2/3 as the generator; adaptation to diffusion model architectures remains an open challenge.
- Layer-wise feature weights $w_k$ in the robust metric are set to $1/N_k$ (dimension-normalized) without exploring alternative weighting schemes.

## Related Work & Insights
- **vs. REVISE (Joshi et al.)**: REVISE performs Euclidean SGD in a VAE latent space, assuming flat geometry—PCG demonstrates that this assumption fails entirely in the image domain.
- **vs. RSGD (Pegios et al.)**: RSGD introduces a Riemannian metric but uses features from a fragile standard classifier—PCG repairs the metric itself by substituting robust features.
- **vs. DiME / diffusion-based counterfactuals**: Diffusion-based methods achieve high generation quality but lack geometric guarantees—PCG's geodesic constraint ensures that every step along the path is perceptually natural.
- **vs. Santurkar et al. (2019) on robust model interpretability**: That work studies the interpretability of robust models themselves, whereas PCG uses robust models to generate explanations for standard models—orthogonal research directions.
- **Broader Implication**: The robust perceptual metric concept is generalizable to any generative task requiring perceptual naturalness, including image editing, style transfer, and inpainting.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The theoretical framework applying Riemannian geodesics to counterfactual explanation is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets, two StyleGAN variants, and quantitative comparison against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are rigorous and figures are clear.
- Value: ⭐⭐⭐⭐⭐ — Provides a principled solution to the metric problem in XAI.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are rigorous.
- Value: ⭐⭐⭐⭐ — Offers a theoretically grounded tool for explainable AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Density-Guided Robust Counterfactual Explanations on Tabular Data under Model Multiplicity](../../ICML2026/causal_inference/density-guided_robust_counterfactual_explanations_on_tabular_data_under_model_mu.md)
- [\[ICLR 2026\] Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)
- [\[ICLR 2026\] Learning Robust Intervention Representations with Delta Embeddings](learning_robust_intervention_representations_with_delta_embeddings.md)
- [\[ICLR 2026\] Direct Doubly Robust Estimation of Conditional Quantile Contrasts](direct_doubly_robust_estimation_of_conditional_quantile_contrasts.md)
- [\[AAAI 2026\] KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education](../../AAAI2026/causal_inference/ktcf_actionable_recourse_in_knowledge_tracing_via_counterfactual_explanations_fo.md)

</div>

<!-- RELATED:END -->
