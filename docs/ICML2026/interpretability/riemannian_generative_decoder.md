---
title: >-
  [Paper Note] Riemannian Generative Decoder
description: >-
  [ICML 2026][Interpretability][Riemannian Manifolds] This paper addresses the pain point where Riemannian VAEs require manual design of complex probability densities for each manifold. It proposes the Riemannian Generative Decoder (RGD)——completely discarding the encoder and treating the latent of each sample as a free parameter optimized directly with a Riemannian optimizer (RiemannianAdam). By introducing "input noise scaled by the inverse local metric" as geometric regulari…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Riemannian Manifolds"
  - "Generative Decoder"
  - "Geometric Regularization"
  - "MAP Estimation"
  - "Computational Biology"
date: 2026-05-08
content_hash: 4164ae20365fa2dd
---

# Riemannian Generative Decoder

**Conference**: ICML 2026  
**arXiv**: [2506.19133](https://arxiv.org/abs/2506.19133)  
**Code**: https://github.com/yhsure/riemannian-generative-decoder (Available)  
**Area**: Representation Learning / Geometric Deep Learning / Interpretable Generative Models  
**Keywords**: Riemannian Manifolds, Generative Decoder, Geometric Regularization, MAP Estimation, Computational Biology

## TL;DR
This paper addresses the pain point where Riemannian VAEs require manual design of complex probability densities for each manifold. It proposes the Riemannian Generative Decoder (RGD)——completely discarding the encoder and treating the latent of each sample as a free parameter optimized directly with a Riemannian optimizer (RiemannianAdam). By introducing "input noise scaled by the inverse local metric" as geometric regularization, RGD recovers more faithful geometries on synthetic branching trees, human mitochondrial DNA, and cell-cycle scRNA-seq data, demonstrating superior numerical stability in high dimensions compared to VAE baselines.

## Background & Motivation

**Background**: Real-world data (evolutionary trees, social networks, periodic signals) often possess non-Euclidean geometric structures. However, mainstream representation learning typically assumes an $\mathbb{R}^d$ latent space, forcing geometric information to be flattened. The Riemannian VAE series ($\mathcal{S}$-VAE using von Mises–Fisher, $\mathcal{P}$-VAE using Poincaré, $\Delta$VAE using Brownian motion sampling) attempts to extend VAEs to manifolds like spheres, hyperboloids, and tori, representing the current mainstream in geometry-aware representation learning.

**Limitations of Prior Work**: Every manifold requires a specifically designed probability density. Difficulties include: (i) von Mises–Fisher normalization constants containing Bessel functions; (ii) Riemannian normal distributions on Poincaré requiring Monte Carlo for normalization constants and volume correction; (iii) wrapped normal distributions needing truncated sums; (iv) $\Delta$VAE requiring Brownian motion simulation. These approximations are numerically unstable (especially exploding in high dimensions) and tether manifold selection to the availability of a "tractable density," preventing the direct use of heterogeneous composite geometries like ProductManifolds.

**Key Challenge**: The VAE framework requires both the encoder $q_\phi(z|x)$ and the prior $p(z)$ to have computable densities on the manifold. However, most Riemannian manifolds lack closed-form densities. The cost of forced approximation includes bias, unstable optimizers, and limited manifold coverage——three factors that are inherently locked together.

**Goal**: (i) Enable representation learning on **any** Riemannian manifold supported by `geoopt` (including Product combinations); (ii) Eliminate dependence on manifold density; (iii) Provide a **geometry-aware** regularization term that aligns decoder smoothness with the local metric of the manifold rather than a fixed Euclidean direction.

**Key Insight**: Following the Deep Generative Decoder (DGD) approach——discard the encoder and treat the latent $z_i$ as a free parameter for direct MAP optimization. This approach is upgraded to Riemannian DGD: using a Riemannian optimizer instead of Euclidean Adam, ensuring each latent remains on the manifold and bypassing density approximations.

**Core Idea**: Discard the encoder + use RiemannianAdam to treat latents as free parameters on the manifold + inject geometric noise with covariance $\sigma^2 G^{-1}(z)$ as regularization during training, allowing the decoder to become naturally smooth in directions with large metrics.

## Method

### Overall Architecture
RGD addresses the pain point of Riemannian VAEs needing manual probability densities by removing the encoder entirely. Given data $X=\{x_i\}_{i=1}^N\in\mathbb{R}^D$ and a chosen $d$-dimensional Riemannian manifold $(\mathcal{M},g)$, it treats each sample's latent variable $z_i$ as a learnable free parameter. Together with the decoder $f_\theta:\mathcal{M}\to\mathcal{X}$, it performs MAP estimation $\mathcal{L}(\theta,Z)=\sum_i(-\log p_\theta(x_i|z_i)-\log p(z_i))-\log p(\theta)$. During training, $\theta$ follows Euclidean steps via standard Adam, while $Z$ follows Riemannian steps via RiemannianAdam, mapping tangent vectors back to the manifold using a retraction $R_z(\cdot)$ to ensure $z_i$ remains on the surface. Simultaneously, geometric noise scaled by the inverse of the local metric is injected into the latents as regularization, aligning the decoder's smoothing direction with the manifold geometry. Consequently, density approximations are no longer needed, and implementation via the `geoopt` library is highly concise.

### Key Designs

**1. Encoder-less MAP + RiemannianAdam: Optimizing latents as free parameters**

The root of VAE complications lies in the encoder: $q_\phi(z|x)$ must be a probability distribution with a calculable density on the surface. RGD abandons amortized inference and directly updates the latent variable of each sample using the Riemannian gradient $\nabla_z^{\mathcal{R}}\mathcal{L}=G(z)^{-1}\nabla_z^E\mathcal{L}$ combined with a retraction (usually the exponential map): $z^{(t+1)}=R_{z^{(t)}}(-\eta\,\nabla_{z^{(t)}}^{\mathcal{R}}\mathcal{L})$. RiemannianAdam maintains adaptive directions in the tangent space, achieving convergence speeds comparable to Adam while strictly remaining on the manifold. The prior is also simplified——compact manifolds use a constant $1/\text{Vol}(\mathcal{M})$ (which does not affect gradients), and non-compact manifolds use wrapped or Riemannian normals. This MAP paradigm bypasses the density approximation chain; a side benefit is that heterogeneous ProductManifolds become immediately usable.

**2. Geometry-aware Input Noise Regularization: Aligning decoder smoothness with metric**

Simply choosing the right manifold provides the space, but to ensure the decoder maps "geometrically similar points" to similar outputs, a metric-aligned regularization is required. RGD injects noise $\epsilon\sim\mathcal{N}(0,\sigma^2 G^{-1}(z))$—where noise is weaker in directions with large metrics and stronger in those with small metrics—into the latents during training, mapped back via $z'=\text{Exp}_z(\epsilon)\approx z+\epsilon$. Following Bishop's (1995) second-order Taylor expansion, this is equivalent to adding a metric-weighted Jacobian norm penalty $\mathbb{E}_\epsilon[L(z')]\approx L(z)+\sigma^2\,\text{Tr}\big(J(z)^\top G^{-1}(z)J(z)\big)$ to the loss, where $J(z)=\partial_z f$. This is effective because standard Euclidean Gaussian noise over-penalizes directions with large metrics on curved surfaces; with $G^{-1}(z)$ scaling, the regularization becomes isotropic on homogeneous manifolds like spheres and adapts by position on non-uniform hyperbolic manifolds.

**3. Unified Framework Supporting Arbitrary Riemannian Manifolds and Product Combinations**

Previously, introducing a new manifold required re-deriving priors and approximations. Because RGD does not rely on manifold density, it can reuse all manifolds implemented in `geoopt` (Euclidean, Sphere, PoincareBall, Lorentz, SPD, Stiefel, ProductManifold, etc.), provided the manifold supplies an exponential map, retraction, and metric. Heterogeneous requirements are expressed via product manifolds $\mathcal{M}=\mathcal{M}_1\times\cdots\times\mathcal{M}_K$, with the metric taken as the direct sum of components. This makes manifold selection a configurable item rather than a hard-coded algorithmic assumption, enabling hypothesis-based exploration.

### Loss & Training
The objective is the negative posterior from Equation (10): $\mathcal{L}=\sum_i(-\log p_\theta(x_i|z_i)-\log p(z_i))-\log p(\theta)$. $\theta$ is updated with Adam and $Z$ with RiemannianAdam. Geometric regularization is implemented by adding $\mathcal{N}(0,\sigma^2 G^{-1})$ noise to the latents followed by a retraction. The reconstruction likelihood is chosen based on data properties (e.g., Gaussian/MSE for continuous, categorical for discrete). There are no KL terms, no ELBO, and no need for Monte Carlo estimation of normalization constants.

## Key Experimental Results

### Main Results
Three real/synthetic biological datasets: (a) Synthetic branching diffusion tree (7 layers, $d=50$, 6350 samples); (b) Human mitochondrial DNA (67k sequences + haplogroup labels); (c) Cell-cycle scRNA-seq (5367 cells $\times$ 189 genes).

| Dataset | Task | Best Geometry | Key Metric | Remarks |
|--------|------|----------|----------|------|
| Cell cycle scRNA-seq | Phase dist vs latent dist correlation | Sphere $\mathbb{S}^2$ | Train Pearson 0.58, Test 0.60, Recon MAE 0.31 | Exceeds $\mathcal{S}$-VAE / $\Delta$VAE |
| Branching diffusion | Tree dist vs latent geodesic dist correlation | Lorentz $\mathbb{H}^2$ ($\sigma=1.0$) | Train Pearson 0.81, Test 0.80 | $\mathcal{P}$-VAE only 0.68 |
| hmtDNA haplogroup classification | Downstream 24/128-class logistic regression | Hyperbolic $\mathbb{H}^2_{\sigma=0.5}$ | 24-way LR 0.70 / XGB 0.85; 128-way LR 0.43 | Outperforms Euclidean and $\mathcal{P}$-VAE |

### Ablation Study

| Configuration | Key Finding |
|------|----------|
| Geometric noise $\sigma$ sweep 0→2.6 (Hyperbolic, Branching) | Correlation rises quickly until $\sigma\approx 0.9$, then noise degrades local precision → Local-global trade-off |
| Hyperbolic $\mathbb{H}^2$ vs Euclidean / Sphere (Branching) | $\mathbb{H}^2$ significantly recovers tree topology (Pearson 0.81 vs 0.53/0.56) |
| UMAP (Branching) | Failed to reveal tree topology → Validates RGD's interpretability advantage |
| Comparison with $\mathcal{P}$-VAE/$\mathcal{S}$-VAE/$\Delta$VAE | VAE baselines collapse in high dimensions; RGD remains stable |
| Generative Discrimination (XGBClassifier) | RGD Sphere 0.58 = $\mathcal{S}$-VAE Sphere 0.58 < $\Delta$VAE 0.62 (closer to 0.5 is better) |

### Key Findings
- **Encoder-less is more stable in high dimensions**: In cell cycle full-gene (high-D) and ProductManifold settings, VAE baselines collapse due to normalizing constants or MC sampling, whereas RGD remains trainable.
- **Geometric noise is critical for learning geometry**: Removing it causes hyperbolic models to perform similarly to Euclidean baselines, showing that the manifold choice provides the space, but regularization aligns the decoder.
- **Manifold selection as hypothesis testing**: Branching data with hyperbolic latents restores tree structure instantly, while UMAP fails, proving that rational geometric priors are more interpretable than general dimensionality reduction.
- **Generation quality is on par with VAEs**: RGD's synthetic sample discriminability matches $\mathcal{S}$-VAE (0.58), indicating no loss in fidelity from discarding the encoder.

## Highlights & Insights
- **Clean methodology**: Extending DGD from Euclidean space to arbitrary Riemannian manifolds bypasses the entire density approximation chain of Riemannian VAEs—a rare "subtractive contribution" with significant scalability benefits.
- **Simple and actionable geometric noise**: Using the classic noise-regularization equivalence with a metric-inverse scaling yields a metric-aware Jacobian penalty on any manifold with a single line of code.
- **Manifolds as hypotheses rather than algorithmic burdens**: Enables scientists to quickly test tori, spheres, hyperbolic, or SPD geometries on the same data, picking the one that best fits the prior—a direct productivity boost for computational biology.
- **Transferability to LLM/VLM latent representations**: The "manifold + free latent + Riemannian Adam + metric noise" combination can be applied to expert routing in Stiefel/Sphere spaces or vision-language alignment in ProductManifolds.

## Limitations & Future Work
- The lack of an amortized encoder means new samples must **undergo latent optimization** (per-sample MAP steps) to be encoded, making online inference slower than VAEs; amortized warm-starts could be added.
- Geometric noise $\sigma$ is coupled with manifold curvature and requires manual sweeping; automatic scheduling is a logical next step.
- Priors on non-compact manifolds still use wrapped/Riemannian normals, theoretically introducing slight bias, though the authors treat generation quality primarily as a sanity check.
- Experiments are focused on biological/synthetic data; large-scale NLP/Vision tasks have not been verified.
- Dimension allocation for ProductManifolds requires manual setting, which may be difficult for complex data.

## Related Work & Insights
- **vs $\mathcal{S}$-VAE / $\mathcal{P}$-VAE / $\Delta$VAE**: These require per-manifold densities; RGD bypasses this and offers significantly better high-dimensional stability.
- **vs DGD (Schuster & Krogh 2023)**: DGD is Euclidean encoder-less; RGD is its Riemannian extension, generalizing free-parameter MAP to arbitrary manifolds.
- **vs Lee & Park (2023) Curvature Regularization**: They use second-order curvature (involving Hessian-vector products) for "global flattening"; RGD uses first-order Jacobian regularization plus metric-inverse noise, which is an order of magnitude cheaper computationally.
- **vs UMAP / Isomap**: Classic non-linear reductions only visualize; RGD is a generative model and explicitly specifies the manifold for hypothesis testing.

## Rating
- Novelty: ⭐⭐⭐⭐ Discarding the encoder to upgrade DGD to arbitrary manifolds plus metric-inverse noise are clean, independent contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of periodic/tree/hierarchical geometries across three datasets with multiple baselines, though lacks large-scale NLP/Vision.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivations, detailed appendix, and a smooth transition between theory and engineering.
- Value: ⭐⭐⭐⭐ Lowers the barrier for manifold usage from "algorithmic hurdle" to "configuration item," significantly impacting geometric representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GenCtrl — A Formal Controllability Toolkit for Generative Models](../../ICLR2026/interpretability/genctrl_--_a_formal_controllability_toolkit_for_generative_models.md)
- [\[NeurIPS 2025\] Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing](../../NeurIPS2025/interpretability/uncovering_graph_reasoning_in_decoder-only_transformers_with_circuit_tracing.md)
- [\[ICLR 2026\] Uncovering Conceptual Blindspots in Generative Image Models Using Sparse Autoencoders](../../ICLR2026/interpretability/uncovering_conceptual_blindspots_in_generative_image_models_using_sparse_autoenc.md)
- [\[ICLR 2026\] Your VAR Model is Secretly an Efficient and Explainable Generative Classifier](../../ICLR2026/interpretability/your_var_model_is_secretly_an_efficient_and_explainable_generative_classifier.md)
- [\[ICML 2025\] Position: We Need An Algorithmic Understanding of Generative AI](../../ICML2025/interpretability/position_we_need_an_algorithmic_understanding_of_generative_ai.md)

</div>

<!-- RELATED:END -->
