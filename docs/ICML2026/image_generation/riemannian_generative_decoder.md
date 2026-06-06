---
title: >-
  [Paper Note] Riemannian Generative Decoder
description: >-
  [ICML 2026][Image Generation][Riemannian manifold] This work addresses the challenge that Riemannian VAEs require hand-crafted, complex probability densities for each manifold. It proposes the Riemannian Generative Decod…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Riemannian manifold"
  - "generative decoder"
  - "geometric regularization"
  - "MAP estimation"
  - "computational biology"
date: 2026-05-08
content_hash: dd5c42ce72cded4c
---

# Riemannian Generative Decoder

**Conference**: ICML 2026  
**arXiv**: [2506.19133](https://arxiv.org/abs/2506.19133)  
**Code**: https://github.com/yhsure/riemannian-generative-decoder (available)  
**Area**: Representation Learning / Geometric Deep Learning / Interpretable Generative Models  
**Keywords**: Riemannian manifold, generative decoder, geometric regularization, MAP estimation, computational biology

## TL;DR
This work addresses the challenge that Riemannian VAEs require hand-crafted, complex probability densities for each manifold. It proposes the Riemannian Generative Decoder (RGD), which entirely discards the encoder and treats each sample's latent as a free parameter, trained directly with a Riemannian optimizer (RiemannianAdam). It introduces "input noise inversely scaled by local metric" as a geometric regularizer. On three real biological datasets—synthetic branching diffusion tree, human mitochondrial DNA, and cell cycle scRNA-seq—RGD recovers more faithful geometry and achieves superior numerical stability over VAE baselines in high dimensions.

## Background & Motivation

**Background**: Real-world data (evolutionary trees, social networks, periodic signals) often exhibit non-Euclidean geometric structures, yet mainstream representation learning typically assumes a latent space in $\mathbb{R}^d$, which forcibly flattens geometric information. The Riemannian VAE family (e.g., $\mathcal{S}$-VAE with von Mises–Fisher, $\mathcal{P}$-VAE with Poincaré, $\Delta$VAE with Brownian motion sampling) attempts to extend VAEs to spherical, hyperbolic, torus, and other manifolds, representing the current mainstream in geometry-aware representation learning.

**Limitations of Prior Work**: Each manifold requires a specifically designed probability density, with challenges including (i) von Mises–Fisher normalization constant involving Bessel functions; (ii) Poincaré Riemannian normal distribution normalization constant and volume correction requiring Monte Carlo; (iii) wrapped normal requiring truncated summation; (iv) $\Delta$VAE relying on Brownian motion simulation. All these approximations are numerically unstable (especially in high dimensions) and tightly couple manifold choice to the tractability of density expressions, making it impossible to directly handle heterogeneous combinations like ProductManifold.

**Key Challenge**: The VAE framework requires both the encoder $q_\phi(z|x)$ and prior $p(z)$ to have computable densities on the manifold, but most Riemannian manifolds lack closed-form densities. Forcing approximations leads to bias, unstable optimizers, and limited manifold coverage—three issues that are mutually reinforcing.

**Goal**: (i) Enable representation learning on **any** Riemannian manifold supported by geoopt (including Product combinations); (ii) eliminate dependence on manifold densities; (iii) provide a **geometry-aware** regularizer so that decoder smoothness aligns with the manifold's local metric rather than a fixed Euclidean direction.

**Key Insight**: The Deep Generative Decoder (DGD) approach—omit the encoder and treat latent $z_i$ as free parameters optimized via MAP. This is upgraded to Riemannian DGD: replace Euclidean Adam with a Riemannian optimizer, ensuring each latent remains on the manifold and bypassing density approximations.

**Core Idea**: Discard the encoder, use RiemannianAdam to treat latents as free parameters on the manifold, and inject geometric noise with covariance $\sigma^2 G^{-1}(z)$ during training as regularization, so the decoder naturally smooths more in directions with larger metric.

## Method

### Overall Architecture
Given data $X=\{x_i\}_{i=1}^N \in \mathbb{R}^D$, select a $d$-dimensional Riemannian manifold $(\mathcal{M},g)$ as the latent space. Treat $Z=\{z_i\}_{i=1}^N$ as per-sample free parameters, and jointly perform MAP estimation with decoder $f_\theta:\mathcal{M}\to\mathcal{X}$:
$$
\mathcal{L}(\theta, Z) = \sum_i \big(-\log p_\theta(x_i|z_i) - \log p(z_i)\big) - \log p(\theta)
$$
The likelihood is typically isotropic Gaussian (i.e., MSE reconstruction loss); the prior is uniform on compact manifolds, and wrapped or Riemannian normal on non-compact ones. Training alternates: $\theta$ is updated with Adam (Euclidean steps), $Z$ with RiemannianAdam (manifold steps), using retraction $R_z(\cdot)$ to map tangent vectors back to the manifold, ensuring $z^{(t+1)}\in\mathcal{M}$. The geoopt library makes this implementation concise.

### Key Designs

1. **Encoder-less MAP + RiemannianAdam for direct manifold latent optimization**:

    - **Function**: Treat each sample's latent variable as a learnable free parameter, enabling immediate use of any Riemannian manifold (Sphere, PoincareBall, Lorentz, SPD, UpperHalf, Stiefel, ProductManifold, etc.).
    - **Mechanism**: Abandon amortized inference. Each training step updates latents using the Riemannian gradient $\nabla_z^{\mathcal{R}}\mathcal{L}=G(z)^{-1}\nabla_z^E\mathcal{L}$, combined with retraction (typically the exponential map): $z^{(t+1)}=R_{z^{(t)}}(-\eta\,\nabla_{z^{(t)}}^{\mathcal{R}}\mathcal{L})$. RiemannianAdam maintains adaptive directions in the tangent space, ensuring convergence speed similar to Adam, but each step remains strictly on the manifold. The prior for compact manifolds is simply $1/\text{Vol}(\mathcal{M})$ (a constant, not affecting gradients); for non-compact, wrapped/Riemannian normal is used.
    - **Design Motivation**: The encoder is the root cause of manifold density approximation issues, as $q_\phi(z|x)$ must be a tractable probability distribution on the surface. Goldberg-DGD has shown that encoder-less MAP works in Euclidean space; this work lifts it directly to manifolds, bypassing all density approximations. An unexpected benefit is that heterogeneous product manifolds (ProductManifold) are immediately usable, as RGD does not require a prior for each manifold.

2. **Geometry-aware input noise regularization**:

    - **Function**: Align the decoder's local Jacobian with the manifold metric, encouraging geometrically similar points to be mapped to similar outputs.
    - **Mechanism**: During training, inject noise into the latent: $\epsilon\sim\mathcal{N}(0, \sigma^2 G^{-1}(z))$ (covariance scaled by the inverse metric, so noise is weaker in directions with large metric and stronger where small), and use the exponential map $z'=\text{Exp}_z(\epsilon)\approx z+\epsilon$ for injection. Following Bishop (1995)'s second-order Taylor expansion, the equivalent regularizer is: $\mathbb{E}_\epsilon[L(z')]\approx L(z)+\sigma^2\,\text{Tr}(J(z)^\top G^{-1}(z) J(z))$, where $J(z)=\partial_z f$; the added term is the Jacobian norm penalized by the manifold metric.
    - **Design Motivation**: Euclidean Gaussian noise on a surface over-penalizes directions with large metric and under-penalizes those with small metric; scaling noise by $G^{-1}(z)$ aligns isotropic regularization with manifold geometry, reducing to approximately isotropic on homogeneous manifolds (sphere), and adapting by location on inhomogeneous curvature (hyperbolic). Compared to Lee & Park (2023)'s second-order curvature regularizer (involving Hessian-vector products and full-page formulas), RGD's approach requires only a single Jacobian computation, making it much more scalable.

3. **Unified framework supporting arbitrary Riemannian manifolds + Product combinations**:

    - **Function**: Users only need to specify prior knowledge as a manifold choice; the framework handles the rest, with no need to hand-write ELBOs or density approximations.
    - **Mechanism**: Directly reuse all manifolds implemented in geoopt (Euclidean, Sphere, Stereographic, PoincareBall, Lorentz, SPD, Stiefel, UpperHalf, ProductManifold, etc.), as long as they provide exponential map/retraction/metric. Product manifolds are written as $\mathcal{M}=\mathcal{M}_1\times\cdots\times\mathcal{M}_K$, with the metric as a direct sum, automatically covering heterogeneous needs such as "some dimensions spherical + some hyperbolic".
    - **Design Motivation**: Previous work required re-deriving priors and approximations for each new manifold, making it hard for researchers to quickly compare which geometry best fits their data. RGD turns the manifold from an **algorithmic assumption** into a **configuration option**, making hypothesis-based exploration truly feasible; for cell cycle data, torus, sphere, and Euclidean can be compared with a single switch.

### Loss & Training
The objective is the negative posterior in Eq. (10): $\mathcal{L}=\sum_i(-\log p_\theta(x_i|z_i)-\log p(z_i))-\log p(\theta)$. $\theta$ is updated with Adam, $Z$ with RiemannianAdam in alternating steps. Geometric regularization is implemented by adding $\mathcal{N}(0,\sigma^2 G^{-1})$ noise to the latent (with a retraction to map noise back to the manifold). Reconstruction loss is chosen based on data type (continuous → Gaussian/MSE; discrete → categorical). There is no KL term, no ELBO, and no Monte Carlo estimation of normalization constants.

## Key Experimental Results

### Main Results
Three real/synthetic biological datasets: (a) Synthetic branching diffusion tree (7 layers, $d=50$, 6350 samples)—naturally suited to hyperbolic manifolds; (b) Human mitochondrial DNA, 67k sequences + haplogroup labels—hyperbolic fits phylogeny; (c) Cell cycle scRNA-seq, 5367 cells × 189 genes—periodic, torus suited.

| Dataset | Task | Best Geometry | Key Metric | Notes |
|---------|------|--------------|------------|-------|
| Cell cycle scRNA-seq | Phase distance vs latent distance correlation | Sphere $\mathbb{S}^2$ | Train Pearson 0.58, Test 0.60, reconstruction MAE 0.31 | Outperforms $\mathcal{S}$-VAE / $\Delta$VAE |
| Branching diffusion | Tree distance vs latent geodesic correlation | Lorentz $\mathbb{H}^2$ ($\sigma=1.0$) | Train Pearson 0.81, Test 0.80 | $\mathcal{P}$-VAE only 0.68 |
| hmtDNA haplogroup classification | Downstream 24/128-way logistic regression accuracy | Hyperbolic $\mathbb{H}^2_{\sigma=0.5}$ | 24-way LR 0.70 / XGB 0.85; 128-way LR 0.43 | Outperforms Euclidean and $\mathcal{P}$-VAE |

### Ablation Study

| Configuration | Key Findings |
|---------------|-------------|
| Geometric noise $\sigma$ sweep 0→2.6 (hyperbolic, branching diffusion) | Correlation rises rapidly up to $\sigma\approx 0.9$, then excessive noise degrades local accuracy—shows local-global trade-off |
| Hyperbolic $\mathbb{H}^2$ vs Euclidean / Sphere (branching diffusion) | $\mathbb{H}^2$ significantly recovers tree topology (Pearson 0.81 vs 0.53/0.56) |
| UMAP (branching diffusion) | Fails to reveal tree topology—demonstrates RGD's clear advantage in interpretability |
| Comparison with $\mathcal{P}$-VAE/$\mathcal{S}$-VAE/$\Delta$VAE | VAE baselines numerically collapse in high dimensions, RGD remains usable |
| Generation discrimination (XGBClassifier distinguishing real vs reconstructed) | RGD Sphere 0.58 = $\mathcal{S}$-VAE Sphere 0.58 < $\Delta$VAE 0.62 (closer to 0.5 is more realistic) |

### Key Findings
- **Encoder-less is more stable in high dimensions**: For cell cycle full-gene (high-dimensional) and ProductManifold settings, $\mathcal{S}$/$\Delta$/$\mathcal{P}$-VAE all collapse numerically due to normalizing constant or Monte Carlo sampling, while RGD can still train and reconstruct reasonably.
- **Geometric noise is key to learning geometry**: Removing it causes hyperbolic models' distance-correlation to approach Euclidean baseline, indicating that manifold choice only provides the space, but aligning the decoder with geometry relies on regularization.
- **Manifold choice = hypothesis testing**: Branching diffusion data with hyperbolic latent instantly recovers tree structure, while UMAP fails, showing that a suitable geometric prior is more interpretable than generic dimensionality reduction.
- **Generation quality matches VAE**: In discrimination tests, RGD's synthetic samples are as separable as those from $\mathcal{S}$-VAE (0.58), indicating that discarding the encoder does not harm generative fidelity.

## Highlights & Insights
- **Conceptually clean approach**: Extends DGD from Euclidean to arbitrary Riemannian manifolds, bypassing the entire density approximation chain of Riemannian VAEs—a rare "subtractive contribution" with major scalability benefits.
- **Simple and actionable geometric noise derivation**: By applying Bishop's classic noise-regularization equivalence with an added metric inverse scaling, a metric-aware Jacobian penalty is obtained for any manifold; the formula is concise and implementation is just a covariance substitution.
- **Manifold as hypothesis, not algorithmic burden**: Enables scientists to quickly try torus, sphere, hyperbolic, SPD on the same data and pick the geometry best matching their prior, directly boosting productivity in computational biology/phylogenetics.
- **Transferable to LLM/VLM latent representations**: LLM routing/expert representations on Stiefel or Sphere, vision-language alignment on ProductManifold—all can leverage this framework.

## Limitations & Future Work
- Absence of amortized encoder means new samples require **re-optimizing latents** (per-sample MAP step) for encoding, making online inference slower than VAE; amortized warm-start can be added.
- The $\sigma$ in geometric regularization is coupled with manifold curvature and requires manual sweeping; automatic scheduling is an obvious next step.
- The prior on non-compact manifolds still uses wrapped/Riemannian normal, theoretically introducing slight bias, but the authors treat generation quality as a sanity check.
- All experiments are on biological/synthetic data; not yet validated on large-scale NLP/vision tasks (though the principle is directly applicable).
- ProductManifold dimension allocation requires manual setting, which may be nontrivial for complex data; automatic geometry selection is a future direction.

## Related Work & Insights
- **vs $\mathcal{S}$-VAE / $\mathcal{P}$-VAE / $\Delta$VAE**: These require hand-crafted probability densities for each manifold, which RGD bypasses entirely, with much better high-dimensional stability.
- **vs DGD (Schuster & Krogh 2023)**: DGD is Euclidean encoder-less; RGD is its Riemannian extension, generalizing free-parameter MAP to arbitrary manifolds.
- **vs Lee & Park (2023) curvature regularization**: They use second-order curvature (with Hessian-vector products and full-page formulas) for "global flattening"; RGD uses first-order Jacobian regularization + metric-inverse noise, with an order of magnitude lower computational cost.
- **vs UMAP / Isomap**: Classic nonlinear dimensionality reduction only visualizes, does not learn generative models or explicitly specify manifolds; RGD both visualizes and generates, and supports hypothesis-testing.
- **Cross-task insights**: Molecular representations (using SE(3) manifolds), robot pose representations (SO(3), SE(3)), and belief space representations in reinforcement learning can all adopt RGD's "manifold + free latent + Riemannian Adam + metric noise" approach.

## Rating
- Novelty: ⭐⭐⭐⭐ "Discarding the encoder and upgrading DGD to arbitrary Riemannian manifolds" + "metric-inverse noise" are both clean, independent contributions, with clear framework-level simplification.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets covering periodic/tree/hierarchical geometries, multiple baselines, noise sweeps, downstream, and generation discrimination; lacks large-scale NLP/vision validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear formula derivations, detailed appendix, comprehensive coverage of prior work, and smooth theory-to-engineering transition.
- Value: ⭐⭐⭐⭐ In geometric DL, reduces the manifold from an "algorithmic barrier" to a "configuration option", significantly boosting productivity in representation learning and computational biology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Riemannian Consistency Model](../../NeurIPS2025/image_generation/riemannian_consistency_model.md)
- [\[ICCV 2025\] REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder](../../ICCV2025/image_generation/regen_learning_compact_video_embedding_with_re-generative_decoder.md)
- [\[ICML 2026\] Generative Visual Code Mobile World Models](generative_visual_code_mobile_world_models.md)
- [\[ICML 2026\] Threshold-Guided Optimization for Visual Generative Models](threshold-guided_optimization_for_visual_generative_models.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)

</div>

<!-- RELATED:END -->
