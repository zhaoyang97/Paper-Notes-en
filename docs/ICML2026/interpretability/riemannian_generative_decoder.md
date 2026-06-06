---
title: >-
  [Paper Note] Riemannian Generative Decoder
description: >-
  [ICML 2026][Interpretability][Riemannian Manifolds] This paper addresses the pain point where Riemannian VAEs require manual design of complex probability densities for each manifold. It proposes the Riemannian Generativ…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Riemannian Manifolds"
  - "Generative Decoder"
  - "Geometric Regularization"
  - "MAP Estimation"
  - "Computational Biology"
date: 2026-05-08
content_hash: 50a6f872b30b1f19
---

# Riemannian Generative Decoder

**Conference**: ICML 2026  
**arXiv**: [2506.19133](https://arxiv.org/abs/2506.19133)  
**Code**: https://github.com/yhsure/riemannian-generative-decoder (Available)  
**Area**: Representation Learning / Geometric Deep Learning / Interpretable Generative Models  
**Keywords**: Riemannian Manifolds, Generative Decoder, Geometric Regularization, MAP Estimation, Computational Biology

## TL;DR
This paper addresses the pain point where Riemannian VAEs require manual design of complex probability densities for each manifold. It proposes the Riemannian Generative Decoder (RGD)—completely discarding the encoder and treating each sample's latent as a free parameter optimized directly with a Riemannian optimizer (RiemannianAdam). It introduces "input noise inverse-scaled by local metrics" as geometric regularization. RGD restores more faithful geometry on synthetic branching trees, human mitochondrial DNA, and cell cycle scRNA-seq, while being numerically stable in high dimensions compared to VAE baselines.

## Background & Motivation

**Background**: Real-world data (evolutionary trees, social networks, periodic signals) often possess non-Euclidean geometric structures. However, mainstream representation learning assumes an $\mathbb{R}^d$ latent space, forcing geometric information to be flattened. The Riemannian VAE series ($\mathcal{S}$-VAE using von Mises–Fisher, $\mathcal{P}$-VAE using Poincaré, $\Delta$VAE using Brownian motion sampling) attempts to migrate VAEs to manifolds like spheres, hyperbolics, or tori, representing the current mainstream of geometry-aware representation learning.

**Limitations of Prior Work**: Every manifold requires specifically designed probability densities. Difficulties include: (i) the normalization constant for von Mises–Fisher contains Bessel functions; (ii) both the normalization constant and volume correction for Riemannian normal distributions on Poincaré require Monte Carlo; (iii) wrapped normal distributions require truncated sums; (iv) $\Delta$VAE relies on Brownian motion simulation. These approximations are both numerically unstable (especially in high dimensions) and tie manifold selection strictly to whether a tractable density can be derived, making it impossible to handle heterogeneous composite geometries like ProductManifold directly.

**Key Challenge**: The VAE framework requires both the encoder $q_\phi(z|x)$ and the prior $p(z)$ to have computable densities on the manifold. However, most Riemannian manifolds lack closed-form densities. The cost of forced approximation is bias, unstable optimizers, and limited manifold coverage—three factors that are directly interlocked.

**Goal**: (i) Enable representation learning on **any** Riemannian manifold supported by `geoopt` (including Product combinations); (ii) Eliminate dependence on manifold densities; (iii) Provide a **geometry-aware** regularization term to align decoder smoothness with the local metric of the manifold rather than a fixed Euclidean direction.

**Key Insight**: Following the Deep Generative Decoder (DGD) approach—discard the encoder and treat latents $z_i$ as free parameters optimized via MAP. This approach is upgraded to Riemannian DGD: using a Riemannian optimizer to replace Euclidean Adam, ensuring every latent remains on the manifold, thereby bypassing density approximations.

**Core Idea**: Discard the encoder + use RiemannianAdam to treat latents as free parameters on the manifold + inject geometric noise with covariance $\sigma^2 G^{-1}(z)$ as regularization during training, allowing the decoder to naturally become smooth in directions where the metric is large.

## Method

### Overall Architecture
Given data $X=\{x_i\}_{i=1}^N \in \mathbb{R}^D$, a $d$-dimensional Riemannian manifold $(\mathcal{M},g)$ is selected as the latent space. $Z=\{z_i\}_{i=1}^N$ are treated as per-sample free parameters, estimated via MAP alongside the decoder $f_\theta:\mathcal{M}\to\mathcal{X}$:
$$\mathcal{L}(\theta, Z) = \sum_i \big(-\log p_\theta(x_i|z_i) - \log p(z_i)\big) - \log p(\theta)$$
Typically, the likelihood is an isotropic Gaussian (reconstruction loss is MSE). The prior is uniform for compact manifolds and wrapped or Riemannian normal for non-compact ones. Training alternates: $\theta$ is updated with Euclidean steps via Adam, and $Z$ is updated with Riemannian steps via RiemannianAdam, using a retraction $R_z(\cdot)$ at each step to map tangent vectors back to the manifold, ensuring $z^{(t+1)}\in\mathcal{M}$. This implementation is highly streamlined via the `geoopt` library.

### Key Designs

1.  **Encoder-less MAP + RiemannianAdam Optimization for Latents**:
    - **Function**: Treats "latent variables for each sample" as learnable free parameters, making any Riemannian manifold (Sphere, PoincareBall, Lorentz, SPD, UpperHalf, Stiefel, ProductManifold...) immediately available.
    - **Mechanism**: Abandons amortized inference. Each training step applies the Riemannian gradient $\nabla_z^{\mathcal{R}}\mathcal{L}=G(z)^{-1}\nabla_z^E\mathcal{L}$ to the latents, updated via retraction (usually the exponential map): $z^{(t+1)}=R_{z^{(t)}}(-\eta\,\nabla_{z^{(t)}}^{\mathcal{R}}\mathcal{L})$. RiemannianAdam maintains adaptive directions in the tangent space, ensuring convergence speeds similar to Adam while strictly remaining on the manifold. For compact manifolds, the prior is simply $1/\text{Vol}(\mathcal{M})$ (a constant not affecting gradients); for non-compact ones, wrapped/Riemannian normal is used.
    - **Design Motivation**: The encoder is the root cause of manifold density approximation issues because $q_\phi(z|x)$ must be a tractable probability distribution on the surface. Goldberg-DGD proved that the encoder-less MAP paradigm works in Euclidean space; this work lifts it to manifolds, bypassing all density approximations. An unexpected benefit is that heterogeneous product manifolds like ProductManifold are also immediately usable since RGD does not require a specific prior for each component.

2.  **Geometric-aware Input Noise Regularization**:
    - **Function**: Automatically aligns the decoder's local Jacobian with the manifold metric, encouraging "geometrically similar points" to be mapped to similar outputs.
    - **Mechanism**: During training, noise $\epsilon\sim\mathcal{N}(0, \sigma^2 G^{-1}(z))$ is injected into the latent space (the covariance uses the inverse manifold metric, making noise weaker in directions with larger metrics). It is injected via the exponential map $z'=\text{Exp}_z(\epsilon)\approx z+\epsilon$. Following Bishop's (1995) second-order Taylor expansion derivation, the equivalent regularization term is: $\mathbb{E}_\epsilon[L(z')]\approx L(z)+\sigma^2\,\text{Tr}(J(z)^\top G^{-1}(z) J(z))$, where $J(z)=\partial_z f$. The second term is a Jacobian norm penalty weighted by the manifold metric.
    - **Design Motivation**: Euclidean Gaussian noise on a curved surface over-penalizes the model in directions with large metrics and under-penalizes it in directions with small metrics. Scaling noise with $G^{-1}(z)$ aligns isotropic regularization with the manifold geometry. It degrades to approximately isotropic on homogeneous manifolds (Sphere) and becomes locally adaptive on hyperbolic manifolds with non-uniform curvature. Compared to the second-order curvature regularization of Lee & Park (2023) (involving complex Hessian-vector products), RGD's scheme requires only one Jacobian calculation, offering much better scalability.

3.  **Unified Framework Supporting Any Riemannian Manifold + Product Combinations**:
    - **Function**: Allows users to express prior knowledge as a choice of manifold, leaving the rest to the framework without hand-writing ELBOs or density approximations.
    - **Mechanism**: Directly reuses all manifolds implemented in `geoopt` (Euclidean, Sphere, Stereographic, PoincareBall, Lorentz, SPD, Stiefel, UpperHalf, ProductManifold...). These manifolds only need to provide an exponential map/retraction and a metric. Product manifolds are defined as $\mathcal{M}=\mathcal{M}_1\times\cdots\times\mathcal{M}_K$ where the metric is the direct sum, naturally supporting heterogeneous requirements like "some dimensions are spherical and some are hyperbolic."
    - **Design Motivation**: Previous works required re-deriving priors and approximations for each new manifold, preventing researchers from quickly comparing which geometry fits their data. RGD transforms the manifold from an **algorithmic hypothesis** into a **configuration item**, making hypothesis-based exploration genuinely feasible; cell cycle data can be tested across torus, sphere, and Euclidean configurations with a single switch.

### Loss & Training
The objective is the negative posterior from Equation (10): $\mathcal{L}=\sum_i(-\log p_\theta(x_i|z_i)-\log p(z_i))-\log p(\theta)$. $\theta$ is updated with Adam, and $Z$ with RiemannianAdam alternatingly. Geometric regularization is implemented by adding $\mathcal{N}(0,\sigma^2 G^{-1})$ noise to the latents (mapped back to the manifold via retraction). Reconstruction loss is chosen based on data properties (continuous → Gaussian/MSE; discrete → categorical). There is no KL term, no ELBO, and no Monte Carlo estimation of normalization constants.

## Key Experimental Results

### Main Results
Three real/synthetic biological datasets: (a) Synthetic branching diffusion tree (7 levels, $d=50$, 6350 samples) → naturally suited for hyperbolic manifolds; (b) Human mitochondrial DNA, 67k sequences + haplogroup labels → hyperbolic suited for phylogeny; (c) Cell cycle scRNA-seq, 5367 cells × 189 genes → periodic, suited for torus.

| Dataset | Task | Best Geometry | Key Metrics | Notes |
|--------|------|----------|----------|------|
| Cell cycle scRNA-seq | Correlation: Phase vs Latent Distance | Sphere $\mathbb{S}^2$ | Train Pearson 0.58, Test 0.60; Reconstruction MAE 0.31 | Outperforms $\mathcal{S}$-VAE / $\Delta$VAE |
| Branching diffusion | Correlation: Tree vs Latent Geodesic Dist | Lorentz $\mathbb{H}^2$ ($\sigma=1.0$) | Train Pearson 0.81, Test 0.80 | $\mathcal{P}$-VAE only 0.68 |
| hmtDNA haplogroup classification | Downstream 24/128-way Logistic Reg. Acc | Hyperbolic $\mathbb{H}^2_{\sigma=0.5}$ | 24-way LR 0.70 / XGB 0.85; 128-way LR 0.43 | Broadly beats Euclidean and $\mathcal{P}$-VAE |

### Ablation Study

| Configuration | Key Findings |
|------|----------|
| Geometric noise $\sigma$ sweep 0→2.6 (Hyperbolic, Branching) | Correlation rises rapidly until $\sigma\approx 0.9$, after which excessive noise prevents the decoder from maintaining local accuracy → local-global trade-off. |
| Hyperbolic $\mathbb{H}^2$ vs Euclidean / Sphere (Branching) | $\mathbb{H}^2$ significantly recovers tree topology (Pearson 0.81 vs 0.53/0.56). |
| UMAP (Branching) | Completely fails to show tree topology → validates RGD's significant advantage in interpretability. |
| Comparison with $\mathcal{P}$-VAE/$\mathcal{S}$-VAE/$\Delta$VAE | VAE baselines suffer numerical collapse in high dimensions; RGD remains usable. |
| Generative Discrimination (XGBClassifier) | RGD Sphere 0.58 = $\mathcal{S}$-VAE Sphere 0.58 < $\Delta$VAE 0.62 (closer to 0.5 is more realistic). |

### Key Findings
- **Encoder-less is more stable for high-dimensional data**: In settings like cell cycle whole genome (high-dimensional) + ProductManifold, $\mathcal{S}$/$\Delta$/$\mathcal{P}$-VAE collapse due to normalizing constants or Monte Carlo sampling; RGD remains trainable and provides reasonable reconstructions.
- **Geometric noise is critical for learning geometry**: Without it, the hyperbolic model's distance correlation approaches the Euclidean baseline, indicating that the choice of manifold provides space, but regularization is what forces the decoder to align with the geometry.
- **Manifold selection = Hypothesis testing**: Branching diffusion data uses hyperbolic latents to restore tree structure instantly, whereas it's invisible in UMAP, proving that a reasonable geometric prior is more interpretable than general dimensionality reduction.
- **Generative quality is on par with VAEs**: In discrimination tests, RGD's synthetic samples show the same separability as $\mathcal{S}$-VAE (0.58), indicating no loss in generation fidelity from discarding the encoder.

## Highlights & Insights
- **The approach is clean**: Lifting DGD from Euclidean space to any Riemannian manifold in one step bypasses the entire density approximation chain of Riemannian VAEs—a rare "subtractive contribution" that offers immense scalability.
- **Actionable geometric noise derivation**: Using the classic conclusion of Bishop's noise-regularization equivalence with one layer of inverse metric scaling provides metric-aware Jacobian penalties on any manifold. The formula is one line, and engineering implementation is just a covariance swap.
- **Manifolds as hypotheses rather than algorithmic burdens**: Scientists can quickly test tori, spheres, hyperbolic, or SPD geometries on the same dataset to pick the most suitable prior, directly improving productivity in computational biology and phylogenetics.
- **Transferability to LLM/VLM latent representations**: Routing/expert representations of LLMs on Stiefel or Sphere manifolds, and vision-language alignment on ProductManifolds, can all be integrated into this framework.

## Limitations & Future Work
- The lack of an amortized encoder means new samples must have their **latents re-optimized** (per-sample MAP steps) to be encoded, making online inference slower than VAEs; an amortized warm-start could be added.
- The $\sigma$ in geometric regularization is coupled with manifold curvature and requires manual sweeping; automatic scheduling is an obvious next step.
- The prior on non-compact manifolds still uses wrapped/Riemannian normal, theoretically introducing slight bias, though the authors treat generation quality as a sanity check.
- Experiments were conducted on biological/synthetic data and haven't been validated on large-scale NLP or vision tasks (though principles apply).
- Dimension assignment in ProductManifold requires manual configuration, which might be difficult for complex data; automatic geometry selection is a future direction.

## Related Work & Insights
- **vs $\mathcal{S}$-VAE / $\mathcal{P}$-VAE / $\Delta$VAE**: These require hand-written densities for each manifold; RGD bypasses all of this and is significantly more stable in high dimensions.
- **vs DGD (Schuster & Krogh 2023)**: DGD is Euclidean encoder-less; RGD is its Riemannian extension, generalizing free-parameter MAP to any manifold.
- **vs Curvature Regularization (Lee & Park 2023)**: They use second-order curvature for "global flattening"; RGD uses first-order Jacobian regularization + inverse metric noise, reducing computational cost by an order of magnitude.
- **vs UMAP / Isomap**: Classic non-linear dimensionality reduction is for visualization only, does not learn generative models, and does not explicitly specify manifolds; RGD provides visualization, generation, and hypothesis testing.
- **Cross-task Inspirations**: Molecular representations (SE(3) manifold), robot pose representations (SO(3), SE(3)), and belief space representations in RL can all utilize the "manifold + free latent + Riemannian Adam + metric noise" combination.

## Rating
- Novelty: ⭐⭐⭐⭐ "Discarding the encoder to upgrade DGD to any Riemannian manifold" and "inverse-metric noise" are clean, independent contributions with significant framework-level simplification.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets cover periodic, tree, and hierarchical geometries, including multi-baseline comparisons, noise sweeps, downstream tasks, and generative discrimination; lacks large-scale NLP/vision validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Derivations are clear, appendices are thorough, and introduction to prior work is complete. Excellent transition between theory and engineering.
- Value: ⭐⭐⭐⭐ Lowers the "algorithmic threshold" of manifolds to a "configuration item" in geometric DL, significantly boosting productivity for representation learning and computational biology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing](../../NeurIPS2025/interpretability/uncovering_graph_reasoning_in_decoder-only_transformers_with_circuit_tracing.md)
- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)
- [\[ICML 2026\] Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](interpretable_self-supervised_learning_via_representer_landmarks_and_nyström_app.md)
- [\[ICML 2026\] Courtroom Analogy: New Perspective on Uncertainty-Aware Classification](courtroom_analogy_new_perspective_on_uncertainty-aware_classification.md)
- [\[ICML 2026\] Verified SHAP: Provable Bounds for Exact Shapley Values in Neural Networks](verified_shap_provable_bounds_for_exact_shapley_values_of_neural_networks.md)

</div>

<!-- RELATED:END -->
