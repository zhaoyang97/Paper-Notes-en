---
title: >-
  [Paper Note] Fast Estimation of Wasserstein Distances via Regression on Sliced Wasserstein Distances
description: >-
  [ICLR 2026][3D Vision][Wasserstein distance] Leveraging the mathematical property that standard Sliced Wasserstein (SW) distances provide lower bounds and lifted SW distances provide upper bounds for the Wasserstein dist…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Wasserstein distance"
  - "Sliced Wasserstein"
  - "optimal transport"
  - "linear regression"
  - "point cloud classification"
date: 2026-05-08
content_hash: 59156d6c57f0cece
---

# Fast Estimation of Wasserstein Distances via Regression on Sliced Wasserstein Distances

**Conference**: ICLR 2026
**arXiv**: [2509.20508](https://arxiv.org/abs/2509.20508)  
**Code**: [Available](https://github.com/hainn2803/Regression-Wasserstein)  
**Area**: 3D Vision
**Keywords**: Wasserstein distance, Sliced Wasserstein, optimal transport, linear regression, point cloud classification

## TL;DR

Leveraging the mathematical property that standard Sliced Wasserstein (SW) distances provide lower bounds and lifted SW distances provide upper bounds for the Wasserstein distance, this paper constructs a minimal linear regression model (the RG framework) that estimates Wasserstein distances with high accuracy using only a small number of exact Wasserstein labels as supervision, comprehensively outperforming the Transformer-based method Wasserstein Wormhole in low-data regimes.

## Background & Motivation

**Background**: The Wasserstein distance is a central metric in optimal transport (OT) theory, widely used in generative modeling, computational biology, 3D point cloud processing, and dataset comparison due to its ability to capture geometric structure between distributions. In practice, large numbers of distribution pairs must be evaluated repeatedly—e.g., computing distances between every test sample and all training samples in k-NN classification, or computing pairwise distances within batches during Wormhole training.

**Limitations of Prior Work**: Exact Wasserstein distance computation has complexity $O(n^3 \log n)$ via linear programming, making it infeasible at scale. Existing acceleration strategies each have drawbacks: Sinkhorn regularization reduces this to $O(n^2)$ via entropic regularization but introduces systematic approximation bias; deep learning methods such as Wasserstein Wormhole encode distributions into Euclidean embedding spaces using Transformers, but require large amounts of training data (hundreds to thousands of exact Wasserstein pairs) and degrade sharply in low-data regimes; Sliced Wasserstein reduces the problem to one-dimensional projections at $O(n \log n)$, but is inherently only a lower bound on the true Wasserstein distance and thus lacks sufficient accuracy.

**Key Challenge**: There is a fundamental speed–accuracy trade-off—fast methods (SW) are insufficiently accurate, accurate methods (exact WD) are too slow, and intermediate deep learning methods (Wormhole) require large training datasets and substantial computational resources.

**Goal**: (1) How can Wasserstein distances be accurately recovered from SW distances without neural networks? (2) How can a high-quality estimator be trained with as few as 10–100 exact Wasserstein labels? (3) Can the approach be combined with existing deep methods (Wormhole) to achieve further acceleration?

**Key Insight**: The authors identify a previously overlooked mathematical structure: the standard SW family (SW, Max-SW, EBSW) provides lower bounds for the Wasserstein distance, while the lifted SW family (PW, Min-SWGG, EST) provides upper bounds, and these bounds satisfy a strict partial order. Since the true Wasserstein distance is sandwiched between these bounds, forming a linear combination of upper and lower bounds to regress the true value is both natural and theoretically motivated.

**Core Idea**: Six SW distance variants (3 lower bounds + 3 upper bounds) are used as features, and the true Wasserstein distance is estimated via least-squares linear regression—trained in closed form, requiring no iteration and no neural networks.

## Method

### Overall Architecture

The RG (Regression) framework follows an extremely simple pipeline: given any pair of probability distributions $(\mu, \nu)$, the framework first computes their values under $K$ SW distance variants $S_p^{(1)}, \ldots, S_p^{(K)}$ (each in $O(n \log n)$), then applies pre-trained linear weights $\omega_1, \ldots, \omega_K$ via weighted summation to produce the estimated Wasserstein distance. During training, only $M_0 \ll N$ pairs are randomly sampled from all $N$ distribution pairs; exact Wasserstein distances are computed for these samples (costly but performed only once), and weights are obtained in closed form via least squares.

### Key Designs

1. **Six SW Distance Variants as Feature Space**:

    - **Function**: Provide multiple reference anchors approximating the Wasserstein distance from both above and below.
    - **Mechanism**: The lower-bound family includes SW (expectation over uniformly sampled projection directions), Max-SW (the direction maximizing projected distance, $\max_{\theta} W_p(P_\theta \sharp \mu, P_\theta \sharp \nu)$), and EBSW (energy-function-weighted projection directions); the upper-bound family includes PW (expectation of lifted cost over uniform directions), Min-SWGG (the direction minimizing lifted cost), and EST (energy-weighted lifted cost). They satisfy a strict partial order: $\text{SW} \leq \text{EBSW} \leq \text{Max-SW} \leq W_p \leq \text{Min-SWGG} \leq \text{EST} \leq \text{PW}$.
    - **Design Motivation**: A single SW variant can only approximate from one side; combining multiple upper and lower bounds allows two-sided bracketing of the true value, substantially improving fitting accuracy. This is why RG-seo (using all 6 variants) outperforms RG-s (using only 2).

2. **Unconstrained Linear Model with Closed-Form Solution**:

    - **Function**: Directly regress the Wasserstein distance using the simplest possible linear model.
    - **Mechanism**: The model is $W_p(\mu,\nu) = \sum_{k=1}^K \omega_k S_p^{(k)}(\mu,\nu) + \varepsilon$, with the least-squares estimator given in closed form as $\hat{\omega}_{LSE} = (\hat{S}^\top \hat{S})^{-1} \hat{S}^\top \hat{W}$, where $\hat{S} \in \mathbb{R}^{M \times K}$ is the SW distance matrix and $\hat{W} \in \mathbb{R}^M$ is the exact Wasserstein vector. Geometrically, this is equivalent to the $L_2$ projection of the Wasserstein vector onto the linear subspace spanned by the SW distance vectors.
    - **Design Motivation**: No iterative training, no hyperparameter tuning (learning rate, epochs, etc.)—a single matrix operation yields the solution. When sufficient data is available ($M_0 \geq 50$), the unconstrained model consistently achieves higher $R^2$ than the constrained variant.

3. **Constrained Linear Model (Midpoint Method)**:

    - **Function**: Exploit prior knowledge of upper and lower bounds to halve the number of parameters, suited for extremely low-data regimes.
    - **Mechanism**: Each (lower bound, upper bound) pair is combined as a convex weight $\omega_k \cdot SL^{(k)} + (1-\omega_k) \cdot SU^{(k)}$ with $0 \leq \omega_k \leq 1$. For $K=1$, a closed-form solution exists: $\hat{\omega} = \frac{\mathbb{E}[(SU-SL)(SU-W)]}{\mathbb{E}[(SU-SL)^2]}$; for $K>1$, quadratic programming is used.
    - **Design Motivation**: The constrained model encodes the inductive bias that the true value lies between upper and lower bounds. With as few as $M_0=10$ labels, it is more stable than the unconstrained model, which has more free parameters and is prone to overfitting.

### Loss & Training

Training minimizes the least-squares (MSE) loss with a closed-form solution requiring no iterative optimization. Total computational cost comprises two components: (1) a one-time training cost of computing exact Wasserstein distances for $M_0$ sample pairs ($O(M_0 n^3 \log n)$); (2) inference cost of computing $K$ SW distances per distribution pair ($O(KLn(\log n + d))$) followed by a linear combination ($O(K)$), far cheaper than exact Wasserstein computation.

The authors also propose **RG-Wormhole**, a hybrid approach: RG weights are calibrated on a small sample set, and all Wasserstein distance calls within Wormhole training (pairwise batch distances, decoder reconstruction loss) are replaced by RG surrogates, with all other architectural, optimizer, and scheduling choices unchanged. This reduces the per-step cost from $O(n^3)$ (Wasserstein) to $O(n \log n)$ (SW), and whereas Wormhole training time grows nearly exponentially with batch size, RG-Wormhole scales nearly linearly.

## Key Experimental Results

### Main Results: ShapeNetV2 Point Cloud k-NN Classification

Evaluated on 10-class ShapeNetV2 point clouds with 500 training samples; RG weights are estimated from only $M_0=10$ sample pairs.

| Method | $R^2$ | k=1 | k=3 | k=5 | k=10 | k=15 |
|--------|-------|-----|-----|-----|------|------|
| WD (exact) | — | 83.6% | 83.5% | 84.2% | 82.9% | 79.2% |
| SW (lower bound only) | — | 72.2% | — | — | — | — |
| RG-s (SW+PW) | 0.868 | 82.1% | 81.7% | 80.8% | 79.4% | 75.5% |
| RG-e (EBSW+EST) | 0.926 | 82.5% | 82.2% | 80.9% | 79.6% | 75.7% |
| RG-se (4 SW variants) | 0.935 | 82.5% | 82.2% | 82.6% | 81.9% | 76.5% |
| RG-seo (all 6 variants) | 0.937 | 82.8% | 83.3% | **83.5%** | 82.3% | 77.9% |

RG-seo achieves 83.5% at k=5, nearly matching the exact WD result of 84.2%, while the SW lower bound alone achieves only 72.2%.

### Low-Data Comparison with Wormhole ($M_0=100$)

$R^2$ comparison across 4 datasets (dimensionality ranging from 2D to 2500D) with 100 training samples:

| Method | MNIST (2D) | ShapeNetV2 (3D) | MERFISH (254D) | scRNA-seq (2500D) |
|--------|-----------|----------------|----------------|-------------------|
| Wormhole | 0.28 | 0.65 | −3.6 | 0.04 |
| RG-s constrained | 0.84 | 0.88 | 0.91 | 1.00 |
| RG-e constrained | 0.86 | 0.90 | 0.92 | 1.00 |
| RG-o constrained | 0.77 | 0.66 | 0.75 | 0.99 |
| RG-s unconstrained | 0.93 | 0.94 | 0.96 | 0.99 |
| RG-se unconstrained | **0.93** | **0.95** | **0.98** | **1.00** |
| RG-seo unconstrained | 0.93 | 0.95 | 0.97 | 0.99 |

Wormhole completely fails on MERFISH ($R^2 = -3.6$), while RG-se unconstrained achieves 0.98. The RG framework dominates across all datasets.

### Ablation Study

| Dimension | Finding |
|-----------|---------|
| Constrained vs. unconstrained | The unconstrained model is consistently stronger when $M_0 \geq 50$ (greater degrees of freedom); the constrained model is more stable at extremely low data ($M_0=10$) due to fewer parameters. |
| Single vs. multiple variants | RG-seo (6 features) > RG-se (4) > RG-s/RG-e (2) > RG-o (2); more SW variants provide richer information. |
| Poor performance of RG-o | Max-SW and Min-SWGG both rely on optimization, yielding high-variance empirical estimates that destabilize regression. |
| RG-Wormhole speedup | Wormhole training time grows near-exponentially with batch size (WD cost is $O(B^2 n^3)$); RG-Wormhole scales near-linearly. Embedding, reconstruction, and interpolation quality are consistent with the original. |

### Key Findings

- **10 sample pairs suffice**: The RG-s constrained model achieves $R^2 > 0.8$ with only 10 training pairs; Wormhole is essentially unusable at the same data scale.
- **Consistent across dimensionalities**: High $R^2$ is maintained from 2D (MNIST point clouds) to 2500D (scRNA-seq gene expression), validating the linear approximation assumption across dimensions.
- **RG-Wormhole as a drop-in replacement**: All Wormhole capabilities (embedding, interpolation, reconstruction, barycenter computation) are preserved; decoded 3D shapes are visually indistinguishable from the original, while training speed is substantially improved.

## Highlights & Insights

- **Surprising effectiveness of extreme simplicity**: The core model is linear regression with a closed-form solution—no neural networks—yet it comprehensively outperforms the Transformer-based Wormhole on 4 datasets. This demonstrates that when a problem possesses strong mathematical structure (the partial order of upper and lower bounds), simple models can substantially outperform complex ones.
- **Elegant exploitation of two-sided bounds**: The most central insight is that the lower-bound SW family and the upper-bound lifted SW family jointly form an "informationally complete" feature space, within which the true Wasserstein distance is approximately a linear function. This theory-driven feature engineering proves more efficient than end-to-end learning.
- **Plug-and-play design of RG-Wormhole**: Only the distance computation is replaced; the architecture and training pipeline remain unchanged. This modular substitution paradigm is generalizable to any deep learning method that employs Wasserstein distance as a subroutine.
- **Inductive bias of the constrained model**: The convex combination form $\omega + (1-\omega)$ automatically ensures that estimates fall between upper and lower bounds, constituting an elegant form of regularization.

## Limitations & Future Work

- **Limitations of the linear assumption**: The true relationship between SW distances and the Wasserstein distance may be nonlinear—the authors acknowledge that kernel regression may yield better results. However, the linear model already achieves $R^2 > 0.9$ across all experiments, suggesting that the nonlinear component is small.
- **Meta-distribution assumption**: Training and test distribution pairs must be drawn from the same meta-distribution; cross-domain generalization has not been validated. For instance, it is unclear whether RG weights calibrated on ShapeNetV2 transfer to ModelNet40.
- **No embedding space**: The standalone RG framework only estimates distances and cannot perform interpolation, reconstruction, or other operations requiring an embedding space; these capabilities require integration with Wormhole, limiting independent applicability.
- **Selection of projection count $L$**: Monte Carlo estimation of SW distances requires specifying the number of projection directions $L$; the optimal $L$ varies across datasets and dimensionalities, and no adaptive selection strategy is provided.

## Related Work & Insights

- **vs. Wasserstein Wormhole**: Wormhole maps distributions into Euclidean space using a Transformer encoder, requiring large numbers of exact WD pairs for training. The RG framework requires no neural networks and consistently outperforms Wormhole when $M_0 \leq 200$. However, Wormhole's embedding space supports interpolation and reconstruction, capabilities that RG must inherit through integration.
- **vs. Sinkhorn distances**: Sinkhorn reformulates the LP as matrix scaling via entropic regularization, reducing complexity to $O(n^2/\varepsilon^2)$, but each call remains $O(n^2)$. After calibration, RG inference requires only $O(Ln \log n)$ and introduces no regularization bias.
- **vs. low-rank OT**: Low-rank approximation methods exploit the low-rank structure of optimal transport plans, but still require solving each pair individually. RG's advantage lies in its "train once, reuse repeatedly" property—once weights are calibrated, new distribution pairs require only SW computation.
- **Broader Implications**: The upper-lower bound regression paradigm generalizes to other computationally expensive distance metrics: whenever fast upper and lower bound approximations of a target metric can be identified, a similar linear regression framework can serve as a surrogate estimator.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The core idea of regressing on SW upper and lower bounds is simple yet novel, with clear mathematical motivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four datasets spanning 2D to 2500D, covering classification, visualization, embedding, interpolation, reconstruction, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Notation is clear and theoretical derivations are rigorous, though preliminary material occupies a disproportionately large portion of the paper.
- **Value**: ⭐⭐⭐⭐⭐ — A plug-and-play Wasserstein acceleration framework with a closed-form solution requiring zero hyperparameter tuning, highly amenable to practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Streaming Sliced Optimal Transport](../../ICML2026/3d_vision/streaming_sliced_optimal_transport.md)
- [\[CVPR 2026\] Fast SceneScript: Fast and Accurate Language-Based 3D Scene Understanding via Multi-Token Prediction](../../CVPR2026/3d_vision/fast_scenescript_fast_and_accurate_language-based_3d_scene_understanding_via_mul.md)
- [\[ICML 2026\] Fast-SAM3D: 3Dfy Anything in Images but Faster](../../ICML2026/3d_vision/fast-sam3d_3dfy_anything_in_images_but_faster.md)
- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](../../ICCV2025/3d_vision/unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [\[ICCV 2025\] CF³: Compact and Fast 3D Feature Fields](../../ICCV2025/3d_vision/cf3_compact_and_fast_3d_feature_fields.md)

</div>

<!-- RELATED:END -->
