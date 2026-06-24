---
title: >-
  [Paper Note] A Connection Between Score Matching and Local Intrinsic Dimension
description: >-
  [NeurIPS 2025][Image Generation][score matching] This paper proves that the lower bound of the denoising score matching (DSM) loss is precisely the local intrinsic dimension (LID) of the data manifold, thereby establishing the DSM loss itself as an efficient LID estimator—requiring neither gradient computation nor multiple forward passes. On Stable Diffusion 3.5, this approach reduces peak memory usage to approximately 60% of FLIPD while yielding more stable estimates under q…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "score matching"
  - "local intrinsic dimension"
  - "diffusion model"
  - "LID estimation"
  - "denoising loss"
date: 2026-05-08
content_hash: 65527c4b585ffb20
---

# A Connection Between Score Matching and Local Intrinsic Dimension

**Conference**: NeurIPS 2025
**arXiv**: [2510.12975](https://arxiv.org/abs/2510.12975)  
**Code**: None  
**Area**: Diffusion Models / Generative Model Theory
**Keywords**: score matching, local intrinsic dimension, diffusion model, LID estimation, denoising loss

## TL;DR
This paper proves that the lower bound of the denoising score matching (DSM) loss is precisely the local intrinsic dimension (LID) of the data manifold, thereby establishing the DSM loss itself as an efficient LID estimator—requiring neither gradient computation nor multiple forward passes. On Stable Diffusion 3.5, this approach reduces peak memory usage to approximately 60% of FLIPD while yielding more stable estimates under quantization.

## Background & Motivation

**Background**: Local intrinsic dimension (LID) is a fundamental quantity in signal processing and learning theory, governing data compressibility and learning efficiency. Estimating LID for high-dimensional complex data has historically been challenging. Recent work has shown that diffusion models can capture LID through spectral analysis of score estimates or noise-sensitivity of density estimates, giving rise to methods such as FLIPD and Normal Bundle.

**Limitations of Prior Work**: Existing parametric LID estimation methods are computationally expensive—FLIPD requires divergence computation (involving gradients), while the Normal Bundle method demands a large number of forward passes, limiting applicability in memory- and compute-constrained settings.

**Key Challenge**: There is an inherent trade-off between LID estimation accuracy and computational efficiency: accurate methods are expensive, while simple methods lack precision.

**Goal**
   - Is there a precise mathematical relationship between DSM loss and LID?
   - Can DSM loss be used directly as a more efficient LID estimator?

**Key Insight**: The analysis departs from information theory (entropy power inequality), examining how denoising error behaves differently along the tangent and normal spaces of the manifold, and proving that the lower bound of DSM loss equals LID.

**Core Idea**: Noise along the tangent directions of the manifold is unpredictable (contributing MSE ≥ 1 per dimension), while noise along the normal directions can be perfectly denoised (contributing MSE = 0). Consequently, the lower bound of the DSM loss equals the dimension of the tangent space, i.e., the LID.

## Method

### Overall Architecture
This work combines **theory and empirical validation**. On the theoretical side, two theorems are established: (1) DSM loss ≥ LID (Theorem 3.1); (2) ISM loss ≥ $-(n-d)$ (Theorem 3.3), where $n-d$ is the dimension of the normal space. Empirically, the competitiveness of the DSM loss as a LID estimator is validated on synthetic manifold benchmarks and on Stable Diffusion 3.5/2.

### Key Designs

1. **Theorem 3.1: Lower Bound of DSM Loss**

    - Function: Proves that $\mathbb{E}_x[\mathcal{L}_{\text{DSM}}(x, \sigma, \theta)] \geq d$, where $d$ is the intrinsic dimension of the manifold.
    - Mechanism: The noise $\epsilon$ is decomposed into a $d$-dimensional tangent component and an $(n-d)$-dimensional normal component. Along the normal directions, the diffusion model can perfectly denoise (since the manifold is a Dirac delta in the normal directions), so the conditional entropy $h(\epsilon_i | \tilde{x}) = -\infty$ and the MSE lower bound is 0. Along the tangent directions, the conditional distribution of $\epsilon_i$ given $\tilde{x}$ remains close to a standard Gaussian (since the manifold has a continuous distribution in the tangent directions); the conditional entropy is approximately $\frac{1}{2}\log 2\pi e$, and the entropy power inequality yields an MSE lower bound of 1. With $d$ tangent directions each contributing at least 1, the total DSM loss $\geq d$.
    - Design Motivation: This lower bound is achieved with equality under an optimal score model, so the DSM loss directly estimates LID. Intuitively, the diffusion model "knows" the dimensionality of the manifold on which the data resides—it can remove noise in the normal directions but not in the tangent directions.

2. **Theorem 3.3: Lower Bound of ISM Loss**

    - Function: Proves that the ISM loss $\geq -(n-d)$, i.e., the lower bound is the negative of the normal space dimension.
    - Mechanism: Using properties of the optimal score function $s_{\theta^*} = \nabla \log p$: along the normal directions the divergence equals $-\sigma^{-2}$ (since the distribution is Gaussian in the normal directions), while along the tangent directions the divergence is 0 (since the distribution is locally uniform in the tangent directions).
    - Design Motivation: This reveals a close relationship between FLIPD (the current state-of-the-art LID estimator) and ISM loss—specifically, FLIPD = ISM loss + score norm + $n$—confirming that FLIPD also admits LID as a lower bound.

3. **Error Bundle (EB) Method and its Connection to Normal Bundle**

    - Function: Unifies the DSM loss and the Normal Bundle method at the level of spectral analysis.
    - Mechanism: The Normal Bundle method estimates the normal space dimension by counting non-zero singular values of the noise prediction matrix via SVD. The authors propose an Error Bundle variant—analyzing the Gram matrix $C' = B^TB/m$ of the error matrix $B$—whose trace equals the DSM loss, while the number of non-zero eigenvalues is at most $d$. The key advantage is that the DSM loss (trace) is accurate even with very few samples (e.g., $m=8$), whereas the NB/EB counting approach requires $m$ to be at least as large as the LID or normal space dimension.

### Loss & Training
- Experiments use a DiT architecture (patch=4, hidden=128, 16 heads, 8 layers) and an MLP trained with a flow matching objective.
- Training runs for 50,000 batches with a batch size of 100.
- LID estimation averages denoising errors over 8 Gaussian noise samples.

## Key Experimental Results

### Main Results: Synthetic Manifold Benchmark (MAE↓)

| Method | HyperSphere (d=16, n=64) | HyperTP (d=128, n=256) | Nonlinear (d=32, n=128) | Avg. MAE |
|------|------------------------|----------------------|----------------------|---------|
| DSM Loss (DiT, σ=0.05) | 2.58 | **4.48** | **2.15** | **3.22** |
| FLIPD (DiT, σ=0.05) | **0.70** | 1.07 | 12.87 | 4.32 |
| ESS (k=100) | **0.32** | 2.47 | 1.37 | 7.12 |
| MLE (k=100) | 3.94 | 84.89 | 13.55 | 22.01 |

DSM Loss achieves the best overall performance under the DiT architecture (avg. MAE = 3.22 vs. 4.32 for FLIPD). FLIPD performs better on low-dimensional simple manifolds but degrades significantly on high-dimensional complex ones.

### Ablation Study: Scalability (Memory)

| Method | Peak GPU Memory (SD-3.5, 10 images) |
|------|------------------------------|
| FLIPD | ~12 GB |
| **DSM Loss** | **~7 GB (~60%)** |

### Quantization Robustness

| Quantization Level | FLIPD MAE from FP32 | DSM Loss MAE from FP32 |
|---------|--------------------|-----------------------|
| float16 | Higher | **Lower** |
| bfloat16 | Highest | **Lower** |

DSM Loss exhibits smaller LID estimation deviation under quantization, as it does not rely on gradient computation (quantization errors accumulate more severely in gradients).

### Key Findings
- **DSM Loss outperforms FLIPD across all architecture and $\sigma$ combinations**: lower average MAE, with a particularly notable advantage on high-curvature manifolds.
- **Significant memory efficiency**: peak memory is ~60% of FLIPD due to the absence of gradient computation.
- **SD-3.5 experiments**: DSM Loss and FLIPD LID estimates are highly correlated, but DSM Loss is systematically lower (providing a tighter lower-bound estimate).
- **FLIPD on SD-2 produces negative values at high noise levels** (invalid estimates), whereas the flow matching parameterization of SD-3.5 avoids this issue.

## Highlights & Insights
- **Elegant theoretical result**: DSM loss = LID is a remarkably concise identity that directly connects the generative model training objective to data geometry. From a practical standpoint, LID information is obtained "for free" after training a diffusion model.
- **Intuition from tangent/normal decomposition**: Noise in the normal direction can be perfectly removed (the manifold resembles a delta function), while noise in the tangent direction is irreducible (the manifold resembles a uniform distribution). This decomposition is particularly elegant.
- **Strong practical utility**: Only forward passes with averages over 8 noise samples are required—no gradients or SVD—making the method directly applicable to large pretrained models such as SD-3.5.
- **New interpretation of the DSM constant $C_{\text{DSM}}$**: The constant term in the training loss (typically ignored) actually encodes the average intrinsic dimension of the data manifold, endowing the "optimal loss value" during model training with geometric meaning.

## Limitations & Future Work
- **Strong theoretical assumptions**: The analysis requires $\sigma$ to be sufficiently small so that manifold curvature is negligible and the density is locally uniform—conditions that may not hold for the noise levels used in practice.
- **Lower bound only**: Theoretically, DSM loss $\geq d$ with equality at optimality; in practice, trained models may not reach the optimum, causing DSM loss to overestimate LID.
- **No knee search**: LID varies with $\sigma$ (Figure 3a), and the authors do not search for an optimal $\sigma$ as FLIPD does, which may affect estimation accuracy.
- **Single-GPU experiments only**: Memory advantages in large-batch and distributed settings have not been evaluated.
- **Validation limited to images**: LID estimation performance in other modalities (text, protein structure, etc.) remains unexplored.

## Related Work & Insights
- **vs. FLIPD (Kamkari et al.)**: FLIPD employs the Fokker-Planck equation and divergence estimation, requiring gradient computation; DSM Loss requires only forward passes, offering superior memory efficiency and quantization robustness, though with slightly lower accuracy on low-dimensional simple manifolds.
- **vs. Normal Bundle (Stanczuk et al.)**: The Normal Bundle method requires at least as many samples as the normal space dimension for SVD; DSM Loss needs only an average over 8 samples, offering far greater scalability.
- **vs. non-parametric methods (MLE, TwoNN, ESS)**: Non-parametric methods are effective in low dimensions but fail on high-dimensional complex manifolds (MAE > 20); parametric methods comprehensively outperform non-parametric ones.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The theoretical contribution is highly elegant; directly connecting DSM loss to LID represents a genuinely novel insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Coverage across synthetic manifolds, SD-3.5/SD-2, and quantization experiments is comprehensive, though limited in scale.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear and rigorous, with well-articulated intuitions.
- Value: ⭐⭐⭐⭐ — High theoretical value and practical utility (free LID estimation), though the application scope is relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation](../../ICML2026/image_generation/local_hessian_spectral_filtering_for_robust_intrinsic_dimension_estimation.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](../../ICCV2025/image_generation/balanced_image_stylization_with_style_matching_score.md)
- [\[NeurIPS 2025\] FerretNet: Efficient Synthetic Image Detection via Local Pixel Dependencies](ferretnet_efficient_synthetic_image_detection_via_local_pixel_dependencies.md)
- [\[NeurIPS 2025\] PixPerfect: Seamless Latent Diffusion Local Editing with Discriminative Pixel-Space Refinement](pixperfect_seamless_latent_diffusion_local_editing_with_discriminative_pixel-spa.md)

</div>

<!-- RELATED:END -->
