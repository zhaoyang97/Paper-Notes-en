---
title: >-
  [Paper Note] Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation
description: >-
  [ICML 2026][Local Intrinsic Dimension] This paper proposes LHSD, which applies a Hill-type spectral filter to the log-density Hessian of a score model…
tags:
  - "ICML 2026"
  - "Local Intrinsic Dimension"
  - "Hessian Spectral Filtering"
  - "Diffusion Models"
  - "Stochastic Lanczos Quadrature"
  - "Memorization Detection"
date: 2026-05-08
content_hash: 2e164154450d6c03
---

# Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation

**Conference**: ICML 2026  
**arXiv**: [2605.01221](https://arxiv.org/abs/2605.01221)  
**Code**: None  
**Area**: Diffusion Models / Manifold Learning / Intrinsic Dimension Estimation  
**Keywords**: Local Intrinsic Dimension, Hessian Spectral Filtering, Diffusion Models, Stochastic Lanczos Quadrature, Memorization Detection

## TL;DR
This paper proposes LHSD, which applies a Hill-type spectral filter to the log-density Hessian of a score model, retaining only near-zero eigenvalues to count the dimension of the tangent space. Stochastic Lanczos Quadrature reduces the computational cost from $\mathcal{O}(D^3)$ to $\mathcal{O}(D)$, enabling stable estimation of local intrinsic dimension in 3072-dimensional image spaces, and is used to diagnose memorization of training samples in diffusion models.

## Background & Motivation

**Background**: The manifold hypothesis posits that high-dimensional data are distributed along low-dimensional manifolds. Local intrinsic dimension (LID) describes the true degrees of freedom in a sample's neighborhood and is a key metric for analyzing generalization, detecting anomalies/adversarial samples, and, more recently, diagnosing memorization phenomena such as "verbatim reproduction of training data" in generative models. Early LID estimators relied on kNN (MLE / TwoNN / LPCA), while recent approaches leverage geometric methods based on the score function of diffusion models, such as LIDL, FLIPD, and NB.

**Limitations of Prior Work**: kNN-based methods suffer from the curse of dimensionality in high-dimensional spaces, where neighborhood distances concentrate, leading to severe estimation bias. FLIPD-type methods, based on score divergence, sum all Hessian eigenvalues indiscriminately; in high co-dimension image data, curvature in the normal direction diverges as $1/\sigma(t)^2$, overwhelming the true tangent signal. NB estimates rank via SVD of the score matrix, but its $\mathcal{O}(D^3)$ complexity makes it infeasible for 3072-dimensional images.

**Key Challenge**: LID is essentially the dimension of the tangent space, so reliable estimation requires explicit separation of tangent and normal contributions. However, quantities like the score and Hessian naturally mix these, and any explicit construction of a $D\times D$ Hessian incurs $\mathcal{O}(D^3)$ cost.

**Goal**: (1) Design an LID estimator robust to normal-direction noise; (2) Achieve linear complexity, scalable to $D{>}1000$; (3) Provide visual diagnostics for principled hyperparameter selection.

**Key Insight**: For small noise $\sigma(t)$, the log-density Hessian $H(\mathbf{x}) = -\nabla^2\log p_t(\mathbf{x})$ exhibits a "two-cluster" spectral structure—small $\mathcal{O}(1)$ tangent eigenvalues and large $\mathcal{O}(1/\sigma(t)^2)$ normal eigenvalues, with a natural spectral gap in between. Thus, placing a cutoff at the gap separates tangent from normal directions.

**Core Idea**: Use a Hill-type smooth filter $f(\lambda)$ to compress Hessian eigenvalues into $[0,1]$, mapping tangent directions to $\approx 1$ and normal directions to $\approx 0$, so that $\mathrm{tr}(f(H))$ yields the LID. SLQ is then used to estimate the trace without constructing the Hessian, reducing complexity to $\mathcal{O}(D)$.

## Method

### Overall Architecture
Input: trained score model $\mathbf{s}_\theta(\mathbf{x}, t)$, noise scale $\sigma(t)^2$, and sample $\mathbf{x}$ to estimate. Output: scalar LID estimate $\hat{d}$. Procedure: Construct a Hessian-vector product oracle at $\mathbf{x}$, $H(\mathbf{v}) = -\nabla_\mathbf{x}(\mathbf{s}_\theta(\mathbf{x}, t)^\top \mathbf{v})$ → use $K$ Rademacher random vectors → for each $\mathbf{v}_k$, run $m$ steps of Lanczos to obtain tridiagonal matrix $T_k$ → perform small-scale eigendecomposition of $T_k$ to get Ritz pairs $(\tilde\lambda_j, \tau_j)$ → feed each $\tilde\lambda_j$ into the filter $f$ and compute weighted sum → Monte Carlo average yields $\hat{d}$.

### Key Designs

1. **Hessian Filter Estimator Based on Tangent–Normal Spectral Separation**:

    - **Function**: Converts "counting tangent space dimensions" into "counting the number of near-zero Hessian eigenvalues."
    - **Mechanism**: Expanding $\log p_t$ near the manifold using tangent–normal coordinates yields $H(\mathbf{x}) = \Pi_\text{nor}(\mathbf{x})/\sigma(t)^2 + \mathcal{O}(1)$, i.e., for small $\sigma(t)$, the Hessian is essentially the normal projection matrix divided by $\sigma(t)^2$. This results in normal eigenvalues $\approx 1/\sigma(t)^2$ and tangent eigenvalues $\approx \mathcal{O}(1)$, creating a clear spectral gap. Define $\text{LHSD}(\mathbf{x}) := \sum_i f(\lambda_i) = \mathrm{tr}(f(H(\mathbf{x})))$, where the filter is Hill-type: $f(\lambda;\sigma(t)) = 1/(1+(|\lambda|/\kappa(t))^p)$, with cutoff $\kappa(t) := c/\sigma(t)^2$ absorbing the $\sigma(t)^{-2}$ scaling of normal curvature and noise.
    - **Design Motivation**: Unlike FLIPD, which sums all eigenvalues indiscriminately ($\nabla\cdot \mathbf{s}_\theta$), LHSD's filter responds $\approx 1$ in tangent directions and $\approx 0$ in normal directions, turning "magnitude summation" into "counting," fundamentally eliminating contamination from divergent normal magnitudes. The Hill filter, compared to sigmoid, has a flatter passband, better suited for SLQ's polynomial approximation.

2. **Verifiable Hyperparameter Selection via Transition Mass**:

    - **Function**: Addresses the seemingly "heuristic" requirement that LHSD's cutoff $\kappa(t)$ must fall within the spectral gap.
    - **Mechanism**: Fix $c, p$, scan $t$; define transition mass $M(t) := \frac{1}{D}\sum_i \mathbb{I}(\lambda_i(t) \in [\kappa(t) - \delta, \kappa(t) + \delta])$ to count the proportion of eigenvalues near the cutoff. When $M(t) \approx 0$ and the cutoff lies between the two eigenvalue "peaks," the cutoff is in the safe zone; if it falls within a peak or outside both peaks, this metric reveals it.
    - **Design Motivation**: Previous spectral filtering methods often chose cutoffs by trial and error on synthetic data. This work quantifies the geometric condition "cutoff in the gap" as a one-dimensional curve $M(t)$; Figure 3 shows the safe zone as a trough in $M(t)$, turning hyperparameter selection from "blind guessing" to "visual inspection."

3. **SLQ Acceleration: Reducing $\mathcal{O}(D^3)$ to $\mathcal{O}(D)$**:

    - **Function**: Computes $\mathrm{tr}(f(H))$ without constructing the full $D\times D$ Hessian.
    - **Mechanism**: Use Hutchinson's estimator $\mathrm{tr}(f(H)) \approx \mathbb{E}_\mathbf{v}[\mathbf{v}^\top f(H) \mathbf{v}]$, with each $\mathbf{v}^\top f(H) \mathbf{v}$ approximated via $m$-step Lanczos tridiagonalization of $H$ in the Krylov subspace, yielding $T_k$. The eigenpairs of $T_k$ are used for Gaussian quadrature: $\mathbf{v}^\top f(H)\mathbf{v} \approx \|\mathbf{v}\|^2 \sum_{j=1}^m \tau_j^2 f(\tilde\lambda_j)$. Hessian-vector products are implemented via automatic differentiation: $H\mathbf{v} = -\nabla(\mathbf{s}_\theta(\mathbf{x})^\top \mathbf{v})$, requiring only one backward pass each time.
    - **Design Motivation**: Traditional NB uses SVD for rank estimation at $\mathcal{O}(D^3)$, which is infeasible for 3072-dimensional images. SLQ leverages Krylov subspace for a low-rank approximation; experiments show $m=5$ steps suffice, yielding linear complexity in $D$ and making high-dimensional LID estimation practically feasible.

### Loss & Training
LHSD is a **pure inference-time** algorithm, introducing no trainable parameters. It assumes the underlying score model $\mathbf{s}_\theta$ is trained via standard denoising score matching. Hyperparameters $c$ (cutoff), $p$ (filter steepness, $p=4$ in the paper), $\delta$ (transition mass margin, $\delta = 0.2$), $K$ (number of Rademacher vectors), and $m$ (Lanczos steps) are set via transition mass curve diagnostics.

## Key Experimental Results

### Main Results
Mean Absolute Error (MAE, lower is better) on synthetic manifold data (linear subspace $\mathcal{L}$ and Funnel $\mathcal{F}$):

| Dimension $D$ | Dataset | FLIPD | NB | LHSD ($m{=}5$) |
|---|---|---|---|---|
| 1024 | $\mathcal{L}^{10+80+200}$ | 86.03 | 528.95 | **3.47** |
| 1024 | $\mathcal{F}^{10+80+200}$ | 373.80 | 937.82 | **6.90** |
| 3072 | $\mathcal{L}^{900}$ | 7.78 | 2171.00 | **11.53** |
| 3072 | $\mathcal{F}^{900}$ | 782.50 | 2171.00 | **18.79** |
| 3072 | $\mathcal{L}^{10+80+200}$ | 256.40 | 2949.10 | **4.70** |
| — | Average (9 settings) | 307.4 | 1319.9 | **6.6** |

The differences are striking: on the $D=3072$ Funnel manifold, FLIPD's MAE is 782, NB's is 2171, and LHSD's is 18.79—two orders of magnitude lower.

### Ablation Study

| Configuration | Mean MAE | Notes |
|---|---|---|
| LHSD ($m{=}2$) | 20.7 | Too few Lanczos steps, coarse spectral approximation |
| LHSD ($m{=}5$) | **6.6** | Default, already sufficient |
| FLIPD ($\nabla\cdot\mathbf{s}_\theta + \|\mathbf{s}_\theta\|^2$, no filtering) | 307.4 | No tangent/normal separation, normal noise explodes |
| NB (SVD rank estimation) | 1319.9 | Not only slow, but fails completely in high dimensions |

### Key Findings
- In high-dimensional settings ($D \geq 1024$), all baselines (kNN / FLIPD / NB) have errors in the $10^2 \sim 10^3$ range on at least one dataset; LHSD remains consistently in the single to tens range, indicating that "explicit normal filtering" is essential, not just beneficial.
- Only $m=5$ Lanczos steps are needed, corresponding to a very small tridiagonal matrix, validating SLQ's efficiency.
- The transition mass diagnostic curve (Fig. 3) reveals that when $t$ is mischosen (e.g., $t=0.22$), the cutoff can move outside the two eigenvalue peaks; $M(t)\approx 0$ alone is insufficient—peak positions must also be considered, a practical point emphasized in the paper.

## Highlights & Insights
- **From magnitude summation to indicator counting for LID**: FLIPD fails because it uses a divergent quantity to approximate a bounded integer (dimension); LHSD compresses the spectrum to $[0,1]$ before summing, restoring the estimation target to a "bounded, noise-decoupled" form. This "normalize then aggregate" approach can be transferred to any Hessian-spectrum-dependent downstream task (e.g., sharpness, model geometry).
- **Cutoff $\kappa(t) := c/\sigma(t)^2$ as adaptive normalization**: It absorbs the noise dependence $\sigma(t)^{-2}$ of normal curvature into the cutoff, ensuring consistent filter behavior across $t$—this "adaptive constant" design is valuable in multi-scale noise scenarios.
- **SLQ + Hill filter as an engineering match**: The Hill filter is smooth and fits well with low-degree polynomial approximation, aligning with SLQ's Gaussian quadrature; a hard step cutoff would break SLQ's polynomial approximation, explaining the necessity of a smooth filter.

## Limitations & Future Work
- LHSD assumes the spectral gap truly exists: when $\sigma(t)$ is large, the Hessian spectrum collapses to isotropy (see Appendix E), rendering the method ineffective. The authors acknowledge the lack of an automatic "gap existence" detector, relying instead on transition mass as an indirect indicator.
- Estimation accuracy depends on the quality of the underlying score model $\mathbf{s}_\theta$; insufficiently trained diffusion models have Hessian spectra contaminated by network noise. This hidden assumption may be problematic in practice (e.g., evaluating third-party diffusion checkpoints).
- Experiments are mainly on small to medium UNet diffusion models; LID computation cost for very large models (e.g., SD/DALL·E scale) remains to be fully validated, though complexity is linear.
- Future work could explore replacing the Hill filter with a learnable spectral filter, supervised end-to-end with memorization/anomaly detection loss, aligning LID estimation directly with downstream tasks.

## Related Work & Insights
- **vs FLIPD**: FLIPD uses $\sigma(t)^2(\nabla\cdot \mathbf{s}_\theta + \|\mathbf{s}_\theta\|^2)$, summing all Hessian eigenvalues indiscriminately; LHSD counts only tangent eigenvalues via filtering, yielding up to two orders of magnitude improvement in high dimensions.
- **vs NB (Normal Bundle)**: NB stacks multiple noisy score SVDs to estimate normal space rank, but $\mathcal{O}(D^3)$ is intractable; LHSD uses second-order Hessian + SLQ, achieving linear complexity and scaling to 3072 dimensions.
- **vs kNN estimators (MLE / TwoNN / LPCA / ESS)**: Traditional estimators rely on neighborhood distances, suffering from the curse of dimensionality; diffusion model scores replace neighborhood search, but only LHSD fully implements tangent–normal separation.
- **Insights**: Hessian spectral analysis in deep learning has long featured the "bulk + outliers" paradigm; this work cleverly transfers it to generative model geometry. The approach can be extended to more tasks using score models as implicit geometric probes (e.g., curvature, reach, manifold Betti numbers).

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of tangent–normal spectral separation, Hill filtering, and SLQ is coherent and novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic + real data, covering low/mid/high dimensions and various manifolds, but lacks large-scale detection on SOTA diffusion models
- Writing Quality: ⭐⭐⭐⭐ Clear geometric motivation, solid derivations, and especially strong on transition mass visualization diagnostics
- Value: ⭐⭐⭐⭐ Enables practical LID estimation in 3000+ dimensions for the first time, with direct applications in diffusion model memorization diagnosis and OOD detection

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Connection Between Score Matching and Local Intrinsic Dimension](../../NeurIPS2025/image_generation/a_connection_between_score_matching_and_local_intrinsic_dimension.md)
- [\[ICML 2026\] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)
- [\[CVPR 2026\] Intrinsic Concept Extraction Based on Compositional Interpretability](../../CVPR2026/image_generation/intrinsic_concept_extraction_based_on_compositional_interpretability.md)
- [\[AAAI 2026\] Stabilizing Self-Consuming Diffusion Models with Latent Space Filtering](../../AAAI2026/image_generation/stabilizing_self-consuming_diffusion_models_with_latent_space_filtering.md)
- [\[ICML 2026\] Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_.md)

</div>

<!-- RELATED:END -->
