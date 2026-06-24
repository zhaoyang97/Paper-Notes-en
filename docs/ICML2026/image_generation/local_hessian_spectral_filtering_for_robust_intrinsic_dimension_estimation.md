---
title: >-
  [Paper Note] Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation
description: >-
  [ICML 2026][Image Generation][Local Intrinsic Dimension] This paper proposes LHSD, which applies a Hill-type spectral filter to the log-density Hessian of a score model to retain only near-zero eigenvalues for counting tangent space dimensions. By leveraging Stochastic Lanczos Quadrature, it reduces the computational cost from $\mathcal{O}(D^3)$ to $\mathcal{O}(D)$, enabling stable estimation of Local Intrinsic Dimension (LID) in 3072-dimensional image spaces and diagnosing t…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Local Intrinsic Dimension"
  - "Hessian Spectral Filtering"
  - "Diffusion Models"
  - "Stochastic Lanczos Quadrature"
  - "Memorization Detection"
date: 2026-05-08
content_hash: f15773952c35d675
---

# Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation

**Conference**: ICML 2026  
**arXiv**: [2605.01221](https://arxiv.org/abs/2605.01221)  
**Code**: None  
**Area**: Diffusion Models / Manifold Learning / Intrinsic Dimension Estimation  
**Keywords**: Local Intrinsic Dimension, Hessian Spectral Filtering, Diffusion Models, Stochastic Lanczos Quadrature, Memorization Detection

## TL;DR
This paper proposes LHSD, which applies a Hill-type spectral filter to the log-density Hessian of a score model to retain only near-zero eigenvalues for counting tangent space dimensions. By leveraging Stochastic Lanczos Quadrature, it reduces the computational cost from $\mathcal{O}(D^3)$ to $\mathcal{O}(D)$, enabling stable estimation of Local Intrinsic Dimension (LID) in 3072-dimensional image spaces and diagnosing training sample memorization in diffusion models.

## Background & Motivation

**Background**: The manifold hypothesis posits that high-dimensional data is distributed along low-dimensional manifolds. Local Intrinsic Dimension (LID) describes the true degrees of freedom in a sample's neighborhood and serves as a key metric for analyzing generalization, detecting anomalies/adversarial samples, and recently, detecting "verbatim reproduction of training data" (memorization) in generative models. Early LID estimation relied on kNN (MLE / TwoNN / LPCA), while recent work has shifted towards geometric methods based on the score functions of diffusion models, such as LIDL, FLIPD, and NB.

**Limitations of Prior Work**: kNN-based methods are crushed by the curse of dimensionality in high-dimensional spaces, where distance concentration leads to severe estimation bias. Methods like FLIPD, based on score divergence, indiscriminately sum all eigenvalues of the Hessian. In high co-dimension image data, normal curvatures diverge at the scale of $1/\sigma(t)^2$, drowning out the true tangential signals. NB estimates rank via SVD of the score matrix, but its $\mathcal{O}(D^3)$ complexity makes it nearly uncomputable for 3072-dimensional images.

**Key Challenge**: LID is fundamentally the dimension of the tangent space. Reliable estimation requires **explicit separation** of tangential and normal contributions. However, quantities like the score or Hessian naturally confound the two, and any explicit construction of a $D\times D$ Hessian triggers an $\mathcal{O}(D^3)$ cost.

**Goal**: (1) Design an LID estimator robust to normal noise; (2) Achieve linear complexity to run on $D{>}1000$; (3) Provide visual diagnostics for hyperparameter selection rather than relying on heuristics.

**Key Insight**: Under small noise $\sigma(t)$, the spectrum of the log-density Hessian $H(\mathbf{x}) = -\nabla^2\log p_t(\mathbf{x})$ exhibits a "two-clump" structure—small eigenvalues of $\mathcal{O}(1)$ for tangential directions and large eigenvalues of $\mathcal{O}(1/\sigma(t)^2)$ for normal directions, with a natural spectral gap in between. This implies that placing a "cutoff" at the gap can distinguish between tangent and normal directions.

**Core Idea**: Use a Hill-type smoothing filter $f(\lambda)$ to compress Hessian eigenvalues into $[0,1]$, mapping tangential ones to $\approx 1$ and normal ones to $\approx 0$. Consequently, $\mathrm{tr}(f(H))$ becomes the LID. Furthermore, use SLQ to estimate the trace without constructing the Hessian, reducing complexity to $\mathcal{O}(D)$.

## Method

### Overall Architecture
LHSD aims to stably estimate the Local Intrinsic Dimension (LID) of a sample in 3000+ dimensional image space, where LID is essentially the dimension of the manifold's tangent space. The paper transforms this into "counting the number of near-zero eigenvalues in the log-density Hessian $H(\mathbf{x})=-\nabla^2\log p_t(\mathbf{x})$"—tangent directions correspond to small eigenvalues and normal directions to large ones. By using a spectral filter to map tangent eigenvalues to $\approx 1$ and normal eigenvalues to $\approx 0$, the trace yields the LID. This estimation is performed at inference time by reusing a trained score model and calculating the trace via Stochastic Lanczos Quadrature (SLQ) without explicitly constructing the $D\times D$ Hessian.

### Key Designs

**1. Tangent–Normal Spectral Separation Hessian Filter: From "Counting Dimensions" to "Filtering and Tracing"**

Directly counting the dimension of the Hessian's null space is numerically infeasible. The breakthrough in this paper is leveraging the "two-clump" structure of the Hessian spectrum. Expanding $\log p_t$ in tangent–normal coordinates near the manifold gives $H(\mathbf{x}) = \Pi_\text{nor}(\mathbf{x})/\sigma(t)^2 + \mathcal{O}(1)$, meaning that under small noise $\sigma(t)$, the Hessian is essentially the normal projection matrix scaled by $\sigma(t)^{-2}$. Thus, normal eigenvalues are $\approx 1/\sigma(t)^2$ while tangential ones are $\approx \mathcal{O}(1)$, creating a natural spectral gap.

Utilizing this gap, the paper defines $\text{LHSD}(\mathbf{x}) := \sum_i f(\lambda_i) = \mathrm{tr}(f(H(\mathbf{x})))$, where the filter is a Hill-type function $f(\lambda;\sigma(t)) = 1/(1+(|\lambda|/\kappa(t))^p)$. The cutoff $\kappa(t) := c/\sigma(t)^2$ directly absorbs the $\sigma(t)^{-2}$ scaling of normal curvature. Unlike FLIPD, which sums all eigenvalues indiscriminately ($\nabla\cdot \mathbf{s}_\theta$), this filter responds with $\approx 1$ for tangents and $\approx 0$ for normals, effectively changing "magnitude summation" into "count-based tallying," fundamentally eliminating contamination from diverging normal magnitudes. The Hill-type filter is preferred over sigmoid due to its flatter passband, which is better suited for the low-order polynomial approximation accuracy of SLQ.

**2. Verifiable Hyperparameter Selection via Transition Mass: Making the "Cutoff in the Gap" Visible**

The validity of LHSD depends entirely on whether the cutoff $\kappa(t)$ falls exactly within the spectral gap. Previously, such cutoffs were selected through trial and error on synthetic data. This paper quantifies the geometric condition of "whether the cutoff is in the gap" into a 1D curve: fixing $c, p$ and scanning $t$, it defines the transition mass $M(t) := \frac{1}{D}\sum_i \mathbb{I}(\lambda_i(t) \in [\kappa(t) - \delta, \kappa(t) + \delta])$, which counts the proportion of eigenvalues near the cutoff boundary.

When $M(t) \approx 0$ and the position is between the two eigenvalue "peaks," the cutoff line passes exactly through the gap, designated as the safe zone. If it falls into a peak or outside both peaks, $M(t)$ reveals it. Figure 3 in the paper shows the safe zone appearing as a valley in $M(t)$, turning hyperparameter selection from a "blind choice" into "selection by visualization," a key step in grounding heuristic operations with empirical evidence.

**3. SLQ Acceleration: Compressing $\mathcal{O}(D^3)$ to $\mathcal{O}(D)$**

To calculate $\mathrm{tr}(f(H))$ in 3072 dimensions, one must avoid explicit construction of the full $D\times D$ Hessian—traditional NB uses SVD for rank estimation, which is $\mathcal{O}(D^3)$ and unusable for images. This paper uses the SLQ path: first, use the Hutchinson estimator $\mathrm{tr}(f(H)) \approx \mathbb{E}_\mathbf{v}[\mathbf{v}^\top f(H) \mathbf{v}]$, taking the Monte Carlo average over $K$ Rademacher probe vectors. Each quadratic form $\mathbf{v}^\top f(H)\mathbf{v}$ is then approximated via $m$ Lanczos steps to tridiagonalize $H$ into $T$ in the Krylov subspace, followed by Gaussian quadrature using the eigenpairs $(\tilde\lambda_j, \tau_j)$ of $T$ to estimate $\mathbf{v}^\top f(H)\mathbf{v} \approx \|\mathbf{v}\|^2 \sum_{j=1}^m \tau_j^2 f(\tilde\lambda_j)$.

The only part of this chain that interacts with the Hessian is the Hessian-vector product, implemented via automatic differentiation as $H\mathbf{v} = -\nabla(\mathbf{s}_\theta(\mathbf{x})^\top \mathbf{v})$, requiring only one backpropagation. Since the Krylov subspace only requires a low-rank approximation, experiments show $m=5$ steps are sufficient. The resulting complexity is linear in $D$, making LID estimation on high-dimensional images practically feasible.

### Loss & Training
LHSD is a **purely inference-time** algorithm and introduces no training parameters. It assumes the underlying score model $\mathbf{s}_\theta$ has been trained using standard denoising score matching. Hyperparameters $c$ (cutoff position), $p$ (filter steepness, $p=4$ used), $\delta$ (transition mass margin, $\delta = 0.2$), $K$ (number of Rademacher probes), and $m$ (Lanczos steps) are set using the transition mass curve diagnostics.

## Key Experimental Results

### Main Results
MAE (Lower is better) on synthetic manifold data (Linear subspace $\mathcal{L}$ and Funnel $\mathcal{F}$):

| Dimension $D$ | Dataset | FLIPD | NB | LHSD ($m{=}5$) |
|---|---|---|---|---|
| 1024 | $\mathcal{L}^{10+80+200}$ | 86.03 | 528.95 | **3.47** |
| 1024 | $\mathcal{F}^{10+80+200}$ | 373.80 | 937.82 | **6.90** |
| 3072 | $\mathcal{L}^{900}$ | 7.78 | 2171.00 | **11.53** |
| 3072 | $\mathcal{F}^{900}$ | 782.50 | 2171.00 | **18.79** |
| 3072 | $\mathcal{L}^{10+80+200}$ | 256.40 | 2949.10 | **4.70** |
| — | Average (9 settings) | 307.4 | 1319.9 | **6.6** |

The gap is significant: on the Funnel manifold with $D=3072$, FLIPD's MAE is 782 and NB's is 2171, while LHSD is 18.79—a two-order-of-magnitude difference.

### Ablation Study

| Configuration | Average MAE | Description |
|---|---|---|
| LHSD ($m{=}2$) | 20.7 | Too few Lanczos steps, coarse spectral approximation |
| LHSD ($m{=}5$) | **6.6** | Default configuration, already sufficient |
| FLIPD ($\nabla\cdot\mathbf{s}_\theta + \|\mathbf{s}_\theta\|^2$, no filtering) | 307.4 | Diverging normal noise without tangent/normal separation |
| NB (SVD rank estimation) | 1319.9 | Not only slow but completely fails in high dimensions |

### Key Findings
- In high-dimensional settings ($D \geq 1024$), all baseline methods (kNN-based / FLIPD / NB) suffer from error magnitudes of $10^2 \sim 10^3$ in at least one dataset; LHSD remains stable across all settings with errors in the single to low double digits, proving that "explicitly filtering normals" is a necessity rather than an improvement.
- Only $m=5$ Lanczos steps are required, corresponding to a very small tridiagonal matrix, validating the efficiency of SLQ.
- The transition mass diagnostic curve (Fig. 3) reveals that when $t$ is chosen incorrectly (e.g., $t=0.22$), the cutoff line passes outside the two eigenvalue peaks. $M(t)\approx 0$ alone is insufficient; one must consider peak positions, which is a practical contribution of the paper.

## Highlights & Insights
- **Shifting LID from magnitude summation to indicator counting**: FLIPD fails because it uses a diverging quantity to approximate a bounded integer (dimension). LHSD recovers a "bounded, noise-decoupled" estimation by compressing the spectrum to $[0,1]$ before summing. This "normalize then aggregate" approach could be applied to any task depending on the Hessian spectrum (e.g., sharpness, model geometry).
- **Cutoff $\kappa(t) := c/\sigma(t)^2$ as "Adaptive Normalization"**: It directly absorbs the $\sigma(t)^{-2}$ noise dependence of normal curvature, ensuring consistent filter behavior across different $t$—this "adaptive constant" design is valuable for multi-scale noise scenarios.
- **Engineering synergy between SLQ and Hill filtering**: Hill filters are smooth and can be fitted with low-order polynomials, perfectly matching the requirements of low-order Gaussian quadrature in SLQ. If a hard step-function cutoff were used, SLQ's polynomial approximation would fail. This detail explains why a smoothing filter is mandatory.

## Limitations & Future Work
- LHSD assumes the existence of a spectral gap: when $\sigma(t)$ is too large, the Hessian spectrum collapses into isotropy (discussed in Appendix E), causing the method to fail. The authors admit they do not yet have an automatic detection for "gap existence" and rely on transition mass.
- Estimation accuracy depends on the quality of the underlying score model $\mathbf{s}_\theta$. The Hessian spectrum of an insufficiently trained diffusion model might be contaminated by network noise.
- Experiments were mainly conducted on small-to-medium diffusion models (UNet); the computational overhead for massive models (like SD or DALL·E scale) has not been fully verified, despite linear complexity.
- Future work could explore replacing the Hill filter with a learnable spectral filter, supervised end-to-end by memorization or anomaly detection losses.

## Related Work & Insights
- **vs FLIPD**: FLIPD uses $\sigma(t)^2(\nabla\cdot \mathbf{s}_\theta + \|\mathbf{s}_\theta\|^2)$, summing all Hessian eigenvalues indiscriminately; LHSD uses a filter to count only tangent eigenvalues, resulting in a 100x improvement in high dimensions.
- **vs NB (Normal Bundle)**: NB estimates the rank of the normal space via SVD of stacked noisy scores, which is $\mathcal{O}(D^3)$ and computationally heavy; LHSD uses the second-order Hessian + SLQ, achieving linear complexity and scaling to 3072 dimensions.
- **vs kNN Estimators (MLE / TwoNN / LPCA / ESS)**: Traditional estimators rely on neighborhood distances and suffer from the curse of dimensionality; diffusion score models replace neighborhood search, but only LHSD fully implements tangent-normal separation.
- **Insight**: Hessian spectral analysis in deep learning often focuses on "bulk + outliers" structures; this paper intelligently migrates that concept to generative model geometry. It could be extended to more tasks using score models as implicit geometric probes (e.g., curvature, reach, manifold topological features).

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of tangent–normal separation + Hill filtering + SLQ is self-consistent and novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic and real data across various dimensions and manifolds, though more large-scale SOTA diffusion model cases would be beneficial.
- Writing Quality: ⭐⭐⭐⭐ Clear geometric motivation, solid derivations, and the transition mass visualization is a significant plus.
- Value: ⭐⭐⭐⭐ Makes high-dimensional LID estimation practically computable for 3000+ dimensions for the first time, offering direct value for memorization diagnostics and OOD detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Connection Between Score Matching and Local Intrinsic Dimension](../../NeurIPS2025/image_generation/a_connection_between_score_matching_and_local_intrinsic_dimension.md)
- [\[ICML 2026\] Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](order_within_chaos_capturing_intrinsic_energy_anomalies_for_ai-manipulated_image.md)
- [\[ICML 2026\] Caracal: Causal Architecture via Spectral Mixing](caracal_causal_architecture_via_spectral_mixing.md)
- [\[ICML 2026\] Spectral Guidance for Flexible and Efficient Control of Diffusion Models](spectral_guidance_for_flexible_and_efficient_control_of_diffusion_models.md)
- [\[ICML 2026\] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)

</div>

<!-- RELATED:END -->
