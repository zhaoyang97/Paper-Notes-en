---
title: >-
  [Paper Note] Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation
description: >-
  [ICML 2026][Image Generation][Local Intrinsic Dimension] This paper proposes LHSD, which applies a Hill-type spectral filter to the log-density Hessian of a score model to retain only near-zero eigenvalues for counting t…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Local Intrinsic Dimension"
  - "Hessian Spectral Filtering"
  - "Diffusion Models"
  - "Stochastic Lanczos Quadrature"
  - "Memorization Detection"
date: 2026-05-08
content_hash: d01ea0d19ea898d8
---

# Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation

**Conference**: ICML 2026  
**arXiv**: [2605.01221](https://arxiv.org/abs/2605.01221)  
**Code**: None  
**Area**: Diffusion Models / Manifold Learning / Intrinsic Dimension Estimation  
**Keywords**: Local Intrinsic Dimension, Hessian Spectral Filtering, Diffusion Models, Stochastic Lanczos Quadrature, Memorization Detection

## TL;DR
This paper proposes LHSD, which applies a Hill-type spectral filter to the log-density Hessian of a score model to retain only near-zero eigenvalues for counting tangent space dimensions. By utilizing Stochastic Lanczos Quadrature (SLQ), it reduces the computational cost from $\mathcal{O}(D^3)$ to $\mathcal{O}(D)$, enabling stable local intrinsic dimension (LID) estimation in 3072-dimensional image spaces and diagnosing training sample memorization in diffusion models.

## Background & Motivation

**Background**: The manifold hypothesis suggests that high-dimensional data is distributed along low-dimensional manifolds. Local Intrinsic Dimension (LID) describes the true degrees of freedom in a sample neighborhood and is a key metric for analyzing generalization, detecting anomalies/adversarial samples, and recently, detecting memorization phenomena where generative models replicate training data exactly. Early LID estimation relied on kNN (MLE / TwoNN / LPCA), while recent methods have shifted toward geometric approaches based on the score function of diffusion models, such as LIDL, FLIPD, and NB.

**Limitations of Prior Work**: kNN-based methods are overwhelmed by the curse of dimensionality in high-dimensional spaces, where distance concentration leads to severe estimation bias. Methods like FLIPD, based on score divergence, sum all Hessian eigenvalues indiscriminately; in high co-dimension image data, curvatures in normal directions diverge at a scale of $1/\sigma(t)^2$, drowning out the true tangent signals. NB estimates rank via SVD of the score matrix, but its $\mathcal{O}(D^3)$ complexity makes it nearly uncomputable for 3072-dimensional images.

**Key Challenge**: LID is essentially the dimension of the tangent space. Reliable estimation requires **explicit separation** of tangent and normal contributions. However, quantities like the score and Hessian naturally mix them, and any explicit construction of a $D \times D$ Hessian triggers an $\mathcal{O}(D^3)$ cost.

**Goal**: (1) Design an LID estimator robust to normal noise; (2) Achieve linear complexity to run on $D > 1000$; (3) Provide verifiable diagnostics for hyperparameter selection instead of relying on heuristics.

**Key Insight**: Under small noise $\sigma(t)$, the spectrum of the log-density Hessian $H(\mathbf{x}) = -\nabla^2\log p_t(\mathbf{x})$ exhibits a "two-cluster" structure—small eigenvalues of $\mathcal{O}(1)$ in the tangent directions and large eigenvalues of $\mathcal{O}(1/\sigma(t)^2)$ in the normal directions. A spectral gap naturally exists between them. This implies that the tangent and normal components can be distinguished by placing a "cutoff" at the gap.

**Core Idea**: Use a Hill-type smooth filter $f(\lambda)$ to compress Hessian eigenvalues into $[0, 1]$, making tangent eigenvalues $\approx 1$ and normal eigenvalues $\approx 0$. Thus, $\mathrm{tr}(f(H))$ represents the LID. Furthermore, use SLQ to estimate the trace without constructing the Hessian, reducing complexity to $\mathcal{O}(D)$.

## Method

### Overall Architecture
Input: A trained score model $\mathbf{s}_\theta(\mathbf{x}, t)$, noise scale $\sigma(t)^2$, and the sample $\mathbf{x}$ to be estimated. Output: Scalar LID estimate $\hat{d}$. The process is: construct a Hessian-vector product oracle $H(\mathbf{v}) = -\nabla_\mathbf{x}(\mathbf{s}_\theta(\mathbf{x}, t)^\top \mathbf{v})$ at $\mathbf{x}$ → use $K$ Rademacher random vectors → run $m$ steps of Lanczos for each $\mathbf{v}_k$ to obtain a tridiagonal matrix $T_k$ → perform a small-scale eigendecomposition of $T_k$ to obtain Ritz pairs $(\tilde\lambda_j, \tau_j)$ → pass each $\tilde\lambda_j$ through the filter $f$ for weighted summation → compute the Monte Carlo average to obtain $\hat{d}$.

### Key Designs

1.  **Hessian Filtering Estimator based on Tangent-Normal Spectral Separation**:
    - **Function**: Transforms "counting tangent space dimensions" into "counting the number of near-zero Hessian eigenvalues."
    - **Mechanism**: Expanding $\log p_t$ in tangent-normal coordinates near the manifold yields $H(\mathbf{x}) = \Pi_\text{nor}(\mathbf{x})/\sigma(t)^2 + \mathcal{O}(1)$, meaning the Hessian under small $\sigma(t)$ is essentially the normal projection matrix scaled by $1/\sigma(t)^2$. This results in normal eigenvalues $\approx 1/\sigma(t)^2$ and tangent eigenvalues $\approx \mathcal{O}(1)$, creating a clear spectral gap. LHSD is then defined as $\text{LHSD}(\mathbf{x}) := \sum_i f(\lambda_i) = \mathrm{tr}(f(H(\mathbf{x})))$. The filter used is a Hill-type $f(\lambda;\sigma(t)) = 1/(1+(|\lambda|/\kappa(t))^p)$, where the cutoff $\kappa(t) := c/\sigma(t)^2$ directly absorbs the $\sigma(t)^{-2}$ scaling of normal curvature and noise.
    - **Design Motivation**: Unlike FLIPD, which sums all eigenvalues indiscriminately ($\nabla\cdot \mathbf{s}_\theta$), the filter in LHSD responds $\approx 1$ to tangent directions and $\approx 0$ to normal directions. By changing "magnitude summation" to "counting," it fundamentally eliminates the contamination of the estimate by diverging normal magnitudes. Compared to a sigmoid, the Hill filter has a flatter passband, which is better suited for the polynomial approximation accuracy of SLQ.

2.  **Verifiable Hyperparameter Selection via Transition Mass**:
    - **Function**: Solves the problem of ensuring the cutoff $\kappa(t)$ falls within the spectral gap.
    - **Mechanism**: Fix $c, p$ and scan $t$. Define transition mass $M(t) := \frac{1}{D}\sum_i \mathbb{I}(\lambda_i(t) \in [\kappa(t) - \delta, \kappa(t) + \delta])$ to count the proportion of eigenvalues near the cutoff boundary. When $M(t) \approx 0$ and is located between two eigenvalue "peaks," the cutoff line falls exactly in the gap, denoted as the "safe zone." Indicators will expose when the line falls within a peak or beyond them.
    - **Design Motivation**: Cutoff selection in previous spectral filtering methods often relied on trial and error with synthetic data. Ours quantifies the geometric condition of whether the cutoff is in the gap as a 1D curve $M(t)$. Fig. 3 shows that the safe zone appears as a trough in $M(t)$, turning hyperparameter selection from "blind guessing" into "visual selection."

3.  **SLQ Acceleration: Reducing $\mathcal{O}(D^3)$ to $\mathcal{O}(D)$**:
    - **Function**: Computes $\mathrm{tr}(f(H))$ without ever constructing the full $D\times D$ Hessian.
    - **Mechanism**: Use the Hutchinson estimator $\mathrm{tr}(f(H)) \approx \mathbb{E}_\mathbf{v}[\mathbf{v}^\top f(H) \mathbf{v}]$. Each $\mathbf{v}^\top f(H) \mathbf{v}$ is approximated via $m$ steps of Lanczos tridiagonalization of $H$ on the Krylov subspace to get $T_k$, followed by Gaussian quadrature using the Ritz pairs of $T_k$: $\mathbf{v}^\top f(H)\mathbf{v} \approx \|\mathbf{v}\|^2 \sum_{j=1}^m \tau_j^2 f(\tilde\lambda_j)$. Hessian-vector products are implemented via automatic differentiation as $H\mathbf{v} = -\nabla(\mathbf{s}_\theta(\mathbf{x})^\top \mathbf{v})$, requiring only one backpropagation per step.
    - **Design Motivation**: Traditional NB uses SVD for rank estimation, which is $\mathcal{O}(D^3)$ and unfeasible for 3072-dim images. SLQ looks only at a low-rank approximation via the Krylov subspace. Experiments show that $m=5$ steps are sufficient, resulting in a complexity linear in $D$. This makes LID estimation on high-dimensional images practically possible.

### Loss & Training
LHSD is a **purely inference-time** algorithm and introduces no trainable parameters. It assumes the underlying score model $\mathbf{s}_\theta$ was trained using standard denoising score matching. Hyperparameters $c$ (cutoff position), $p$ (filter steepness, $p=4$ used), $\delta$ (transition mass margin, $\delta = 0.2$), $K$ (number of Rademacher vectors), and $m$ (Lanczos steps) are set via transition mass curve diagnostics.

## Key Experimental Results

### Main Results
MAE (lower is better) on synthetic manifold data (Linear subspace $\mathcal{L}$ and Funnel $\mathcal{F}$):

| Dimension $D$ | Dataset | FLIPD | NB | LHSD ($m=5$) |
|---|---|---|---|---|
| 1024 | $\mathcal{L}^{10+80+200}$ | 86.03 | 528.95 | **3.47** |
| 1024 | $\mathcal{F}^{10+80+200}$ | 373.80 | 937.82 | **6.90** |
| 3072 | $\mathcal{L}^{900}$ | 7.78 | 2171.00 | **11.53** |
| 3072 | $\mathcal{F}^{900}$ | 782.50 | 2171.00 | **18.79** |
| 3072 | $\mathcal{L}^{10+80+200}$ | 256.40 | 2949.10 | **4.70** |
| — | Average (9 settings) | 307.4 | 1319.9 | **6.6** |

The gap is significant: on the 3072-dim Funnel manifold, the MAE for FLIPD is 782 and for NB is 2171, while LHSD is 18.79—a difference of two orders of magnitude.

### Ablation Study

| Configuration | Average MAE | Description |
|---|---|---|
| LHSD ($m=2$) | 20.7 | Too few Lanczos steps; coarse spectral approximation |
| LHSD ($m=5$) | **6.6** | Default configuration; sufficient |
| FLIPD ($\nabla\cdot\mathbf{s}_\theta + \|\mathbf{s}_\theta\|^2$, no filtering) | 307.4 | Sums tangent/normal directly; normal noise explodes |
| NB (SVD rank estimation) | 1319.9 | Slow and completely fails in high dimensions |

### Key Findings
- In high-dimensional settings ($D \geq 1024$), all baseline methods (kNN-based / FLIPD / NB) show error magnitudes of $10^2 \sim 10^3$ on at least one dataset; LHSD consistently stays in the single to double digits across all settings, proving that "explicit normal filtering" is necessary, not just supplementary.
- Lanczos requires only $m=5$ steps, corresponding to a very small tridiagonal matrix, verifying the efficiency of SLQ.
- The transition mass diagnostic curve (Fig. 3) reveals that when $t$ is chosen incorrectly (e.g., $t=0.22$), the cutoff line drifts outside the two eigenvalue peaks. Monitoring $M(t) \approx 0$ alone is insufficient; it must be combined with peak locations, providing a practical operational point for the paper.

## Highlights & Insights
- **From magnitude summation to indicator counting for LID**: The failure of FLIPD stems from using a diverging quantity to approximate a bounded integer (dimension). LHSD restores the estimation goal to a "bounded, noise-scale decoupled" form by compressing the spectrum to $[0, 1]$ before summation. This "normalize-then-aggregate" approach can be transferred to any downstream task relying on the Hessian spectrum (e.g., sharpness, model geometry).
- **Adaptive Normalization via Cutoff $\kappa(t) := c/\sigma(t)^2$**: This directly absorbs the noise dependency $\sigma(t)^{-2}$ of the normal curvature, ensuring consistent filter behavior across different $t$—an "adaptive constant" design that is highly valuable in multi-scale noise scenarios.
- **Engineering Synergy of SLQ + Hill Filter**: The Hill filter is smooth and can be fitted with low-degree polynomials, matching SLQ's requirement for low-order Gaussian quadrature. If a hard step cutoff were used, SLQ's polynomial approximation would fail. This detail explains why a smooth filter is required.

## Limitations & Future Work
- LHSD assumes that a spectral gap truly exists: when $\sigma(t)$ is too large, the Hessian spectrum collapses into isotropy (discussed in Appendix E), causing the method to fail. The authors admit they haven't provided an automatic detection for "gap existence," relying on transition mass instead.
- Estimation accuracy depends on the quality of the underlying score model $\mathbf{s}_\theta$; the Hessian spectrum of an insufficiently trained diffusion model may be contaminated by network noise.
- Experiments were primarily conducted on small-to-medium UNet-based diffusion models. The computational overhead of LHSD for ultra-large models (e.g., SD/DALL·E scale) has not been fully verified, despite its linear complexity.
- Future work could explore replacing the Hill filter with a learnable spectral filter, supervised end-to-end by memorization/anomaly detection losses to align LID estimation directly with downstream tasks.

## Related Work & Insights
- **vs FLIPD**: FLIPD uses $\sigma(t)^2(\nabla\cdot \mathbf{s}_\theta + \|\mathbf{s}_\theta\|^2)$, summing all Hessian eigenvalues indiscriminately. LHSD uses a filter to count only tangent eigenvalues, performing two orders of magnitude better in high dimensions.
- **vs NB (Normal Bundle)**: NB estimates the normal space rank via SVD of multiple stacked noisy scores, which is computationally prohibitive ($\mathcal{O}(D^3)$). LHSD uses the second-order Hessian + SLQ path with linear complexity, enabling 3072-dim estimation.
- **vs kNN Estimators (MLE / TwoNN / LPCA / ESS)**: Traditional estimators rely on neighborhood distances and suffer from the curse of dimensionality. Diffusion score models replace neighborhood search, but only LHSD fully implements tangent-normal separation.
- **Insight**: Hessian spectral analysis in deep learning has long had the "bulk + outliers" stereotype; this paper's application to generative model geometry is a clever transfer. It can be extended to more tasks using score models as implicit geometric probes (e.g., curvature, reach, manifold topological features).

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of tangent-normal spectral separation + Hill filter + SLQ is self-consistent and novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic and real data across low/medium/high dimensions and various manifolds, though it lacks large-scale case studies on modern SOTA diffusion models.
- Writing Quality: ⭐⭐⭐⭐ Clear geometric motivation, solid derivations, and the transition mass visualization diagnostics are a plus.
- Value: ⭐⭐⭐⭐ Makes high-dimensional LID estimation truly computable at 3000+ dimensions for the first time, with direct value for diffusion model memorization diagnostics and OOD detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Connection Between Score Matching and Local Intrinsic Dimension](../../NeurIPS2025/image_generation/a_connection_between_score_matching_and_local_intrinsic_dimension.md)
- [\[ICML 2026\] Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](order_within_chaos_capturing_intrinsic_energy_anomalies_for_ai-manipulated_image.md)
- [\[ICML 2026\] Caracal: Causal Architecture via Spectral Mixing](caracal_causal_architecture_via_spectral_mixing.md)
- [\[ICML 2026\] Spectral Guidance for Flexible and Efficient Control of Diffusion Models](spectral_guidance_for_flexible_and_efficient_control_of_diffusion_models.md)
- [\[ICML 2026\] DiScoFormer: Plug-In Density and Score Estimation with Transformers](discoformer_plug-in_density_and_score_estimation_with_transformers.md)

</div>

<!-- RELATED:END -->
