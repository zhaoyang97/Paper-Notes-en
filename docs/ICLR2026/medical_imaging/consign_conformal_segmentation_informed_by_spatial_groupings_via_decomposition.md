---
title: >-
  [Paper Note] CONSIGN: Conformal Segmentation Informed by Spatial Groupings via Decomposition
description: >-
  [ICLR 2026][Medical Imaging][Conformal Prediction] CONSIGN utilizes SVD to extract spatially correlated "principal uncertainty directions" from multiple samples of segmentation models, constructing a space-aware conformal prediction set that can vary jointly. This reduces the prediction set volume by several orders of magnitude compared to pixel-wise methods while maintaining statistical coverage guarantees.
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Conformal Prediction"
  - "Conformal Risk Control"
  - "Image Segmentation"
  - "Uncertainty Quantification"
  - "SVD/PCA"
date: 2026-05-08
content_hash: f1dd2634890010fd
---

# CONSIGN: Conformal Segmentation Informed by Spatial Groupings via Decomposition

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=pHF5CXB0YH](https://openreview.net/forum?id=pHF5CXB0YH)  
**Code**: TBD  
**Area**: Medical Imaging / Uncertainty Quantification / Conformal Prediction  
**Keywords**: Conformal Prediction, Conformal Risk Control, Image Segmentation, Uncertainty Quantification, SVD/PCA, Medical Imaging  

## TL;DR
CONSIGN utilizes SVD to extract spatially correlated "principal uncertainty directions" from multiple samples of segmentation models, constructing a space-aware conformal prediction set that can vary jointly. This reduces the prediction set volume by several orders of magnitude compared to pixel-wise methods while maintaining statistical coverage guarantees.

## Background & Motivation
**Background**: Pixel-wise softmax confidence scores provided by segmentation models are merely heuristic scores and lack statistical reliability; in high-risk scenarios like medical imaging, "90% model certainty" does not equate to "90% true coverage." Conformal Prediction (CP) provides a rigorous framework to transform heuristic uncertainty into finite-sample coverage guarantees. This paper employs Conformal Risk Control (CRC), aiming for the prediction set $C(X)$ to satisfy $\mathbb{E}[\ell(C(X_{test}), Y_{test})] \le \alpha$.

**Limitations of Prior Work**: When applying CP directly to segmentation, most methods (such as RAPS, Mossina, etc.) construct prediction sets **independently per pixel**: $C_\lambda(X_{ij}) = \{l : f(X_{ij})_l \ge 1-\lambda\}$. However, strong spatial correlations exist between image pixels—labels of adjacent pixels often change together. While pixel-wise independent processing preserves coverage guarantees, it populates the prediction set with numerous inconsistent combinations that are "practically impossible to occur simultaneously" (e.g., haphazard stitching of a sheep and a cow at a boundary), leading to **volume explosion**, over-conservatism, and poor interpretability.

**Key Challenge**: Coverage guarantees require "enlarging rather than missing," but pixel-wise independent enlargement ignores spatial structure, resulting in unnecessarily large sets. The challenge is how to maintain **statistical guarantees** while allowing the prediction set to expand only along **truly meaningful directions of spatial variation**.

**Goal**: To construct a space-aware conformal prediction set where samples vary jointly along coordinated spatial structures, significantly shrinking the prediction set volume while remaining compatible with any pre-trained segmentation model capable of producing multiple sampled outputs (e.g., dropout, Bayesian, ensemble).

**Core Idea**: **[Spatial Decomposition]** Drawing inspiration from using SVD to extract principal uncertainty directions in image restoration (Belhasin 2023, Nehme 2023), the method extracts a few principal components from the model's sampling matrix as an "uncertainty basis." The prediction set expands only within the coefficient intervals of these bases. **[Nonlinear Quantization]** Exploiting the discrete nature of segmentation as a classification task, an argmax projection $P(\sigma)$ is introduced, enabling even a truncated set of $K \in \{2,5\}$ principal components to cover the ground truth without needing the specialized procedures required in regression versions.

## Method

### Overall Architecture
CONSIGN consists of two steps: **constructing the space-aware prediction set** and **calibration**. Given an image, $N_s$ softmax outputs are first sampled from a pre-trained model $f$. These form a sample matrix for SVD, from which the top $K$ principal component directions $\{u_k\}$ are taken as the uncertainty basis. The prediction set is defined as "all segmentation maps obtainable by the argmax projection of the mean plus linear combinations of these basis directions." During the calibration phase, the scale parameter $\lambda$ is incrementally increased on the calibration set to expand the coefficient intervals until the empirical risk falls below the threshold, yielding $\hat\lambda$ that satisfies the coverage guarantee.

```mermaid
flowchart TD
    A[Pre-trained model f multiple sampling<br/>to obtain N_s softmax outputs] --> B[Construct sample matrix Ŝ-μ<br/>Perform SVD: UΣVᵀ]
    B --> C[Take top K∈{2,5} principal components u_k<br/>Extract spatial uncertainty directions]
    C --> D[Calculate coefficient quantile intervals<br/>a_k, b_k for each u_k]
    D --> E["Prediction set C*_λ: Mean + Σ c_k u_k<br/>via argmax projection P(·), c_k∈[A_k,B_k]"]
    F --> G[Obtain λ̂ satisfying coverage guarantee<br/>E[ℓ] ≤ α]
    E --> F[Calibration: Incrementally increase λ<br/>Solve constrained optimization to determine if Y is in set]
```

### Key Designs

**1. Using SVD to extract spatially correlated principal uncertainty directions: Compressing "where is uncertain and how it varies together" into several basis vectors.** CONSIGN does not require specific model architectures; it only requires samples $\hat s_1, \dots, \hat s_{N_s} \in \mathbb{R}^{WHL}$ (from dropout, probabilistic U-Net, or ensembles). It subtracts the mean from samples to form a matrix for reduced SVD: $\hat S - \mu(X)\cdot \mathbf{1}^T = U\Sigma V^T$, where each column $u_k$ of $U$ is a principal direction with maximum variance in the sampling space, naturally encoding which pixel regions are uncertain and how they vary cooperatively. Crucially, only the first $K < \min\{WHL, N_s\}$ singular values are needed; experiments show $K \in \{2,5\}$ is sufficient, drastically reducing computation compared to a full basis representation ($K=WHL$). This step is the source of the method's "spatial awareness" and "low-dimensional efficiency."

**2. Defining prediction sets in the principal component coefficient space: Allowing predictions to expand jointly along meaningful directions rather than exploding pixel-wise.** For each principal direction, empirical quantile intervals $a_k = Q_{\alpha/2}\{\langle u_k, \hat s_n - \mu\rangle\}$ and $b_k = Q_{1-\alpha/2}\{\cdots\}$ are calculated. A symmetric interval linearly scaled by $\lambda$ is then constructed around the quantile midpoint:

$$A_k = \frac{a_k+b_k}{2} - \lambda \Sigma_{k,k}\frac{b_k-a_k}{2}, \quad B_k = \frac{a_k+b_k}{2} + \lambda \Sigma_{k,k}\frac{b_k-a_k}{2}.$$

Weighting by singular values $\Sigma_{k,k}$ means principal components with larger variance are allowed wider coefficient ranges—more degrees of freedom are given to directions where the model is more uncertain. The prediction set is defined as:

$$C^*_\lambda(X) = \Big\{ Y : \exists c \in \prod_{k=1}^K [A_k, B_k],\ Y \overset{\beta}{=} P\big(\mu(X) + \sum_{k=1}^K c_k u_k(X)\big) \Big\}.$$

Each $c$ falling within the coefficient box yields a complete, spatially coordinated segmentation map via argmax projection $P(\cdot)$, which is why it is more "realistic and coherent" than pixel-wise methods.

**3. Nonlinear argmax projection allowing truncated PCA to cover ground truth: Turning the discreteness of classification into an advantage.** Unlike the regression version (Belhasin 2023), CONSIGN inserts a nonlinear quantization step $P(\sigma)$ (argmax along the label dimension) between coefficients and labels. Because this step quantizes continuous softmax values into discrete labels, even if coefficients for $u_{K+1}, \dots, u_{WHL}$ are forced to zero using only the first $K$ principal components, the reconstructed segmentation map can typically hit the ground truth when $\lambda$ is large enough. This is impossible in regression, which requires extra procedures to ensure coverage. The paper also replaces precision with **per-label precision** $\beta$: $Y_1 \overset{\beta}{=} Y_2$ when $\frac{1}{L}\sum_l \frac{\sum_{ij}\mathbb{I}(Y_1^{ij}=l \wedge Y_2^{ij}=l)}{\sum_{ij}\mathbb{I}(Y_1^{ij}=l)} > \beta$, avoiding bias toward high-frequency labels.

**4. Constrained optimization calibration + termination mechanism: Maintaining guarantees when membership in the set cannot be explicitly determined.** Since the prediction set form is complex, it is impossible to enumerate all $c$ to check if $Y \in C^*_\lambda(X)$. CONSIGN instead solves a constrained minimization $c^* = \arg\min_{c\in B} L(Y, P(\mu(X) + \sum_k c_k u_k))$. If the numerical solution satisfies $Y \overset{\beta}{=} P\sigma$, a "hit" is determined. The calibration algorithm increases $\lambda$ until the empirical risk $\hat R(\lambda) \le \alpha - \frac{1-\alpha}{N_{cal}}$. Even if the numerical solver misses an existing $c$ (leading to a slightly higher $\lambda$ and conservative intervals), the coverage guarantee $\mathbb{P}[Y_{test}\in C^*_{\hat\lambda}] \ge 1-\alpha$ still holds by Angelopoulos's theorem (Lemma 1). For pathological cases where principal directions degenerate and do not converge even as $\lambda \to \infty$, a $\lambda_{max}$ is set to force termination, **explicitly informing the user** that a meaningful prediction set cannot be obtained under current parameters, rather than providing a failed set quietly.

## Key Experimental Results

### Main Results Setup
- **Datasets (5)**: Three medical — M&Ms-2, MS-CMR19 (cardiac, dropout U-Net sampling), LIDC (lung nodules, probabilistic U-Net sampling); two COCO subsets — animals, vehicles (ensemble sampling from DeepLabV3+ with different backbones).
- **Baselines**: Pixel-wise RAPS (PW), space-aware SACP (Liu 2025).
- **CONSIGN Variants**: Two principal component counts, $K=2$ and $K=5$.
- **Metrics**: Chao estimator (estimates the number of unique segmentation maps in the prediction set, lower is better), sEC (sample estimated coverage, should converge to $1-\alpha$), mean Pearson correlation $\hat\rho$ (measures if samples are constrained to a low-dimensional manifold).

| Dimension | CONSIGN | PW (RAPS) / SACP |
|------|---------|------------------|
| Chao Estimator (Set Volume) | Consistently **bounded** by (smaller than) both baselines; largest gap on LIDC (orders of magnitude) | Significantly larger |
| sEC Convergence Speed | Reaches $1-\alpha$ coverage with fewer samples; COCO-vehicle satisfied with only 10 samples | Slower convergence |
| Correlation between samples $\hat\rho$ | High correlation → constrained to low-dimensional subspace | Near independent → higher intrinsic dimensionality |

### Key Findings
- **Gains from spatial structure depend on sampling quality**: Using probabilistic U-Nets (LIDC) designed to produce diverse samples results in volumes **orders of magnitude** smaller than baselines; gains are more modest with dropout sampling (cardiac data).
- **Trade-off between $K=2$ vs $K=5$**: $K=5$ provides more freedom to coefficients, potentially leading to wider prediction ranges and higher Chao estimates; $K=2$ is more compact. In most cases, a few principal components are sufficient.
- **COCO high uncertainty scenarios**: When objects are large and intrinsic uncertainty is high, CONSIGN's Chao estimate also increases, indicating that gains correlate with the strength of the task's spatial structure.
- **Heuristics for parameter selection**: Use small $\alpha$ (e.g., 0.05) for strict reliability in safety-critical scenarios like medicine; use high $\beta$ (>0.8) for fine structures; $\beta$ can be relaxed for large object segmentation; stronger models allow for more aggressive parameters.

## Highlights & Insights
- **Operationalizing "spatial correlation" into computable low-dimensional bases**: SVD principal components locate uncertain regions and encode how pixels vary together. A few directions can represent the entire prediction set, which is the source of its volume advantage.
- **Exploiting the discreteness of classification tasks**: Argmax quantization allows truncated PCA to still cover the ground truth, eliminating the patch-up procedures of regression versions—an elegant design where a task characteristic becomes a method advantage.
- **Model-agnostic + Uncompromised Guarantees**: Applicable as long as the model can be sampled, and statistical coverage is theoretically backed even if numerical solvers are imperfect, making it practical for engineering.
- **Explicit alerts on failure**: The $\lambda_{max}$ termination mechanism exposes the inability to obtain an effective prediction set to the user, rather than providing a seemingly reasonable but actually invalid result, which is a responsible design for high-risk scenarios.

## Limitations & Future Work
- **Prediction set volume can only be estimated via sampling**: The true volume of $C^*$ cannot be calculated analytically and must rely on the Chao estimator or sample approximation; evaluation precision is limited by sample count.
- **Dependency on sampling quality**: Gains are highly dependent on whether the pre-trained model can produce meaningful diverse samples; advantages shrink significantly under weak sampling like dropout.
- **Calibration involves numerical optimization**: Constrained minimization cannot guarantee a global optimum, which may make $\lambda$ conservative (sacrificing tightness without breaking guarantees); additionally, solving optimization for every calibration sample is more computationally expensive than direct pixel-wise judgment.
- **Pathological cases require manual $\lambda_{max}$**: The method terminates when principal directions degenerate; selecting $\alpha, \beta, \lambda_{max}$ requires domain expertise.
- **Future Work**: Extending spatial decomposition to stronger generative samplers, automated parameter selection, and reducing the overhead of constrained optimization during calibration are natural next steps.

## Related Work & Insights
- **Conformal Prediction / CRC**: Vovk, Angelopoulos, et al. established the CP and CRC frameworks; this paper proves coverage guarantees based on the calibration theorem in Angelopoulos (2024).
- **Pixel-wise segmentation CP**: Wundram (2024) and Mossina (2024) extended CRC to multi-class segmentation but remained pixel-wise; Blot (2025) performed group-conditioned risk control; this paper points out their common volume explosion issue.
- **Space-aware CP**: SACP (Liu 2025) aggregates scores in neighborhoods and is the most direct comparison; Brunekreef (2024) aggregates non-conformity scores by similar regions but relies on custom calibration; Davenport (2024) makes scores dependent on distance to boundaries.
- **SVD for uncertainty extraction**: Belhasin (2023) and Nehme (2023) used SVD for principal directions in image restoration/regression; this paper migrates this to segmentation and adapts it with argmax projection, forming the core methodological source.
- **Mechanism Insight**: In structured output tasks, "compressing output space correlation into low-dimensional bases, then allowing uncertainty to expand along these bases" is a general paradigm applicable to depth estimation, optical flow, video segmentation, or any dense prediction task capable of multiple samplings.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Transitioning SVD principal component uncertainty from regression to segmentation and cleverly solving the truncated PCA coverage problem via argmax quantization is a clear and theoretically supported innovative combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 3 medical + 2 COCO datasets, three sampling models, two baselines, and multiple metrics with random splits, though it lacks side-by-side comparisons with more recent space-aware CP methods, and volume is restricted to sample estimation.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation and method derivation are clear; formulas and algorithm pseudocode are complete; coverage guarantees are supported by Lemmas; some notation is dense.
- **Value**: ⭐⭐⭐⭐ — Model-agnostic, statistically guaranteed, and significantly compresses prediction set volume; highly valuable for trustworthy UQ in high-risk segmentation like medical imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[ICLR 2026\] HFSTI-Net: Hierarchical Frequency-spatial-temporal Interactions for Video Polyp Segmentation](hfsti-net_hierarchical_frequency-spatial-temporal_interactions_for_video_polyp_s.md)
- [\[AAAI 2026\] Provably Minimum-Length Conformal Prediction Sets for Ordinal Classification](../../AAAI2026/medical_imaging/provably_minimum-length_conformal_prediction_sets_for_ordinal_classification.md)
- [\[CVPR 2026\] PMRNet: Physics-informed Multi-scale Refinement Network for Medical Image Segmentation](../../CVPR2026/medical_imaging/pmrnet_physics-informed_multi-scale_refinement_network_for_medical_image_segment.md)
- [\[AAAI 2026\] Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation](../../AAAI2026/medical_imaging/unsupervised_motion-compensated_decomposition_for_cardiac_mri_reconstruction_via.md)

</div>

<!-- RELATED:END -->
