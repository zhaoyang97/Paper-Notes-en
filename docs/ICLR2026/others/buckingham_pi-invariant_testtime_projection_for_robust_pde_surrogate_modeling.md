---
title: >-
  [Paper Note] Buckingham $\pi$-Invariant Test-Time Projection for Robust PDE Surrogate Modeling
description: >-
  [ICLR 2026][PDE surrogate models] Utilizing the Buckingham π theorem, this work identifies "OOD shifts caused by different units/scales" as physically equivalent scaling transformations. It proposes a **training-free, model-agnostic** test-time projection: translating test samples along $\pi$-preserving equivalence classes in log space to the nearest training class. This approach reduces the MAE of surrogate models such as FNO/U-Net by up to 91% under extreme OOD conditions.
tags:
  - "ICLR 2026"
  - "PDE surrogate models"
  - "Buckingham π theorem"
  - "dimensional analysis"
  - "test-time projection"
  - "OOD generalization"
  - "neural operators"
date: 2026-05-08
content_hash: 793d871b508f346e
---

# Buckingham $\pi$-Invariant Test-Time Projection for Robust PDE Surrogate Modeling

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=2FFhwssQda](https://openreview.net/forum?id=2FFhwssQda)  
**Code**: To be confirmed  
**Area**: AI for Science / PDE Surrogate Modeling  
**Keywords**: PDE surrogate models, Buckingham π theorem, dimensional analysis, test-time projection, OOD generalization, neural operators  

## TL;DR
Utilizing the Buckingham π theorem, this work identifies "OOD shifts caused by different units/scales" as physically equivalent scaling transformations. It proposes a **training-free, model-agnostic** test-time projection: translating test samples along $\pi$-preserving equivalence classes in log space to the nearest training class. This approach reduces the MAE of surrogate models such as FNO/U-Net by up to 91% under extreme OOD conditions.

## Background & Motivation
- **Background**: PDE surrogate models like FNO and PINN exhibit strong performance in in-distribution interpolation. However, prediction accuracy collapses when the physical units and numerical scales of test inputs differ significantly from the training set (e.g., a heat source $q$ spanning 5 orders of magnitude). OOD generalization remains a core challenge in physical machine learning.
- **Limitations of Prior Work**: ① **Dimensionless learning** often remains restricted to scalar scenarios. Directly applying pixel-wise Buckingham-π scaling to 2D/3D spatial fields fails due to numerical instability—zeros in the numerator collapse inputs to 0, while near-zero values in the denominator cause π to diverge. ② **Test-time training/adaptation** (TTT/TTA) requires backpropagation during inference, incurring optimization overhead and latency, and is rarely designed for regression and spatial fields.
- **Key Challenge**: Many instances of OOD regarded as "distribution shifts" are essentially **scaling transformations that maintain invariant $\pi$ groups** (e.g., dynamics are equivalent if the Reynolds number is identical). However, surrogate models treat these as entirely new distributions—this is a representation problem, not a true distributional issue.
- **Goal**: To "align" OOD test inputs to the vicinity of the training distribution without retraining the model or altering the architecture, while strictly maintaining their Buckingham π invariants, thereby restoring OOD accuracy with minimal overhead.
- **Core Idea**: **【$\pi$-Preserving Test-Time Projection】** Dimensional scaling is viewed as a translation in log space. The physical content of the input resides solely within the π values. By moving test samples within their own π-equivalence class (an affine subspace) and projecting them onto the nearest training π-equivalence class, alignment is achieved by solving a small-scale log-space least-squares problem.

## Method

### Overall Architecture
The method serves as a **pure inference-time preprocessing step** applicable to any pretrained surrogate model, consisting of three stages: ① **Domain Profile Reduction**—compressing high-dimensional spatial fields into a few representative variables via arithmetic means to avoid pixel-wise π degradation; ② **$\pi$-Preserving Test-Time Projection**—translating test samples along $\pi$-preserving directions in log space to the nearest training class to determine the scaling coefficient $\exp(v^*)$; ③ **Offline π-uniform + Centroid Reduction**—balancing the training set π distribution using dominant parameters and applying K-means to reduce projection complexity from $O(MN)$ to $O(KN)$. After prediction, the inverse transformation restores the physical quantities based on the scaling coefficients.

```mermaid
flowchart LR
    A[Test Field X̃] --> B[Domain Profile Reduction<br/>Field → Representative x̃]
    B --> C[Log-space Decomposition<br/>π-preserving Component ∥ / π-altering Component ⊥]
    C --> D[Find Nearest Training Equivalence Class i*<br/>O(KN) Centroid Reduction]
    D --> E[Scaling Factor exp v*<br/>Per-channel Scaling X̃*=X̃⊙exp v*]
    E --> F[Surrogate Model<br/>CNN/U-Net/FNO]
    F --> G[Inverse Scaling → Predicted Solution]
    H[(Training Set)] -.π-uniform Sampling + K-means.-> D
```

### Key Designs

**1. Domain Profile Reduction: Replacing pixel-wise π with global statistics to cure degradation.** Directly applying Buckingham-π to spatial fields encounters a fatal issue: if the heat source field $q$ or body force field $f$ is zero at certain locations, pixel-wise π values collapse or diverge, termed "π-information loss." The solution introduces a feature extractor $\psi: X \mapsto x$, mapping discrete fields to finite feature variables using the **arithmetic mean** $\bar k, \bar q, \bar E, \bar f$ as representative scalars. These global statistics are naturally robust to local zeros and outliers, ensuring non-zero representative scales for numerically stable log-linear projections. This is a critical step for generalizing π methods from scalars to 2D spatial fields.

**2. Log-space $\pi$-Preserving Projection: Decomposing alignment into "$\pi$-preserving translation + $\pi$-altering physical difference."** From the log-form of the Buckingham π theorem, $\log \Pi(x)=\Phi^\top \log x$, where $\Phi$ spans the null space of the dimensional matrix. Componentwise scaling $x\mapsto x\odot\exp(v)$ is a translation $z\mapsto z+v$ in log space. When $v\in\ker(\Phi^\top)$, π values remain invariant, and each input $z$ generates an affine π-equivalence class $z+\ker(\Phi^\top)$. The optimization goal is to find the point within the test equivalence class closest to the training set:

$$\tilde x^* = \arg\min_{\tilde x' \in [\psi(\tilde X)]_\pi}\ \mathrm{dist}\big(\tilde x', \{\psi(X_i)\}_{i=1}^M\big)$$

Let $v_i^t = z_i - \tilde z$. The projection operator $P_\parallel = I - \Phi(\Phi^\top\Phi)^{-1}\Phi^\top$ decomposes this into an intra-class component $P_\parallel v_i^t$ (scaling change) and an inter-class component $(I-P_\parallel)v_i^t$ (true physical difference). Setting $v_i^*=P_\parallel v_i^t$ yields the quotient distance between the test class and the $i$-th training class. The optimal training class is $i^*=\arg\min_i \|v_i^t - v_i^*\|^2$, resulting in the scaled input $\tilde x^* = \tilde x\odot\exp(v^*)$. This process is a constrained least-squares problem in log space with minimal computational cost.

**3. π-Uniform Strategy: Balancing the training π distribution via dominant parameters to broaden coverage.** Projection is only effective if the training set π coverage is sufficiently wide; however, raw distributions are often skewed. SHAP analysis identifies **dominant parameters** whose scales most directly control the π group (e.g., $q$ contributes 48.7% in thermal problems). Other inputs are fixed while dominant parameters are adjusted so that each training sample's π value matches a target uniform distribution. Notably, $\log\pi$ is uniformized rather than $\pi$ to maximize coverage. Combined with $\pi$-preserving constraints (e.g., $\log\beta+2\log\delta-\log\alpha-\log\gamma=0$), the training space covers a wider π interval.

**4. Centroid Reduction: Reducing projection complexity from $O(MN)$ to $O(KN)$ with K-means.** Comparison between $N$ test samples and $M$ training samples takes $O(MN)$. The authors perform K-means on the uniformized log features, retaining only $K\in\{1,\dots,10\}$ centroids as representative points, reducing complexity to $O(KN)$. Experiments show that the MAE of clustered projection is comparable to the full-set baseline within statistical noise, while projection time is reduced by approximately $100\times$. When π is ill-conditioned (e.g., $q\approx 0$), the method degrades gracefully by performing alignment on non-degenerate channels while preserving spatial heterogeneity.

## Key Experimental Results

### Main Results
OOD testing for 2D steady-state heat conduction (Thermal) and linear elasticity (Stress) where training/test π ranges are disjoint (e.g., $\log_{10}q$ training $[0,7.5]$, test $[7.5,12]$):

| Method | Thermal MAE | Thermal RMSE | Time | Stress MAE | Stress RMSE | Time |
|---|---|---|---|---|---|---|
| CNN | 8.43 | 9.99 | - | 0.96 | 1.17 | - |
| CNN + Pairwise Proj. | 2.63 | 3.24 | 100.3 | 0.53 | 0.71 | 73.6 |
| CNN + π-uniform + 10-Centroids | **1.79** | **2.23** | 1.80 | **0.60** | **0.84** | 1.36 |
| U-Net | 13.60 | 15.29 | - | 0.81 | 0.99 | - |
| U-Net + Pairwise Proj. | 1.75 | 2.31 | 99.1 | 0.17 | 0.28 | 94.9 |
| U-Net + π-uniform + 10-Centroids | **1.18** | **1.53** | 2.31 | **0.22** | **0.39** | 1.58 |
| FNO | 9.88 | 11.43 | - | 3.20 | 4.19 | - |
| FNO + Pairwise Proj. | 1.38 | 1.74 | 151.4 | 0.28 | 0.42 | 94.0 |
| FNO + π-uniform + 10-Centroids | **1.25** | **1.60** | 2.31 | **0.33** | **0.53** | 1.54 |

For U-Net, the Thermal MAE dropped from 13.60 to 1.18 (~**91%** reduction); for FNO, it dropped from 9.88 to 1.25.

### Ablation Study
Comparison of representative candidates for reduction (Baseline=Full set / Clustered=K Centroids / Random=K Random Samples):

| Projection Type | MAE | Projection Time |
|---|---|---|
| Pairwise (Full Baseline) | Reference Low | $O(MN)$, ~100–150s |
| K-Centroids (Clustered) | On par with baseline | ~$100\times$ speedup |
| K-Randoms (Random) | Significantly higher MAE | Same order |

As the number of candidates increases, the MAE of clustered projection converges to the full baseline, whereas random projection retains a gap, validating the representativeness of centroids.

### Key Findings
- **Improvements concentrated on worst OOD cases**: The top-3 worst cases showed the largest improvements, precisely where original models failed to extrapolate.
- **Near-zero additional training**: The method is training-free and model-agnostic, consistently effective across CNN, U-Net, and FNO architectures.
- **Accuracy and speed**: Centroid reduction reduces projection time from ~100s to ~1.8s with almost no loss in precision.

## Highlights & Insights
- **Redefining OOD**: OOD shifts resulting from unit/scale differences are reinterpreted as $\pi$-preserving scaling equivalence. OOD is transformed from a "hard adaptation problem" to a "representation problem resolvable at the input level," offering a novel, physically interpretable perspective.
- **Geometric Elegance**: The orthogonal decomposition of intra-class (π-preserving scale) ⊥ inter-class (π-altering difference) translates the abstract quotient space of dimensions into a computable projection operator.
- **True Plug-and-Play**: Acting as an inference-time preprocessing step, it enhances any surrogate model without modifying training or architecture, ensuring low deployment costs.

## Limitations & Future Work
- **Strong Assumptions**: Requires the system to be PDE-governed; systems with empirical or hybrid components may fail.
- **Information Loss in Statistics**: Summarizing fields with arithmetic means blurs highly irregular spatial patterns.
- **Degradation in Extreme OOD**: Performance still drops for extreme samples far beyond the training π range, even with π-uniform expansion.
- **Limited Task Scope**: Only validated on static PDEs (thermal and elasticity). Future work includes extensions to transient/convective PDEs (advection-diffusion, Navier-Stokes) and uncertainty-aware projections.

## Related Work & Insights
- **Dimensional Analysis in Learning** (Bakarji 2022, Xie 2022, SINDy): This work generalizes scalar π learning to 2D spatial fields and systematically addresses failure modes like zero-set collapse.
- **Neural Operators** (FNO and variants, PINO): This method does not compete with them but serves as a training-free robustness enhancement layer.
- **Test-Time Training/Adaptation** (TTT, Tent): This provides a sample-level, $\pi$-preserving alignment alternative that avoids backpropagation-based optimization.
- **Insight**: When OOD "shifts" involve known physical symmetries (dimensions, scaling, translation groups), explicit analytical correction at the input level is often superior to forcing the model to learn invariants.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Combining Buckingham-π invariants with test-time projection while solving π degradation in spatial fields is a fresh perspective.
- **Experimental Thoroughness**: ⭐⭐⭐ Robust validation across three architectures and two PDEs; however, lacks transient/convection validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical derivations (log-form, orthogonal decomposition) and effective visualizations.
- **Value**: ⭐⭐⭐⭐ Training-free, model-agnostic, and plug-and-play; highly practical for the OOD deployment of AI4Science surrogate models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Prior-Free Tabular Test-Time Adaptation](prior-free_tabular_test-time_adaptation.md)
- [\[ICLR 2026\] PriorGuide: Test-Time Prior Adaptation for Simulation-Based Inference](priorguide_test-time_prior_adaptation_for_simulation-based_inference.md)
- [\[ICLR 2026\] Exposing Mixture and Annotating Confusion for Active Universal Test-Time Adaptation](exposing_mixture_and_annotating_confusion_for_active_universal_test-time_adaptat.md)
- [\[ICLR 2026\] Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute](scaling_atomistic_protein_binder_design_with_generative_pretraining_and_test-tim.md)
- [\[ECCV 2024\] MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory](../../ECCV2024/others/membn_robust_test-time_adaptation_via_batch_norm_with_statistics_memory.md)

</div>

<!-- RELATED:END -->
