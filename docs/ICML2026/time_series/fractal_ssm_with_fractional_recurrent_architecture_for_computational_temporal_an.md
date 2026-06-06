---
title: >-
  [Paper Note] FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences
description: >-
  [ICML 2026][Time Series][HiPPO] This paper generalizes the probability measures behind the HiPPO framework to fractional power-law measures with a tunable singularity index $\alpha$…
tags:
  - "ICML 2026"
  - "Time Series"
  - "HiPPO"
  - "Fractional Calculus"
  - "State Space Models"
  - "Long Range Dependencies"
  - "Long Range Arena"
date: 2026-05-08
content_hash: c74d0e0b2a8e33f0
---

# FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences

**Conference**: ICML 2026  
**arXiv**: [2605.08833](https://arxiv.org/abs/2605.08833)  
**Code**: None  
**Area**: Sequence Modeling / State Space Models (SSM)  
**Keywords**: HiPPO, Fractional Calculus, State Space Models, Long Range Dependencies, Long Range Arena

## TL;DR
This paper generalizes the probability measures behind the HiPPO framework to fractional power-law measures with a tunable singularity index $\alpha$, achieving "full history retention + recency sensitivity + scale invariance" for the first time. This theory is implemented as an LTI diagonalized SSM—FRACTAL—which achieves an average score of 87.11% on Long Range Arena, matching S5 and reaching 61.85% on ListOps.

## Background & Motivation
**Background**: Modern SSMs (S4 / S4D / DSS / S5 / Mamba) are almost all built upon HiPPO's "online polynomial projection + probability measure" framework. While architectures have iterated through several generations—from structured matrices to diagonalization and then to time-varying selective SSMs— the choice of the underlying measure used to weight history has remained unchanged for years.

**Limitations of Prior Work**: The classic triad of measures faces inherent deadlocks: LegS (uniform) preserves the full history and scale invariance, but recent signals are diluted by $1/t$; LagT (exponential) highlights the present but fixes the time scale, failing if the signal is slowed down or sped up; LegT (sliding window) provides high local resolution but suffers from complete forgetting outside the window. Table 1 contrasts these measures against three properties and identifies an "Impossible Trinity."

**Key Challenge**: Simultaneously achieving "memory extending infinitely," "sufficient sensitivity to recent inputs," and "robustness to signal scaling" is fundamentally impossible for integer-order differential equations—they either spread memory uniformly or decay it exponentially.

**Goal**: Without abandoning HiPPO's "online polynomial projection + closed-form dynamics," the goal is to find a class of measures that satisfies all three properties and implement it as a parallelizable LTI diagonal SSM.

**Key Insight**: Fractional calculus naturally describes non-local, heavy-tailed memory. By replacing the "indicator function" in the measure with a power-law singularity $(t-x)^{-\alpha}$, the singularity index $\alpha\in[0,1)$ can continuously interpolate between LegS ($\alpha=0$) and a near-delta function ($\alpha\to 1$).

**Core Idea**: Replace HiPPO's uniform/exponential measures with power-law singular measures. This is equivalent to generalizing the basis from Legendre to Jacobi polynomials $P_n^{(-\alpha,0)}$ while retaining closed-form derivable SSM coefficients.

## Method

### Overall Architecture
FRACTAL consists of two phases. **Phase 1 (Offline Initialization)**: Given multi-channel singularity indices $\boldsymbol{\alpha}=(\alpha_1,\dots,\alpha_K)$, the fractional measure $\mu^{(t)}(x)=(1-\alpha)t^{\alpha-1}(t-x)^{-\alpha}\mathbb{I}_{[0,t]}(x)$ is defined (Def. 3.1). Its orthogonal basis (normalized Jacobi polynomials) is substituted into the HiPPO projection equations to derive LTV dynamics $\dot{x}=-\frac{1}{t}A(\alpha)x+\frac{1}{t}B(\alpha)u$. Eigen-decomposition is performed on $A(\alpha)$ to obtain $\Lambda$, and a closed-form formula is used for physically meaningful initialization $\tilde{B}_{\text{init}}=V^{-1}B$. **Phase 2 (Online Training)**: The $1/t$ term in the LTV system is relaxed to a learnable time scale $\Delta$ to obtain an LTI system, and scanning is completed in $O(N\log L)$ time using ZOH discretization and parallel prefix-sum. Finally, the SSM output is wrapped in a GLU to form a standard gated SSM block.

### Key Designs

1.  **Fractional HiPPO Measure (Definition 3.1)**:
    *   **Function**: Uses a family of probability measures with an adjustable singularity index $\alpha$ to continuously interpolate between "uniform memory" and "concentration on the present" while maintaining scale invariance.
    *   **Mechanism**: $\mu^{(t)}(x)=(1-\alpha)t^{\alpha-1}(t-x)^{-\alpha}\mathbb{I}_{[0,t]}(x)$, where the normalization factor is derived from $\int_0^t(t-x)^{-\alpha}dx=t^{1-\alpha}/(1-\alpha)$. $\alpha=0$ reduces to LegS, and $\alpha\to 1$ makes the measure approach $\delta_t$. Mapping the domain $[0,t]$ to $[-1,1]$ via $y=2x/t-1$ makes the weight function the Jacobi weight $(1-y)^{-\alpha}$, thus the orthogonal bases are naturally Jacobi polynomials $P_n^{(-\alpha,0)}$.
    *   **Design Motivation**: Power-law singularities simultaneously preserve "heavy-tailed long-range memory" and "high recency response," while the normalized form of $\mu$ remains unchanged under $t\mapsto\lambda t$, thus preserving scale invariance and breaking the "Impossible Trinity" of Table 1.

2.  **Provable State Matrix Structure (Theorem 3.4)**:
    *   **Function**: Provides the analytical structure of $A(\alpha)\in\mathbb{R}^{N\times N}$ after the LTI relaxation, ensuring diagonalizability and spectral stability without needing engineering approximations like NPLR (normal-plus-low-rank).
    *   **Mechanism**: Performing a Galerkin projection $\mathcal{L}[P_n]=P_n+(1+\eta)P_n'$ under the normalized Jacobi basis proves that $A(\alpha)$ is strictly lower triangular. The diagonal elements $A_{nn}=n+1$ are independent of $\alpha$. The non-diagonal elements $A_{nk}$ ($k<n$) are given by $\langle\mathcal{L}[P_n^{(-\alpha,0)}],P_k^{(-\alpha,0)}\rangle_w/\|P_k\|_w^2$, which reduces to $\sqrt{(2n+1)(2k+1)}$ only when $\alpha=0$; otherwise, they are calculated via Gauss–Jacobi numerical integration. The $B$ term has a closed form $B_n=\sqrt{(2n+1-\alpha)/(1-\alpha)}\binom{n-\alpha}{n}$.
    *   **Design Motivation**: The fact that "diagonal elements are always $1,\dots,N$" is the most critical stability guarantee—it means eigenvalues do not change regardless of how $\alpha$ is tuned; only the eigenvectors rotate. This allows for a direct diagonal SSM via $A=V\Lambda V^{-1}$ without complex approximations.

3.  **Fractional Filter Bank Multi-channel Architecture (§4.3)**:
    *   **Function**: Captures features at different time scales by capturing sub-states with different $\alpha_k$ in parallel, equivalent to structuring the SSM state space as a set of "frequency domain filters."
    *   **Mechanism**: The state dimension $H$ is partitioned into $K$ blocks, each using a different $\alpha_k$. Low $\alpha$ channels act like low-pass filters (preserving global context and denoising), while high $\alpha$ channels act like band-pass/high-pass filters (highlighting local transitions). The output projection $C$ learns to combine these temporal bases. $\Delta$ controls "how far" (resolution) while $\alpha$ controls "how to look" (memory topology), decoupling the two.
    *   **Design Motivation**: Experimental observation shows that ListOps requires both long-range bracket matching and local numerical values; a single $\alpha$ cannot handle both. Using a filter bank naturally decouples different scales, which is the fundamental reason for FRACTAL's performance gains on ListOps.

### Loss & Training
No new loss functions were introduced; standard CE or BCE from LRA tasks were used. $\alpha_k$ was assigned via linear spacing in the $[0, 0.9]$ interval and remains fixed (the paper lists learnable $\alpha$ as future work). Other hyperparameters were kept consistent with S5 to ensure a fair comparison.

## Key Experimental Results

### Main Results: Long Range Arena (Table 2)

| Model | ListOps | Text | Retrieval | Image | Pathfinder | Path-X | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Transformer | 36.37 | 64.27 | 57.46 | 42.44 | 71.40 | ✗ | – |
| S4 | 59.60 | 86.82 | 90.90 | 88.65 | 94.20 | 96.35 | 86.09 |
| S4D | 60.47 | 86.18 | 89.46 | 88.19 | 93.06 | 91.95 | 84.89 |
| DSS | 57.60 | 84.80 | 87.60 | 84.40 | 85.00 | 85.00 | 80.73 |
| S5 (Reproduction) | 61.10 | 88.72 | 91.27 | 87.59 | 95.04 | 98.62 | 87.04 |
| **FRACTAL** | **61.85** | **89.10** | 91.19 | 87.30 | 94.80 | 98.39 | **87.11** |

### Ablation Study

| Setting | Key Metric | Description |
| :--- | :--- | :--- |
| $\alpha=0$ (Reduces to LegS / S4-like) | On par with S4 | Validates that the framework strictly generalizes existing methods. |
| Single $\alpha$ (No filter bank) | Significant drop in ListOps | A single time scale cannot capture both global brackets and local digits. |
| Random $B$ vs Analytical $\tilde{B}_{\text{init}}$ | Similar final accuracy | The analytical version has significantly lower initial loss, acting as a pre-conditioner (Remark 4.1). |
| Numerical validation of $A_{nn}=n+1$ | Holds for any $\alpha$ | Perfectly consistent with Theorem 3.4; no spectral drift occurs on long sequences. |

### Key Findings
*   FRACTAL achieved the largest gains on ListOps, where hierarchical and heavy-tailed structures are most prominent, exceeding S5 by 0.75pt and S4 by 2.25pt. Its performance on Image / Pathfinder (primarily local dependencies) was nearly equal to S5. This matches theoretical predictions: "The advantages of power-law measures are concentrated in long-range and multi-scale tasks."
*   On Path-X (length 16K), FRACTAL still achieved 98.39%, indicating that the introduction of singularities does not destabilize gradients on extremely long sequences—providing experimental endorsement for the spectral stability of $A_{nn}=n+1$.
*   While strict theoretical scale invariance is lost after moving away from the LTV assumption, the filter bank retains multi-scale inductive biases, making it more practical for engineering than "strictly scale-invariant but difficult to train" LTV systems.

## Highlights & Insights
*   The narrative shifting from "Impossible Trinity" to "Fractional Unlock" is clean: the authors clearly divide the progress of SSMs into "Architecture vs. Measure" and point out that the measure line has been stagnant for years. This research pattern of "revisiting implicit assumptions" can be applied to many fields (e.g., normalization in attention, noise scheduling in diffusion models).
*   The spectral invariance property where "diagonal elements are independent of $\alpha$" is an elegant byproduct: it means $\alpha$ could be set as a learnable hyperparameter (or even per-token adaptive) without destroying numerical stability—opening the door for "data-driven selection of $\alpha$."
*   Treating different $\alpha$ as different frequency bands using a filter bank can be applied to non-SSM models (such as linear attention or RNN-style models)—by treating the kernel decay shape as an adjustable "band," similar multi-channel structures can obtain multi-scale inductive biases.

## Limitations & Future Work
*   Strict scale invariance only holds under LTV (preserving $1/t$); for parallel scanning, the authors chose to abandon this property. The engineering version reduces to "spectral multi-scale" rather than "true scale invariance." Utilizing scale invariance for physical/physiological signals in the future would require a specialized LTV-friendly scanning algorithm.
*   $\alpha_k$ is currently a fixed linear interval and not learned. The optimal $\alpha$ spectrum may vary significantly across different tasks; making it end-to-end learnable is an obvious next step.
*   The evaluation is limited to LRA. The paper explicitly positions itself as a "train-from-scratch LTI" and does not compare against the Mamba family of selective SSMs on large-scale language tasks; thus, conclusions regarding "language modeling quality" require cautious extrapolation.

## Related Work & Insights
*   **vs S4 / S4D**: S4 treats HiPPO-LegS as a static initialization followed by NPLR engineering; FRACTAL turns the measure into a tunable design parameter and uses spectral invariance for direct diagonalization, bypassing NPLR.
*   **vs DSS / S5**: DSS / S5 argue that "exact matrix structure is not important, but spectral structure is." FRACTAL continues this line of thought but further "designs the spectral structure according to measure laws" rather than using random or approximate initialization.
*   **vs Mamba / Selective SSMs**: Mamba introduces data dependency by making matrices input-dependent; FRACTAL takes an orthogonal direction—sticking to LTI but using measures for inductive bias—though the two could be combined (input-dependent $\alpha$).
*   **vs LMU / Voelker 2019**: LMU uses sliding window Legendre, equivalent to LegT; FRACTAL generalizes this to a tunable $\alpha$, making LMU a special case.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ First introduction of fractional measures into the HiPPO framework with closed-form derivable structures.
*   Experimental Thoroughness: ⭐⭐⭐ Matches S5 on LRA and leads on ListOps, but lacks large-scale language modeling comparisons.
*   Writing Quality: ⭐⭐⭐⭐ The logical chain from the "Impossible Trinity" to FRACTAL is smooth, and the derivations in the appendix are self-consistent.
*   Value: ⭐⭐⭐⭐ Provides a long-overlooked design dimension for SSMs, offering clear significance for long-range signal modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic.md)
- [\[ICML 2026\] HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models](hippo_zoo_explicit_memory_mechanisms_for_interpretable_state_space_models.md)
- [\[ICLR 2026\] WARP: Weight-Space Linear Recurrent Neural Networks](../../ICLR2026/time_series/weight-space_linear_recurrent_neural_networks.md)
- [\[NeurIPS 2025\] RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting](../../NeurIPS2025/time_series/rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)
- [\[ICML 2026\] HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series](hepa_a_self-supervised_horizon-conditioned_event_predictive_architecture_for_tim.md)

</div>

<!-- RELATED:END -->
