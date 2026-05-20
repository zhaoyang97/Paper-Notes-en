---
title: >-
  [Paper Note] FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences
description: >-
  [ICML 2026][Time Series][HiPPO] This work generalizes the probabilistic measure underlying the HiPPO framework to a fractional power-law measure with a tunable singularity index $\alpha$, thereby, for the first time…
tags:
  - "ICML 2026"
  - "Time Series"
  - "HiPPO"
  - "Fractional Calculus"
  - "State Space Model"
  - "Long-Range Dependency"
  - "Long Range Arena"
date: 2026-05-08
content_hash: d2d4b5e7af66c4f3
---

# FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences

**Conference**: ICML 2026  
**arXiv**: [2605.08833](https://arxiv.org/abs/2605.08833)  
**Code**: None  
**Area**: Sequence Modeling / State Space Models (SSM)  
**Keywords**: HiPPO, Fractional Calculus, State Space Model, Long-Range Dependency, Long Range Arena

## TL;DR
This work generalizes the probabilistic measure underlying the HiPPO framework to a fractional power-law measure with a tunable singularity index $\alpha$, thereby, for the first time, achieving "full-history retention + recent sensitivity + scale invariance" simultaneously. This theory is instantiated as an LTI diagonalized SSM—FRACTAL matches S5 with an 87.11% average on Long Range Arena and achieves 61.85% on ListOps.

## Background & Motivation
**Background**: Modern SSMs (S4 / S4D / DSS / S5 / Mamba) are almost all built on the HiPPO "online polynomial projection + probabilistic measure" framework. Architecturally, there have been several iterations—from structured matrices to diagonalization, and then to time-varying selective SSMs—but the foundational choice of which measure to use for weighting history has remained unchanged for years.

**Limitations of Prior Work**: The three classic measures each have inherent flaws: LegS (uniform) preserves full history and scale invariance, but recent signals are diluted by $1/t$; LagT (exponential) highlights recent inputs but fixes the time scale, breaking down when time is sped up or slowed; LegT (sliding window) offers high local resolution but completely forgets information outside the window. Table 1 aligns these three with the three desired properties, directly revealing the "Impossible Trinity."

**Key Challenge**: To simultaneously achieve "memory extending arbitrarily far," "sensitivity to recent inputs," and "robustness to time scaling," integer-order differential equations are fundamentally insufficient—they either spread memory uniformly or decay exponentially.

**Goal**: Without abandoning HiPPO's "online polynomial projection + closed-form dynamics," identify a class of measures that satisfy all three properties and instantiate them in a parallelizable LTI diagonal SSM.

**Key Insight**: Fractional calculus naturally describes non-local, heavy-tailed memory. By replacing the indicator function in the measure with a power-law singularity $(t-x)^{-\alpha}$, the singularity index $\alpha\in[0,1)$ continuously interpolates between LegS ($\alpha=0$) and near-delta ($\alpha\to 1$).

**Core Idea**: Replace HiPPO's uniform/exponential measure with a power-law singular measure—equivalent to generalizing the basis from Legendre to Jacobi polynomials $P_n^{(-\alpha,0)}$, while retaining closed-form derivable SSM coefficients.

## Method

### Overall Architecture
FRACTAL consists of two phases. **Phase 1 (Offline Initialization)**: Given multi-channel singularity indices $\boldsymbol{\alpha}=(\alpha_1,\dots,\alpha_K)$, define the fractional measure $\mu^{(t)}(x)=(1-\alpha)t^{\alpha-1}(t-x)^{-\alpha}\mathbb{I}_{[0,t]}(x)$ as in Def. 3.1, substitute its orthogonal basis (normalized Jacobi polynomials) into the HiPPO projection equation, and derive the LTV dynamics $\dot{x}=-\frac{1}{t}A(\alpha)x+\frac{1}{t}B(\alpha)u$; then, perform eigendecomposition on $A(\alpha)$ to obtain $\Lambda$, and initialize $B(\alpha)$ using a closed-form physical initialization $\tilde{B}_{\text{init}}=V^{-1}B$. **Phase 2 (Online Training)**: Relax $1/t$ in the LTV to a learnable time scale $\Delta$ to obtain an LTI system, then use ZOH discretization + parallel prefix-sum to scan in $O(N\log L)$ time. Finally, wrap the SSM output with a GLU to form a standard gated SSM block.

### Key Designs

1. **Fractional HiPPO Measure (Definition 3.1)**:

    - **Function**: Uses a family of probability measures with tunable singularity index $\alpha$ to continuously interpolate between "uniform memory" and "concentration on the present," while maintaining scale invariance.
    - **Mechanism**: $\mu^{(t)}(x)=(1-\alpha)t^{\alpha-1}(t-x)^{-\alpha}\mathbb{I}_{[0,t]}(x)$, with normalization factor derived from $\int_0^t(t-x)^{-\alpha}dx=t^{1-\alpha}/(1-\alpha)$; $\alpha=0$ reduces to LegS, $\alpha\to 1$ approaches $\delta_t$. Mapping $[0,t]$ to $[-1,1]$ via $y=2x/t-1$, the weight function becomes Jacobi weight $(1-y)^{-\alpha}$, so the orthogonal basis is naturally Jacobi polynomials $P_n^{(-\alpha,0)}$.
    - **Design Motivation**: The power-law singularity preserves both "heavy-tailed long-range memory" and "high responsiveness to recent inputs," and the normalized form of $\mu$ remains invariant under $t\mapsto\lambda t$, thus ensuring scale invariance—breaking the Impossible Trinity in Table 1.

2. **Provable State Matrix Structure (Theorem 3.4)**:

    - **Function**: Provides the analytic structure of $A(\alpha)\in\mathbb{R}^{N\times N}$ after LTI, ensuring diagonalizability, spectral stability, and eliminating the need for engineering approximations like NPLR or normal-plus-low-rank.
    - **Mechanism**: Perform Galerkin projection $\mathcal{L}[P_n]=P_n+(1+\eta)P_n'$ under the normalized Jacobi basis; $A(\alpha)$ is strictly lower triangular; diagonal entries $A_{nn}=n+1$ are independent of $\alpha$; off-diagonal entries $A_{nk}$ ($k<n$) are given by $\langle\mathcal{L}[P_n^{(-\alpha,0)}],P_k^{(-\alpha,0)}\rangle_w/\|P_k\|_w^2$, degenerating to $\sqrt{(2n+1)(2k+1)}$ only when $\alpha=0$, otherwise computed via Gauss–Jacobi numerical integration. The $B$ term has a closed form $B_n=\sqrt{(2n+1-\alpha)/(1-\alpha)}\binom{n-\alpha}{n}$.
    - **Design Motivation**: "Diagonal entries always being $1,\dots,N$" is the key stability guarantee—meaning that regardless of $\alpha$, eigenvalues remain unchanged, only eigenvectors rotate. No need for complex NPLR approximations as in S4; direct diagonalization $A=V\Lambda V^{-1}$ yields a diagonal SSM.

3. **Fractional Filter Bank Multi-Channel Architecture (§4.3)**:

    - **Function**: Captures features at different time scales in parallel using sub-states with different $\alpha_k$, effectively structuring the SSM state space as a set of "frequency-domain filters."
    - **Mechanism**: Split the state dimension $H$ into $K$ blocks, each with its own $\alpha_k$; low $\alpha$ channels act as low-pass filters (preserving global context and denoising), high $\alpha$ channels act as band-pass/high-pass filters (highlighting local changes). The output projection $C$ learns how to combine these temporal bases. $\Delta$ controls "how far to look" (resolution), $\alpha$ controls "how to look" (memory topology), and the two are decoupled.
    - **Design Motivation**: Experiments show that ListOps requires both long-range bracket matching and local numerics, which a single $\alpha$ cannot handle; using a filter bank naturally decouples different scales, which is the fundamental reason FRACTAL achieves the largest improvement on ListOps.

### Loss & Training
No new loss is introduced; standard CE or BCE for each LRA task is used. $\alpha_k$ are linearly spaced in $[0,0.9]$ and fixed (the paper lists learnable $\alpha$ as future work). Other hyperparameters follow S5 for fair comparison.

## Key Experimental Results

### Main Results: Long Range Arena (Table 2)

| Model | ListOps | Text | Retrieval | Image | Pathfinder | Path-X | Avg |
|------|---------|------|-----------|-------|------------|--------|-----|
| Transformer | 36.37 | 64.27 | 57.46 | 42.44 | 71.40 | ✗ | – |
| S4 | 59.60 | 86.82 | 90.90 | 88.65 | 94.20 | 96.35 | 86.09 |
| S4D | 60.47 | 86.18 | 89.46 | 88.19 | 93.06 | 91.95 | 84.89 |
| DSS | 57.60 | 84.80 | 87.60 | 84.40 | 85.00 | 85.00 | 80.73 |
| S5 (paper reproduction) | 61.10 | 88.72 | 91.27 | 87.59 | 95.04 | 98.62 | 87.04 |
| **FRACTAL** | **61.85** | **89.10** | 91.19 | 87.30 | 94.80 | 98.39 | **87.11** |

### Ablation Study & Diagnostics

| Setting | Key Metric | Notes |
|------|----------|------|
| $\alpha=0$ (degenerates to LegS / S4-like) | Matches S4 | Confirms strict generalization of existing methods |
| Single $\alpha$ (no filter bank) | ListOps drops significantly | Single time scale cannot capture both global brackets and local numbers |
| Random $B$ vs analytic $\tilde{B}_{\text{init}}$ | Same final accuracy, but analytic version has much lower initial loss | Closed-form acts as a preconditioner, see Remark 4.1 |
| Numerical validation $A_{nn}=n+1$ | Holds for any $\alpha$ | Perfectly consistent with Theorem 3.4, no spectral drift on long sequences |

### Key Findings
- On ListOps, which has the most hierarchical + heavy-tailed structure, FRACTAL outperforms S5 by 0.75pt and S4 by 2.25pt, the largest margin; on Image / Pathfinder, which are mainly local, FRACTAL matches S5. This matches the theoretical prediction: "The advantage of power-law measures is concentrated on long-range + multi-scale tasks."
- On Path-X (length 16K), FRACTAL still achieves 98.39%, indicating that introducing singularities does not destabilize gradients on extremely long sequences—this experimentally supports the spectral stability of $A_{nn}=n+1$.
- Strict scale invariance is lost after moving away from LTI, but the filter bank still provides multi-scale inductive bias, making it more practical than "strictly scale-invariant but hard-to-train" LTV systems.

## Highlights & Insights
- The narrative from "Impossible Trinity → Fractional Unlock" is very clear: the authors explicitly separate SSM progress into "architecture vs measure" and point out that the measure line has been neglected for years. This "re-examining implicit assumptions" research mode is reusable in many fields (e.g., attention normalization, diffusion model noise scheduling).
- The spectral invariance of "diagonal entries independent of $\alpha$" is a beautiful byproduct: it means $\alpha$ can be set as a learnable hyperparameter (even per-token adaptive) with almost no impact on numerical stability—opening the door to "data-driven selection of $\alpha$."
- Using a filter bank to treat different $\alpha$ as different frequency bands can be borrowed by non-SSM models (e.g., linear attention or RNN-style models)—as long as the kernel decay shape is viewed as a tunable "frequency band," similar multi-channel structures can also yield multi-scale inductive bias.

## Limitations & Future Work
- Strict scale invariance only holds under LTV (with $1/t$); for parallel scanning, the authors deliberately abandon this, so the engineering version is "spectral multi-scale" rather than "true scale-invariant." To truly leverage scale invariance for physical/physiological signals, a dedicated LTV-friendly scanning algorithm is needed.
- $\alpha_k$ are currently fixed and linearly spaced; the optimal $\alpha$ spectrum may vary greatly across tasks, so end-to-end learnability is an obvious next step.
- Evaluation is limited to LRA; the paper positions itself as "train-from-scratch LTI" and does not compare with Mamba-style selective SSMs on large-scale language tasks, so conclusions about "language modeling quality" should be extrapolated with caution.

## Related Work & Insights
- **vs S4 / S4D**: S4 uses HiPPO-LegS as static initialization, then applies NPLR engineering; FRACTAL makes the measure a tunable design parameter and diagonalizes directly via spectral invariance, skipping NPLR.
- **vs DSS / S5**: DSS / S5 argue that "precise matrix structure is unimportant, spectral structure is key"; FRACTAL continues this line—further "designing spectral structure by measure law" rather than random or approximate initialization.
- **vs Mamba and selective SSMs**: Mamba makes matrices input-dependent to introduce data dependence; FRACTAL takes the orthogonal approach—sticking to LTI but using the measure as inductive bias—the two can be combined (input-dependent $\alpha$).
- **vs LMU / Voelker 2019**: LMU uses sliding window Legendre, i.e., LegT; FRACTAL generalizes this to tunable $\alpha$, making LMU a special case.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces fractional measures into the HiPPO framework for the first time, with closed-form derivable structure.
- Experimental Thoroughness: ⭐⭐⭐ Matches S5 on LRA and leads on ListOps, but lacks large-scale language modeling comparison.
- Writing Quality: ⭐⭐⭐⭐ The logical chain from "Impossible Trinity" to FRACTAL is very clear, and the appendix derivations are self-consistent.
- Value: ⭐⭐⭐⭐ Provides a long-overlooked design dimension for SSMs, with clear guidance for long-range signal modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Investigating Advanced Reasoning of Large Language Models via Black-Box Environment Interaction](investigating_advanced_reasoning_of_large_language_models_via_black-box_environm.md)
- [\[ICML 2026\] Holmes: 用层次化证据学习重新审视部分相关视频检索中的不确定性](revisiting_uncertainty_on_evidential_learning_for_partially_relevant_video_retri.md)
- [\[ICML 2026\] Dual-branch Robust Unlearnable Examples](dual-branch_robust_unlearnable_examples.md)
- [\[ICML 2026\] DoLQ: 用 LLM 做定性 + 定量评估发现常微分方程](discovering_ordinary_differential_equations_with_llm-based_qualitative_and_quant.md)
- [\[ICML 2026\] iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework](iworld-bench_a_benchmark_for_interactive_world_models_with_a_unified_action_gene.md)

</div>

<!-- RELATED:END -->
