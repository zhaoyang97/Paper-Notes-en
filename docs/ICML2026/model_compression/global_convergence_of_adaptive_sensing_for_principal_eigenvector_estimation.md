---
title: >-
  [Paper Note] Global Convergence of Adaptive Sensing for Principal Eigenvector Estimation
description: >-
  [ICML 2026][Model Compression][Principal Component Analysis] This paper establishes the optimal convergence rate for compressed streaming PCA. It shows that the error upper bound of Oja's algorithm using two **adaptive**…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Principal Component Analysis"
  - "Adaptive Sensing"
  - "Streaming Learning"
  - "Compressed Measurement"
  - "Convergence Analysis"
date: 2026-05-08
content_hash: c0bf6cca2564b5e3
---

# Global Convergence of Adaptive Sensing for Principal Eigenvector Estimation

**Conference**: ICML 2026  
**arXiv**: [2505.10882](https://arxiv.org/abs/2505.10882)  
**Code**: To be confirmed  
**Area**: Optimization Theory / Streaming PCA  
**Keywords**: Principal Component Analysis, Adaptive Sensing, Streaming Learning, Compressed Measurement, Convergence Analysis

## TL;DR
This paper establishes the optimal convergence rate for compressed streaming PCA. It shows that the error upper bound of Oja's algorithm using two **adaptive** measurements per step under noisy observations matches the information-theoretic lower bound (both being $\Theta(\lambda_1 \lambda_2 d^2 / (\Delta^2 t))$). It reveals for the first time that the fundamental cost of compression relative to full observation is an additional factor of $d$, while adaptive sensing recovers one factor of $d$ compared to non-adaptive sensing.

## Background & Motivation

**Background**: Classical PCA requires full $d$-dimensional samples. However, in hardware-constrained scenarios like mmWave communications, neural signals, and radar, each sample can only be obtained through a small number of scalar measurements. While Oja's streaming algorithm is a benchmark for such constraints, existing analyses are based on full observations.

**Limitations of Prior Work**: The design and analysis of PCA under compressed observations are difficult. Adaptive GROUSE only analyzes the noiseless case ($\lambda_2 = 0$) and cannot handle real-world data with tail eigenvalues. Crucially, there is a lack of information-theoretic lower bounds for compressed PCA—leaving the fundamental limit of "two measurements per step" unknown.

**Key Challenge**: Adaptive measurement (along the direction of the current estimate) can improve the convergence rate, but it introduces signal-noise coupling that complicates the analysis. Furthermore, one must balance "exploitation" (along the current estimate) and "exploration" (along orthogonal directions).

**Goal**: (1) Prove global convergence guarantees for compressed Oja under noisy observations; (2) Establish the optimal rate for adaptive compressed PCA in comparison with full observation and non-adaptive sensing; (3) Provide the first information-theoretic lower bound for compressed eigenvector estimation.

**Key Insight**: The problem is formalized as two adaptive linear measurements per step, with a measurement matrix designed to balance exploitation and exploration. The root of the $d^2$ factor is revealed through Assouad's Lemma combined with a measurement energy budget argument.

**Core Idea**: By tracking the stochastic recurrence of the expected cosine alignment and employing a measurement budget argument, it is shown that the additional cost of adaptive compressed PCA relative to full observation is **exactly** $d$, and it is one factor of $d$ less than non-adaptive sensing.

## Method

### Overall Architecture
Samples $v_t \sim \mathcal{N}(0, \Sigma)$ arrive in a stream. The environment can only obtain $x_t = A_t v_t \in \mathbb{R}^2$ at each step, where $A_t \in \mathbb{R}^{2 \times d}$ is adaptively selected based on the current estimate $u_t$. The goal is to estimate the principal eigenvector $\bar{u}$ of $\Sigma$. Compressed Oja (Algorithm 1) performs the following at each step:
- **Measurement**: $A_t = [u_t^\top; b_t^\top]$, where $b_t$ is a unit vector sampled uniformly from the orthogonal complement of $u_t$.
- **Observation**: $x_t = [u_t^\top v_t; b_t^\top v_t]$.
- **Update**: Reconstruct the projection $\tilde{v}_t = (u_t u_t^\top + b_t b_t^\top) v_t$ from the two measurements, then apply the standard Oja update $u_{t+1} = \text{Norm}(u_t + \eta_t \tilde{v}_t v_t^\top u_t)$.

The analysis defines auxiliary quantities $c = \bar{u}^\top u$ (cosine alignment), $z = \bar{u}^\top b$, $g = v^\top u$, and $h = v^\top b$ to track the stochastic recurrence of $\mathbb{E}[c_t^2]$.

### Key Designs

1. **Adaptive Sensing Strategy**:
    - **Function**: Achieves a balance between exploitation and exploration in streaming PCA, increasing the information extraction rate under a finite measurement budget.
    - **Mechanism**: Two directions are measured at each step—strengthening the existing signal along the current estimate $u_t$ (exploitation) and obtaining corrective information along the orthogonal direction $b_t$ (exploration). Even if the initial alignment is poor, useful gradients are gradually accumulated.
    - **Design Motivation**: The expected overlap between a completely random measurement direction and the true principal direction is only $O(1/d)$, wasting the budget. Adaptive sensing amplifies "existing alignment" into an effective signal through a positive feedback mechanism.

2. **Two-stage Step Size + Convergence Analysis**:
    - **Function**: Separately handles the warm-up phase and the local convergence phase to derive piecewise optimal step sizes.
    - **Mechanism**: A constant step size $\eta_0 = (d-1)/(2 S \Delta)$ is used during the warm-up phase to ensure monotonic improvement. Once alignment $c_t^2 \geq 0.5$ is reached, it switches to a decaying step size $\eta_t = 2(d-1) / [\Delta (4S + t - t_0)]$, causing $1 - c_t^2$ to decay at $O(1/t)$, where $S = 3 \lambda_1 \lambda_2 d^2 / \Delta^2 + 15 \lambda_1 d / \Delta$.
    - **Design Motivation**: The warm-up phase requires monotonicity to ensure numerical stability; the local phase uses a decaying step size to avoid oscillation while maintaining "momentum" to reach the optimal rate.

3. **Signal-Noise Decomposition + Lower Bound Construction**:
    - **Function**: (a) Accurately tracks the recurrence variance terms in noisy scenarios with tail eigenvalues $\lambda_2 > 0$; (b) Provides the first information-theoretic lower bound for compressed PCA.
    - **Mechanism**: For the upper bound, Isserlis' Theorem and Cauchy-Schwarz are used to bound $\mathbb{E}[g h \mid c, z] = u^\top \Sigma b$ by $\sqrt{(u^\top \Sigma u)(b^\top \Sigma b)}$, where the squared terms contribute $2 a^2 b^2$ ($a^2 = \Delta c^2 + \lambda_2, b^2 = \Delta z^2 + \lambda_2$). A combination of Jensen's inequality and monotonicity avoids adaptive coupling of matrix concentration. For the lower bound, Assouad's Lemma and a measurement energy budget are used—$2t$ total energy from two unit-norm measurements per step must be distributed across $d-1$ coordinates. An average energy of $O(t/d)$ per coordinate yields a per-coordinate error of $O(d/t)$, summing to $\Theta(d^2/t)$.
    - **Design Motivation**: Terms in the GROUSE family cancel out because $\lambda_2 = 0$. This work tracks these cross-terms to handle real data. The lower bound proof transforms "compression to 2D" into an energy allocation problem, which is the most elegant part of the paper.

### Convergence Rate
**Upper Bound Theorem 4.1**: After the warm-up period $t_0 = (4S+1) \log(d/2)$, $\mathbb{E}[1 - (\bar{u}^\top u_t)^2] \leq \frac{C_1}{4S + (t - t_0)} + \frac{C_2}{[4S + (t-t_0)]^2}$, asymptotically $\mathcal{O}(\lambda_1 \lambda_2 d^2 / (\Delta^2 t))$.

**Lower Bound Theorem 4.2**: $\inf_{\hat{u}} \sup_P \mathbb{E}[1 - (\bar{u}^\top \hat{u}_t)^2] \geq \Omega(\lambda_1 \lambda_2 d^2 / (\Delta^2 t))$. The bounds match tightly.

**Comparison of Three Schemes**: Full observation $\Theta(d / t)$; Adaptive compressed $\Theta(d^2 / t)$; Non-adaptive compressed $\Omega(d^3 / t)$. Adaptive sensing recovers one factor of $d$.

## Key Experimental Results

### Main Results

| Dimension $d$ | Adaptive Iterations | Non-adaptive Iterations | Speedup |
|---------|------------|--------------|---------|
| 16 | $3.8 \times 10^4$ | $1.6 \times 10^5$ | 4.2× |
| 32 | $1.9 \times 10^5$ | $1.3 \times 10^6$ | 7.1× |
| 64 | $8.4 \times 10^5$ | $1.2 \times 10^7$ | 14× |

Median number of iterations required to reach a target error of $10^{-2}$ (20 trials, $\eta = 0.01/d$).

### Dimension Scaling Validation

| Dimension $d$ | Iterations | $t_d / t_{d/2}$ |
|---------|-------|----------------|
| 16 | 35,500 | — |
| 32 | 172,750 | 4.87 |
| 64 | 879,190 | 5.09 |
| 128 | 4,091,830 | 4.65 |
| 256 | 17,950,000 | 4.39 |
| 512 | 68,500,000 | 3.82 |
| 1024 | 284,650,000 | 4.15 |

The ratios between adjacent dimensions range from 3.8 to 5.1, aligning with the theoretical $d^2$ (expected ratio of 4). The fitted exponent is 2.16; the slight excess over 2 is due to the $O(d)$ term in $S$ and the $\log d$ contribution from the warm-up period.

### Key Findings
- The speedup from adaptive sensing grows with the dimension (4× → 14×), quantifying the advantage that "adaptive directions overlap significantly more than fixed directions."
- The upper and lower bounds differ only by a constant factor (~$10^4$), representing a tight match.
- In a non-stationary setting where the optimum changes over time (at speed $V$), the optimal step size $\eta^* = \sqrt{V/S}$ yields a steady-state error of $V + \sqrt{V S}$, validating the non-stationary generalization.

## Highlights & Insights
- **First Information-Theoretic Lower Bound for Compressed PCA**: The $\Omega(d^2)$ lower bound derived via a measurement energy budget argument, coupled with the $d$-fold cost comparison for non-adaptive sensing, intuitively reveals the "value of adaptivity."
- **Breaking the Noiseless Constraint of the GROUSE Family**: The combination of signal-noise decomposition, Isserlis' fourth-moment correction, and explicit integration of the exploration direction makes the noisy setting analyzable for the first time.
- **Clear Separation of Three Schemes**: The complexity comparison of full observation ($d^1$), adaptive compressed ($d^2$), and non-adaptive compressed ($d^3$) provides theoretical guidance for selecting sampling strategies.
- **New Application of Assouad's Lemma**: A byproduct is a new proof for the classic full-observation PCA lower bound $\Omega(\lambda_1 \lambda_2 d / (\Delta^2 t))$, replacing existing Fano arguments with a more versatile technique.

## Limitations & Future Work
- The quadratic dependence on dimension limits practicality in ultra-high dimensions ($d > 10^4$), which is a fundamental bottleneck of "two measurements."
- Limited to rank-1 estimation; extending to rank $k > 1$ requires handling the coupling of $k$ recurrences and orthogonalization. The authors speculate that taking $m = 2k$ measurements would incur a $(d/m)^2$ penalty.
- The Gaussian assumption could be generalized using sub-Gaussian moment bounds and Le Cam’s $\chi^2$ divergence, though details were not expanded.
- The compressed version of sparse PCA remains an open problem.

## Related Work & Insights
- **vs. Full Observation Oja**: $\Theta(\lambda_1 \lambda_2 d / \Delta^2 t)$ vs. $\Theta(d^2 / \Delta^2 t)$ in this work; both reveal that compression equals a $d$-fold cost.
- **vs. Adaptive GROUSE (Ongie et al. 2017)**: Only analyzes the noiseless case; this work is the first to handle $\lambda_2 > 0$, borrowing recurrence ideas but adding signal-noise decomposition.
- **vs. Randomized SVD (Halko et al. 2011)**: Batch vs. streaming; both aim to bypass the curse of dimensionality. This paper shows that adaptivity can compensate for part of the cost.
- **Insight**: The energy budget argument is a general tool for lower bounds in constrained observation problems (radar, MRI, covariance estimation).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First tight bound for noisy compressed PCA; original lower bound technique using energy budgets + Assouad.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validates adaptive vs. non-adaptive, dimension scaling, and tracking stability; lacks experiments on real radar/MRI data.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, well-balanced technical detail and intuitive explanation, precise theorem statements.
- Value: ⭐⭐⭐⭐⭐ Fills a gap in tight bounds for the intersection of compressed sensing and streaming learning; provides theoretical guidance for hardware-constrained applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs](gemq_global_expert-level_mixed-precision_quantization_for_moe_llms.md)
- [\[ICML 2026\] Beyond Tokens: Enhancing RTL Quality Estimation via Structural Graph Learning](beyond_tokens_enhancing_rtl_quality_estimation_via_structural_graph_learning.md)
- [\[ICML 2026\] Active Budget Allocation for Efficient Scaling Law Estimation via Surrogate-Guided Pruning](active_budget_allocation_for_efficient_scaling_law_estimation_via_surrogate-guid.md)
- [\[ICML 2026\] WUSH: Near-Optimal Adaptive Transforms for LLM Quantization](wush_near-optimal_adaptive_transforms_for_llm_quantization.md)
- [\[ACL 2026\] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](../../ACL2026/model_compression/grasprune_global_gating_for_budgeted_structured_pruning_of_large_language_models.md)

</div>

<!-- RELATED:END -->
