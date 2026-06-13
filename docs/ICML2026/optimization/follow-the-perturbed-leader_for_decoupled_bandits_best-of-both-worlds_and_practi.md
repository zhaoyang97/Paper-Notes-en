---
title: >-
  [Paper Note] Follow-the-Perturbed-Leader for Decoupled Bandits: Best-of-Both-Worlds and Practicality
description: >-
  [ICML 2026][Optimization][Follow-the-Perturbed-Leader] This paper designs the first Best-of-Both-Worlds (BOBW) FTPL algorithm for the decoupled multi-armed bandit problem (where an "exploitation" arm and an "exploration"…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Follow-the-Perturbed-Leader"
  - "decoupled bandits"
  - "Best-of-Both-Worlds"
  - "Pareto perturbation"
  - "convex-optimization-free"
date: 2026-05-08
content_hash: b924ee88d29b252f
---

# Follow-the-Perturbed-Leader for Decoupled Bandits: Best-of-Both-Worlds and Practicality

**Conference**: ICML 2026  
**arXiv**: [2510.12152](https://arxiv.org/abs/2510.12152)  
**Code**: Not released  
**Area**: Online Learning / Bandits / Optimization  
**Keywords**: Follow-the-Perturbed-Leader, decoupled bandits, Best-of-Both-Worlds, Pareto perturbation, convex-optimization-free

## TL;DR
This paper designs the first Best-of-Both-Worlds (BOBW) FTPL algorithm for the decoupled multi-armed bandit problem (where an "exploitation" arm and an "exploration" arm are selected separately in each round). By using Pareto perturbations for exploitation and a proxy $q_{t,i}$, which depends only on the cumulative estimated loss ranking, to define the exploration distribution, the algorithm requires neither the per-step convex optimization of FTRL nor the geometric resampling typical of FTPL. It achieves regret bounds of $\mathcal{O}(\sqrt{KT})$ and $\mathcal{O}(K/\Delta_{\min})$ in adversarial and stochastic environments, respectively, matching the current optimal FTRL algorithms while being approximately 130× faster than baselines for $K=2$.

## Background & Motivation

**Background**: In the standard multi-armed bandit (MAB), a single arm must simultaneously handle exploration and exploitation in each round. Avner et al. (2012) introduced the decoupled bandit: in each round, $i_t$ is chosen to incur loss (not observed), and $j_t$ is chosen to observe loss (not incurred). This setting originates from ultra-wideband communications (sensing and transmission use different channels), sim-to-real robotics (simulator explores, real system exploits), and recommendation systems (a small group of users explore while others exploit). Rouyer & Seldin (2020) achieved BOBW guarantees using Decoupled-Tsallis-INF: $\mathcal{O}(\sqrt{KT})$ for adversarial and $\mathcal{O}(K/\Delta_{\min})$ for stochastic, representing the current theoretical SOTA.

**Limitations of Prior Work**: Decoupled-Tsallis-INF belongs to the FTRL framework, requiring a Tsallis entropy-regularized convex optimization on the $(K-1)$-dimensional simplex in every round to obtain the exploitation probability vector $w_t$, which is then used to calculate the exploration probability $p_t$. Even with Newton iteration and warm starts, this convex optimization is a burden for real-world scenarios like ultra-wideband communication requiring millisecond responses; empirically, it is about 130× slower than FTPL at $K=2$.

**Key Challenge**: FTPL (a lightweight framework using random perturbations instead of regularization to select actions via simple argmin) is naturally efficient. However, all existing decoupled bandit algorithms dictate that "the exploration probability $p_{t,i}$ must be computed from the exploitation probability $w_{t,i}$." Since FTPL generally lacks a closed-form expression for $w_t$, it must be estimated via geometric resampling (GR), costing $\mathcal{O}(K^2)$ or $\mathcal{O}(K\log K)$ per step. If the entire vector $w_t$ is estimated to define the exploration distribution, the cost rises to $\mathcal{O}(K^2\log K)$, making it slower than FTRL and negating the speed advantage of FTPL.

**Goal**: Design an FTPL algorithm for the decoupled setting that retains the $\mathcal{O}(K\log K)$ per-step complexity of FTPL while achieving BOBW regret bounds comparable to Decoupled-Tsallis-INF.

**Key Insight**: The prior requirement that "$p_t$ is a function of $w_t$" is a sufficient but not necessary condition. One can bypass $w_t$ by finding a proxy vector $q_t$ calculable using only **currently available quantities** (cumulative estimated loss $\hat L_t$, learning rate $\eta_t$, and ranking $\sigma_{t,i}$), provided it maintains a tight analytical inequality relationship with $w_t$.

**Core Idea**: Use Pareto($\alpha$) perturbations for FTPL exploitation (corresponding to Tsallis entropy FTRL with $\beta=1-1/\alpha$). Simultaneously, use $q_{t,i}=\big(\min\{1/(1+\eta_t\hat{\underline L}_{t,i}),\,1/\sigma_{t,i}^{1/\alpha}\}\big)^{(\alpha+1)/2}$ as a computable proxy for the upper bound of $w_{t,i}^{1/2+1/(2\alpha)}$. The exploration distribution is then defined by normalization: $p_{t,i}=q_{t,i}/\sum_j q_{t,j}$. This involves only sorting and assignment, without convex optimization or resampling.

## Method

### Overall Architecture
In each round, the algorithm performs three steps: ① Sample $K$ perturbations $r_{t,i}$ from the Pareto distribution $\mathcal{P}_\alpha$ and select the exploitation arm $i_t=\arg\min_i\{\hat L_{t,i}-r_{t,i}/\eta_t\}$. ② Calculate the exploration distribution $p_t$ directly using rankings, sample $j_t\sim p_t$, and observe $\ell_{t,j_t}$. ③ Update cumulative losses using the IW estimator $\hat L_{t+1}=\hat L_t+\ell_{t,j_t}p_{t,j_t}^{-1}e_{j_t}$. The entire process involves no convex optimization or resampling; the per-step cost is dominated by ranking maintenance, which is $\mathcal{O}(K)$ (average) or $\mathcal{O}(\log K)$ via binary insertion.

Note that $p_t$ and $w_t$ are independent: the exploitation arm selection depends on the argmin of $$\hat L_t$$ plus noise, while the exploration arm sampling depends on the ranking and loss differences of $$\hat L_t$$. They share the same cumulative estimated loss but follow completely independent computational paths—this is the core structure for bypassing the historical convention that "$p_t$ must be derived from $w_t$."

### Key Designs

1.  **Pareto Perturbation FTPL for Exploitation**:
    *   **Function**: Obtains an exploitation strategy of the same order as $\beta$-Tsallis-INF without solving an optimization problem.
    *   **Mechanism**: Perturbations $r_{t,i}$ are sampled from a Pareto distribution $f(x)=\alpha/x^{\alpha+1}$ with shape $\alpha>1$. The exploitation arm is $i_t=\arg\min_i\{\hat L_{t,i}-r_{t,i}/\eta_t\}$. Previous work (Kim & Tewari 2019; Lee 2025) has shown that Pareto-perturbed FTPL corresponds exactly to Tsallis entropy FTRL with $\beta=1-1/\alpha$ in terms of implicit exploitation probabilities, thus providing the necessary "decay rate of exploitation probability" for BOBW—yet $w_{t,i}=\phi_i(\eta_t\hat L_t)$ lacks a closed form.
    *   **Design Motivation**: While Gumbel perturbations (Exp3) in FTPL bandits have a closed-form softmax for $w_t$, they yield suboptimal regret in stochastic environments. Pareto perturbations are used for BOBW, but the lack of a closed form for $w_t$ motivates the need to bypass it.

2.  **Ranking-based Proxy Vector $q_t$ for Exploration Distribution**:
    *   **Function**: Constructs a computable quantity from $\hat L_t$ and $\eta_t$ to generate exploration probabilities $p_t$, completely bypassing geometric resampling when $w_t$ has no closed form.
    *   **Mechanism**: Define the loss identity $\hat{\underline L}_{t,i}=\hat L_{t,i}-\min_j\hat L_{t,j}$ and ranking $\sigma_{t,i}$ (minimum = 1, maximum = $K$). Let $q_{t,i}=\big(\min\{1/(1+\eta_t\hat{\underline L}_{t,i}),\,1/\sigma_{t,i}^{1/\alpha}\}\big)^{(\alpha+1)/2}$ and $p_{t,i}=q_{t,i}/\sum_{j}q_{t,j}$. Intuitively, $q_{t,i}$ approximates $w_{t,i}^{1/2+1/(2\alpha)}$, corresponding to $w_{t,i}^{1-\beta/2}$ in Decoupled-Tsallis-INF (with $\alpha=1/(1-\beta)$). The critical tight inequality $q_{t,i}\le w_{t,i}^{1/2+1/(2\alpha)}\lesssim w_{t,i}^{1-1/\alpha}$ is provided by Lemma D.2.
    *   **Design Motivation**: Exploration distributions require a shape that "decays for bad arms but remains non-zero." This was originally provided by $w_{t,i}$. This paper finds that the intuition "arms with low cumulative loss and high ranking should explore more" can be captured by a $\min\{\cdot,\cdot\}$ operator over both dimensions—one controlling loss magnitude and the other controlling rank. Raising this to the $(\alpha+1)/2$ power matches the upper bound required for tight regret analysis.

3.  **Incremental Ranking Maintenance + Self-bounding Regret Analysis**:
    *   **Function**: Reduces per-step complexity to $\mathcal{O}(K)$ average cost and incorporates the difficulty introduced by $p_t$ into the classic FTPL regret decomposition.
    *   **Mechanism**: Implementation-wise, since the IW estimator updates only one arm $j_t$ per round, the rankings of the other $K-1$ arms change by at most 1. The algorithm uses binary search to locate the new position of $j_t$ in $\mathcal{O}(\log K)$ and applies $\mathcal{O}(K)$ corrections to the affected interval. Analytically, the authors decompose regret into stability and penalty (Lemma 3.4), avoiding the extra $\log T$ factor found in Honda (2023)/Lee (2024). They bound the stability term using the proxy $q_{t,j}q_{t,i}$ and employ a self-bounding technique to converge to $\mathcal{O}(\sqrt{KT})$ (adversarial) and $\mathcal{O}(K/\Delta_{\min})$ (SCA).
    *   **Design Motivation**: Ensures that both the implementation and analysis are free from convex optimization or resampling. By setting $\alpha=3$ (corresponding to $\beta=2/3$, the optimal configuration for Decoupled-Tsallis-INF), the algorithm achieves BOBW regret of the same order as FTRL.

## Key Experimental Results

### Main Results

| Experimental Setup | Metric | EXP3 (β=1) | FTRL (β=2/3, Decoupled-Tsallis-INF) | FTPL (Ours, α=3) |
| :--- | :--- | :--- | :--- | :--- |
| Adversarial 8-arm, $\Delta=0.125$ | Cum. Regret (Lower is better) | Highest | Medium | **Lowest** |
| Stochastic 5-arm, Easy $\mu_1$, $\Delta_{\min}=0.05$ | Cum. Regret | Higher | Lower | **Lowest** |
| Stochastic 5-arm, Hard $\mu_2$, $\Delta_{\min}=0.002$ | Cum. Regret | High | Medium | **Lowest** |
| SCS Convex Solver, $K\in\{2,\ldots,64\}$ | Per-step Time (ms) | — | Significantly Higher | **↓~130× ($K=2$)** |
| Newton+warm start, increasing $K$ | Per-step Time Slope | — | Largest Slope | **Smallest Slope** |

### Ablation Study

| Dimension | FTRL (Newton + warm start) | FTPL (sorting) | FTPL (improved, Ours) |
| :--- | :--- | :--- | :--- |
| Per-round Dependency | Convex optimization iterations | Full vector re-sorting | Incremental binary search |
| Avg. Per-step Complexity | No formal guarantee (≥ $\mathcal{O}(K)$ × #iter) | $\mathcal{O}(K\log K)$ | **$\mathcal{O}(K)$ average** |
| Requires $w_t$ estimation | Yes (Optimization solution) | No | **No** |
| Requires Geo-Resampling | No | No (Bypassed by Ours) | **No** |
| Adversarial Regret Bound | $\mathcal{O}(\sqrt{KT})$ | $\mathcal{O}(\sqrt{KT})$ | $\mathcal{O}(\sqrt{KT})$ |
| SCA / Stochastic Regret | $\mathcal{O}(K/\Delta_{\min})$ | $\mathcal{O}(K/\Delta_{\min})$ | $\mathcal{O}(K/\Delta_{\min})$ |

### Key Findings
*   The intuition that "FTPL is slower than FTRL" only holds for unoptimized implementations. This paper proves that by bypassing $w_t$ estimation, FTPL's complexity ($\mathcal{O}(K)$ incremental) is lower than FTRL (convex iterations). Empirically, it is 130× faster at $K=2$, with the advantage scaling as $K$ increases.
*   Using rankings instead of estimated loss values as proxies is the core design: rankings are robust statistics, insensitive to loss noise/magnitude, and cheap to maintain incrementally, allowing both tight theoretical bounds and efficient implementation.
*   The analysis is tightest when $\alpha\in(1,3]$, where $\alpha=3$ matches the optimal Tsallis configuration ($\beta=2/3$) from Rouyer & Seldin (2020), suggesting that FTPL and FTRL share the same "theoretically optimal shape" for this problem.
*   In "hard" stochastic instances ($\mu_2$, where all arm means are near 0.9, $\Delta_{\min}=0.002$), while all BOBW algorithms take longer to converge, FTPL consistently leads FTRL/EXP3, suggesting empirically optimal constants from self-bounding.

## Highlights & Insights
*   "Using proxies instead of true probabilities" is a generalizable methodology. The difficulty of calculating $w_t$ in FTPL appears in many bandit variants (contextual, combinatorial, semi-bandit). This work demonstrates that as long as a computable upper bound with the "right tight inequality" is found, BOBW can be achieved.
*   Treating "ranking" as a valid state variable is perhaps the most efficient and clever step—it simplifies a geometric object (the optimal solution on the probability simplex) into $\sigma_{t,i}\in\{1,\ldots,K\}$, streamlining both theory and implementation.
*   The incremental ranking + binary search implementation achieves $\mathcal{O}(K)$ average complexity, which is an engineering optimization that benefits both theoretical asymptotic behavior and empirical constants.

## Limitations & Future Work
*   The second term of the SCA stochastic regret bound, $K/\Delta_{\min}$, is larger than the $\sqrt{K}/\Delta_{\min}$ achieved by FTRL using log-barrier + arm-dependent learning rates (Jin 2023) by a factor of $\sqrt{K}$. Adding arm-dependent learning rates to FTPL is non-trivial (equivalent to non-i.i.d. perturbations) and remains for future work.
*   When $\alpha>3$, the $K$ dependence degrades from $\sqrt{K}$ to $K^{(\alpha-2)/(\alpha-1)}$, limiting the flexibility of $\alpha$ to the range $(1,3]$.
*   The algorithm assumes a unique best arm; guarantees for instances with multiple optimal arms are not provided. The SCA assumes constant mean gaps, requiring further analysis for non-stationary (drifting) environments.
*   Benchmark results are provided for $T=10^4$ and $K\le 256$. The efficiency in large-scale bandits ($K\gtrsim 10^4$ in recommendation systems) needs further validation, particularly whether the overhead of ranking maintenance remains smaller than FTRL's iterations at such scales.

## Related Work & Insights
*   **vs Decoupled-Tsallis-INF (Rouyer & Seldin 2020)**: Achieves comparable BOBW regret but bypasses the convex optimization for $w_t$ via Pareto perturbations and ranking proxies.
*   **vs Avner et al. 2012 (decoupled Exp3)**: While Avner used Exp3 with variance-optimal exploration, this work brings BOBW to the FTPL framework for the first time using Pareto perturbations.
*   **vs Honda 2023 / Lee 2024 (FTPL-BOBW for standard MAB)**: They achieved BOBW for standard MAB but relied on GR to estimate $1/w_{t,i_t}$. In the decoupled setting, where the entire vector $w_t$ is needed, GR is prohibitively expensive; this work bypasses GR using $q_t$.
*   **vs Jin et al. 2023 (log-barrier + arm-dependent LR FTRL)**: This work trades a constant factor in the $K$-dependence ($\sqrt{K}$ vs $K$) for a significant reduction in system latency by avoiding log-barrier optimizations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Oracle-Efficient Combinatorial Semi-Bandits](../../NeurIPS2025/optimization/oracle-efficient_combinatorial_semi-bandits.md)
- [\[ICML 2026\] URS: A Unified Neural Routing Solver](urs_a_unified_neural_routing_solver_for_cross-problem_zero-shot_generalization.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)
- [\[ICML 2026\] Interpretability and Generalization Bounds for Learning Spatial Physics](interpretability_and_generalization_bounds_for_learning_spatial_physics.md)

</div>

<!-- RELATED:END -->
