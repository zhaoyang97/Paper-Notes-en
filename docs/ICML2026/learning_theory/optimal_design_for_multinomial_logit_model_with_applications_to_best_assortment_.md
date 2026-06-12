---
title: >-
  [Paper Note] Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification
description: >-
  [ICML2026][MNL bandits] This paper provides the first **computationally feasible** G-optimal experimental design for the combinatorial action space of Multinomial Logit (MNL) bandits. By formulating the Frank–Wolfe Linea…
tags:
  - "ICML2026"
  - "MNL bandits"
  - "G-optimal design"
  - "Frank-Wolfe"
  - "Mixed-Integer Linear Programming"
  - "Best Assortment Identification"
date: 2026-05-08
content_hash: 284d8bf8d162ee46
---

# Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification

**Conference**: ICML2026  
**arXiv**: [2605.25592](https://arxiv.org/abs/2605.25592)  
**Code**: TBD  
**Area**: others (Multinomial Logit Bandits / Experimental Design / Pure Exploration)  
**Keywords**: MNL bandits, G-optimal design, Frank-Wolfe, Mixed-Integer Linear Programming, Best Assortment Identification

## TL;DR
This paper provides the first **computationally feasible** G-optimal experimental design for the combinatorial action space of Multinomial Logit (MNL) bandits. By formulating the Frank–Wolfe Linear Maximization Oracle (LMO) as a 0–1 MILP or a polynomial-time Schur complement relaxation, the authors construct the first Best Assortment Identification algorithm for "linear utility + non-uniform rewards" with a sample complexity of $\tilde{\mathcal{O}}(d\log N / \Delta^2)$.

## Background & Motivation

**Background**: In online advertising, recommendation, and dynamic pricing, a decision-maker presents an **assortment (subset)** $S$ (up to size $K$) to a user, who then selects one item (or none) based on the MNL model. While MNL bandit literature has matured regarding regret minimization (Agrawal 2019, Oh 2021, Perivier 2022, Lee 2024), **pure exploration/best assortment identification** remains largely unexplored—especially when items have feature vectors and utilities are linear $\mathbf{a}_i^\top \theta^*$, where no sample complexity bounds previously existed.

**Limitations of Prior Work**: Directly applying standard tools from linear bandits—G-optimal design + Frank–Wolfe + KW Equivalence Theorem—encounters major obstacles: (1) The Fisher information for MNL $\mathbf{I}_\theta(S) = \sum_{i\in S} p(i|S,\theta)(\mathbf{a}_i - \bar{\mathbf{a}}_\theta(S))(\mathbf{a}_i - \bar{\mathbf{a}}_\theta(S))^\top$ **cannot be decomposed into a sum of individual arms** due to the probabilistic coupling of the entire subset; (2) The design space contains $|\mathcal{S}|=\mathcal{O}(N^K)$ combinations, making the LMO required in each Frank–Wolfe step naturally NP-hard; (3) The only related work, DopeWolfe (Thekumparampil 2024), uses random sampling for the LMO, which theoretically requires $\mathcal{O}(N^K)$ samples into order to guarantee approximation error, effectively leaving the problem unsolved.

**Key Challenge**: Experimental design demands **statistical efficiency** (ensuring the Fisher information matrix $\mathbf{M}$ "covers" all directions well), while combinatorial action spaces demand **computational efficiency**. No bridge existed between these two requirements.

**Goal**: (i) Formulate the MNL G-optimal design as a 0–1 MILP processable by modern solvers or a polynomial-time relaxation; (ii) Construct a Best Assortment Identification algorithm with sample complexity guarantees using the resulting design.

**Key Insight**: The authors observe that the MNL Fisher information $\mathbf{I}_\theta(S)$ can be represented as the Schur complement of a simpler second moment matrix $\widetilde{\mathbf{I}}_\theta(S) = \sum_{i\in S} p(i|S,\theta) \tilde{\mathbf{a}}_i \tilde{\mathbf{a}}_i^\top$ after **lifting** the features to $\tilde{\mathbf{a}}_i = (\mathbf{a}_i^\top, 1)^\top$. This transforms the nonlinear centralization term $-\bar{\mathbf{a}}_\theta(S)\bar{\mathbf{a}}_\theta(S)^\top$ into a linear block of the lifted matrix, clarifying the combinatorial optimization structure.

**Core Idea**: Solve the "hard nonlinear LMO" via two paths: an exact MILP (NP-hard but manageable via solvers with certified early stopping) or a ratio-based optimization after Schur complement relaxation (polynomial time with bounded error).

## Method

### Overall Architecture

The pipeline centers on **local G-optimal design**: (1) Given nominal parameters $\theta_0$ (estimated via a warm-up phase); (2) Run Frank–Wolfe to maximize the D-optimal objective $f_{\theta_0}(\pi) = \log\det(\mathbf{M}_{\theta_0}(\pi))$ (the KW Equivalence Theorem ensures this shares the same optimal solution as G-optimal); (3) In each FW iteration, solve an LMO $S_m \in \arg\max_S \text{tr}(\mathbf{M}_m^{-1} \mathbf{I}_{\theta_0}(S))$; (4) Feed the resulting design $\hat\pi_{\theta_0}$ into the BSI-MNL (Best aSsortment Identification for MNL) algorithm for pure exploration.

### Key Designs

1. **MILP Exact Reformulation + Solver-Certified Early Stopping**
    - **Function**: Rewrites the combinatorial LMO $\arg\max_S \text{tr}(\mathbf{M}_m^{-1}\mathbf{I}_{\theta_0}(S))$ as a 0–1 MILP where the number of variables and constraints are polynomial in $N$ (Theorem 3.3).
    - **Mechanism**: Since MNL probability $p(i|S,\theta_0) = w_i/\sum_{j\in S} w_j$ depends on $S$, the term $\sum_{j\in S} w_j$ is introduced as a denominator via auxiliary variables. Combined with big-M constraints, $\text{tr}(\cdot)$ is expressed as a linear function of 0–1 subset indicators. The centralization term $\bar{\mathbf{a}}_\theta(S)\bar{\mathbf{a}}_\theta(S)^\top$ is linearized through McCormick envelopes. Modern branch-and-bound solvers maintain a **current best feasible solution** and an **upper bound**, allowing safe early stopping when the solver-certified optimality gap is below $\epsilon_{\text{LMO}}$.
    - **Design Motivation**: While MILP is NP-hard in the worst case, industrial solvers (Gurobi, CPLEX) often find certified optima for $N\sim 10^3, K\sim 5$ in seconds. Combined with the FW iteration bound $\tilde{\mathcal{O}}(d/\tilde\epsilon)$, this bridges the gap between theoretical feasibility and practical execution.

2. **Schur Complement Lifting + Ratio Maximization Relaxation**
    - **Function**: Substitutes the original LMO with a lifted version $S_m \in \arg\max_S \text{tr}(\widetilde{\mathbf{M}}_m^{-1} \widetilde{\mathbf{I}}_{\theta_0}(S))$, resulting in a **strictly polynomial-time** proxy (Theorem 3.5).
    - **Mechanism**: Defining $\tilde{\mathbf{a}}_i = (\mathbf{a}_i^\top, 1)^\top \in \mathbb{R}^{d+1}$ and lifted Fisher $\widetilde{\mathbf{I}}_{\theta_0}(S) = \sum_{i\in S} p(i|S,\theta_0)\tilde{\mathbf{a}}_i \tilde{\mathbf{a}}_i^\top$, the original Fisher is its Schur complement. The LMO in the lifted space becomes a ratio form $\text{tr}(\widetilde{\mathbf{M}}_m^{-1}\widetilde{\mathbf{I}}_{\theta_0}(S)) = \sum_{i\in S} w_i s_i / \sum_{j\in S} w_j$, which is a classic MNL ratio-of-sums assortment optimization solvable in $\mathcal{O}(NK)$ time.
    - **Design Motivation**: Trading statistical efficiency for computational scalability. The mismatch between the lifted and true design is controlled by a PSD upper bound $\Delta_{\theta_0}(\pi)$.

3. **BSI-MNL: Design-Based Best Assortment Identification**
    - **Function**: Identifies $S^* = \arg\max_S R(S,\theta^*)$ with fixed confidence $\delta$ using minimal samples, supporting non-uniform rewards $r_i$.
    - **Mechanism**: Three stages: (a) Brief uniform exploration to obtain $\theta_0$; (b) Sampling assortments according to the G-optimal design $\hat\pi_{\theta_0}$ and updating MLE; (c) Applying a stopping criterion based on the estimated reward gap $\hat\Delta$ and the $\|\cdot\|_{\mathbf{M}^{-1}}$ confidence radius.
    - **Design Motivation**: Applying the G-optimal pure exploration template to MNL for the first time. The final sample complexity $\tilde{\mathcal{O}}(d\log N / \Delta_{\min}^2)$ shows only logarithmic dependence on $N$, highlighting the advantage of optimized design over uniform sampling.

### Loss & Training
The algorithm uses Maximum Likelihood Estimation (MLE) by maximizing the negative log-likelihood $\ell_t(\theta) = -\sum_{i\in S_t} y_{ti}\log p(i|S_t,\theta)$. The stopping rule is based on a GLRT-style statistic $\|\hat\theta - \theta\|_{\mathbf{V}_t}^2$, where $\mathbf{V}_t$ is the cumulative Fisher information. No neural network training is required; the algorithm relies on convex optimization and design solvers.

## Key Experimental Results

### Main Results: Sample Complexity Comparison

| Setting | Algorithm | Sample Complexity | Features | Non-uniform Rewards |
|------|------|------------|-----------|--------------------|
| MNL bandit, context-free | Saure & Zeevi (2013), Yang (2021) | $\tilde{\mathcal{O}}(N/\Delta_{\min}^2)$ | No | Partial |
| Linear bandit pure exploration | Soare et al. (2014) | $\tilde{\mathcal{O}}(d\log N/\Delta_{\min}^2)$ | Yes | N/A |
| MNL bandit, linear utility | **BSI-MNL (Ours)** | $\tilde{\mathcal{O}}(d\log N/\Delta_{\min}^2)$ | Yes | Yes |

### Ablation Study: LMO Implementation Comparison

| LMO Implementation | Per-step Complexity | FW Iterations | Worst-case Guarantee | Practice |
|----------|------------|-----------|--------------|----------|
| Enumeration | $\mathcal{O}(N^K)$ | $\tilde{\mathcal{O}}(d/\epsilon)$ | Exact | Infeasible for $N>30$ |
| DopeWolfe (Thekumparampil 2024) | Sampling $\mathcal{O}(N^K)$ | Pre-specified | $\epsilon$-accurate (w.h.p) | Stuck at combinatorial scales |
| **MILP + Stop (Ours)** | NP-hard (Secs in practice) | $\tilde{\mathcal{O}}(d/\tilde\epsilon)$ | Exact / Certified | Fits $N\sim 10^3$ |
| **Schur Relaxation (Ours)** | $\mathcal{O}(NK)$ | $\tilde{\mathcal{O}}(d/\epsilon)$ | Bounded mismatch | Scalable for very large $N$ |

### Key Findings
- **$\log N$ is the victory of contextualization**: Reducing complexity from $\mathcal{O}(N)$ to $\mathcal{O}(\log N)$ relies entirely on the feature structure, consistent with linear bandit intuition.
- **Dual LMO paths cover the precision-scalable spectrum**: MILP provides precision, while relaxation provides scale. The KW Equivalence Theorem ensures the design is $\epsilon$-optimal as long as the objective bound is met.
- **Bounded support size**: Proposition 3.2 proves that an optimal design exists with $|\text{supp}(\pi^*_{\theta_0})| \le d(d+1)/2$, meaning only a few assortments need to be rotated in practice.

## Highlights & Insights
- **The "lifting + Schur" method is highly transferable**: This approach can be applied to any structure where Fisher information equals the second moment minus the outer product of the mean (e.g., softmax, Plackett–Luce).
- **Serious treatment of solver-certified stopping**: Using the dual bound of a MILP solver as a theoretical tool to quantify the "run-time vs. accuracy" trade-off is a sophisticated strategy.
- **Asymptotic optimality for MNL**: Achieving $\tilde{\mathcal{O}}(d\log N/\Delta_{\min}^2)$ proves that "combinatorial + MNL feedback" imposes no extra sample cost relative to linear bandits in the pure exploration setting.
- **Honest handling of $\kappa$**: The inclusion of $(\kappa\Delta_{\min})^{-1}$ reflects the inherent difficulty of logistic-type feedback, consistent with prior work on logistic bandits.

## Limitations & Future Work
- **Dependency on $\theta_0$**: If the warm-up phase provides a poor estimate of the nominal parameters, the G-optimal design may be significantly misaligned.
- **MILP Scalability**: While solvers are fast, there is no guarantee on solve time for extremely large $N$ or $K$.
- **Schur Relaxation Error**: Mismatch bounds depend on parameter ranges like $\|\theta_0\|$ and $\|\mathbf{a}_i\|$, which may be loose for long-tail features.
- **Assortment Constraints**: Extending the MILP framework to handle complex real-world constraints (diversity, compatibility) remains to be verified.

## Related Work & Insights
- **vs. DopeWolfe (2024)**: They avoid exact LMO through randomization but still suffer $\mathcal{O}(N^K)$ sample requirements; this work breaks the combinatorial barrier using MILP and relaxation.
- **vs. Soare et al. (2014) / Fiez et al. (2019)**: Foundational linear bandit works; this paper proves structural advantages are preserved in MNL settings.
- **vs. Yang et al. (2021)**: Context-free MNL exploration; this work leverages feature structures to reduce dependency from $N$ to $\log N$.
- **vs. Faury et al. (2022)**: Consistent with logistic bandit research, the $\kappa$ term is acknowledged as an inherent difficulty of logistic-type feedback.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimal Regularization for Performative Learning](optimal_regularization_for_performative_learning.md)
- [\[ICML 2026\] Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding](conditional_krr_injecting_unpenalized_features_into_kernel_methods_with_applicat.md)
- [\[CVPR 2026\] ELogitNorm: Enhancing OOD Detection with Extended Logit Normalization](../../CVPR2026/others/enhancing_outofdistribution_detection_with_extende.md)
- [\[ICCV 2025\] Toward Material-Agnostic System Identification from Videos](../../ICCV2025/others/toward_material-agnostic_system_identification_from_videos.md)
- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2025\] Near Optimal Best Arm Identification for Clustered Bandits](../../ICML2025/others/near_optimal_best_arm_identification_for_clustered_bandits.md)
- [\[ICML 2026\] Optimal Regularization for Performative Learning](optimal_regularization_for_performative_learning.md)
- [\[ICML 2026\] Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding](conditional_krr_injecting_unpenalized_features_into_kernel_methods_with_applicat.md)
- [\[ICML 2026\] Comprehensive AI Governance Requires Addressing Non-Model Gains](comprehensive_ai_governance_requires_addressing_non-model_gains.md)
- [\[ICML 2025\] Optimal Auction Design in the Joint Advertising](../../ICML2025/others/optimal_auction_design_in_the_joint_advertising.md)

</div>

<!-- RELATED:END -->
