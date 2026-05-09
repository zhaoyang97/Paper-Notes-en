---
title: >-
  [Paper Note] Last Iterate Convergence in Monotone Mean Field Games
description: >-
  [NeurIPS 2025][Reinforcement Learning][mean field games] This paper proposes a KL-divergence-based proximal point (PP) method that achieves asymptotic last iterate convergence (LIC) in non-strictly monotone mean field games (MFGs), and proves that regularized mirror descent (RMD) converges to regularized equilibria at an exponential rate. The combined approximate proximal point (APP) algorithm reliably converges to non-regularized equilibria on standard benchmarks.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - mean field games
  - last iterate convergence
  - proximal point
  - mirror descent
  - monotone games
date: 2026-05-08
content_hash: b1af6ff24c0e855e
---

# Last Iterate Convergence in Monotone Mean Field Games

**Conference**: NeurIPS 2025
**arXiv**: [2410.05127](https://arxiv.org/abs/2410.05127)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: mean field games, last iterate convergence, proximal point, mirror descent, monotone games

## TL;DR
This paper proposes a KL-divergence-based proximal point (PP) method that achieves asymptotic last iterate convergence (LIC) in non-strictly monotone mean field games (MFGs), and proves that regularized mirror descent (RMD) converges to regularized equilibria at an exponential rate. The combined approximate proximal point (APP) algorithm reliably converges to non-regularized equilibria on standard benchmarks.

## Background & Motivation

**State of the Field**: Mean field games (MFGs), introduced by Lasry and Lions, are a central framework for modeling large-population interactions. In discrete-time settings, online mirror descent (OMD) is the dominant algorithm for computing MFG equilibria, and when combined with deep learning it scales to high-dimensional problems.

**Limitations of Prior Work**:
- Existing algorithms either require strict monotonicity assumptions (which excludes transition-symmetric settings) or only guarantee convergence of time-averaged policies (e.g., Fictitious Play);
- Time-averaged policies are ill-defined in neural network parameter spaces due to nonlinearity — in practice, direct convergence of the last-iterate policy is required;
- Prior OMD/RMD methods under monotone MFGs achieve only polynomial convergence rates or provide guarantees only for the average iterate.

**Root Cause**: In MFGs, the mean field distribution $\mu$ changes with policy $\pi$, causing the objective $J(\mu^k, \cdot)$ to shift at every step. This prevents direct application of classical proximal point theory and creates a dependency of the Q-function on policy both forward (through $\mu$) and backward (through dynamic programming), obstructing standard backward induction arguments.

**Paper Goals**:
- Prove LIC (asymptotic convergence) of the PP method under non-strict monotonicity;
- Prove exponential convergence of RMD in regularized MFGs;
- Provide a practically feasible approximate proximal point (APP) algorithm.

**Starting Point**: The Łojasiewicz inequality (a tool from real analytic geometry) is used to handle PP convergence; the regularization effect of KL divergence is leveraged to overcome the non-Lipschitz continuity of the Q-function.

**Core Idea**: By decomposing inexact PP updates into equilibrium-finding subproblems for regularized MFGs, and approximating each subproblem via RMD at an exponential rate, the method achieves last iterate convergence in non-strictly monotone MFGs.

## Method

### Overall Architecture

The paper proposes a three-level nested structure:
- **Outer PP iteration**: Each step solves a KL-regularized MFG subproblem and updates the reference policy $\sigma^k$;
- **Inner RMD iteration**: Regularized mirror descent approximately solves each PP subproblem;
- **APP algorithm**: PP and RMD are combined, with a small number of RMD steps executed per outer PP iteration.

The input is an MFG definition $(\mathcal{S}, \mathcal{A}, H, P, r, \mu_1)$ and an initial policy $\pi^0$; the output is an approximate equilibrium policy.

### Key Designs

1. **KL-Proximal Point (PP) Method (Theorem 3.1)**:

   - **Function**: Iteratively updates the policy via $\sigma^{k+1} = \arg\max_\pi \{J(\mu^{k+1}, \pi) - \lambda D_{m[\pi]}(\pi, \sigma^k)\}$;
   - **Mechanism**: At each step, cumulative reward maximization is augmented with a KL divergence penalty to prevent large deviations from the previous policy. The KL divergence is weighted by the mean field: $D_\mu(\pi, \sigma) = \sum_h \mathbb{E}_{s \sim \mu_h}[D_{\text{KL}}(\pi_h(s), \sigma_h(s))]$;
   - **Design Motivation**: Classical OMD linearizes the objective $J$ in the PP update (using only gradient information), whereas PP uses the full non-linearized objective, making it more robust to approximation errors. The Łojasiewicz inequality is employed to handle the difficulty arising from $\mu$ depending on $\pi$;
   - **Distinction from Prior Work**: Prior LIC results required strict monotonicity; this result holds under non-strict monotonicity.

2. **Regularized Mirror Descent RMD (Theorem 4.3)**:

   - **Function**: Finds regularized equilibria at an exponential rate within KL-regularized MFGs;
   - **Mechanism**: Each step updates $\pi_h^{t+1}(a|s) \propto (\sigma_h(a|s))^{\lambda\eta} (\pi_h^t(a|s))^{1-\lambda\eta} \cdot \exp(\eta Q_h^{\lambda,\sigma}(s,a,\pi^t,\mu^t))$, mixing the reference policy $\sigma$ via KL regularization into the softmax update;
   - **Design Motivation**: The subproblem of each PP step (finding a regularized equilibrium) is implicit; RMD provides an efficient explicit approximation. The exponential convergence rate significantly improves upon prior polynomial rates ($O(\lambda \log^2 T / \sqrt{T})$ → exponential decay);
   - **Technical Challenge**: The Q-function $Q_h^{\lambda,\sigma}(s,a,\pi,\mu)$ is not Lipschitz continuous in $\pi$ (derivatives explode as $\pi$ approaches the boundary of the probability simplex); this is resolved by exploiting the regularization effect of KL divergence.

3. **Approximate Proximal Point (APP) Algorithm**:

   - **Function**: Combines PP and RMD into a practical algorithm;
   - **Mechanism**: In the outer loop, each step uses a small number of RMD iterations to approximately solve the PP subproblem, followed by an update of the reference policy $\sigma$. The key step is updating $\sigma$ — without this, RMD only converges to a regularized equilibrium, which deviates from the true equilibrium by $O(\lambda)$;
   - **Practical Effect**: Only a minor modification to standard RMD (periodically updating the reference $\sigma$) enables convergence to the non-regularized equilibrium in experiments.

### Loss & Training

- The regularization strength $\lambda$ in PP governs the trade-off between convergence speed and subproblem difficulty;
- The theoretical upper bound $\eta^*$ on the learning rate $\eta$ in RMD may be small (due to the complex dependence of the Q-function), requiring careful tuning in practice;
- The overall method is model-based: the transition kernel $P$ and reward function $r$ are assumed to be known.

## Key Experimental Results

### Main Results — Beach Bar Process Benchmark

Validation on the classical one-dimensional torus random walk MFG benchmark: $|\mathcal{S}| = 100$, $|\mathcal{A}| = 3$, $H = 50$.

| Algorithm | Convergence Behavior | Convergence Target | Requires Time Averaging |
|-----------|---------------------|--------------------|------------------------|
| RMD | Converges to regularized equilibrium | $\varpi^*$ (depends on $\lambda$) | No |
| APP | Converges to non-regularized equilibrium | $\pi^*$ | No |
| Fictitious Play | Average convergence | $\pi^*$ | Yes |

### Ablation Study

| Configuration | Key Observation | Explanation |
|---------------|----------------|-------------|
| RMD (no $\sigma$ update) | Converges to regularized equilibrium, deviates from true equilibrium | Policy remains in the interior of the probability simplex |
| APP (periodic $\sigma$ update) | Policy moves toward vertices of the probability simplex | Exploitability decreases after each $\sigma$ update |
| Continuous-time limit | Exponential convergence | Validates theoretical prediction of Theorem 4.5 |

### Key Findings

- **Monotone exploitability decrease in APP**: After each outer update of $\sigma$, exploitability decreases substantially, converging to equilibrium along the PP trajectory.
- **Limitation of RMD**: Without updating the reference policy, RMD shifts the equilibrium to the interior of the probability simplex (due to regularization), causing the policy distribution to "diffuse" rather than concentrate on the correct vertex.
- **No time averaging required**: Unlike Fictitious Play, the last iterate of APP is directly usable, which is critical for deep learning settings where time-averaging neural network parameters is ill-defined in nonlinear spaces.

## Highlights & Insights

- **Breakthrough LIC under non-strict monotonicity**: All prior MFG LIC results required strict monotonicity or contraction conditions. This paper is the first to achieve LIC under non-strict (weak) monotonicity, covering important settings such as transition symmetry, representing a significant advance in MFG convergence theory.
- **Novel application of the Łojasiewicz inequality**: Real analytic geometry tools are introduced into MFG convergence analysis to handle the core difficulty of $\mu$ varying with $\pi$. This technique generalizes to related variational inequality problems.
- **PP–regularized equilibrium equivalence**: Each PP step is shown to be equivalent to finding the equilibrium of a regularized MFG; this equivalence directly yields the APP algorithm — approximating each subproblem via RMD suffices to approach the non-regularized equilibrium.
- **Qualitative improvement: exponential vs. polynomial rates**: The exponential convergence of RMD (Theorem 4.3) far exceeds prior $O(1/t)$ or $O(\log^2 T / \sqrt{T})$ results, with substantial advantages at large iteration counts.

## Limitations & Future Work

- **Model-based assumption**: The transition kernel and reward function are assumed to be fully known; model-free settings requiring learning from data are not addressed, and no sample complexity or statistical guarantees are provided.
- **No computational cost analysis**: Only iteration complexity is improved; the computational cost per PP subproblem and scalability to large state/action spaces are not analyzed.
- **Deep learning compatibility not validated**: Although PP/RMD can in principle be combined with neural network function approximation, the paper does not study approximation errors, stability, or high-dimensional nonlinear settings.
- **Strict LIC proof for APP remains open**: Only experimental evidence (decreasing exploitability) is provided; a rigorous proof is left as an open problem, conjectured to require a uniform positive lower bound on $\eta^*$.
- **Future Directions**:
  - Extension to model-free and sample-based settings;
  - Integration with deep RL for large-scale MFGs (e.g., traffic, multi-robot coordination);
  - Analysis of asynchronous feedback (stability under delayed index effects).

## Related Work & Insights

- **vs. Fictitious Play**: FP guarantees time-average convergence but not LIC. The PP/APP methods directly guarantee LIC, making them better suited for deep learning training.
- **vs. OMD (Perolat et al.)**: OMD achieves LIC in continuous time under strict monotonicity, but lacks guarantees in discrete time and under non-strict monotonicity. This paper fills that gap.
- **vs. Contraction-based methods**: Methods such as FPPI and value iteration require contraction conditions that many MFG instances do not satisfy. The monotonicity condition used here is weaker and more practical.
- **vs. LIC in two-player zero-sum Markov games**: In two-player zero-sum games, the Q-function depends only on future policies (enabling backward induction), whereas in MFGs the Q-function simultaneously depends on past policies through the mean field — a technical difficulty specific to MFGs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First LIC result in non-strictly monotone MFGs; outstanding theoretical contribution
- Experimental Thoroughness: ⭐⭐⭐ Validated on only one standard benchmark (Beach Bar Process); large-scale and deep learning experiments are absent
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations; notation-heavy but logically clear
- Value: ⭐⭐⭐⭐ Significant advancement in MFG learning theory, though experimental support is limited

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis](learning_in_stackelberg_mean_field_games_a_non-asymptotic_analysis.md)
- [\[NeurIPS 2025\] Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics](solving_continuous_mean_field_games_deep_reinforcement_learning_for_non-stationa.md)
- [\[NeurIPS 2025\] Scalable Neural Incentive Design with Parameterized Mean-Field Approximation](scalable_neural_incentive_design_with_parameterized_mean-field_approximation.md)
- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[NeurIPS 2025\] Non-convex Entropic Mean-Field Optimization via Best Response Flow](non-convex_entropic_mean-field_optimization_via_best_response_flow.md)

<!-- RELATED:END -->
