---
title: >-
  [Paper Note] A Unifying View of Linear Function Approximation in Off-Policy RL Through Matrix Splitting and Preconditioning
description: >-
  [NeurIPS 2025 (Spotlight, top 3%)][Reinforcement Learning][temporal difference learning] This work is the first to introduce matrix splitting theory, unifying TD, FQI, and PFQI under linear function approximation as iterative methods for solving the same target linear system $(\Sigma_{cov} - \gamma\Sigma_{cr})\theta = \theta_{\phi,r}$, differing only in their preconditioners. It establishes necessary and sufficient conditions for the convergence of each algorithm…
tags:
  - "NeurIPS 2025 (Spotlight, top 3%)"
  - "Reinforcement Learning"
  - "temporal difference learning"
  - "fitted Q-iteration"
  - "matrix splitting"
  - "preconditioning"
  - "convergence analysis"
date: 2026-05-08
content_hash: 07bb8dfaa8180307
---

# A Unifying View of Linear Function Approximation in Off-Policy RL Through Matrix Splitting and Preconditioning

**Conference**: NeurIPS 2025 (Spotlight, top 3%)
**arXiv**: [2501.01774](https://arxiv.org/abs/2501.01774)  
**Code**: None  
**Area**: Reinforcement Learning / Policy Evaluation / Theoretical Analysis
**Keywords**: temporal difference learning, fitted Q-iteration, matrix splitting, preconditioning, convergence analysis

## TL;DR
This work is the first to introduce matrix splitting theory, unifying TD, FQI, and PFQI under linear function approximation as iterative methods for solving the same target linear system $(\Sigma_{cov} - \gamma\Sigma_{cr})\theta = \theta_{\phi,r}$, differing only in their preconditioners. It establishes necessary and sufficient conditions for the convergence of each algorithm, introduces the novel concept of rank invariance, and reveals that target networks are fundamentally a continuous transformation of the preconditioner from a constant to a data-adaptive form.

## Background & Motivation
**Background**: In off-policy policy evaluation (OPE), TD learning may diverge while FQI is generally more stable. The conventional understanding holds that the three algorithms differ only in the number of updates applied to the target value function (TD = 1, FQI = ∞, PFQI = finite).

**Limitations of Prior Work**: (1) The conventional view cannot explain why TD convergence does not imply FQI convergence, nor vice versa. (2) Existing convergence analyses rely on strong assumptions such as feature linear independence, yielding only sufficient rather than necessary and sufficient conditions. (3) Several erroneous claims exist in the literature (e.g., Ghosh 2020 asserts that linear independence alone guarantees a unique off-policy TD fixed point — this paper shows that rank invariance is also required).

**Key Challenge**: Do three seemingly distinct algorithms share a unified mathematical essence? What are the precise convergence conditions for each?

**Key Insight**: Drawing on matrix splitting and preconditioning techniques from numerical linear algebra, the three RL algorithms are reformulated as instances of the unified iterative scheme $\theta_{k+1} = (I - MA)\theta_k + Mb$ with different preconditioners $M$.

## Method

### Overall Architecture
The core insight is that TD, FQI, and PFQI all solve the same target linear system (the LSTD system): $(\Sigma_{cov} - \gamma\Sigma_{cr})\theta = \theta_{\phi,r}$. The algorithms differ **only in the choice of preconditioner $M$** — TD uses $M = \alpha I$ (constant), FQI uses $M = \Sigma_{cov}^{-1}$ (data-adaptive), and PFQI uses $M = \alpha\sum_{i=0}^{t-1}(I-\alpha\Sigma_{cov})^i$ (a transition from TD to FQI). Convergence is entirely determined by the consistency of the target linear system and the semiconvergence of the iteration matrix $H = I - MA$.

### Key Designs
1. **Rank Invariance**:
    - Function: Proposes a new, mild condition to replace the traditional feature linear independence assumption.
    - Mechanism: Defines $\text{Rank}(\Phi) = \text{Rank}(\Phi^\top \mathbf{D}(I - \gamma\mathbf{P}_\pi)\Phi)$, equivalently requiring that $\gamma\Sigma_{cov}^\dagger\Sigma_{cr}$ has no eigenvalue equal to 1. Proves that rank invariance is a necessary and sufficient condition for the target linear system to have a solution for any reward function $R$ (universal consistency) (Proposition 4.2); rank invariance together with feature linear independence implies a unique solution (Proposition 4.5); rank invariance holds automatically in the on-policy setting (Proposition 4.7).
    - Design Motivation: Feature linear independence is an overly strong assumption, as overparameterization is common in practice. Rank invariance is milder and nearly always satisfied, providing a new foundation for analysis that removes redundant assumptions from much prior work.

2. **Continuous Preconditioner Transformation and Proper Splitting**:
    - Function: Reveals the preconditioner transformation chain TD→PFQI→FQI and proves that under rank invariance, FQI benefits from the proper splitting property.
    - Mechanism: $\alpha I \underset{t=1}{\rightleftharpoons} \alpha\sum_{i=0}^{t-1}(I-\alpha\Sigma_{cov})^i \xrightarrow{t\to\infty} \Sigma_{cov}^{-1}$. When rank invariance holds, $(\Sigma_{cov}, \Sigma_{cr})$ constitutes a proper splitting (Lemma 5.2), relaxing the FQI convergence condition to $\rho(\gamma\Sigma_{cov}^\dagger\Sigma_{cr}) < 1$ and guaranteeing a unique fixed point. This explains why FQI is more robust than TD — its preconditioner adapts to the data distribution.
    - Design Motivation: The theoretical nature of target networks (widely used in DQN) is characterized for the first time: increasing the target update interval $t$ is equivalent to a continuous transition from a constant preconditioner to a data-adaptive preconditioner based on the inverse feature covariance.

3. **TD Learning Rate Interval and the Encoder–Decoder Perspective**:
    - Function: Provides necessary and sufficient conditions for TD stability and a precise characterization of admissible learning rates.
    - Mechanism: TD is stable if and only if three conditions hold simultaneously — consistency, positive semistability of $A_{LSTD}$, and $\text{Index}(A_{LSTD}) \leq 1$ (Corollary 6.2). Admissible learning rates form the interval $\alpha \in (0, \epsilon)$, where $\epsilon = \min_{\lambda \in \sigma(A_{LSTD})\backslash\{0\}} \frac{2\Re(\lambda)}{|\lambda|^2}$ (Corollary 6.3). In the on-policy setting, TD remains stable even without feature linear independence (Theorem 6.4), relaxing the classical condition of Tsitsiklis & Van Roy (1996).
    - Design Motivation: This constitutes the first proof that "when large learning rates fail, small ones may succeed" — admissible learning rates form a continuous interval rather than isolated points, providing theoretical grounding for practical hyperparameter tuning.

### Loss & Training
This is a purely theoretical work with no training procedure. The central contribution is proving that the three iterative schemes — TD: $\theta_{k+1} = (I - \alpha(\Sigma_{cov} - \gamma\Sigma_{cr}))\theta_k + \alpha\theta_{\phi,r}$, FQI: $\theta_{k+1} = \gamma\Sigma_{cov}^\dagger\Sigma_{cr}\theta_k + \Sigma_{cov}^\dagger\theta_{\phi,r}$, and PFQI: nested $t$-step TD updates — all converge to the solution set $\Theta_{LSTD}$ of the target linear system (when they converge). A key finding is that increasing $t$ (the target network update interval) in PFQI may cause divergence rather than convergence when features are not linearly independent.

## Key Experimental Results

### Summary of Core Theoretical Results

| Algorithm | Necessary & Sufficient Convergence Conditions | Preconditioner | Key Properties |
|---|---|---|---|
| TD | Consistency + semiconvergence of $H_{TD}$ | $\alpha I$ (constant) | Convergence depends on learning rate $\alpha \in (0, \epsilon)$ |
| FQI | Consistency + semiconvergence of $H_{FQI}$ | $\Sigma_{cov}^{-1}$ (adaptive) | Condition relaxed to $\rho < 1$ under rank invariance |
| PFQI | Consistency + semiconvergence of $H_{PFQI}$ | Transition from TD to FQI | Increasing $t$ does not guarantee stabilization |

### Convergence Implication Relations Between Algorithms

| Implication | Holds? | Key Condition |
|---|---|---|
| TD converges → FQI converges | ✗ Not necessarily | Counterexample constructed |
| FQI converges → TD converges | ✗ Not necessarily | Counterexample constructed |
| TD stable → PFQI converges | ✓ Conditionally | For any finite $t$, there exists $\epsilon_t$ such that PFQI converges for $\alpha \in (0, \epsilon_t)$ |
| Increasing $t$ → PFQI more stable | ✗ Not necessarily | Increasing $t$ may cause divergence when features are not linearly independent |

### Key Findings
- Rank invariance alone suffices to guarantee non-singularity of the FQI linear system (Proposition 4.6), while the target linear system requires both rank invariance and feature linear independence (Proposition 4.5).
- In the on-policy setting, rank invariance holds automatically, guaranteeing the existence of fixed points for TD, FQI, and PFQI.
- Literature corrections: conditions stated in Ghosh (2020), Asadi (2024), and Xiao (2021) are identified as sufficient rather than necessary and sufficient.

## Highlights & Insights
- The unifying framework is remarkably elegant — a single difference in preconditioners unifies three algorithms, yielding a mathematically clean and powerful result.
- Classical tools from numerical linear algebra (matrix splitting, proper splitting, semiconvergent matrices) are introduced into RL convergence analysis for the first time.
- Necessary and sufficient conditions — strictly sharper than prior work — are established, and multiple errors in the literature are corrected.
- The theoretical nature of target networks is characterized for the first time as a continuous transformation of the preconditioner from constant to data-adaptive.
- The "admissible learning rate interval" result provides theoretical support for practical tuning, and the encoder–decoder perspective offers new intuition for TD convergence.

## Limitations & Future Work
- Restricted to linear function approximation — although the final layer of neural networks is typically linear, the core theory does not directly extend to the nonlinear setting.
- Limited to policy evaluation and does not address control (policy improvement/optimization).
- Purely theoretical, lacking empirical validation — whether conditions such as rank invariance are commonly satisfied in large-scale practical problems remains empirically untested.
- The main analysis concerns expected TD (the deterministic version); extensions to stochastic/batch TD are claimed but treated only briefly.

## Related Work & Insights
- **vs. Tsitsiklis & Van Roy (1996)**: The classical on-policy TD convergence result requires feature linear independence; Theorem 6.4 of this paper proves that this assumption can be removed.
- **vs. Fellows et al. (2023)**: Prior work provides only sufficient conditions for PFQI convergence under FQI convergence conditions when increasing $t$; this paper provides necessary and sufficient conditions and reveals that increasing $t$ may instead cause divergence.
- **vs. DQN target networks**: This paper provides the first theoretical explanation of target networks as a preconditioner transformation, offering guidance for designing improved target update strategies.
- **Insights**: The matrix splitting perspective may inspire new RL algorithms by motivating the design of superior preconditioners to accelerate convergence; rank invariance may become a new standard assumption in RL theoretical analysis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The matrix splitting perspective is entirely new; the elegance of unifying three algorithms is exceptional.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical work with only counterexample constructions and no empirical validation.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, but the mathematical density is extremely high (50+ page appendix), requiring a strong background in linear algebra.
- Value: ⭐⭐⭐⭐⭐ A Spotlight paper with significant impact in the RL theory community, correcting multiple errors in the literature and establishing a new analytical paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Unifying View of Coverage in Linear Off-Policy Evaluation](../../ICLR2026/reinforcement_learning/a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)
- [\[ICLR 2026\] Replicable Reinforcement Learning with Linear Function Approximation](../../ICLR2026/reinforcement_learning/replicable_reinforcement_learning_with_linear_function_approximation.md)
- [\[ICLR 2026\] Is Pure Exploitation Sufficient in Exogenous MDPs with Linear Function Approximation?](../../ICLR2026/reinforcement_learning/is_pure_exploitation_sufficient_in_exogenous_mdps_with_linear_function_approxima.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model (BOOM)](bootstrap_off-policy_with_world_model.md)
- [\[NeurIPS 2025\] Scalable Policy-Based RL Algorithms for POMDPs](scalable_policy-based_rl_algorithms_for_pomdps.md)

</div>

<!-- RELATED:END -->
