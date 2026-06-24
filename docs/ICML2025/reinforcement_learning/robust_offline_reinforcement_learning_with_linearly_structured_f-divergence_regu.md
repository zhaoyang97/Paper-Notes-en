---
title: >-
  [Paper Note] Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization
description: >-
  [ICML 2025][Reinforcement Learning][robust offline RL] Proposes the d-rectangular linear RRMDP (d-RRMDP) framework, incorporating latent linear structures into both the transition kernel and the f-divergence regularization. It designs the R2PVI algorithm to learn robust policies from offline data, proves an instance-dependent suboptimality upper bound, and validates its near-optimality through an information-theoretic lower bound.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "robust offline RL"
  - "f-divergence regularization"
  - "linear structure"
  - "d-rectangular"
  - "pessimistic value iteration"
date: 2026-05-08
content_hash: e33804a58e127c32
---

# Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization

**Conference**: ICML 2025  
**arXiv**: [2411.18612](https://arxiv.org/abs/2411.18612)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: robust offline RL, f-divergence regularization, linear structure, d-rectangular, pessimistic value iteration

## TL;DR

Proposes the d-rectangular linear RRMDP (d-RRMDP) framework, incorporating latent linear structures into both the transition kernel and the f-divergence regularization. It designs the R2PVI algorithm to learn robust policies from offline data, proves an instance-dependent suboptimality upper bound, and validates its near-optimality through an information-theoretic lower bound.

## Background & Motivation

**Background**: In offline reinforcement learning, agents learn policies from pre-collected datasets, which faces the challenge of potential dynamical discrepancies between training and deployment environments. The Robust Regularized MDP (RRMDP) framework learns policies robust to environmental perturbations by adding regularization terms to the value function, becoming an important tool for addressing distribution shifts.

**Limitations of Prior Work**: Prior RRMDP methods mostly employ unstructured regularization, enforcing a holistic f-divergence constraint on the transition kernel. This allows state transitions to experience worst-case perturbations simultaneously across all dimensions. Although this approach offers the strongest robustness guarantees, it often leads to overly conservative policies that optimize against unrealistic, extreme transition scenarios, resulting in severe performance degradation in normal deployment environments.

**Key Challenge**: The fundamental trade-off between robustness and conservatism. Unstructured regularization considers all possible perturbation directions (including joint extreme perturbations that are unrealistic), leading to overly pessimistic policies. However, abandoning robustness entirely renders the policy unable to handle inevitable environmental discrepancies in real-world deployments.

**Goal**: To design a structured regularization approach that aligns robustness constraints with real-world dynamic variation patterns—focusing only on "reasonable" perturbation directions rather than all theoretically possible ones. This is decomposed into three subproblems: (1) How to incorporate linear structures into RRMDP? (2) How to design efficient offline algorithms under this framework? (3) Is the sample complexity of the algorithm near-optimal?

**Key Insight**: Under the linear MDP assumption, the transition kernel naturally possesses a $d$-dimensional decomposition structure ($P_h(s'|s,a) = \langle \phi(s,a), \mu_h(s') \rangle$). This allows the f-divergence regularization to be naturally decomposed along these dimensions—namely, the d-rectangular structure. This decomposition applies independent constraints to each latent dimension, avoiding joint extreme perturbations across dimensions.

**Core Idea**: Enforcing dimension-decomposed f-divergence regularization within the latent space of linear MDPs. This requires the robust policy to defend only against independent perturbations in each dimension rather than joint perturbations, substantially reducing conservatism while maintaining robustness.

## Method

### Overall Architecture

The input consists of an offline dataset (transition tuples $(s, a, r, s')$ collected from a nominal environment) and the definition of the d-RRMDP (including the feature map $\phi$, regularization radius $\rho$, and f-divergence type). The R2PVI algorithm performs backward induction over $H$ value-iteration steps to estimate the robust value functions. In each step, it uses linear regression to estimate value function parameters and incorporates a pessimistic penalty term to handle poorly covered state-action regions. The output is the robust policy $\hat\pi$.

### Key Designs

1. **d-Rectangular Linear RRMDP**:

    - **Function**: Defines a structured robust MDP framework that restricts the uncertainty set of environmental perturbations to a controllable structure.
    - **Mechanism**: In linear MDPs, the transition kernel can be decomposed as $P_h(\cdot|s,a) = \Phi(s,a)^\top \mu_h(\cdot)$, where $\Phi \in \mathbb{R}^d$ is a feature vector. The d-rectangular regularization imposes independent f-divergence constraints on each dimension $i \in [d]$: $D_f(\xi^{(i)}_h \| \mu^{(i)}_h) \leq \rho$ holds for all $i$. This implies that the measure shift of each latent dimension is independently constrained rather than applying a single constraint on the joint distribution.
    - **Design Motivation**: (s,a)-rectangular unstructured regularization allows independent worst-case perturbations for each state-action pair, resulting in an overly large uncertainty set. The d-rectangular approach leverages linear structure to restrict the uncertainty to the $d$-dimensional latent space, substantially narrowing the range of considered perturbations.

2. **R2PVI (Robust Regularized Pessimistic Value Iteration)**:

    - **Function**: Learns robust policies from offline data under the d-RRMDP framework.
    - **Mechanism**: The algorithm iterates backward from time step $H$ to $1$. At each step $h$: (a) It estimates the parameters of the robust Q-function from data using linear regression: $\hat w_h = \Lambda_h^{-1} \sum_{i} \phi(s_h^i, a_h^i) [r_h^i + \hat V_{h+1}^{\text{rob}}(s_{h+1}^i)]$, where $\Lambda_h = \sum_i \phi \phi^\top + \lambda I$ is the regularized Gram matrix; (b) It computes the robust value function $\hat V_h^{\text{rob}}$ using the convex dual form of f-divergence—duality transforms the min-max problem into convex optimization; (c) It adds a pessimistic penalty $\beta \|\phi(s,a)\|_{\Lambda_h^{-1}}$ to ensure conservative value function estimates in poorly covered data regions.
    - **Design Motivation**: Linear function approximation enables feasibility in large state spaces. The combination of pessimistic penalties and robust regularization provides a dual safeguard—the former handles distribution shift in offline data, while the latter handles environment dynamics shifts.

3. **Instance-Dependent Theoretical Analysis**:

    - **Function**: Provides tight upper bounds for the suboptimality of R2PVI and proves its near-optimality via information-theoretic lower bounds.
    - **Mechanism**: The upper bound is expressed as $\text{SubOpt}(\hat\pi) \leq \tilde O(\sum_h \sqrt{d / n} \cdot C_h^{\text{rob}})$, where $C_h^{\text{rob}}$ characterizes the degree to which the dataset covers the "robustly reachable" state-action space under the optimal robust policy. This coverage condition is stronger than standard offline RL: it requires coverage not only of optimal paths in the nominal environment, but also of paths under all robustly admissible transitions.
    - **Design Motivation**: The instance-dependent analysis reveals the fine-grained relationship between algorithm performance and problem hardness—automatically yielding tighter bounds for simpler problems.

### Loss & Training

The core computation of R2PVI involves solving the robust Bellman equation. For f-divergences (e.g., KL divergence or $\chi^2$ divergence), the robust value function is computed via a dual formulation: $V_h^{\text{rob}}(s) = \max_a [\hat Q_h(s,a) - \beta \|\phi(s,a)\|_{\Lambda_h^{-1}} - \rho \cdot \text{DualPenalty}]$, where DualPenalty depends on the specific f-divergence type—having a log-sum-exp form for KL divergence, and a quadratic penalty form for $\chi^2$ divergence.

## Key Experimental Results

### Synthetic MDP Experiments: Structured vs. Unstructured Regularization

| Method | Suboptimality Gap | Computational Efficiency | Conservatism |
|------|-----------|---------|--------|
| Unstructured RRMDP | Higher | Slow | Overly Conservative |
| d-RRMDP (KL Divergence) | Significantly Lower | Fast | Moderate |
| d-RRMDP ($\chi^2$ Divergence) | Significantly Lower | Fast | Moderate |
| Non-Robust Baseline | Lowest (Nominal Env) | Fastest | No Robustness |

### Comparison of Crucial Theoretical Metrics

| Metric | R2PVI (Ours) | Prior Unstructured Methods | Non-robust Offline RL |
|------|------------|-------------|-------------|
| Uncertainty Dimension | $d$ (Latent Space) | $|\mathcal{S}|$ (State Space) | — |
| Sample Complexity Order | $\tilde O(d H^2 / \epsilon^2)$ | $\tilde O(|\mathcal{S}| H^2 / \epsilon^2)$ | $\tilde O(d H^2 / \epsilon^2)$ |
| Near Lower Bound | Yes | Unknown | Yes |
| Handles Dynamic Shift | Yes (Structured) | Yes (Unstructured) | No |

### Key Findings

1. The d-rectangular structure leverages the latent low-dimensionality of linear MDPs to reduce the dimension of the robustness constraints from $|\mathcal{S}|$ (which can be extremely large) to $d$ (the feature dimension), substantially mitigating conservatism while maintaining robustness.
2. The information-theoretic lower bound proves that the sample complexity order of R2PVI is minimax optimal—incapable of being further improved without additional assumptions.
3. Different types of f-divergences (KL and $\chi^2$) exhibit similar performance in trials, indicating that the framework is robust to the choice of divergence.
4. The regularization strength $\rho$ controls the robustness-performance trade-off: an overly large $\rho$ leads to excessive conservatism, whereas an overly small $\rho$ fails to guard against dynamics shifts.

## Highlights & Insights

1. **Injecting problem structures into robustness constraints**: The d-rectangular structure is naturally derived from the low-rank factorization of linear MDPs—it is an intrinsic property of the data generation process rather than an artificially designed constraint.
2. **Theoretical completeness**: Concurrent presentation of upper and lower bounds confirms the optimality of the algorithm—a first under the linear function approximation setting for robust offline RL.
3. **Generality of the f-divergence framework**: Unifies various distance metrics like KL divergence, $\chi^2$ divergence, and total variation distance under one family, providing flexibility for practical realizations.
4. **Computational advantages of convex dualization**: Converts min-max problems in the robust Bellman equation into convex optimization, making R2PVI more computationally efficient than methods requiring explicit adversarial game solving.

## Limitations & Future Work

- The linear MDP assumption may not hold in complex practical tasks—extending this to non-linear function approximations (e.g., neural networks) remains an important open question.
- Experiments are only verified on synthetic MDPs, lacking empirical validation on standard offline RL benchmarks such as MuJoCo or D4RL.
- The regularization radius $\rho$ requires prior knowledge to set—methods for adaptive selection of $\rho$ have not been explored.
- Although the d-rectangular structure is more reasonable than unstructured alternatives, it may still fail to fully match the true patterns of actual environmental perturbations.

## Related Work & Insights

- Classical RMDP (Nilim & El Ghaoui, 2005) and RRMDP (Panaganti & Kalathil, 2022) established the theoretical foundations of robust MDPs.
- The pessimism principle in offline RL (Jin et al., 2021) and robust regularization complement each other—addressing offline data distribution shifts and environmental dynamic shifts, respectively.
- **Insight**: The concept of structured uncertainty sets can be generalized to other robust decision-making problems (e.g., factored uncertainty sets in distributionally robust optimization).

## Rating

⭐⭐⭐⭐ Solid theoretical contribution—the first systematic study of d-rectangular linear structured regularization, with matching upper and lower bounds proving near-optimality. The limited experimental scale is the primary drawback, but as a theoretical piece, the overall quality is high, representing a major advancement in function approximation for robust offline RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL](learning_to_trust_bellman_updates_selective_state-adaptive_regularization_for_of.md)
- [\[ICLR 2026\] Beyond Penalization: Diffusion-based Out-of-Distribution Detection and Selective Regularization in Offline Reinforcement Learning](../../ICLR2026/reinforcement_learning/beyond_penalization_diffusion-based_out-of-distribution_detection_and_selective_.md)
- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](../../NeurIPS2025/reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[ACL 2025\] Learning to Generate Structured Output with Schema Reinforcement Learning](../../ACL2025/reinforcement_learning/learning_to_generate_structured_output_with_schema_reinforcement_learning.md)
- [\[ICLR 2026\] Belief-Based Offline Reinforcement Learning for Delay-Robust Policy Optimization](../../ICLR2026/reinforcement_learning/belief-based_offline_reinforcement_learning_for_delay-robust_policy_optimization.md)

</div>

<!-- RELATED:END -->
