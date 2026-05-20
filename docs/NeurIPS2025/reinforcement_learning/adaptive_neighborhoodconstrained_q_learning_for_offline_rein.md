---
title: >-
  [Paper Note] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][offline RL] This paper proposes ANQ (Adaptive Neighborhood-constrained Q learning), which introduces advantage-function-based adaptive neighborhood constraints for offline RL. ANQ o…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "offline RL"
  - "neighborhood constraint"
  - "OOD actions"
  - "adaptive conservatism"
  - "bilevel optimization"
date: 2026-05-08
content_hash: 456c2372604d18cc
---

# Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2511.02567](https://arxiv.org/abs/2511.02567)  
**Code**: [https://github.com/thu-rllab/ANQ](https://github.com/thu-rllab/ANQ)  
**Area**: Reinforcement Learning / Offline RL
**Keywords**: offline RL, neighborhood constraint, OOD actions, adaptive conservatism, bilevel optimization

## TL;DR
This paper proposes ANQ (Adaptive Neighborhood-constrained Q learning), which introduces advantage-function-based adaptive neighborhood constraints for offline RL. ANQ offers a flexible middle ground between density constraints (overly conservative) and support constraints (requiring precise behavior policy modeling), and realizes efficient Q learning via a bilevel optimization framework, achieving state-of-the-art performance on the D4RL benchmark.

## Background & Motivation
**Background**: Offline RL learns policies from static datasets. The central challenge is extrapolation error and Q-value overestimation caused by out-of-distribution (OOD) actions. Existing methods mitigate this by constraining action selection, but each approach has its own limitations.

**Systematic Analysis of Three Constraint Categories**:
- **Density constraints** (BRAC/TD3BC/CQL): Require the learned policy's probability density to be close to the behavior policy. Direct but overly conservative—even when the dataset contains near-optimal behaviors, if the overall quality of the behavior policy is poor, the learned policy remains highly suboptimal. Theoretically, policy performance is bounded by the overall quality of the behavior policy $\eta(\pi_\beta)$.
- **Support constraints** (BCQ/BEAR/SPOT): Only require actions to lie within the support of the behavior policy distribution. Theoretically the most permissive, but require precise modeling of the behavior policy via CVAEs, diffusion models, etc., which is computationally expensive and difficult on high-dimensional, multimodal real-world data.
- **Sample constraints** (IQL/XQL/SQL): Bellman targets use only actions already present in the dataset. Simple to implement but unable to generalize beyond the dataset; overly conservative when near-optimal actions are absent.

**Key Challenge**: Density and sample constraints are too conservative and limit policy improvement; support constraints are the most flexible but incur high modeling costs. A gap between flexibility and implementation complexity remains unfilled.

**Key Insight**: Using the union of neighborhoods around dataset actions as the constraint set allows exploration of better actions near data points (more flexible than sample constraints) without requiring explicit behavior policy modeling (simpler than support constraints), and can theoretically approximate support constraints.

**Core Idea**: Replace behavior policy modeling with adaptive neighborhoods around data points, enabling per-point conservatism adjustment in offline Q learning.

## Method

### Overall Architecture
The core of ANQ is to define an adaptive neighborhood constraint $\mathcal{C}_{AN}(s) = \{\tilde{a} \in \mathcal{A} \mid \|\tilde{a} - a\| \leq \epsilon \exp(-\alpha A(s,a)), (s,a) \in \mathcal{D}\}$, and then perform Q learning under this constraint via bilevel optimization. The inner optimization maximizes the Q function within the adaptive neighborhood of each data point; the outer optimization implicitly takes the maximum over all neighborhoods via expectile regression.

### Key Designs
1. **Neighborhood Constraint**:

    - Definition: The constraint set is the union of $\epsilon$-neighborhoods of all actions in the dataset: $\mathcal{C}_N(s) = \{\tilde{a} \mid \|\tilde{a} - a\| \leq \epsilon, (s,a) \in \mathcal{D}\}$.
    - Theoretical guarantee (Theorem 1): Under standard regularity assumptions, when the sample size $n$ is sufficiently large, the Hausdorff distance between the neighborhood union $U_{n,\epsilon}$ and the behavior policy support set $S$ is $\leq \epsilon$, meaning the neighborhood constraint can approximate the support constraint.
    - Extrapolation control (Lemma 2): Under the NTK regime, the Q-value deviation for actions within the neighborhood satisfies $\|Q(s,\tilde{a}) - Q(s,a)\| \leq C(\sqrt{\min(\|s \oplus a\|, \|s \oplus \tilde{a}\|)}\sqrt{\epsilon} + 2\epsilon)$; a smaller radius yields tighter control.
    - Distribution shift (Proposition 1): The TV distance between the state occupancy distribution under the neighborhood constraint and that under the sample constraint is $\leq \gamma K_P \epsilon / (2(1-\gamma))$.

2. **Adaptive Neighborhood Radius**:

    - Core idea: High-advantage (high-quality) data points use a small radius—they are already near-optimal, requiring less exploration, and a smaller radius reduces extrapolation error; low-advantage (low-quality) data points use a large radius—encouraging broader search for better actions.
    - Radius formula: $r(s,a) = \epsilon \exp(-\alpha A(s,a))$, where $\alpha$ is an inverse temperature parameter.
    - Robustness of advantage estimation: Estimation is performed only within the data distribution (relatively reliable), and is used only to qualitatively distinguish action quality; the exponential form serves as a soft heuristic.

3. **Bilevel Optimization Framework**:

    - **Inner optimization**: An auxiliary policy $\mu_\omega(s,a)$ outputs an action perturbation $\delta$ and maximizes the Q function within the adaptive neighborhood of each data point. The constraint is internalized via a Lagrange multiplier $\lambda$: $\max_{\mu_\omega} \mathbb{E}[Q_\theta(s, a + \mu_\omega(s,a)) - \lambda \exp(\alpha(Q_{\theta'}(s,a) - V_\psi(s)))\|\mu_\omega(s,a)\|]$.
    - **Outer optimization**: Actions refined by the auxiliary policy are sampled, and expectile regression (IQL-style) is used to implicitly take the maximum over neighborhoods: $\min_{V_\psi} \mathbb{E}[L_2^\tau(Q_{\theta'}(s, a + \mu_{\omega'}(s,a)) - V_\psi(s))]$.
    - **Policy extraction**: After the Q function is trained, a policy is extracted from the optimized neighborhood actions via weighted regression.

### Loss & Training
- The Q function is updated with standard Polyak-averaged target networks.
- The auxiliary policy $\mu_\omega$ and its target $\mu_{\omega'}$ are updated synchronously via Polyak averaging.
- Key hyperparameters: expectile $\tau$, inverse temperature $\alpha$, Lagrange multiplier $\lambda$, and neighborhood radius $\epsilon$.

## Key Experimental Results

### Main Results — D4RL Gym Locomotion

| Task | ANQ | IQL | CQL | TD3BC | SPOT | IDQL |
|------|-----|-----|-----|-------|------|------|
| halfcheetah-m | 48.4 | 47.4 | 44.0 | 48.3 | 45.4 | 51.0 |
| hopper-m | 71.7 | 66.3 | 58.5 | 59.3 | 86.7 | 65.7 |
| walker2d-m | 83.7 | 78.3 | 72.5 | 83.7 | 65.0 | 82.5 |
| **Average** | **82.9** | ~78 | ~76 | ~75 | - | - |

### D4RL AntMaze

| Task | ANQ | IQL | CQL | TD3BC |
|------|-----|-----|-----|-------|
| antmaze-large-play | **87.5** | 81.6 | ~70 | ~30 |
| antmaze-umaze | 97.5 | 87.5 | 74.0 | 78.6 |

### Robustness Experiments

| Scenario | ANQ | IQL | CQL | SPOT |
|----------|-----|-----|-----|------|
| 70% expert + 30% noise | **Strong** | Medium | Weak | Weak |
| 70% data dropped | **Strong** | Medium | Medium | Weak |

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|-----------|-------|
| Full ANQ | Best | Adaptive neighborhood + bilevel optimization |
| Fixed radius (no adaptation) | −3–5% | Loss of per-point conservatism |
| $\alpha = 0$ (uniform radius) | Degraded | Equivalent to fixed neighborhood |
| $\alpha$ too large | Higher variance | Instability from excessive radius variation |
| $\alpha = 5$ (optimal) | Best balance | Moderate differentiation of action quality |

### Key Findings
- Neighborhood constraints yield the greatest advantage in low-quality data scenarios—density constraints are dragged down by behavior policy quality, while ANQ effectively exploits sparse high-quality data via adaptive radii.
- The Lagrange multiplier $\lambda$ is critical for controlling overall neighborhood size: too large degrades to sample constraints, too small leads to uncontrolled extrapolation.
- In continuous action spaces, the advantage of neighborhood constraints over sample constraints is more pronounced.

## Highlights & Insights
- **Precise positioning in the constraint spectrum**: Neighborhood constraints fill the gap between density constraints and support constraints on a continuous spectrum, providing a clear conceptual contribution.
- **Per-point conservatism**: Different degrees of conservatism are applied to different data points within the same policy, which is more fine-grained than globally conservative (CQL) or globally permissive (BEAR) approaches.
- **Theory–practice consistency**: Theorem 1 (support approximation), Lemma 2 (extrapolation control), and Proposition 1 (distribution shift) together form a complete theoretical framework, with consistent empirical validation.
- **Implementation simplicity**: The auxiliary policy $\mu_\omega$ is a standard MLP; no generative model or diffusion model training is required, making the approach simpler than SPOT/IDQL.

## Limitations & Future Work
- **Neighborhood shape assumption**: The method is fixed to spherical neighborhoods (L2 norm), which may be suboptimal for anisotropic action spaces; ellipsoidal or learned metrics could be more appropriate.
- **Advantage function estimation**: Although the authors argue for robustness, estimation of $A(s,a)$ on highly suboptimal offline data may still be biased.
- **Continuous action space restriction**: The method is designed for continuous action spaces; application to discrete action spaces requires redefining the neighborhood concept.
- **Lack of large-scale validation**: Evaluation is limited to the standard D4RL benchmark, without experiments on real robot platforms or high-dimensional tasks.

## Related Work & Insights
- **vs. IQL**: IQL employs sample constraints (expectile regression); ANQ extends IQL by introducing neighborhood expansion and an auxiliary policy, retaining the expectile outer loop while adding an inner optimization step.
- **vs. SPOT**: SPOT requires training a CVAE to model the behavior policy support set; ANQ approximates the support set via neighborhoods without explicit modeling, making it simpler and more robust.
- **vs. CQL**: CQL implicitly imposes density constraints (by reducing Q values of OOD actions); ANQ explicitly constrains the action range, providing stronger theoretical guarantees.

## Rating
- Novelty: ⭐⭐⭐⭐ Neighborhood constraints fill the gap between density and support constraints; the adaptive mechanism is well motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Standard D4RL validation + robustness analysis + complete ablation study.
- Writing Quality: ⭐⭐⭐⭐⭐ Constraint taxonomy is clearly organized, theoretical derivations are rigorous, and the exposition is logically coherent.
- Value: ⭐⭐⭐⭐ Provides a new perspective on constraint design for offline RL; the method is concise and practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Automaton Constrained Q-Learning](automaton_constrained_q-learning.md)
- [\[NeurIPS 2025\] Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents](risk-averse_constrained_reinforcement_learning_with_optimized_certainty_equivale.md)
- [\[NeurIPS 2025\] Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning](sample-efficient_tabular_self-play_for_offline_robust_reinforcement_learning.md)
- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[NeurIPS 2025\] Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning](structural_information-based_hierarchical_diffusion_for_offline_reinforcement_le.md)

</div>

<!-- RELATED:END -->
