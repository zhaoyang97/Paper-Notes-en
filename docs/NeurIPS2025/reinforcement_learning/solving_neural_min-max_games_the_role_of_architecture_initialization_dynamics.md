---
title: >-
  [Paper Note] Solving Neural Min-Max Games: The Role of Architecture, Initialization & Dynamics
description: >-
  [NeurIPS 2025][Reinforcement Learning][Neural network games] This paper provides the first convergence guarantees for zero-sum games parameterized by two-layer neural networks, proving that under sufficient overparameterization, random Gaussian initialization, and alternating gradient descent-ascent (AltGDA), the dynamics converge to an $\epsilon$-approximate Nash equilibrium with high probability.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Neural network games"
  - "min-max optimization"
  - "overparameterization"
  - "hidden convexity"
  - "AltGDA"
date: 2026-05-08
content_hash: 8dceb0dd670967f6
---

# Solving Neural Min-Max Games: The Role of Architecture, Initialization & Dynamics

**Conference**: NeurIPS 2025
**arXiv**: [2512.00389](https://arxiv.org/abs/2512.00389)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Neural network games, min-max optimization, overparameterization, hidden convexity, AltGDA

## TL;DR

This paper provides the first convergence guarantees for zero-sum games parameterized by two-layer neural networks, proving that under sufficient overparameterization, random Gaussian initialization, and alternating gradient descent-ascent (AltGDA), the dynamics converge to an $\epsilon$-approximate Nash equilibrium with high probability.

## Background & Motivation

**Background**: The intersection of deep learning and game theory has given rise to numerous applications—GANs, robust RL, adversarial attacks, AI safety debates, etc. These systems are fundamentally zero-sum games parameterized by neural networks, with the goal of finding von Neumann minimax points or Nash equilibria.

**Limitations of Prior Work**: (i) Once non-convex deep learning architectures are introduced, classical game-theoretic guarantees on existence and efficient computation no longer hold; (ii) even when equilibria exist, gradient methods may exhibit instability, cycling, or divergence; (iii) the "hidden convexity" paradigm relies on a global uniform well-conditioning assumption on the Jacobian, which frequently fails in practice due to rank collapse.

**Key Challenge**: The gap between the hidden convex-concave structure of the latent game and the non-convexity of the parameter space—the minimum singular value of the Jacobian may approach zero, causing the PL condition to degenerate and theoretical guarantees to break down.

**Goal**: To provide explicit, verifiable conditions—on architecture design, initialization, and training dynamics—that guarantee efficient convergence in large-scale neural min-max games.

**Key Insight**: Combining overparameterization theory with hidden-convex game theory to prove (a) that the AltGDA trajectory length is bounded, and (b) that under overparameterization, the singular values of the Jacobian remain well-conditioned with high probability.

**Core Idea**: Sufficiently wide two-layer networks + Gaussian random initialization + AltGDA training dynamics = convergence to an approximate Nash equilibrium with high probability.

## Method

### Overall Architecture

The paper considers hidden convex-concave zero-sum games of the form:

$$\min_{\theta} \max_{\phi} \mathcal{L}_{\mathcal{D}}(F_\theta, G_\phi) = I_1^{\mathcal{D}_F}(F_\theta) + I_2^{\mathcal{D}}(F_\theta, G_\phi) - I_3^{\mathcal{D}_G}(G_\phi)$$

where $F_\theta, G_\phi$ are two-layer neural networks, $I_1, I_3$ are strongly convex, and $I_2$ is bilinear. The latent game is convex-concave, while the end-to-end objective is non-convex.

### Key Designs

**Module 1: AltGDA Trajectory Length Control (Lemma 3.3)**
- **Function**: Proves that AltGDA iterates remain within a bounded region near initialization.
- **Core Tool**: A Lyapunov potential $P_t = (\max_\phi \mathcal{L}(\theta_t, \phi) - \mathcal{L}(\theta^*, \phi^*)) + \lambda(\max_\phi \mathcal{L}(\theta, \phi_t) - \mathcal{L}(\theta_t, \phi_t))$.
- **Path Length Bound**: $\ell(T) \leq \frac{\sqrt{2\alpha_1}}{1-\sqrt{c}} \cdot \sqrt{P_0}$.
- **Design Motivation**: In minimization, path length bounds follow directly from unrolling GD iterates; the alternating structure of min-max makes this analysis substantially harder, necessitating the potential function approach.

**Module 2: Upper Bound on Initial Potential $P_0$ (Lemma 3.3)**
- **Function**: Expresses $P_0$ as a function of gradient norms at initialization.
- **Core Formula**: $P_0 \leq L_{\mathcal{L}}(C_1 \|\nabla_\theta \mathcal{L}\| + C_2 \|\nabla_\phi \mathcal{L}\|)$, where $C_1, C_2 = \Theta(L_{\mathcal{L}}/\mu_\theta^3)$.
- **Design Motivation**: Controlling $P_0 \leq \kappa R^2$ is necessary to ensure iterates do not leave the well-conditioned region.

**Module 3: Jacobian Spectral Analysis for Input Optimization Games (Lemma 3.4)**
- **Function**: For a fixed randomly initialized network, establishes bounds on the singular values of the Jacobian with respect to inputs.
- **Core Result**: Using GeLU activations, when $d_1^{(F)} \geq 256\max\{d_0^{(F)}, d_2^{(F)}\}$, with probability $\geq 1 - e^{-\Omega(d_1)}$:
    - $\sigma_{\min}(\nabla_\theta F_\theta) = \Omega(\sigma_{1,F} \cdot \sigma_{2,F} \cdot d_1)$
    - $\sigma_{\max}(\nabla_\theta F_\theta) = \mathcal{O}(\sigma_{1,F} \cdot \sigma_{2,F} \cdot d_1)$

**Module 4: Overparameterization Condition for Network Parameter Games (Theorem 3.8)**
- **Function**: Specifies the minimum network width required to guarantee convergence.
- **Core Condition**: $d_1^{(F)} = \widetilde{\Omega}\left(\mu_\theta^2 \frac{n^3}{d_0^{(F)}}\right)$, $d_1^{(G)} = \widetilde{\Omega}\left(\mu_\phi^2 \frac{n^3}{d_0^{(G)}}\right)$.
- **Design Motivation**: The required width scales as $\Omega(n^3)$ rather than $\Omega(n)$ as in minimization—this is a fundamental cost of the min-max setting.

### Loss & Training

**Alternating Gradient Descent-Ascent (AltGDA)**:
$$\theta^{(t)} = \theta^{(t-1)} - \eta_\theta \nabla_\theta \mathcal{L}(\theta^{(t-1)}, \phi^{(t-1)})$$
$$\phi^{(t)} = \phi^{(t-1)} + \eta_\phi \nabla_\phi \mathcal{L}(\theta^{(t)}, \phi^{(t-1)})$$

Step sizes: $\eta_\theta = \frac{\mu_\phi^2}{18L_{\nabla\mathcal{L}}^3}$, $\eta_\phi = \frac{1}{L_{\nabla\mathcal{L}}}$.

## Key Experimental Results

### Main Results

| Setting | Result |
|---------|--------|
| Input optimization game (Theorem 3.5) | AltGDA converges to $\epsilon$-NE with iteration complexity $\tilde{O}\!\left(\frac{1}{\epsilon}\log\frac{1}{\epsilon}\right)$, w.h.p. |
| Network parameter game (Theorem 3.8) | With width $\Omega(n^3/d_{\text{input}})$, AltGDA converges exponentially fast to a saddle point, w.h.p. |

### Ablation Study / Comparison

| Dimension | Ours vs. Practice |
|-----------|-------------------|
| Network architecture | Single hidden layer, fully connected vs. deep networks |
| Training algorithm | AltGDA vs. double-loop / other methods |
| Initialization | Gaussian (with variance constraints) vs. He/Xavier |
| Overparameterization scaling | $\Omega(n^3)$ vs. $\Omega(n)$ (minimization) |

### Key Findings

1. The overparameterization requirement of $\Omega(n^3)$ in min-max settings is fundamentally higher than the $\Omega(n)$ required for pure minimization.
2. Alternation is critical for min-max optimization: simultaneous GDA may diverge even under hidden convex-concavity and two-sided PL conditions.
3. $\sigma_{\min}$ governs the model's ability to explore the strategy space—when $\sigma_{\min} \approx 0$, certain strategy directions are left unexplored.

## Highlights & Insights

1. **First open-box result**: Rather than relying on abstract assumptions about the latent mapping, the paper provides explicit conditions on architecture, initialization, and training dynamics to guarantee convergence.
2. **From minimization to games**: The extension of overparameterization and NTK theory from minimization to min-max games is significant, revealing the trajectory–spectrum coupling as the central technical challenge.
3. **Key distinction**: In games, networks output high-dimensional vectors (action distributions), making spectral analysis considerably more subtle than in the scalar-label classification setting.
4. **Lyapunov potential circumvents the difficulty of directly unrolling AltGDA iterates**, providing a template for trajectory analysis in general non-convex min-max problems.

## Limitations & Future Work

1. The analysis is restricted to single-hidden-layer fully connected networks; extension to deep networks remains an important open problem.
2. The activation function must be twice differentiable (ruling out ReLU), though GeLU and softplus are compatible.
3. The $\Omega(n^3)$ width requirement may limit practical applicability; whether this can be reduced to $\Omega(n)$ is worth investigating.
4. No numerical experiments are provided (the work is purely theoretical); validation in practical settings such as GANs is a natural next step.
5. The strongly convex regularization terms are central to the theory; extension to the unregularized case remains unclear.

## Related Work & Insights

- **Comparison with the hidden convexity theory of Vlatakis-Gkaragkounis et al. (2019, 2021)**: The latter requires a global assumption of uniform Jacobian well-conditioning; this paper establishes that condition with high probability via overparameterization.
- **Comparison with Yang et al. (2020)**: That work establishes AltGDA convergence under two-sided PL conditions; this paper's contribution is proving that overparameterized networks satisfy two-sided PL.
- **Comparison with Song et al. (2021)**: That work develops spectral analysis for minimization; this paper extends it to the game setting and handles vector-valued outputs.
- **Insight**: Overparameterization benefits not only optimization (minimization) but also equilibrium computation (games).

## Rating

⭐⭐⭐⭐ (4/5)

The theoretical contribution is important and novel—this is the first work to provide quantitative convergence guarantees for min-max games parameterized by neural networks. Extending overparameterization theory from minimization to game theory is technically significant. Limitations include the restriction to shallow networks, the absence of empirical validation, and the relatively strong $\Omega(n^3)$ width requirement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach](multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)
- [\[NeurIPS 2025\] Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics](solving_continuous_mean_field_games_deep_reinforcement_learning_for_non-stationa.md)
- [\[ICML 2025\] Solving Zero-Sum Convex Markov Games](../../ICML2025/reinforcement_learning/solving_zero-sum_convex_markov_games.md)
- [\[NeurIPS 2025\] Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation](decoderhybriddecoder_architecture_for_efficient_reasoning_wi.md)
- [\[NeurIPS 2025\] Thompson Sampling in Function Spaces via Neural Operators](thompson_sampling_in_function_spaces_via_neural_operators.md)

</div>

<!-- RELATED:END -->
