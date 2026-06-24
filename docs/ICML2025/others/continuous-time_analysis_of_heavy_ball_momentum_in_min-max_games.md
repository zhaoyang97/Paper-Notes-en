---
title: >-
  [Paper Note] Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games
description: >-
  [ICML2025][heavy ball momentum] Through continuous-time ODE modeling, this work systematically reveals that heavy ball momentum behaves completely differently in min-max games compared to minimization problems: **smaller momentum** (including negative momentum) expands the stable stepsize range and guides trajectories toward flatter gradient regions, while **alternating updates** converge faster than simultaneous updates and amplify this regularization effect.
tags:
  - "ICML2025"
  - "heavy ball momentum"
  - "min-max games"
  - "continuous-time analysis"
  - "implicit gradient regularization"
  - "alternating updates"
  - "negative momentum"
  - "local convergence"
date: 2026-05-08
content_hash: a0c3e7c44b7834b9
---

# Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games

**Conference**: ICML2025  
**arXiv**: [2505.19537](https://arxiv.org/abs/2505.19537)  
**Code**: None  
**Area**: Others/Optimization Theory  
**Keywords**: heavy ball momentum, min-max games, continuous-time analysis, implicit gradient regularization, alternating updates, negative momentum, local convergence

## TL;DR

Through continuous-time ODE modeling, this work systematically reveals that heavy ball momentum behaves completely differently in min-max games compared to minimization problems: **smaller momentum** (including negative momentum) expands the stable stepsize range and guides trajectories toward flatter gradient regions, while **alternating updates** converge faster than simultaneous updates and amplify this regularization effect.

## Background & Motivation

**Background**: Heavy Ball (HB) momentum has been widely studied in minimization optimization since Polyak (1964), with its mechanism typically explained by continuous-time differential equations describing the physical process of mass moving in a frictional medium. HB momentum is also a key component of practical min-max algorithms like Adam, heavily used in GAN training and adversarial training.

**Limitations of Prior Work**:

**Theoretical Gap**: The role of HB momentum in min-max games remains largely unexplored. Findings from minimization (e.g., larger momentum is beneficial) cannot be directly applied—Gidel et al. (2019) found that **negative momentum** instead improves min-max training, which is counterintuitive.

**Unclear Impact of Update Rules**: Simultaneous updates (Sim-HB) and alternating updates (Alt-HB) have fundamentally different impacts on learning dynamics, but quantitative analysis is lacking. Gidel et al. (2019) demonstrated convergence for alternating + negative momentum only under the special case of bilinear games.

**Insufficient Understanding of Global Behavior**: While global properties of HB (such as implicit regularization) have been studied in minimization, they are entirely missing in min-max settings.

**Core Problem**: How do momentum parameter $\beta$ and update rules (simultaneous vs. alternating) affect the learning dynamics of min-max games?

## Method

### Overall Architecture

The authors adopt a **continuous-time analysis** framework, modeling the discrete HB algorithm as an ODE to conduct analysis from the perspectives of local convergence and implicit gradient regularization.

### 1. Derivation of Continuous-Time Models

For the min-max game $\min_x \max_y f(x,y)$, two HB algorithms are considered:

- **Simultaneous HB (Sim-HB)**: both players update their strategies simultaneously.
- **Alternating HB (Alt-HB)**: the x-player updates first, and the y-player updates based on the updated $x_{n+1}$.

Define the modified loss function:

$$\mathcal{F}(x,y) = \frac{1}{1-\beta} f(x,y) + \frac{h(1+\beta)}{4(1-\beta)^3}\left(\|\nabla_x f\|^2 - \|\nabla_y f\|^2\right)$$

Corresponding continuous-time models:

- **Continuous Sim-HB**: $\dot{x} = -\nabla_x \mathcal{F}$, $\dot{y} = \nabla_y \mathcal{F}$
- **Continuous Alt-HB**: the x equation is the same as above, while the y equation contains an additional $-\frac{h}{2(1-\beta)^2}\|\nabla_x f\|^2$ term.

**Key Designs**: Unlike the second-order ODE modeling commonly used in minimization, this work adopts a **first-order ODE** because: (a) second-order methods couple stepsize and momentum parameters, impeding independent analysis; (b) second-order methods require $\beta > 0$ to conform to physical intuition, preventing the exploration of negative momentum.

**Approximating Accuracy (Theorem 3.1)**: Within a finite time, the local error between the continuous-time trajectory and the discrete algorithm trajectory is $\mathcal{O}(h^3)$, which improves upon the $\mathcal{O}(h^2)$ model by Bailey et al. (2020)—the latter fails to correctly predict the divergent behavior of simultaneous GDA in bilinear games.

### 2. Local Convergence Analysis

The core tool is Jacobian matrix analysis. The Jacobian of the gradient flow can be decomposed as:

$$\mathcal{J} = \mathcal{S} + \mathcal{A}$$

where $\mathcal{S}$ (potential part) is a block-diagonal matrix, and $\mathcal{A}$ (Hamiltonian part) is an anti-symmetric matrix capturing player interactions.

**Core Assumption (Assumption 4.4)**: $\mathcal{A} \gg \mathcal{S}$, meaning player interactions dominate the game dynamics. In this case, the Jacobian eigenvalues satisfy $|\text{Im}(\lambda)| > |\text{Re}(\lambda)|$.

**Key Theorems**:

- **Theorem 4.6 (Sim-HB Convergence Condition)**: Continuous Sim-HB achieves local convergence when the stepsize satisfies $h < \min_\lambda \frac{2(1-\beta)^2}{(1+\beta)} \frac{|\text{Re}(\lambda)|}{\text{Im}(\lambda)^2 - \text{Re}(\lambda)^2}$.
- **Corollary 4.7 (Negative Momentum Expands Stability)**: Since $\frac{2(1-\beta)^2}{1+\beta}$ decreases with respect to $\beta$, **smaller $\beta$ (including negative values) allows a larger range of stepsizes**. This contradicts the conclusion in minimization where larger momentum expands the stepsize range.
- **Theorem 4.8 (Optimality of Positive Momentum)**: When the stepsize is sufficiently small, the momentum $\tilde{\beta}(h)$ corresponding to the optimal convergence rate is **positive**.
- **Theorem 4.9 (Alternating Update Acceleration)**: The maximum real part of the Jacobian eigenvalues for Alt-HB satisfies $\text{Re}(\lambda'_A) = \text{Re}(\lambda_S) - \frac{h}{(1-\beta)^2}|\sigma|^2 + \mathcal{O}(h^2 + h\alpha)$, which contains an extra negative term $-\frac{h}{(1-\beta)^2}|\sigma|^2$ compared to Sim-HB, achieving **exponentially accelerated local convergence**.

### 3. Implicit Gradient Regularization Analysis

Analyze two competing forces in the evolution of $\|\nabla_x f\|^2$ under Continuous Sim-HB:

- **Gradient Descent Force** (Eq. 5): coefficient $-\frac{h(1+\beta)}{4(1-\beta)^3}$, which tends to decrease the gradient norm.
- **Gradient Ascent Force** (Eq. 6): coefficient $+\frac{h(1+\beta)}{4(1-\beta)^3}$, which tends to increase the gradient norm.

Since $\frac{h(1+\beta)}{4(1-\beta)^3}$ increases with respect to $\beta$, smaller $\beta$ weakens both forces. Under the condition $\mathcal{A} \gg \mathcal{S}$ (interaction-dominated), the weakening of the ascent force outweighs that of the descent force, and thus **smaller momentum guides the trajectory toward flatter gradient regions**.

For Alt-HB, the coefficient of $\nabla_y\|\nabla_x f\|^2$ shifts from $\frac{h(1+\beta)}{4(1-\beta)^3}$ (always positive) to $\frac{h(3\beta-1)}{4(1-\beta)^3}$ (negative when $\beta < 1/3$), meaning the y-player changes from maximizing the gradient to **minimizing the gradient**, further clarifying the regularization effect.

## Key Experimental Results

### 2D Toy Function Experiments

| Test Function | Dynamic Behavior | Observed Conclusion |
|----------|----------|----------|
| $f(x,y) = -xy^2, x \geq 0$ | Converges to equilibrium | Smaller $\beta$ → Trajectory tends to equilibrium points with smaller gradients |
| $f = 3x(4y-0.45)+g(x)-g(y)$ (sextic polynomial) | Converges to limit cycle | Smaller $\beta$ → Limit cycle with lower average slope |

Both test functions verify: (1) smaller momentum → flatter gradient regions; (2) alternating updates < simultaneous updates in terms of average slope.

### GAN Training Experiments

- **Setup**: WGAN-GP, CIFAR-10, ResNet-32 architecture, learning rate 2e-4 with linear decay, batch size 64
- **Metric**: Cumulative average slope $\text{AvgS}_\beta(t) = \frac{1}{t}\sum_{s=1}^t (\|\nabla_x f\|^2 + \|\nabla_y f\|^2)$

| Experiment | Observation |
|------|------|
| Different $\beta$ (Alternating) | Smaller momentum → Lower average slope |
| Alternating vs. Simultaneous (Fixed $\beta$) | Alternating updates yield lower average slope |
| Slope vs. FID Relation | Lower average slope → Lower FID (better generation quality) |

### Consistency Verification

- When $\beta = 0$, the model degenerates into the continuous-time GDA equation from Rosca et al. (2021) (consistency).
- In bilinear games: Sim-HB diverges + Alt-HB converges with negative momentum → Consistent with discrete conclusions from Gidel et al. (2019) (Proposition 3.2).

## Highlights & Insights

1. **Core Comparative Finding**: The role of momentum in min-max games is **completely opposite** to that in minimization—smaller/negative momentum is advantageous, rather than larger/positive momentum. This finding has significant practical guidance.
2. **First-order ODE Modeling Innovation**: Bypasses the limitations of traditional second-order ODEs, enabling independent analysis of stepsize and momentum, and supporting the study of negative momentum.
3. **$\mathcal{O}(h^3)$ Accuracy**: More accurate than existing $\mathcal{O}(h^2)$ models, successfully predicting the divergent behavior of Sim-GDA in bilinear games.
4. **Theoretical Explanation of Practical Phenomena**: For the first time, explains from the perspective of implicit regularization why negative momentum improves GAN training stability—by guiding trajectories away from sharp gradient regions.
5. **Quantitative Advantage of Alternating Updates**: Theorem 4.9 precisely characterizes the difference in convergence rates between Alt-HB and Sim-HB, yielding an acceleration of $\frac{h|\sigma|^2}{(1-\beta)^2}$.

## Limitations & Future Work

1. **Stochastic Gradient Not Considered**: All analyses are based on deterministic gradients, neglecting stochastic settings like SGD/Adam, whereas mini-batches are almost exclusively used in practice.
2. **Strong Interaction Assumption**: Core conclusions rely on Assumption 4.4 ($\mathcal{A} \gg \mathcal{S}$), which might not hold for weak interaction games.
3. **Conditionality of Alt-HB Acceleration**: Theorem 4.9 requires sufficiently small $h$ and $\alpha$. Counterexamples in Appendix I.4 show that alternating updates are not necessarily superior to simultaneous ones when this condition is violated.
4. **Dimension Restriction**: Theorem 4.9 requires $m = n$ (both players have the same dimension), limiting generality.
5. **Connection between Implicit Regularization and Generalization**: GAN experiments observe lower slope → lower FID, but rigid theoretical support is lacking, leaving it as an empirical observation.
6. **Limitations of Perturbation Theory**: The coefficient of the $\mathcal{O}(h\alpha)$ term in Theorem 4.9 requires computing perturbed normalized eigenvectors, which the authors leave for future work.

## Related Work & Insights

- **Gidel et al. (2019)**: First to empirically discover the benefits of negative momentum in min-max, which this work supports with systematic theory.
- **Ghosh et al. (2023)**: Analyzed implicit HB regularization in minimization (larger momentum → flatter slope); this paper proves that the opposite holds in min-max.
- **Barrett & Dherin (2021)**: Source of the implicit gradient regularization concept.
- **Rosca et al. (2021)**: Continuous-time equation for GDA (special case of this work with $\beta=0$).
- **Wang & Chizat (2024)**: Source of Assumption 4.5 for local convergence of gradient flows.

**Insights**: (1) Stochastic noise can be introduced into this framework to derive SDE models; (2) The connection between implicit regularization effects and GAN training quality deserves further theorization; (3) The combination of negative momentum + alternating updates can serve as a practical guide for min-max training.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic analysis of the continuous-time behavior of HB momentum in min-max, revealing fundamental differences from minimization.
- Theoretical Depth: ⭐⭐⭐⭐⭐ — Multiple theorems are closely linked, forming a complete theoretical system from Jacobian analysis to implicit regularization.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 2D functions and GAN training cover various dynamic modes, though large-scale experiments are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with a complete logical chain from motivation to theory and experiments.
- Value: ⭐⭐⭐⭐ — Provides actionable theoretical guidance for selecting momentum and update rules in min-max training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Semi-infinite Nonconvex Constrained Min-Max Optimization](../../NeurIPS2025/others/semi-infinite_nonconvex_constrained_min-max_optimization.md)
- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](../../NeurIPS2025/others/deep_learning_for_continuous-time_stochastic_control_with_jumps.md)
- [\[AAAI 2026\] Bayesian Network Structural Consensus via Greedy Min-Cut Analysis](../../AAAI2026/others/bayesian_network_structural_consensus_via_greedy_min-cut_analysis.md)
- [\[ICML 2025\] Runtime Analysis of Evolutionary NAS for Multiclass Classification](runtime_analysis_of_evolutionary_nas_for_multiclass_classification.md)
- [\[NeurIPS 2025\] Deep Continuous-Time State-Space Models for Marked Event Sequences](../../NeurIPS2025/others/deep_continuous-time_state-space_models_for_marked_event_sequences.md)

</div>

<!-- RELATED:END -->
