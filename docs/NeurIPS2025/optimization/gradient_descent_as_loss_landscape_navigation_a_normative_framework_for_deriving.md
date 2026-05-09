---
title: >-
  [Paper Note] Gradient Descent as Loss Landscape Navigation: a Normative Framework for Deriving Learning Rules
description: >-
  [NeurIPS 2025][Optimization][learning rules] This paper proposes treating learning rules as optimal navigation policies in a (partially observable) loss landscape. By solving a continuous-time optimal control problem via variational calculus, it derives gradient descent, momentum, natural gradient, Adam, and continual learning strategies within a unified framework.
tags:
  - NeurIPS 2025
  - Optimization
  - learning rules
  - optimal control
  - loss landscape navigation
  - momentum
  - adaptive optimizers
date: 2026-05-08
content_hash: 8431fd01378cd042
---

# Gradient Descent as Loss Landscape Navigation: a Normative Framework for Deriving Learning Rules

**Conference**: NeurIPS 2025
**arXiv**: [2510.26997](https://arxiv.org/abs/2510.26997)
**Code**: None
**Area**: Optimization Theory / Learning Rules
**Keywords**: learning rules, optimal control, loss landscape navigation, momentum, adaptive optimizers

## TL;DR

This paper proposes treating learning rules as optimal navigation policies in a (partially observable) loss landscape. By solving a continuous-time optimal control problem via variational calculus, it derives gradient descent, momentum, natural gradient, Adam, and continual learning strategies within a unified framework.

## Background & Motivation

- Learning rules (e.g., gradient descent, Adam) are typically assumed as given, lacking a unified theoretical foundation derived from first principles.
- Different optimizers (momentum, adaptive learning rates, natural gradient, etc.) have developed independently, with their mutual relationships remaining unclear.
- Core problem: Why do certain learning rules outperform others? Under what assumptions is a given rule "optimal"?
- This paper proposes a normative framework that reformulates these questions as optimal control problems, naturally deriving distinct optimizers by varying the underlying assumptions.

## Method

### Overall Architecture

The learning process is modeled as a continuous-time optimal navigation problem in the loss landscape. A functional objective is defined as:

$$J(\{\boldsymbol{\theta}_t\}) = \mathbb{E}\left\{\int_0^\infty \left(\frac{1}{2\eta}[\dot{\boldsymbol{\theta}}_t - \boldsymbol{f}(\boldsymbol{\theta}_t)]^T \boldsymbol{G}(\boldsymbol{\theta}_t)[\dot{\boldsymbol{\theta}}_t - \boldsymbol{f}(\boldsymbol{\theta}_t)] + k\hat{\mathcal{L}}_t(\boldsymbol{\theta}_t)\right) e^{-\gamma t} dt\right\}$$

This objective comprises three core components: a cost on parameter change (kinetic term), the loss function (potential term), and a temporal discount factor $\gamma$. Three key insights form the foundation of the framework:

1. **Multi-step optimization**: Optimization considers not just the next step, but plans the entire future parameter trajectory.
2. **Parameter space geometry**: Non-Euclidean geometry of the parameter space is encoded via the metric $\boldsymbol{G}$.
3. **Partial observability**: The loss landscape is only partially observable, requiring Bayesian inference for estimation.

### Key Designs

1. **Momentum emerges naturally from multi-step optimization**:
   - Function: Derives Euler-Lagrange equations from the simplest multi-step objective.
   - Mechanism: The optimal trajectory satisfies $\dot{\boldsymbol{\theta}}_t = \boldsymbol{p}_t$, $\dot{\boldsymbol{p}}_t = \gamma \boldsymbol{p}_t + \eta k \nabla \mathcal{L}$, where $\boldsymbol{p}_t$ is the momentum.
   - Design Motivation: Momentum requires no additional assumptions—it arises solely from "planning multiple steps ahead"; the temporal discount $\gamma$ controls interpolation between momentum and standard gradient descent.

2. **Natural gradient arises from parameter space geometry**:
   - Function: Introduces a non-Euclidean metric $\boldsymbol{G}(\boldsymbol{\theta})$ (e.g., the Fisher information matrix) into the objective.
   - Mechanism: The metric $\boldsymbol{G}$ and the Hessian $\boldsymbol{H}$ play fundamentally different roles—$\boldsymbol{G}$ defines the geometry of the environment, while $\boldsymbol{H}$ describes loss curvature.
   - Design Motivation: Clarifies the long-standing debate over whether natural gradient is "a second-order method in disguise."

3. **Adam derives from Bayesian inference over the loss landscape shape**:
   - Function: Assumes the learner maintains Bayesian beliefs over the local loss landscape gradient $\boldsymbol{m}_t$ and curvature $\boldsymbol{V}_t$.
   - Mechanism: Temporal evolution of landscape beliefs is modeled via an Ornstein-Uhlenbeck prior; the derived optimal update takes the form $\Delta\boldsymbol{\theta} \propto \boldsymbol{V}_t^{-1/2} \boldsymbol{m}_t$.
   - Design Motivation: Provides a theoretical justification for the square-root normalization in Adam—it is the optimal result under the ballistic regime (i.e., long-horizon planning).

### Loss & Training

- Optimal trajectories are obtained by solving the Euler-Lagrange equations; different assumptions yield different EL equations.
- Three limiting cases correspond to three learning rules: large $\Delta t$ → Newton's method; small $\Delta t$ + large $\gamma$ → gradient descent; small $\Delta t$ + $\gamma=0$ → ballistic rule.
- For continual learning: distributional estimation of weights (mean $\mu_i$ + variance $v_i$) yields variance-sensitive learning dynamics, providing theoretical justification for weight-reset strategies.

## Key Experimental Results

### Main Results

| Method | MNIST (MLP) Accuracy | CIFAR-10 (CNN) Accuracy |
|--------|----------------------|-------------------------|
| SGD | ~97% | ~68% |
| Adam | ~98% | ~73% |
| Ballistic (Ours) | ~97.5% | ~71% |

### Ablation Study

- Effect of temporal discount rate $\gamma$: $\gamma \to 0$ (ballistic limit) yields faster convergence but longer trajectories; $\gamma \gg 1$ (overdamped limit) reduces to standard gradient descent.
- Convergence rate ratio under anisotropic loss: a 4× curvature difference leads to a 4× convergence speed difference under gradient descent, but only $\sqrt{4}=2\times$ under the ballistic rule.
- In a double-well loss landscape: multi-step optimization finds the global minimum, whereas gradient descent becomes trapped in a local minimum.

### Key Findings

- The ballistic learning rule ($\gamma \approx 0$) generally outperforms SGD, but does not necessarily outperform Adam.
- Physical analogy: the learning process corresponds to a particle of mass $1/\eta$ moving in a potential field $k\mathcal{L}$, with $\gamma$ as the friction coefficient.
- A drift term $\boldsymbol{f}$ can render non-gradient learning rules optimal—providing theoretical support for non-gradient synaptic plasticity rules observed in biological neural networks.

## Highlights & Insights

- **Grand unified perspective**: For the first time, gradient descent, momentum, natural gradient, Adam, and continual learning strategies are all derived from a single objective function.
- **Physics–AI bridge**: The learning dynamics correspond precisely to classical mechanics, enabling the application of tools such as Noether's theorem to the analysis of learning algorithms.
- The square root in Adam is a feature, not a bug—it is the optimal result under the ballistic (long-horizon planning) assumption.
- The essential distinction between natural gradient and second-order methods is formally clarified.

## Limitations & Future Work

- The notion of "optimality" in the framework does not account for practical constraints such as memory requirements and computational efficiency.
- Optimal rules may require expensive matrix operations (e.g., Hessian computation), limiting practical applicability.
- The ballistic rule is validated only on small-scale datasets (MNIST/CIFAR-10); large-scale experiments are absent.
- The connection between the theoretical framework and specific empirical settings requires further investigation.

## Related Work & Insights

- The framework shares a similar spirit with the variational framework for accelerated optimization by Wibisono et al., but a key distinction lies in the sign of the objective (summation rather than subtraction), making optimal trajectories minima rather than stationary points.
- The unified natural gradient framework of Khan & Rue does not incorporate multi-step planning or partial observability.
- The framework provides a normative explanation for non-gradient rules such as Hebbian learning and STDP in biological neural networks.

## Rating

- Theoretical Innovation: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Value: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Overall: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[ICLR 2026\] Rolling Ball Optimizer: Learning by Ironing Out Loss Landscape Wrinkles](../../ICLR2026/optimization/rolling_ball_optimizer_learning_by_ironing_out_loss_landscape_wrinkles.md)
- [\[NeurIPS 2025\] Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression](large_stepsizes_accelerate_gradient_descent_for_regularized_logistic_regression.md)
- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)

</div>

<!-- RELATED:END -->
