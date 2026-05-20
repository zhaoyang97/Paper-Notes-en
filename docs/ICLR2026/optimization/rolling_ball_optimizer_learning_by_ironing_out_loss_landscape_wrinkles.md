---
title: >-
  [Paper Note] Rolling Ball Optimizer: Learning by Ironing Out Loss Landscape Wrinkles
description: >-
  [ICLR 2026][Optimization][optimizer] This paper proposes the Rolling Ball Optimizer (RBO), which breaks the spatial locality of conventional optimizers by simulating the rolling motion of a finite-radius rigid sphere ove…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "optimizer"
  - "loss landscape"
  - "rolling ball"
  - "smoothing effect"
  - "generalization"
date: 2026-05-08
content_hash: 0610fd7fd6317915
---

# Rolling Ball Optimizer: Learning by Ironing Out Loss Landscape Wrinkles

**Conference**: ICLR 2026
**arXiv**: [2505.19527](https://arxiv.org/abs/2505.19527)  
**Area**: Optimization
**Keywords**: optimizer, loss landscape, rolling ball, smoothing effect, generalization

## TL;DR

This paper proposes the Rolling Ball Optimizer (RBO), which breaks the spatial locality of conventional optimizers by simulating the rolling motion of a finite-radius rigid sphere over the loss landscape. This induces an ironing property on the loss function and demonstrates superior convergence speed and generalization performance on MNIST and CIFAR-10/100.

## Background & Motivation

Training deep learning models fundamentally requires minimizing high-dimensional, data-dependent loss functions whose optimization landscapes are typically highly complex:

- Numerous spurious local minima (some sharply curved)
- Ill-conditioned valleys and saddle points
- Potentially fractal structure
- Noise in training data propagates into the fine-grained geometry of the landscape

**Fundamental limitation of existing optimizers — their "point-mass" nature**:

All mainstream optimizers (SGD, Adam, HeavyBall, NAG, etc.) operate as point-mass dynamics on the loss landscape, relying solely on local information (gradients) at the current position. This **spatial locality** leads to:

1. **Sensitivity to microscopic structure**: any arbitrarily small perturbation to the loss function (including data-noise-induced perturbations) affects the update
2. **Insensitivity to macroscopic structure**: inability to capture global geometric features of the landscape
3. Susceptibility to sharp minima, ill-conditioned valleys, and saddle points

Although SAM and Entropy-SGD abandon spatial locality, they focus solely on avoiding sharp minima, neglecting other geometric properties of the loss landscape.

## Method

### Core Idea

The conventional point-mass is replaced by a **rigid sphere** of finite radius $\rho > 0$ rolling over the loss landscape. The sphere's dynamics respond to landscape features at a scale proportional to $\rho$, such that:

- Noise much smaller than $\rho$ does not affect the sphere's trajectory
- Sharp minima and ill-conditioned valleys narrower than the sphere cannot "accommodate" it
- Adjusting $\rho$ controls the granularity at which the optimizer interacts with the landscape

### Algorithm

RBO alternates between two steps:

**Step 1: Descent step** (analogous to gradient descent)

$$\tilde{c}_{t+1} = c_t - \eta \tau(p_t)$$

where $p_t$ is the contact point between the sphere and the loss landscape $\Gamma$, and $\tau(p)$ is the steepest descent direction of $\Gamma$ at $p$.

**Step 2: Constraint projection** (restoring the distance invariant)

$$p_{t+1} = \arg\min_{p \in \Gamma} \|p - \tilde{c}_{t+1}\|^2$$

The landscape point nearest to the new center is found and taken as the new contact point; the sphere center is then updated as $c_{t+1} = p_{t+1} + \rho \nu(p_{t+1})$, where $\nu(p)$ is the upward unit normal to $\Gamma$ at $p$.

### Distance Invariant

The core constraint of RBO: the distance from the sphere center $c_t$ to the loss landscape $\Gamma$ always equals the radius $\rho$:

$$\forall t \geq 0, \quad d(c_t, \Gamma) = \inf_{p \in \Gamma} \|p - c_t\| = \rho$$

### Iterative Solution of the Projection Step

The constraint projection is solved via iterative optimization:

$$\theta^{(k+1)} = \theta^{(k)} - \gamma[\theta^{(k)} - \tilde{\theta} + (f(\theta^{(k)}) - \tilde{y}) \nabla f(\theta^{(k)})]$$

This makes the dynamics of RBO dependent on landscape information over a broader region, rather than only at the current contact point.

### Ironing Property

**Weak ironing (Lemma)**: For any continuous bounded function $\phi: \mathbb{R}^d \to \mathbb{R}$, as $\rho \to +\infty$, the offset manifold containing the sphere center trajectory converges to a constant — i.e., the loss landscape is completely "ironed out."

**Linear ironing (Proposition)**: For a landscape composed of an affine function $f$ plus a bounded perturbation $\phi$, RBO with sufficiently large $\rho$ behaves approximately as if optimizing on the pure affine landscape.

### Unreachable Point Theory

If the spectral norm of the Hessian at a point $p$ on the landscape is $\sigma = \|\nabla^2 f(\theta_0)\|$, then for $\rho > 1/\sigma$, $p$ is unreachable by RBO. This implies:

- **Automatic avoidance of sharp minima**: the larger the curvature, the smaller the $\rho$ needed to bypass the minimum
- **Open-set property of unreachable points**: neighborhoods of unreachable points are also unreachable

## Key Experimental Results

### Main Results: Test Set Performance Comparison

| Dataset/Model | SGD (Acc) | Entropy-SGD (Acc) | SAM (Acc) | **RBO (Acc)** |
|---|---|---|---|---|
| MNIST/MLP | 91.77% | 95.22% | 97.22% | **97.51%** |
| MNIST/ResNet-6 | 97.59% | 98.18% | **99.11%** | 99.07% |
| MNIST/VGG-9 | 98.78% | 98.57% | **99.39%** | 99.27% |
| CIFAR-10/ResNet-8 | 56.54% | 59.16% | 69.09% | **71.58%** |
| CIFAR-10/VGG-9 | 66.04% | 65.46% | 77.81% | **81.87%** |
| CIFAR-100/ResNet-8 | 19.28% | 28.33% | 36.26% | **37.11%** |
| CIFAR-100/VGG-9 | 29.37% | 28.98% | 47.17% | **50.07%** |

### Effect of Radius $\rho$ on Performance (MNIST/MLP, 3 epochs)

| $\rho$ range | Learning rate range | Observation |
|---|---|---|
| 0.1 – 1.0 | 0.001 – 1.0 | Microscopic regime, close to point-mass optimizer |
| 1.0 – 5.0 | 0.01 – 50 | Macroscopic regime, best performance |
| > 10 | > 100 | Super-macroscopic regime, potentially unstable |

### Key Experimental Findings

1. RBO consistently outperforms SGD and Entropy-SGD in accuracy, and is competitive with SAM (RBO achieves higher accuracy; SAM achieves lower loss values).
2. RBO converges extremely fast: on CIFAR-10/100, it reaches the final training performance of other optimizers in half the number of epochs.
3. RBO can stably operate with very large learning rates (e.g., $\eta = 6$ or even $\eta = 100$), whereas SGD is limited to $\eta = 0.01$.
4. Performance improves monotonically as $\rho$ and $\eta$ increase, until the super-macroscopic instability regime is entered.

## Highlights & Insights

1. **Elegant physical intuition**: The analogy of replacing a point mass with a rolling rigid sphere is both intuitive and profound — just as a car wheel is insensitive to Planck-scale road roughness, a large sphere naturally ignores fine-grained noise.
2. **Theoretical innovation**: The ironing property and unreachable point theory provide rigorous mathematical characterizations of the non-locality of the optimizer.
3. **Extreme learning rate stability**: RBO remains stable at $\eta = 100$, which is inconceivable for conventional optimizers.
4. **Honest experimental design**: The authors explicitly state that no hyperparameter tuning was performed for any experiment, so the reported results do not represent RBO's optimal performance.
5. **Interesting source of inspiration**: The algorithm draws on the physical simulation of sphere motion from the open-source game *Marble Marcher*.

## Limitations & Future Work

1. **High computational cost**: The constraint projection step requires additional iterative optimization, making the per-step cost substantially higher than SGD.
2. **Incomplete theory**: The strong ironing conjecture (generalization to arbitrary continuous functions) remains unproven.
3. **Limited experimental scale**: Evaluation is restricted to MNIST and CIFAR-10/100 using relatively small architectures (MLP, ResNet-6/8, VGG-9).
4. **Uncertainty in high dimensions**: Approximation errors in the projection step may accumulate with increasing dimensionality; the impact of the curse of dimensionality remains unknown.
5. **Modest validation performance gains**: While training performance is excellent, improvements on the validation set are less pronounced than expected.
6. Only short training runs of 10 epochs are reported; long-horizon training behavior is unknown.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Reconceptualizes optimizer design from a physics simulation perspective; highly creative.
- **Experimental Thoroughness**: ⭐⭐⭐ — Results are convincing but limited in scale; large-model/large-dataset validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Physical intuition is clearly conveyed, theoretical derivations are rigorous, and figures are well-crafted.
- **Value**: ⭐⭐⭐⭐ — Opens a new direction for non-local optimizers, though practical adoption requires addressing the computational cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [\[NeurIPS 2025\] Gradient Descent as Loss Landscape Navigation: a Normative Framework for Deriving Learning Rules](../../NeurIPS2025/optimization/gradient_descent_as_loss_landscape_navigation_a_normative_framework_for_deriving.md)
- [\[ICLR 2026\] Optimizer Choice Matters for the Emergence of Neural Collapse](optimizer_choice_matters_for_the_emergence_of_neural_collapse.md)
- [\[NeurIPS 2025\] Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers](../../NeurIPS2025/optimization/generalization_or_hallucination_understanding_out-of-context_reasoning_in_transf.md)
- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](deepafl_deep_analytic_federated_learning.md)

</div>

<!-- RELATED:END -->
