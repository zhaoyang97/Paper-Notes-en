---
title: >-
  [Paper Note] DiLQR: Differentiable Iterative Linear Quadratic Regulator via Implicit Differentiation
description: >-
  [ICML 2025][Differentiable control] This paper proposes the DiLQR framework, which applies implicit differentiation to the fixed points of the iLQR controller to derive analytical gradient solutions. This reduces the backpropagation computational complexity from linear growth with the number of iterations to a constant $O(1)$, achieving up to a 128× speedup while improving learning performance by up to $10^6$ times compared to conventional neural network policies.
tags:
  - "ICML 2025"
  - "Differentiable control"
  - "iLQR"
  - "Implicit differentiation"
  - "Fixed-point differentiation"
  - "Imitation learning"
date: 2026-05-08
content_hash: e1d4245189eb3255
---

# DiLQR: Differentiable Iterative Linear Quadratic Regulator via Implicit Differentiation

**Conference**: ICML 2025  
**arXiv**: [2506.17473](https://arxiv.org/abs/2506.17473)  
**Code**: [https://sites.google.com/view/dilqr/](https://sites.google.com/view/dilqr/) (Project Page)  
**Area**: LLM Evaluation  
**Keywords**: Differentiable control, iLQR, Implicit differentiation, Fixed-point differentiation, Imitation learning

## TL;DR

This paper proposes the DiLQR framework, which applies implicit differentiation to the fixed points of the iLQR controller to derive analytical gradient solutions. This reduces the backpropagation computational complexity from linear growth with the number of iterations to a constant $O(1)$, achieving up to a 128× speedup while improving learning performance by up to $10^6$ times compared to conventional neural network policies.

## Background & Motivation

**Background**: Differentiable control is a new paradigm that combines model-free flexibility with model-based efficiency. Iterative Linear Quadratic Regulator (iLQR), as a powerful numerical controller, is widely used in trajectory optimization; however, the development of its differentiable counterparts lags behind LQR.

**Limitations of Prior Work**: To integrate iLQR as a trainable neural network module, naive automatic differentiation (AutoDiff) methods rely on backpropagating through the entire unrolled iteration chain. Propagating forward for hundreds of iterations to solve LQR optimization problems and backpropagating through all these layers leads to memory consumption and computational time growing linearly with both the number of iterations and the horizon length, severely limiting scalability.

**Key Challenge**: The forward pass of iLQR requires multiple iterations to converge to the optimal trajectory, but AutoDiff couples forward and backward propagation. Consequently, more iterations and longer horizons result in slower and more memory-intensive backpropagation. Although DiffMPC proposed analytical gradient methods, it treats the input of the last layer of iLQR as a constant rather than a function of the learnable parameters, leading to inaccurate gradients.

**Goal**: How to efficiently and accurately compute the gradient of the iLQR controller with respect to learnable parameters $\theta$, allowing it to serve as a differentiable module in end-to-end learning frameworks?

**Key Insight**: The converged trajectory of iLQR is a fixed point, where the input itself is the output. One can directly compute $\partial \tau^*/\partial \theta$ at the fixed point using the implicit function theorem, escaping the need to unroll the iteration process. This completely decouples the computational complexity of backpropagation from the number of forward iterations.

**Core Idea**: By applying implicit differentiation to the iLQR fixed point to obtain exact analytical gradients, the computational complexity of backpropagation is reduced from $O(\text{iterations})$ to $O(1)$.

## Method

### Overall Architecture

DiLQR encapsulates iLQR as a differentiable module. Forward pass: iLQR iterations are executed normally to solve for the optimal trajectory $\tau^*$. Backward pass: without unrolling the iteration chain, the gradient $\partial \tau^*/\partial \theta$ is computed directly via implicit differentiation using the fixed-point condition $\tau^* = \text{iLQR}(\tau^*, \theta)$. This module can be embedded into larger neural networks (e.g., combined with visual encoders) to enable end-to-end learning.

### Key Designs

1. **Fixed-Point Implicit Differentiation**:

    - Function: Accurately compute the gradient of the optimal trajectory with respect to parameters without unrolling the iteration process.
    - Mechanism: Take the total derivative with respect to $\theta$ at the fixed point $X^* = F(X^*, U^*, \theta)$, $U^* = G(X^*, U^*, \theta)$, yielding the system of linear equations $(I - F_X)\nabla_\theta X^* - F_U \nabla_\theta U^* = F_\theta$. Solving this yields the analytical formulation: $\nabla_\theta X^* = M(F_\theta + F_U(K - G_X M F_U)^{-1}(G_X M F_\theta - G_\theta))$, where $M = (I - F_X)^{-1}$ and $K = I - G_U$.
    - Design Motivation: Compared to DiffMPC, which treats the fixed-point trajectory as constant and only takes the partial derivative $\partial A^i / \partial \theta$, DiLQR correctly handles the total derivative $\nabla_\theta A^i = \partial A^i / \partial \theta + \partial A^i / \partial \tau^i \cdot \partial \tau^i / \partial \theta$ (the boxed term), making the gradients more accurate.

2. **Forward Algorithm**:

    - Function: Efficiently compute the derivative of the linearized dynamics matrix $D_t$ with respect to the parameter $\theta$.
    - Mechanism: Utilizing the recursive relation between timesteps, $\nabla_\theta x_t$ can be computed recursively from $\nabla_\theta x_{t-1}$ as: $\nabla_\theta x_t = \partial x_t / \partial \theta + [\partial x_t / \partial x_{t-1} + \partial x_t / \partial u_{t-1} \cdot \partial u_{t-1} / \partial x_{t-1}] \nabla_\theta x_{t-1}$. The partial derivatives at each timestep can be analytically computed beforehand, needing only numerical substitution at runtime.
    - Design Motivation: PyTorch's `torch.autograd.jacobian` does not reuse gradient information across timesteps, leading to significant redundant computation. The Forward Algorithm leverages the recursive relation to reduce the computation by dozens of times.

3. **Sparsity & Parallelization**:

    - Function: Utilize structural sparsity and independent computations for acceleration.
    - Mechanism: (a) Sparsity: $\partial D / \partial X$ is a block-diagonal matrix ($\partial D_t / \partial x_{t'}=0$ when $t' \neq t$), avoiding full matrix instantiation and using only diagonal blocks; (b) Parallelization: Construct batches of binary loss functions to calculate $\partial H_{i,j} / \partial D$ in parallel, setting $L_{i,j}$ to 1 and the rest to 0, making the computation of each element fully independent and parallelizable.
    - Design Motivation: The analytical solution for implicit differentiation involves large Jacobian matrix operations. Without exploiting problem structure, the computational overhead remains substantial.

### Loss & Training

Imitation learning loss: $L(\tau(\theta))$, which minimizes the gap between predicted and expert trajectories. The total gradient is computed via the chain rule $\nabla_\theta(L \circ \tau)(\theta) = \nabla_\tau L(\tau(\theta)) \cdot \partial \tau / \partial \theta$, where $\nabla_\tau L$ is provided by AutoDiff and $\partial \tau / \partial \theta$ is provided by the analytical solution of DiLQR.

## Key Experimental Results

### Main Results

| Horizon | iLQR Iterations | AutoDiff Time (s) | DiLQR Time (s) | Speedup |
|---------|----------|------------------|---------------|-------|
| 10 | 50 | 1.41 | 0.067 | 21× |
| 10 | 300 | 8.57 | 0.067 | **128×** |
| 30 | 50 | ~4 | ~0.2 | ~20× |
| 30 | 300 | ~25 | ~0.2 | ~125× |

### Imitation Learning Performance

| Task | Method | Imitation Loss | Gain over NN |
|------|------|---------|----------|
| Pendulum | NN (LSTM) | ~1e-1 | Baseline |
| Pendulum | DiLQR.dx | ~1e-7 | **10^6×** |
| Cartpole | NN (LSTM) | ~1e-1 | Baseline |
| Cartpole | DiLQR.dx | ~1e-5 | **10^4×** |

### Ablation Study

| Configuration | Horizon=10 Time | Horizon=30 Time | Explanation |
|------|----------------|----------------|------|
| Full DiLQR | Fastest | Fastest | All optimizations enabled |
| w/o Forward Algorithm | Significantly increased | Sharp increase | Forward Algorithm contributes the most |
| w/o Parallelization | Further increase | Slower for long horizons | Parallelization is critical for long horizons |
| w/o Sparsity | Slight increase | Slight increase | Sparsity exploitation has minor impact |

### Model Loss & Physical Consistency

| Metric | DiLQR | DiffMPC | Explanation |
|------|-------|---------|------|
| dcost model loss | Reduced by 32% | Baseline | More accurate recovery of cost function parameters |
| dx model loss (train=50) | Eventually 41% lower | Stabilizes earlier | More continuous optimization of dynamics parameter learning |
| Ratio of negative parameters (train=100) | 2.76% | 16.85% | Physical consistency far superior to DiffMPC |
| Ratio of negative parameters (train=50) | 7.23% | 17.82% | Maintains physical plausibility even with scarce data |

### Key Findings

- The backpropagation time of DiLQR remains strictly constant with respect to the number of iterations (flat line vs. the linear growth of AutoDiff), achieving a minimum of 21× and a maximum of 128× speedup.
- In dx mode, DiLQR's imitation loss is $10^6$ orders of magnitude lower than the LSTM policy, showcasing the immense advantage of embedding structured controllers into learning frameworks.
- Physical consistency is significantly superior to DiffMPC (negative parameters 2.76% vs 16.85%), indicating that accurate gradients yield not only numerical precision but also semantic plausibility.
- The Forward Algorithm is the largest contributor to computational efficiency gains, especially in long-horizon settings.

## Highlights & Insights

- **Implicit differentiation decouples forward and backward propagation**: This is the core contribution of this paper—shifting the backpropagation of iLQR from "unrolling all iterations" to "solving linear equations at the fixed point", reducing complexity from $O(N_{\text{iter}})$ to $O(1)$. This concept can be generalized to any fixed-point-iteration-based differentiable computation module.
- **Real-world impact of exact vs. approximate gradients**: DiffMPC ignores the dependency of the fixed point on parameters, leading to a 32% higher model loss and numerous unphysical parameters (16.85% negative values). This demonstrates that in the control domain, gradient accuracy directly impacts learning quality.
- **Modular demonstration of visual end-to-end control**: Embedding DiLQR into an encoder-decoder architecture enables end-to-end control from pixel inputs. Given a single real image, it can "imagine" downstream trajectory images, showcasing the compositional capability of modular design.

## Limitations & Future Work

- The experiments are validated only on two classic control tasks, CartPole and Inverted Pendulum. These are relatively simple low-dimensional systems, and their performance on complex tasks, such as high-dimensional robotic control, remains to be verified.
- The method assumes that iLQR converges to a fixed point. For certain non-convex problems, iLQR may not converge or may converge to local optima, under which the assumption of implicit differentiation may not hold.
- First- and second-order derivatives of the system dynamics are required. When dynamics models are fitted using neural networks, computing second-order derivatives may introduce a new bottleneck.
- The visual control experiments are only at the proof-of-concept level, leaving a significant gap compared to more complete perception-control frameworks like DiffTORI.

## Related Work & Insights

- **vs DiffMPC (Amos et al., 2018)**: DiffMPC is a pioneer in differentiable LQR but treats fixed-point trajectories as constants when computing partial derivatives. DiLQR computes the total derivative using implicit differentiation, correcting this approximation and reducing model loss by 32%.
- **vs SafePDP / IDOC**: Differentiable methods based on Pontryagin's Maximum Principle converge slower than iLQR (which has a 1.5-order convergence rate). When using expert trajectories generated by iLQR, DiLQR significantly outperforms both.
- **vs Deep Equilibrium Models (DEQ)**: Conceptually similar—DEQ applies implicit differentiation to the fixed points of deep networks, while DiLQR applies it to iLQR fixed points. However, the self-referential structure of iLQR ($x^* = f_{x^*}(x^*)$) is more complex than a general fixed point.

## Rating

- Novelty: ⭐⭐⭐⭐ Applying implicit differentiation to the iLQR fixed point is novel, and the derivation of the analytical solution is complete and rigorous.
- Experimental Thoroughness: ⭐⭐⭐ The experimental scenarios are relatively simple (Pendulum/CartPole), but the computational efficiency comparison is highly thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ The mathematical derivations are clear and comprehensive, and the methodological comparison (vs DiffMPC) is precise down to specific formula terms.
- Value: ⭐⭐⭐⭐ Provides an efficient and accurate foundational tool for differentiable control, with the 128× speedup holding significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Sampling from Binary Quadratic Distributions via Stochastic Localization](sampling_from_binary_quadratic_distributions_via_stochastic_localization.md)
- [\[ICML 2025\] AutoAL: Automated Active Learning with Differentiable Query Strategy Search](autoal_automated_active_learning_with_differentiable_query_strategy_search.md)
- [\[NeurIPS 2025\] Exact Learning of Arithmetic with Differentiable Agents](../../NeurIPS2025/others/exact_learning_of_arithmetic_with_differentiable_agents.md)
- [\[NeurIPS 2025\] A Differentiable Model of Supply-Chain Shocks](../../NeurIPS2025/others/a_differentiable_model_of_supply-chain_shocks.md)
- [\[ACL 2025\] Predicting Implicit Arguments in Procedural Video Instructions](../../ACL2025/others/implicit_arguments_video_instructions.md)

</div>

<!-- RELATED:END -->
