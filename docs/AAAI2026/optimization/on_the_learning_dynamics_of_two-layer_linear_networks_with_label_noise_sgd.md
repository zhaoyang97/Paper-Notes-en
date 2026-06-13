---
title: >-
  [Paper Note] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD
description: >-
  [AAAI 2026][Optimization][Label Noise SGD] This paper theoretically analyzes the learning dynamics of label noise SGD on two-layer overparameterized linear networks, revealing a two-phase behavior: in Phase I…
tags:
  - "AAAI 2026"
  - "Optimization"
  - "Label Noise SGD"
  - "Learning Dynamics"
  - "Lazy-to-Rich Transition"
  - "Implicit Bias"
  - "Sharpness-Aware Minimization"
date: 2026-05-08
content_hash: 4b85308a07ef3468
---

# On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD

**Conference**: AAAI 2026
**arXiv**: [2603.10397](https://arxiv.org/abs/2603.10397)  
**Code**: [https://github.com/a-usually/Label-Noise-SGD](https://github.com/a-usually/Label-Noise-SGD)  
**Area**: Alignment RLHF / Optimization Theory
**Keywords**: Label Noise SGD, Learning Dynamics, Lazy-to-Rich Transition, Implicit Bias, Sharpness-Aware Minimization

## TL;DR

This paper theoretically analyzes the learning dynamics of label noise SGD on two-layer overparameterized linear networks, revealing a two-phase behavior: in Phase I, weight norms progressively diminish, enabling the model to escape the lazy regime and enter the rich regime; in Phase II, weights align with the ground-truth interpolator and converge. The theory is further extended to the SAM optimizer.

## Background & Motivation

**Background**: The success of deep learning is partly attributed to the implicit bias induced by noise in gradient-based training algorithms. Recent empirical findings show that injecting label noise during training can improve generalization (e.g., ~1.5% test accuracy gain on CIFAR-10 + ResNet-18) and yields sparser models.

**Limitations of Prior Work**: Existing theoretical work (Blanc et al. 2020; Damian et al. 2021; HaoChen et al. 2021) largely focuses on the local implicit regularization effects of label noise SGD near global minima (e.g., regularizing sharpness), or analyzes highly simplified models such as diagonal linear networks. For more realistic networks with two trainable layers and inter-layer coupling effects, a theoretical analysis of the complete learning dynamics from initialization to convergence remains absent.

**Key Challenge**: Overparameterized networks initialized under the NTK regime tend to remain in the lazy regime—where parameters barely move and the model behaves like a linear kernel method—failing to explain the generalization advantages of deep learning. Yet in practice, models do learn meaningful features (rich regime). How does label noise drive this critical transition?

**Goal**: To rigorously characterize the complete learning trajectory of two-layer linear networks under label noise SGD, and to explain how and why label noise drives the lazy-to-rich transition.

**Key Insight**: The analysis begins from the oscillation effect in the second-layer parameters—label noise amplifies the oscillations of the second-layer weights $\mathbf{a}$, which through inter-layer coupling cause the first-layer weight norms $\mathbf{W}$ to progressively diminish, thereby enabling the transition from the lazy regime (NTK initialization) to the rich regime (small-initialization behavior).

**Core Idea**: Label noise, by amplifying oscillations in the second-layer parameters, indirectly drives a sustained decay of the first-layer weight norms, allowing the network to naturally transition from the lazy regime to the rich regime and ultimately converge to a sparse ground-truth interpolator.

## Method

### Overall Architecture

The paper considers a two-layer linear network $\hat{y}_i = \mathbf{a}^\top \mathbf{W} \mathbf{x}_i$, where $\mathbf{W} \in \mathbb{R}^{m \times d}$ and $\mathbf{a} \in \mathbb{R}^m$. NTK initialization is used: $w_{i,j}(0) \sim \frac{1}{\sqrt{d}} \mathcal{N}(0, I)$, $a_i(0) \sim \frac{1}{\sqrt{m}} \mathcal{N}(0, I)$. Training employs label noise SGD: at each step, labels are randomly flipped as $\tilde{y}_i = y_i + \epsilon$ ($\epsilon \sim \{-\sigma, +\sigma\}$), and gradient updates are computed on the noisy labels.

### Key Designs

1. **Phase I: Progressive Diminishing and Lazy-to-Rich Transition (Theorem 4.2, Lemma 4.3)**

    - **Function**: Proves that during Phase I, the first-layer weight norms $\|\mathbf{w}_i(t)\|$ for all neurons decrease monotonically, enabling the model to escape the lazy regime.
    - **Mechanism**: The change in weight norm is $\Delta W_i(j) = -\nabla\hat{\ell}^2 \cdot ((\mathbf{x}^\top \mathbf{w}_i)^2 - a_i^2 \|\mathbf{x}\|^2)$. Since $\mathbf{a}(0)$ is initialized to small values, the $(\mathbf{x}^\top \mathbf{w}_i)^2$ term dominates, making $\Delta W_i(j)$ negative with high probability. The key identity $\nabla\hat{\ell}^2 \cdot (\mathbf{x}^\top \mathbf{w}_i)^2 = (a_i(j+1) - a_i(j))^2$ shows that the decay of the first-layer norm is directly controlled by the oscillation amplitude of the second layer, and label noise is the source of these oscillations.
    - **Design Motivation**: This is the central finding of the paper—it establishes the causal chain "label noise → second-layer oscillations → first-layer norm decay → lazy-to-rich transition." Theorem 4.2 gives the escape time $T_1 = O(\frac{\sqrt{\log m}}{\sigma^2 \eta^2 \sqrt{m}})$, explicitly depending on noise intensity $\sigma$.

2. **Phase II: Alignment and Convergence (Lemma 4.5, 4.6)**

    - **Function**: Proves that once weight norms are sufficiently small (i.e., $\|\mathbf{w}_i\|, |a_i| \leq \sqrt{\eta}$), neurons rapidly align with the ground-truth interpolator $\theta^*$ and converge.
    - **Mechanism**: Lemma 4.5 proves that after $T_2 = \frac{1}{\|\theta^*\|} \ln(1/\eta)$ steps, the alignment $\frac{|\langle \theta^*, \mathbf{w}_i \rangle|}{\|\theta^*\| \cdot \|\mathbf{w}_i\|} \geq 1 - O(\ln(1/\eta) \cdot \sqrt{\eta})$. Lemma 4.6 proves that after perfect alignment, an additional $T_3 = O(\frac{-\ln\eta}{\eta})$ steps suffice to achieve $\|\theta(t_3) - \theta^*\| \leq O(\eta \ln(1/\eta))$.
    - **Design Motivation**: Phase II resembles learning dynamics under small initialization—which is precisely the effect of Phase I. Label noise SGD effectively transforms a large initialization into an equivalent small initialization.

3. **Extension to SAM**

    - **Function**: Validates that SAM (Sharpness-Aware Minimization) exhibits the same two-phase dynamics.
    - **Mechanism**: SAM's inner adversarial perturbation amplifies gradient noise, analogously to the effect of label noise. In synthetic and CIFAR-10 experiments, the loss curves of WideResNet trained with SAM deviate significantly from those of its linearized counterpart (a signature of the rich regime), while first-layer weight norms visibly decay.
    - **Design Motivation**: Generalizes the identified mechanism from a specific noise source to a broader class of "noise-amplifying" optimization strategies.

### Loss & Training

The mean squared loss $\hat{\ell}_i(\theta(t)) = \frac{1}{2} |f(\theta(t); \mathbf{x}_i) - y_i - \epsilon|^2$ is used, where $\epsilon \sim \{-\sigma, +\sigma\}$. Key conditions include: overparameterization $m = \Omega(1/\sqrt{\eta})$, small learning rate $\eta \leq 1/C^{96}$, sufficient data $n \geq 1/\eta^2$, and sparse ground-truth $\|\theta^*\| \leq m^{-1/4}$. These conditions ensure Phase I is long enough for sufficient norm decay and that the network width is large enough for high-probability results. Training uses online SGD (batch size 1) with label noise, sampling one training example per step and randomly flipping its label.

## Key Experimental Results

### Main Results (CIFAR-10 + ResNet-18)

| Configuration | Test Accuracy | Test Loss | Notes |
|---|---|---|---|
| Vanilla SGD | ~93.5% | Higher | Baseline |
| Label Noise SGD ($\tau=0.05$) | ~94.5% | Lower | +1.0% |
| Label Noise SGD ($\tau=0.1$) | ~95.0% | Lowest | +1.5% |
| Label Noise SGD ($\tau=0.2$) | ~94.8% | Low | +1.3% |

All label noise probabilities $\tau \in \{0.05, 0.1, 0.2\}$ outperform vanilla SGD.

### Ablation Study (Synthetic Two-Layer Linear Network, Two-Phase Verification)

| Phase | Key Metric | Observation |
|---|---|---|
| Phase I | Average neuron norm $\text{Avg}(\|\mathbf{w}_i\|)$ | Initial decrease, verifying progressive diminishing |
| Phase I | Train/test loss | Training loss fluctuates; test loss decreases steadily |
| Phase II | Average alignment $\text{Avg}(\langle \mathbf{w}_i, \theta^* \rangle)$ | Rapidly rises toward 1 |
| Phase II | Parameter distance $\|\theta(t) - \theta^*\|$ | Converges to $O(\eta \ln(1/\eta))$ |
| Lazy regime (noiseless GD) | Loss curve | Nearly overlaps with linearized model, confirming lazy regime |
| Rich regime (with noise) | Loss curve | Deviates from linearized model, confirming rich regime |
| Simulated oscillation (Markov Chain) | Weight norm | Confirms oscillation → norm decay |

### Key Findings

- **Label noise drives the lazy→rich transition**: Loss curves of WideResNet trained with noiseless GD nearly coincide with those of its linearized model (lazy regime), while adding label noise causes significant deviation (rich regime) accompanied by visible first-layer weight norm decay.
- **Sparsity advantage**: At equal pruning ratios ($\alpha\%$ parameters retained), models trained with label noise SGD consistently outperform those trained with vanilla SGD in accuracy, indicating the learning of sparser representations.
- **SAM exhibits the same behavior**: On both synthetic and CIFAR-10 experiments, SAM's learning dynamics are qualitatively consistent with label noise SGD (weight norms first decrease, then align), supporting the generalizability of the theory.
- **Causal chain from noise to second-layer oscillations to first-layer decay**: A three-state Markov process simulation confirms that even without SGD sampling noise, second-layer oscillations alone can drive first-layer norm decay (Lemma 4.4).

## Highlights & Insights

- **A new mechanism for implicit bias**: Unlike prior work focusing on "label noise regularizing sharpness," this paper reveals a more fundamental effect—noise-driven regime transition. This perspective is more explanatory, as it accounts for why noise not only reduces sharpness but also promotes feature learning.
- **Elegant analysis of inter-layer coupling**: The analysis linking second-layer oscillations to first-layer norm decay captures the essence of inter-layer dependence in deep networks. This analytical paradigm may generalize to deeper networks.
- **A unified view from SGD to SAM**: By unifying label noise SGD and SAM under the framework of "noise amplification promoting the rich regime," the paper provides a common theoretical foundation for understanding the implicit bias of different optimizers.
- **Outstanding experimental visualization**: Figure 2, which traces each neuron's trajectory in the norm–alignment space, intuitively illustrates the two-phase dynamics and serves as an exemplary integration of theory and experiment.

## Limitations & Future Work

- The theory is restricted to **two-layer linear networks**—without nonlinear activations, there remains a gap from realistic architectures. How nonlinearity affects the two-phase dynamics is an important open problem.
- The theoretical analysis switches to GD (noiseless) during Phase II, simplifying the analysis at the cost of a complete characterization of the convergence phase under full label noise SGD.
- The experimental scale is limited: a 64-image subset of CIFAR-10 with ResNet-18/WideResNet; validation on large-scale tasks is absent.
- Theoretical conditions are strong: $\eta \leq 1/C^{96}$ and $\|\theta^*\| \leq m^{-1/4}$ are difficult to verify in practice.
- Only **regression tasks** are analyzed; extending to classification with cross-entropy loss remains an open challenge.

## Related Work & Insights

- **vs HaoChen et al. (2021) / Vivien et al. (2023)**: These works analyze label noise SGD on diagonal linear networks (effectively single-layer parameters) and show it recovers a sparse ground-truth. The present paper analyzes two-layer networks, handling more complex inter-layer coupling.
- **vs Blanc et al. (2020) / Damian et al. (2021)**: These works focus on the implicit regularization of label noise near global minima (trace of Hessian), constituting a local analysis around Phase II. This paper provides global dynamics from initialization to convergence.
- **vs Geiger et al. (2020)**: They show that initialization scale determines the lazy vs. rich regime. This paper complements their finding: even under large (NTK) initialization, label noise can drive the system into the rich regime.
- **vs Weight Decay / Large LR methods**: Li et al. and Lewkowycz et al. show that weight decay and large learning rates can also induce the rich regime. The label noise mechanism in this paper operates differently—acting indirectly through second-layer oscillations—offering a new perspective on regime transitions.
- **vs Varre et al. (2023)**: They show that label noise SGD tends to reduce the rank of parameter matrices, consistent with the weight norm decay in Phase I of this paper, but they do not analyze the complete two-phase dynamics.
- **Inspiration**: The oscillation–decay mechanism may be applicable to analyzing the implicit bias of other regularization techniques such as Dropout and Mixup.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The theoretical analysis of the lazy→rich transition is a new contribution; the oscillation mechanism via inter-layer coupling is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐ Synthetic experiments sufficiently validate the theory, but real-data experiments are limited in scale due to NTK computational complexity.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are rigorous and the two-phase picture is clearly articulated, though notation is heavy.
- **Value**: ⭐⭐⭐⭐ Makes a theoretical contribution to understanding SGD implicit bias and feature learning; practical impact warrants further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](../../ICML2026/optimization/balancing_learning_rates_across_layers_exact_two-step_dynamics_and_optimal_scali.md)
- [\[ICML 2026\] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks](../../ICML2026/optimization/dynamics_and_representation_structure_of_local_approximations_to_gradient-based_.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](../../ICLR2026/optimization/directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)
- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](../../NeurIPS2025/optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)
- [\[ICML 2026\] Sharp Description of Local Minima in the Loss Landscape of High-Dimensional Two-Layer ReLU Networks](../../ICML2026/optimization/sharp_description_of_local_minima_in_the_loss_landscape_of_high-dimensional_two-.md)

</div>

<!-- RELATED:END -->
