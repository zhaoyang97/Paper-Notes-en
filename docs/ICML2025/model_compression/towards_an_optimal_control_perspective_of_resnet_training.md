---
title: >-
  [Paper Note] Towards an Optimal Control Perspective of ResNet Training
description: >-
  [ICML 2025][Model Compression][ResNet] Formulates ResNet training as an optimal control problem, achieving self-regularization by adding stage cost losses to intermediate layers, and proves that redundant deep weights asymptotically vanish, laying the foundation for theory-driven layer pruning.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "ResNet"
  - "optimal control"
  - "stage cost"
  - "layer pruning"
  - "self-regularization"
date: 2026-05-08
content_hash: 6885ed4632b5c3d5
---

# Towards an Optimal Control Perspective of ResNet Training

**Conference**: ICML 2025  
**arXiv**: [2506.21453](https://arxiv.org/abs/2506.21453)  
**Code**: -  
**Area**: Theory / Neural Network Training  
**Keywords**: ResNet, optimal control, stage cost, layer pruning, self-regularization  

## TL;DR

Formulates ResNet training as an optimal control problem, achieving self-regularization by adding stage cost losses to intermediate layers, and proves that redundant deep weights asymptotically vanish, laying the foundation for theory-driven layer pruning.

## Background & Motivation

- Residual connections in ResNet can be viewed as the forward Euler discretization of continuous-time Neural ODEs.
- From an optimal control perspective: forward data propagation corresponds to the state trajectory of a dynamical system, while trainable parameters act as control signals.
- Existing optimal control training methods are only applicable to architectures with fixed hidden dimensions and make restrictive assumptions about the loss functions.
- **Goal**: Generalize the stage cost formulation to standard ResNets (with non-trivial skip connections) and general loss functions.

## Method

### 1. ResNet as a Dynamical System

Forward propagation of standard ResNet-N:

$$x_{k+1}^{(i)} = f_k(x_k^{(i)}, \mathbf{w}_k) = \begin{cases} x_k^{(i)} + \mathcal{F}_k(x_k^{(i)}, \mathbf{w}_{\mathcal{F},k}), & k \notin \mathcal{K}_\mathcal{S} \\ \mathcal{S}_k(x_k^{(i)}, \mathbf{w}_{\mathcal{S},k}) + \mathcal{F}_k(x_k^{(i)}, \mathbf{w}_{\mathcal{F},k}), & k \in \mathcal{K}_\mathcal{S} \end{cases}$$

where $\mathcal{K}_\mathcal{S}$ is the set of layer indices with non-trivial skip connections (1×1 convolutions).

### 2. Designing Intermediate Output Heads

**Key Innovation**: Construct intermediate outputs by utilizing the skip connections of subsequent layers and the final output head:

$$\hat{y}_k^{(i)} = \mathcal{H}(\mathcal{S}_{N-1}(\cdots \mathcal{S}_k(x_k^{(i)}, \mathbf{w}_{\mathcal{S},k}), \cdots, \mathbf{w}_{\mathcal{S},N-1}), \mathbf{w}_\mathcal{H})$$

This indicates that intermediate predictions **reuse existing parameters of the backbone network** (skip connection weights and the output head), requiring no additional parameters.

### 3. Stage Cost Training Objective

$$\min_\mathbf{w} J_N(\mathbf{w}) = \sum_{k=0}^{N-1} \gamma \mathcal{L}(\hat{\mathbf{y}}_k) + \mathcal{L}(\hat{\mathbf{y}})$$

Degenerates to standard training when $\gamma = 0$.

### 4. Theoretical Result: Asymptotic Loss Bound

**Theorem 3.1**: Let ResNet-N be trained with stage cost and weight decay, and let ResNet-M ($M<N$) be its SubResNet, then:

$$J_N(\mathbf{w}^N) \leq \sum_{k=0}^{M-1}\left[\gamma\mathcal{L}(\hat{\mathbf{y}}_k^M) + \frac{\lambda}{2}\|\mathbf{w}_{\mathcal{F},k}^M\|^2\right] + (1+\gamma(N-M))\bar{\mathcal{L}}$$

**Implication**: When a shallower SubResNet is already capable of completing the learning task (small $\bar{\mathcal{L}}$), the weights of deeper residual blocks will approach zero—meaning the network automatically discovers the optimal depth required.

Simplified result without weight decay:

$$\mathcal{L}_\text{avg} = \frac{1}{N+1}\sum_{k=0}^N \mathcal{L}(\hat{\mathbf{y}}_k^N) \leq \bar{\mathcal{L}} + \frac{C}{N+1}$$

## Key Experimental Results

### CIFAR-10 Loss Trajectory

- The ResNet trained with stage cost **achieves a good fit after only 12 residual blocks** (78.74% test accuracy).
- Standard training only achieves a good fit at the final output.
- Performance plateaus within the same stage (same number of filters) and jumps at the start of a new stage (increased number of filters).

### Layer Pruning Comparison

| Model | MNIST | CIFAR-10 | CIFAR-100 |
|------|-------|----------|-----------|
| ResNet-54 Standard Training | 99.64 | 93.05 | 71.94 |
| SubResNet-12 Standard Training | 99.63 | 91.43 | 68.77 |
| ResNet-54 Stage Cost | 99.62 | 91.59 | 69.97 |
| SubResNet-12 Pruned | 99.54 | 91.02 | 66.55 |

- In homogeneous models, the gap between the pruned SubResNet-12 and the standard-trained SubResNet-12 is only **≤3.5%**.
- ResNets trained standardly struggle with lossless layer pruning.

### Tightness of the Theoretical Bound

Experiments verify that the bound provided by Theorem 3.1 is **relatively tight**.

## Highlights & Insights

- Elegantly maps the concept of stage cost from optimal control to standard ResNet architectures.
- Requires no additional parameters—intermediate output heads reuse the skip connections and the output layer.
- Theoretically proves that deeper weights asymptotically approach zero, laying the foundation for **theory-driven layer pruning**.
- Deep convergence analysis of the training dynamics provides new insights into the optimal depth of ResNets.
- Applicable to general loss functions (including standard cross-entropy).

## Limitations & Future Work

- Experiments are limited to small-scale datasets such as MNIST and CIFAR-10/100.
- Layer pruning for standard ResNets (with non-homogeneous dimensions) is not yet straightforward.
- The choice of the stage cost weight $\gamma$ lacks theoretical guidance.
- The connection with early-exit methods is not discussed in depth.
- Not validated on large-scale models (such as ImageNet + ResNet-152).

## Rating

⭐⭐⭐⭐ — Theoretically elegant, systematically applying the control theory perspective to standard ResNets, but the scale of the experiments limits its practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](../../ICLR2026/model_compression/compute-optimal_quantization-aware_training.md)
- [\[ICML 2025\] Best Subset Selection: Optimal Pursuit for Feature Selection and Elimination](best_subset_selection_optimal_pursuit_for_feature_selection_and_elimination.md)
- [\[ICML 2025\] Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective](rethinking_the_stability-plasticity_trade-off_in_continual_learning_from_an_arch.md)
- [\[ICML 2025\] Training a Generally Curious Agent (Paprika)](training_a_generally_curious_agent.md)
- [\[ICCV 2025\] Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](../../ICCV2025/model_compression/perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)

</div>

<!-- RELATED:END -->
