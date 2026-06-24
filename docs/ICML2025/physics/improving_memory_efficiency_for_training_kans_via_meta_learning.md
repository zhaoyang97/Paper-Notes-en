---
title: >-
  [Paper Note] Improving Memory Efficiency for Training KANs via Meta Learning
description: >-
  [ICML2025][Physics & Scientific Computing][Kolmogorov-Arnold Networks] Proposes MetaKANs, which use a small meta-learner to generate the parameters of all learnable activation functions in KANs. This compresses the trainable parameter count from $(G+k+1)$ times that of KANs to a level close to MLPs (approximately 1/3 to 1/9), while maintaining or even improving performance.
tags:
  - "ICML2025"
  - "Physics & Scientific Computing"
  - "Kolmogorov-Arnold Networks"
  - "Meta Learning"
  - "parameter efficiency"
  - "HyperNetwork"
  - "learnable activation functions"
date: 2026-05-08
content_hash: 345bbd91e84ca31a
---

# Improving Memory Efficiency for Training KANs via Meta Learning

**Conference**: ICML2025  
**arXiv**: [2506.07549](https://arxiv.org/abs/2506.07549)  
**Code**: [GitHub](https://github.com/Murphyzc/MetaKAN)  
**Area**: KAN Training / Meta Learning / Memory Efficiency  
**Keywords**: Kolmogorov-Arnold Networks, Meta Learning, parameter efficiency, HyperNetwork, learnable activation functions

## TL;DR

Proposes MetaKANs, which use a small meta-learner to generate the parameters of all learnable activation functions in KANs. This compresses the trainable parameter count from $(G+k+1)$ times that of KANs to a level close to MLPs (approximately 1/3 to 1/9), while maintaining or even improving performance.

## Background & Motivation

**KAN's Parameter Expansion Problem**: KANs (Kolmogorov-Arnold Networks) replace traditional fixed activation functions in MLPs with univariate functions parameterized by learnable B-splines, yielding better interpretability and function approximation capabilities. However, each activation function requires $G+k+1$ trainable coefficients ($G$ is the grid size, $k$ is the spline order), resulting in KANs having $(G+k+1)$ times the parameter count of MLPs under the same structure. For example, when $G=5, k=3$, the parameter size is 9 times that of MLPs, which severely limits the scalability of KANs on large-scale tasks.

**Limitations of Prior KAN Variants**: Although variants like ChebyshevKAN, WavKAN, and FastKAN improve basis function selection and computational efficiency, the parameter expansion problem caused by learnable activation functions remains unresolved.

**Key Insight**: All activation functions in KANs belong to the same function family $\mathcal{F}$ and share the same parameter generation rules. Learning these parameters can be viewed as a multi-task learning problem, where each activation function represents a "task," and the weight generation of all tasks follows a common rule. This provides theoretical motivation for using a small network to generate the weights in a unified manner.

## Method

### Review of KAN Parameter Structure

In standard KANs, each activation function is parameterized as:

$$\phi(t; \mathbf{w}) = w_b \cdot \text{SiLU}(t) + \sum_{i=1}^{G+k} c_i B_i(t) = \mathbf{w}^\top \mathbf{B}(t)$$

where $\mathbf{w} = [w_b, c_1, \ldots, c_{G+k}]^\top \in \mathbb{R}^{G+k+1}$.

For a KAN structured as $[n_0, n_1, \ldots, n_L]$, the total number of parameters is:

$$|\mathcal{W}| = \sum_{l=0}^{L-1}(n_l \times n_{l+1}) \times (G+k+1)$$

### MetaKAN Framework

**Core Idea**: Use a small MLP (meta-learner) $M_\theta$ to generate activation function weights from a learnable prompt $z \in \mathbb{R}$:

$$M_\theta: \mathbb{R} \to \mathbb{R}^{G+k+1}, \quad z \mapsto \mathbf{w}$$

Each activation function is assigned a scalar prompt $z_\alpha^{(l)}$ as a unique identifier. The meta-learner (a two-layer MLP with hidden dimension $d_{\text{hidden}}$) learns the mapping rule from prompt to weight. The activation function becomes:

$$\phi(t; z, \theta) = M_\theta(z)^\top \mathbf{B}(t)$$

**Parameter Comparison**:

| Model | Parameter Count |
|------|--------|
| MLP | $\sum_{l}(n_l \times n_{l+1})$ |
| KAN | $\sum_{l}(n_l \times n_{l+1}) \times (G+k+1)$ |
| MetaKAN | $\sum_{l}(n_l \times n_{l+1}) + C \times (d_{\text{hidden}}+1) \times (G+k+1)$ |

where $C$ is the number of meta-learners ($C=1$ for shallow networks). When the network is sufficiently large, the parameter size of MetaKAN approaches that of MLP.

### Extension for Deep KANs: Layer Clustering Strategy

Since weight generation rules vary across different layers (due to different input distributions), a single meta-learner is insufficient. The proposed solution:

1. Group $L$ layers into $C$ clusters $\{L_1, \ldots, L_C\}$ using K-Means based on the number of output channels per layer.
2. Assign an independent meta-learner $M_{\theta_{(c)}}$ to each cluster.
3. Layers within the same cluster share one meta-learner.

$$\mathbf{w}_\alpha^{(l)} = M_{\theta_{(c)}}(z_\alpha^{(l)}), \quad \forall l \in L_c$$

### Loss & Training

End-to-end optimization of prompt $\mathcal{Z}$ and meta-learner parameters $\Theta$:

$$\mathcal{Z}^\star, \Theta^\star = \arg\min_{\mathcal{Z}, \Theta} \mathbb{E}_{\mathbf{x}} \left[ \ell(\text{MetaKAN}(\mathbf{x}; \mathcal{Z}, \Theta), f(\mathbf{x})) \right]$$

MSE is used for regression tasks, and cross-entropy for classification tasks.

### Model Agnosticism

The MetaKAN framework can be directly applied to various KAN variants. It has been validated on: MetaFastKAN (RBF-based), MetaWavKAN (wavelet-based), MetaConvKAN (convolutional), MetaKALNConv, and MetaKAGNConv. The parameter compression ratio is approximately $1/\text{dim}(\mathbf{w})$.

## Key Experimental Results

### Symbolic Regression (Feynman Dataset, G=5)

| Function | KAN MSE | KAN #Param | MetaKAN MSE | MetaKAN #Param |
|------|---------|------------|-------------|----------------|
| I.6.20a | 5.94e-4 | 58 | **3.88e-4** | 218 |
| I.8.4 | 1.30e-2 | 979 | **3.76e-3** | 461 |
| I.9.18 | 2.39e-3 | 781 | **1.70e-3** | 993 |
| I.12.5 | 1.32e-3 | 57 | **1.16e-4** | 30 |
| I.15.3x | 1.57e-2 | 223 | **5.38e-3** | 80 |

Under the $G=5$ setting, MetaKAN outperforms KAN in 16 out of 18 functions.

### Image Classification (ConvKAN, 4 layers)

| Model | MNIST Acc | #Param | CIFAR-10 Acc | #Param |
|------|-----------|--------|--------------|--------|
| KANConv | 98.43 | 3.49M | 41.92 | 3.49M |
| MetaKANConv | 96.03 | **391K** | **45.97** | **393K** |
| FastKANConv | 99.36 | 3.49M | 68.12 | 3.49M |
| MetaFastKANConv | 98.54 | **392K** | 66.69 | **392K** |
| KAGNConv | 99.15 | 1.94M | 72.08 | 1.94M |
| MetaKAGNConv | **99.21** | **391K** | — | — |

MetaKAGNConv outperforms KAGNConv on MNIST with approximately 1/5 of the parameters. MetaKANConv outperforms KANConv on CIFAR-10 with approximately 1/9 of the parameters (45.97 vs 41.92).

### Summary of Parameter Efficiency

- Parameter compression ratio: approximately **1/3 to 1/9** (depending on $G+k+1$ and the network scale).
- On most benchmarks, MetaKAN achieves comparable or superior performance with significantly fewer parameters.
- It is equally effective on PDE solving tasks.

## Highlights & Insights

1. **Clear Theoretical Motivation**: Starting from the mechanism of KAN, weight learning is formulated as a multi-task learning problem. The meta-learner learns shared weight generation rules, which is more efficient than optimizing the parameters of each activation function individually.
2. **Model-Agnostic**: They can be seamlessly integrated with variants like KAN, FastKAN, WavKAN, ConvKAN, KALN, and KAGN, showcasing strong generality.
3. **Parameters Approaching MLP**: When the network scale is sufficiently large, the parameter size of MetaKAN approaches that of MLP, fundamentally addressing the training cost gap between KAN and MLP noted in the original KAN paper.
4. **Clever Prompt Design**: Each activation function is identified by a scalar prompt, which is simple yet effective, echoing the in-context learning concept in LLMs.
5. **Layer Clustering Strategy**: Avoids the parameter overhead of assigning an independent meta-learner to each layer, using K-Means clustering to balance representational capacity and efficiency.

## Limitations & Future Work

1. **Small Networks May Not Save Parameters**: When the KAN is very small (with few total activation functions), the fixed overhead of the meta-learner might lead MetaKAN to have more parameters instead (as shown in some small configurations in Table 2).
2. **No Speedup during Inference**: MetaKAN still needs to generate weights via the meta-learner during forward propagation, meaning inference latency might increase.
3. **Fixed Meta-Learner Structure**: Only a two-layer MLP was considered as the meta-learner; more complex or adaptive architectures have not been explored.
4. **Scalar Prompt Dimension**: Each activation function is identified by only a 1D prompt, which may lack representational capacity for extremely deep networks (although ablation studies show limited improvement with higher dimensions).
5. **Insufficient Validation on Large-scale Vision Tasks**: Image experiments have only been validated at the MNIST/CIFAR scale, and have not been tested on large-scale datasets such as ImageNet.

## Related Work & Insights

- **HyperNetworks** (Ha et al., 2017): Pioneer work using auxiliary networks to generate weights for the main network, but employing heuristic strategies; MetaKAN is designed specifically based on the mechanism of KAN, making it more targeted.
- **Original KAN Paper** (Liu et al., 2024): Explicitly points out the high training cost of KANs. MetaKAN addresses this issue from the perspective of parameter efficiency.
- **Meta-learning Paradigm**: Treats the weight learning of each activation function as a "task," utilizing a shared meta-learner to learn cross-task weight generation rules.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of reformulating KAN weight learning as a multi-task learning problem and generating weights using a meta-learner is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive validation on multiple tasks, including symbolic regression, PDE, and classification, covering various KAN variants with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivations and thorough parameter analysis.
- Value: ⭐⭐⭐⭐ — Highly practical, effectively lowering the barrier for training KANs and promoting the application of KANs to larger-scale problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](../../NeurIPS2025/physics/neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)
- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](../../NeurIPS2025/physics/enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)
- [\[ICLR 2026\] Efficient Regression-based Training of Normalizing Flows for Boltzmann Generators](../../ICLR2026/physics/efficient_regression-based_training_of_normalizing_flows_for_boltzmann_generator.md)
- [\[ICML 2025\] PAC Learning with Improvements](pac_learning_with_improvements.md)
- [\[ICML 2025\] Rethink the Role of Deep Learning towards Large-scale Quantum Systems](rethink_the_role_of_deep_learning_towards_large-scale_quantum_systems.md)

</div>

<!-- RELATED:END -->
