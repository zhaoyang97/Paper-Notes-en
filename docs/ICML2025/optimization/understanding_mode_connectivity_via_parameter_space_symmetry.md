---
title: >-
  [Paper Note] Understanding Mode Connectivity via Parameter Space Symmetry
description: >-
  [ICML2025][Optimization][mode connectivity] Analyzes the topological connectivity of the set of minima of neural network loss functions through continuous symmetries in parameter space (e.g., $GL_h(\mathbb{R})$). It derives that the number of connected components of the minima for linear networks is $2^{l-1}$, proves that skip connections can reduce this number, and provides explicit symmetry-induced low-loss connecting paths as well as sufficient conditions for linear mode c…
tags:
  - "ICML2025"
  - "Optimization"
  - "mode connectivity"
  - "parameter space symmetry"
  - "loss landscape"
  - "linear networks"
  - "connected components"
date: 2026-05-08
content_hash: f1ddbfc8c5a2aa45
---

# Understanding Mode Connectivity via Parameter Space Symmetry

**Conference**: ICML2025  
**arXiv**: [2505.23681](https://arxiv.org/abs/2505.23681)  
**Code**: To be confirmed  
**Area**: Others  
**Keywords**: mode connectivity, parameter space symmetry, loss landscape, linear networks, connected components

## TL;DR
Analyzes the topological connectivity of the set of minima of neural network loss functions through continuous symmetries in parameter space (e.g., $GL_h(\mathbb{R})$). It derives that the number of connected components of the minima for linear networks is $2^{l-1}$, proves that skip connections can reduce this number, and provides explicit symmetry-induced low-loss connecting paths as well as sufficient conditions for linear mode connectivity to approximately hold.

## Background & Motivation
- **Mode Connectivity Phenomenon**: Continuous paths with nearly constant training/test loss often exist between different local minima obtained by training with SGD from random initializations (Draxler et al., 2018; Garipov et al., 2018). This property finds critical applications in scenarios such as model merging, ensemble learning, and fine-tuning.
- **Lack of Theoretical Explanation**: Existing theories (e.g., dropout stability, sub-level net connectivity) rely on strong assumptions (such as over-parameterization or specific width conditions), leaving the fundamental cause of mode connectivity unclear.
- **Discrete vs. Continuous Symmetry**: A significant body of work focuses on the role of neuron permutation (discrete symmetry) in mode connectivity, such as Entezari's conjecture. However, the role of **continuous symmetry** (e.g., positive scaling invariance of ReLU) in the loss landscape structure remains under-explored.
- **Perspective of this Work**: Maps the topological properties (number of connected components) of the parameter space symmetry group to the topological properties of the set of minima, explaining the origins and failures of mode connectivity from an algebraic topology perspective.

## Method

### Core Idea: Symmetry Group Topology $\rightarrow$ Minima Topology
Leveraging a fundamental theorem in topology: continuous mappings preserve connectivity (Theorem 3.1). If a symmetry group $G$ continuously acts on the parameter space and preserves the loss, then the number of connected components of each $G$-orbit in the set of minima is $\leq$ the number of connected components of $G$.

**Key Corollary** (Corollary 3.7): If the set of minima consists of a single $G$-orbit, then:

$$|\pi_0(\text{Minimum})| \leq |\pi_0(G)|$$

### Connected Components of Minima in Linear Networks
Consider the loss function of an $l$-layer linear network:

$$L(W_1, \ldots, W_l) = \|Y - W_l \cdots W_1 X\|_2^2$$

where $X, Y \in \mathbb{R}^{h \times h}$ are of full rank. The parameter space symmetry group is $GL_h(\mathbb{R})^{l-1}$, acting by inserting an invertible matrix and its inverse between adjacent layers:

$$g \cdot (W_1, \ldots, W_l) = (g_1 W_1, g_2 W_2 g_1^{-1}, \ldots, W_l g_{l-1}^{-1})$$

**Proposition 4.1**: The set of minima $L^{-1}(0)$ is homeomorphic to $(GL_h)^{l-1}$.

**Corollary 4.2**: Since $GL_n(\mathbb{R})$ always has 2 connected components (corresponding to positive/negative determinants), the set of minima has exactly $2^{l-1}$ connected components, regardless of the network width.

### Skip Connections Reduce Connected Components
Consider a 3-layer network with residual connections:

$$L(W_3, W_2, W_1) = \|Y - W_3(W_2 W_1 X + \varepsilon X)\|_2$$

**Proposition 4.3** (1D case):
- $\varepsilon = 0$ (no skip connections): The minima set has **4** connected components.
- $\varepsilon \neq 0$ (with skip connections): The minima set has **3** connected components.

Mechanism: The additional solution set $S_0$ (a straight line) introduced by the skip connection bridges 2 out of the 4 connected components of $S_1$.

### Mode Connectivity Under Permutation
**Proposition 5.2**: When the hidden dimension satisfies $h \geq 2$, any two points in the minima set of a linear network are connected after permutation alignment. This is because for hidden dimensions $\geq 2$, there exist permutation matrices with negative determinants, which can map points in different connected components to the same component.

### Failure Modes of Linear Mode Connectivity
**Proposition 5.3**: For networks with homogeneous activation functions $\sigma(cz) = c^k \sigma(z)$, the loss barrier of linear interpolation can be unbounded. Construction method: push two points on the same orbit arbitrarily far apart using scaling symmetry.

**Proposition 5.4**: Even if neuron permutation between the last two layers is allowed, the linear interpolation error barrier can still be unbounded (under additional conditions).

### Symmetry-Induced Explicit Connecting Curves
Given the symmetry of a Lie group $G$, construct a curve connecting two minima $\boldsymbol{w}_1$ and $\boldsymbol{w}_2 = g \cdot \boldsymbol{w}_1$:

$$\gamma(t, g, \boldsymbol{w}) = \exp(t \log(g)) \cdot \boldsymbol{w}$$

The loss value at every point on this curve is equal to $L(\boldsymbol{w}_1)$, which provides an **exact equal-loss path**.

### Approximate Linear Connectivity Under Curvature Conditions
**Theorem 6.2**: If two minima are connected by a curve with curvature $\leq \kappa_{\max}$, the distance from any point on the linear interpolation to the set of minima is bounded:

$$d_{\max} = \frac{1}{\kappa_{\max}} \left(1 - \sqrt{1 - \left(\frac{\kappa_{\max} \|\boldsymbol{w}_2 - \boldsymbol{w}_1\|_2}{2}\right)^2}\right)$$

If the Lipschitz constant of the loss function is $C_L$, the loss deviation on the linear interpolation satisfies $|L(\boldsymbol{w}) - c| \leq C_L \cdot d_{\max}$.
When $\kappa_{\max} \|\boldsymbol{w}_2 - \boldsymbol{w}_1\|$ is small, $d_{\max} \approx \frac{\kappa_{\max} \|\boldsymbol{w}_2 - \boldsymbol{w}_1\|_2^2}{8}$.

## Key Experimental Results

| Experiment | Settings | Results |
|------|------|------|
| Proposition 6.1 Verification | 100 two-layer networks, sigmoid activation, random dimensions 2–100 | All experimental points satisfy $\|U\sigma(VX) - U'\sigma(V'X)\| \leq \|U\sigma(VX)\|$ |
| Approximate Symmetric Curve vs. Linear Interpolation | Two-layer leaky ReLU network, $W_1 \in \mathbb{R}^{32 \times 16}$, $X \in \mathbb{R}^{16 \times 8}$ | Losses along the symmetric curve $\gamma$ are consistently lower than those of linear interpolation (Figs. 3b–c) |
| Impact of Group Element Magnitude | Increasing the magnitude of group elements $\rightarrow$ endpoint distance increases | Linear interpolation loss barrier is higher, but symmetric curve loss remains low (Fig. 3c) |
| 1D ResNet Visualization | 3-layer 1D linear networks $\pm$ skip connection | Without skip: 4 isolated components; with skip: 3 components, validating Prop 4.3 |

## Highlights & Insights
1. **Elegant Integration of Algebraic Topology and Deep Learning**: Uses the number of connected components of $GL_n(\mathbb{R})$ (always 2) to derive the exact topological structure of the minima of linear networks of arbitrary width, yielding concise and powerful conclusions.
2. **Topological Explanation of Skip Connections**: Explores for the first time from the perspective of connected components why residual connections are effective—they "weld" together previously isolated regions of minima.
3. **Explicit Low-Loss Paths**: Constructs exact equal-loss curves using the matrix exponential $\exp(t \log(g))$, transforming empirical path searches into algebraic calculations.
4. **Counterexamples to Linear Mode Connectivity**: Rigorously proves that even within the same connected component, the loss barrier of linear interpolation can be unbounded (stemming from the non-compactness of scaling symmetry).
5. **Quantitative Relationship Between Curvature and Connectivity**: Theorem 6.2 quantitatively links curve curvature with linear interpolation error, offering a theoretical foundation to assess the feasibility of model merging in practice.

## Limitations & Future Work
1. **Exact Results Only for Linear Networks**: The complete symmetry group of non-linear networks remains undominated, meaning that the main theorems (Prop 4.1, Cor 4.2) do not directly generalize.
2. **Strong Full-Rank Assumptions**: Requires $X, Y$ to be of full rank and weights to be invertible, whereas low-rank weights are common in practical networks.
3. **Lack of Large-Scale Non-linear Network Experiments**: Numerical experiments are limited to small-scale two-layer networks and not validated on practical architectures like ResNet/Transformer.
4. **Implicit Bias of SGD Not Factored In**: The theoretical analysis covers the entire set of minima, but actual SGD only visits a small subset, meaning that the counterexample in Prop 5.3 might not be accessible by SGD.
5. **Loose Error Bounds for Approximate Symmetry (Section 6)**: The upper bound given by Proposition 6.1 is $\|U\sigma(VX)\|$, which might be much larger than the actual error in practice.

## Related Work & Insights
- **Entezari's Conjecture** (2022): Conjectures that all SGD minima are linearly connected after permutation alignment. This work verifies it on linear full-rank networks in Prop 5.2.
- **Ainsworth et al. (2023) Git Re-Basin**: Practical permutation alignment algorithms; this work provides theoretical foundations.
- **Zhao et al. (2023, 2024)**: Parametrizes minima via continuous symmetries and estimates curvature using symmetry-induced curves. This work extends their methods to mode connectivity analysis.
- **Freeman & Bruna (2017)**: Analyzes the connectivity of minima in 2-layer linear networks. This work generalizes it to an arbitrary number of layers and yields the exact number of components.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Systematically investigates mode connectivity from an algebraic topology perspective, establishing a bridge between symmetry groups and the connectivity of minima.
- Experimental Thoroughness: ⭐⭐⭐ — Theoretical-driven small-scale verification is sufficient, but lacks experiments on practical deep networks.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous mathematical formulations, clear structure, and well-balanced intuition and formalism.
- Value: ⭐⭐⭐⭐ — Provides theoretical guidance for model merging/ensembles, though practical application requires further extension to non-linear cases.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](../../ICLR2026/optimization/entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [\[ICLR 2026\] Exploring Mode Connectivity in Krylov Subspace for Domain Generalization](../../ICLR2026/optimization/exploring_mode_connectivity_in_krylov_subspace_for_domain_generalization.md)
- [\[ICML 2025\] On Understanding Attention-Based In-Context Learning for Categorical Data](on_understanding_attention-based_in-context_learning_for_categorical_data.md)
- [\[ICML 2025\] Understanding the Statistical Accuracy-Communication Trade-off in Personalized Federated Learning with Minimax Guarantees](understanding_the_statistical_accuracy-communication_trade-off_in_personalized_f.md)
- [\[ICML 2025\] Understanding Sharpness Dynamics in NN Training with a Minimalist Example: The Effects of Dataset Difficulty, Depth, Stochasticity, and More](understanding_sharpness_dynamics_in_nn_training_with_a_minimalist_example_the_ef.md)

</div>

<!-- RELATED:END -->
