---
title: >-
  [Paper Note] Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures
description: >-
  [ICLR 2026][Optimization][simplicity bias] This paper proposes a unified theoretical framework that explains the pervasive simplicity bias observed across multiple neural network architectures (fully connected…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "simplicity bias"
  - "saddle-to-saddle dynamics"
  - "neural network learning dynamics"
  - "invariant manifolds"
  - "gradient descent"
date: 2026-05-08
content_hash: 6df6015af74dec93
---

# Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures

**Conference**: ICLR 2026
**arXiv**: [2512.20607](https://arxiv.org/abs/2512.20607)
**Code**: None
**Area**: Optimization Theory / Deep Learning Theory
**Keywords**: simplicity bias, saddle-to-saddle dynamics, neural network learning dynamics, invariant manifolds, gradient descent

## TL;DR

This paper proposes a unified theoretical framework that explains the pervasive simplicity bias observed across multiple neural network architectures (fully connected, convolutional, and attention-based) through saddle-to-saddle learning dynamics — the phenomenon whereby gradient descent tends to learn simple solutions first before progressively learning more complex ones.

## Background & Motivation

Simplicity bias is a widely observed phenomenon in deep learning: neural networks tend to learn "simple" solutions first during training and then progressively learn more complex ones. This behavior has been observed across a variety of architectures:

**Phenomenon Description**:
   - Linear networks first learn low-rank solutions and then incrementally increase rank
   - ReLU networks first learn solutions with few "kinks" and then add more
   - Convolutional networks first utilize few filters and then activate more
   - Attention models first use few attention heads and then engage more

**Limitations of Prior Work**:
   - Although simplicity bias is widely reported empirically, existing theoretical analyses are fragmented — each architecture is analyzed in isolation, with no unified framework
   - Low-rank bias in linear networks has been studied extensively, but simplicity bias in ReLU networks, CNNs, and Transformers lacks theoretical explanation
   - The respective roles of data distribution and initialization in inducing simplicity bias have not been clearly distinguished

**Saddle-to-Saddle Dynamics**:
   - Gradient descent often exhibits "plateaus" during training — periods where the loss remains nearly constant before suddenly dropping sharply
   - This staircase-like learning behavior is closely related to saddle-point dynamics
   - However, a unified understanding of how such dynamics produce simplicity bias across architectures has been lacking

## Method

### Overall Architecture

The paper establishes a unified theoretical framework built on three core components:
1. **Fixed Point Analysis**: Characterizing critical points in the loss landscape
2. **Invariant Manifolds**: Constraining gradient descent trajectories to specific low-dimensional subspaces
3. **Saddle-to-Saddle Dynamics**: Describing the training process as iterative transitions between invariant manifolds

### Key Designs

1. **Unified Definition of Simplicity**:

    - Function: Provide a unified definition of "simplicity" across different architectures
    - Mechanism: **Simple = expressible with fewer hidden units**. Specifically:
        - Fully connected networks: number of hidden neurons
        - Convolutional networks: number of effective filters
        - Attention networks: number of effective attention heads
    - Simple solutions correspond to low-rank weight matrices (or sparse structures) in parameter space
    - **Design Motivation**: The notion of "simplicity" across different architectures can be unified under the concept of "number of effective hidden units"

2. **Identification of Invariant Manifolds**:

    - Function: Prove that gradient descent dynamics are characterized by a sequence of nested invariant manifolds
    - Mechanism:
        - Define the rank-$k$ invariant manifold $\mathcal{M}_k$ as the set of parameters in parameter space where the weight matrix has rank exactly $k$
        - Show that under appropriate conditions, gradient descent trajectories evolve near these manifolds
        - $\mathcal{M}_0 \subset \mathcal{M}_1 \subset \mathcal{M}_2 \subset \cdots$ form a nested structure
    - For linear networks: $\mathcal{M}_k$ corresponds to the solution space of rank-$k$ weights
    - For ReLU networks: $\mathcal{M}_k$ corresponds to the space with $k$ active neurons
    - For CNNs: $\mathcal{M}_k$ corresponds to the space with $k$ active filters
    - **Design Motivation**: Invariant manifolds are the key mathematical tool for understanding gradient descent dynamics

3. **Formalization of Saddle-to-Saddle Dynamics**:

    - Function: Prove that gradient descent produces simplicity bias through the following cyclic mechanism
    - Core dynamic process:
      a. **Intra-manifold evolution**: Gradient descent evolves near the current invariant manifold $\mathcal{M}_k$, approaching a saddle point on that manifold
      b. **Saddle-point approximation**: The trajectory lingers near the saddle point for an extended period (forming a plateau), during which the loss barely decreases
      c. **Escape along unstable direction**: The trajectory escapes along the unstable direction of the saddle point (corresponding to the largest eigenvalue)
      d. **Manifold transition**: After escaping, the trajectory enters the next, more complex invariant manifold $\mathcal{M}_{k+1}$
      e. **Repetition**: Evolution continues on $\mathcal{M}_{k+1}$...
    - **Design Motivation**: This "staircase" evolution naturally gives rise to progressive learning from simple to complex

4. **Distinguishing Data-Induced vs. Initialization-Induced Dynamics**:

    - Function: Differentiate two distinct sources of saddle-to-saddle dynamics
    - **Data-induced**:
        - Determined by the covariance structure of the data
        - Leads to low-rank weights
        - Sequentially captures principal components of the data (starting from the direction of the largest eigenvalue)
    - **Initialization-induced**:
        - Determined by the weight initialization scheme
        - Leads to sparse weights
        - Different initialization schemes activate different neurons/filters/heads
    - **Design Motivation**: Distinguishing these two mechanisms enables independent understanding and control of simplicity bias

5. **Prediction of Training Plateaus**:

    - Function: Theoretically predict the number and duration of plateaus during training
    - Core results:
        - Number of plateaus = number of effective complexity levels expressible by the network
        - Duration of each plateau depends on the spectral gap of the data (larger gap → shorter plateau) and initialization conditions
    - The shape of the learning curve can be quantitatively predicted from the data covariance spectrum and initialization scheme
    - **Design Motivation**: Advance from descriptive understanding to quantitative predictive capability

### Loss & Training

- This is a purely theoretical work analyzing the behavior of standard gradient descent under standard loss functions such as mean squared error
- No new training strategies are proposed; instead, the work provides explanations for phenomena observed during existing training procedures
- Theoretical analysis is conducted under certain simplifying assumptions (e.g., small learning rate, specific initialization distributions)

## Key Experimental Results

### Main Results

Theoretical predictions validated against experiments (synthetic and small-scale real experiments):

| Architecture | Simplicity Bias Manifestation | Theoretical Prediction | Experimental Validation |
|---|---|---|---|
| Linear network | Rank increases progressively | ✅ Predicts plateau count/duration | ✅ Consistent |
| ReLU network | Number of kinks increases progressively | ✅ Predicts activation pattern changes | ✅ Consistent |
| Convolutional network | Active filters increase progressively | ✅ Predicts filter activation order | ✅ Consistent |
| Attention network | Active heads increase progressively | ✅ Predicts head activation order | ✅ Consistent |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---|---|---|
| Varying data spectrum | Change in plateau duration | Larger spectral gap → shorter plateau |
| Varying initialization scheme | Change in sparsity pattern | Initialization determines which units activate first |
| Varying learning rate | Qualitative dynamics unchanged | Theory holds under small learning rate approximation |
| Varying hidden layer width | Change in maximum attainable complexity | Width determines the maximum expressible rank |

### Key Findings

1. **Unified mechanism across architectures**: The simplicity bias in fully connected, convolutional, and attention architectures can all be explained by the same saddle-to-saddle framework
2. **Distinct effects of data vs. initialization**: Data-induced dynamics lead to low-rank solutions, while initialization-induced dynamics lead to sparsity — these two effects are independently separable
3. **Plateaus are predictable**: The covariance spectrum of the data and the initialization scheme can quantitatively predict the staircase shape of the learning curve
4. **Learning from simple to complex is an intrinsic property of gradient descent**: No specially designed regularization or training strategies are required

## Highlights & Insights

1. **Elegance of the unified framework**: A single mathematical tool (invariant manifolds + saddle-point dynamics) explains a universal phenomenon across architectures, rather than constructing separate models for each
2. **Precise definition of "simplicity"**: The vague notion of "simple" is formalized as "number of effective hidden units," enabling meaningful comparisons across architectures
3. **Clarity in causal separation**: Decomposing the sources of simplicity bias into data effects (low-rank) and initialization effects (sparsity) has practical implications — for instance, simplicity bias can be controlled by adjusting the initialization scheme
4. **Quantitative predictive power**: The framework not only explains *why* simplicity bias arises, but also predicts *when* and *for how long* — predictive power is its core contribution
5. **Practical implications**: Understanding the mechanism of simplicity bias opens the door to designing smarter training strategies — for example, adaptive learning rates to accelerate escaping plateaus

## Limitations & Future Work

1. **Simplifying assumptions**:
    - Theoretical analysis is conducted in the small learning rate, continuous-time limit; the discrete large learning rate regime is more complex
    - Analysis is restricted to certain network structures (e.g., single hidden layer or shallow networks)
    - Loss functions are limited to mean squared error; cross-entropy and other losses are not fully covered

2. **Scale limitations**:
    - Experimental validation is primarily conducted on small-scale networks and synthetic data
    - Whether saddle-to-saddle dynamics remains the primary explanation for simplicity bias in GPT-scale models remains to be verified

3. **Gap from practical training configurations**:
    - Practical training employs Adam, learning rate warmup, Batch Normalization, and other techniques that may alter the dynamics
    - The gradient flow assumed in theory deviates under the noise of SGD

4. **Nonlinear interactions**:
    - Analysis of attention mechanisms may oversimplify the nonlinear effects of softmax
    - Analysis of convolutional networks assumes specific filter initialization conditions

5. **Future directions**:
    - Extend the framework to residual connections (ResNet) and full Transformer architectures
    - Quantitatively study the impact of simplicity bias on generalization performance
    - Connect simplicity bias to other training phenomena such as double descent and grokking

## Related Work & Insights

- **Linear network theory**: The seminal work of Saxe et al. (2014, 2019) on learning dynamics in linear networks is the direct foundation of this paper
- **Empirical simplicity bias**: Experimental observations of simplicity bias, e.g., Shah et al. (2020)
- **Loss landscape analysis**: Saddle-point analysis by Choromanska et al. (2015) and visualization by Li et al. (2018)
- **Implicit regularization**: Theoretical work on gradient descent implicitly favoring low-rank solutions, e.g., Gunasekar et al. (2017) and Arora et al. (2019)
- **Insights**: The saddle-to-saddle framework may provide a theoretical foundation for understanding curriculum learning — which is essentially the process of artificially accelerating simplicity bias

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First unified theoretical framework for simplicity bias across architectures; outstanding contribution
- **Experimental Thoroughness**: ⭐⭐⭐ — Primarily theoretical; experiments are largely confirmatory and limited in scale
- **Writing Quality**: ⭐⭐⭐⭐ — Achieves a good balance between theoretical depth and readability, aided by illustrative figures
- **Value**: ⭐⭐⭐⭐⭐ — Significantly advances foundational understanding of deep learning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Escaping Saddle Points without Lipschitz Smoothness: The Power of Nonlinear Preconditioning](../../NeurIPS2025/optimization/escaping_saddle_points_without_lipschitz_smoothness_the_power_of_nonlinear_preco.md)
- [\[NeurIPS 2025\] A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](../../NeurIPS2025/optimization/a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)
- [\[ICLR 2026\] COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics](cold-steer_steering_large_language_models_via_in-context_one-step_learning_dynam.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)
- [\[ICLR 2026\] Optimizer Choice Matters for the Emergence of Neural Collapse](optimizer_choice_matters_for_the_emergence_of_neural_collapse.md)

</div>

<!-- RELATED:END -->
