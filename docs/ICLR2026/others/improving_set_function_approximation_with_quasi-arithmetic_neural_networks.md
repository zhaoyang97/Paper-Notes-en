---
title: >-
  [Paper Note] Improving Set Function Approximation with Quasi-Arithmetic Neural Networks
description: >-
  [ICLR 2026][set functions] This paper proposes QUANN (Quasi-Arithmetic Neural Networks), which employs invertible neural networks to implement a learnable Kolmogorov mean as the pooling operation. It is the first to real…
tags:
  - "ICLR 2026"
  - "set functions"
  - "Kolmogorov mean"
  - "invertible networks"
  - "learnable pooling"
  - "permutation invariance"
date: 2026-05-08
content_hash: ffe10174e647c230
---

# Improving Set Function Approximation with Quasi-Arithmetic Neural Networks

**Conference**: ICLR 2026
**arXiv**: [2602.04941](https://arxiv.org/abs/2602.04941)  
**Code**: None  
**Area**: Deep Learning Theory / Set Functions
**Keywords**: set functions, Kolmogorov mean, invertible networks, learnable pooling, permutation invariance

## TL;DR
This paper proposes QUANN (Quasi-Arithmetic Neural Networks), which employs invertible neural networks to implement a learnable Kolmogorov mean as the pooling operation. It is the first to realize a machine-learning instantiation of generalized measures of central tendency. QUANN serves as a universal approximator for mean-decomposable set functions, and the learned embeddings exhibit stronger cross-task transferability.

## Background & Motivation

### State of the Field

**Background**: Set function learning requires permutation invariance. DeepSets uses sum pooling and PointNet uses max pooling — both are fixed, non-trainable pooling operations that shift the approximation burden onto the encoder and estimator.

**Limitations of Prior Work**: (1) Fixed pooling forces the encoder to learn embeddings that simultaneously accommodate the downstream task and the specific pooling operation, limiting embedding transferability. (2) Sum and max are extreme special cases of the Kolmogorov mean, leaving a wide range of intermediate forms (geometric mean, harmonic mean, etc.) unexploited. (3) Existing learnable pooling methods are either complex and difficult to use, or limited in expressiveness (e.g., Power DeepSets learns only a single exponent).

**Key Challenge**: There is a need for a learnable pooling operation that is theoretically grounded, straightforward to implement, and sufficiently expressive.

**Key Insight**: The Kolmogorov mean $M_f = f^{-1}(\frac{1}{n}\sum_i f(x_i))$ unifies various means through the choice of different invertible functions $f$. Implementing $f$ with an invertible neural network yields a learnable generalized measure of central tendency.

## Method

### Overall Architecture
QUANN: $\hat{F}(X) = \rho(\psi^{-1}(\frac{1}{|P_k(X)|}\sum_{\pi} \psi(\phi(\pi))))$, where $\phi$ is the encoder, $\psi$ is an invertible neural network (the generating function), and $\rho$ is the estimator.

### Key Designs

1. **Neuralized Kolmogorov Mean (NKM)**:

    - **Function**: Implements the generating function of the Kolmogorov mean via an invertible neural network $\psi$.
    - **Mechanism**: $M_\psi(X) = \psi^{-1}(\frac{1}{n}\sum_{i=1}^n \psi(x_i))$, with a RevNet serving as $\psi$.
    - **Design Motivation**: NKM is the first learnable implementation of the Kolmogorov mean — the form of $\psi$ determines the mean type (linear = arithmetic mean, log = geometric mean, power = power mean), and invertible networks provide sufficient expressiveness.

2. **Theoretical Guarantees**:

    - QUANN-1 is a universal approximator for mean-decomposable set functions.
    - QUANN-2, which accounts for pairwise element interactions, is strictly more powerful.
    - Under mild conditions, it can also approximate max-decomposable functions.

3. **Embedding Quality**:

    - The invertible $\psi$ allows NKM to preserve the structural information of the input, enabling the encoder to learn more general-purpose embeddings.
    - Empirically, QUANN encoders transfer effectively to non-set tasks as well.

### Loss & Training
- Standard supervised learning with end-to-end training.
- Invertibility of $\psi$ is achieved via the RevNet architecture.

## Key Experimental Results

### Set Function Tasks


### Main Results

| Method | Set Classification | Set Regression | Point Cloud Classification | Average |
|---|---|---|---|---|
| DeepSets (sum) | Baseline | Baseline | Baseline | Baseline |
| PointNet (max) | Medium | Medium | Medium | Medium |
| HPDS (power mean) | Good | Good | Good | Good |
| **QUANN-1** | **Best** | **Best** | **Best** | **SOTA** |

### Encoder Transferability


### Ablation Study

| Configuration | Performance on Non-Set Tasks |
|---|---|
| DeepSets encoder | Poor → embeddings tightly coupled to sum pooling |
| PointNet encoder | Poor → embeddings tightly coupled to max pooling |
| **QUANN encoder** | **Good → embeddings are general-purpose** |

### Key Findings
- The pooling form learned by NKM lies between sum and max, adapting automatically to the task.
- The invertible $\psi$ ensures no information loss, so the encoder need not "compensate" for a specific pooling operation.
- QUANN surpasses prior SOTA across all benchmarks, including tasks requiring higher-order interactions.

## Highlights & Insights
- **Neuralization of the Kolmogorov Mean**: This work is the first to combine a century-old mathematical concept (quasi-arithmetic means) with modern deep learning. Using invertible networks as the generating function is both theoretically elegant and practically effective.
- **Decoupling Encoder from Pooling**: Fixed pooling forces the encoder to "adapt" to the pooling operation, yielding non-transferable embeddings. Learnable pooling allows the encoder to focus on learning good representations while the pooling adapts automatically, enhancing embedding transferability.
- **Dual Value of Invertibility**: (1) It guarantees the Kolmogorov mean is well-defined (requiring an invertible generating function); (2) it preserves information, unlike max pooling which discards it.

## Limitations & Future Work
- RevNet introduces additional computational overhead, though invertibility does eliminate the need to store intermediate activations.
- The quadratic complexity of QUANN-2 over element pairs limits its scalability to large sets.
- Experiments are conducted only on finite sets; the case of function sets (continuous sets) is not considered.
- Comparisons with non-Janossy methods such as Slot Attention are insufficient.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The neuralization of the Kolmogorov mean is an elegant theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers diverse tasks, transferability evaluation, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ The theoretical framework is clear and the unified tables are easy to interpret.
- Value: ⭐⭐⭐⭐ Provides a foundational improvement to set function learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)
- [\[ICLR 2026\] Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)
- [\[ICLR 2026\] Consistent Low-Rank Approximation](consistent_low-rank_approximation.md)
- [\[ICLR 2026\] Probabilistic Kernel Function for Fast Angle Testing](probabilistic_kernel_function_for_fast_angle_testing.md)
- [\[ICLR 2026\] Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)

</div>

<!-- RELATED:END -->
