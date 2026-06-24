---
title: >-
  [Paper Note] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets
description: >-
  [ICLR 2026][Lipschitz Continuity] This paper systematically investigates the Lipschitz continuity of three common set aggregation functions (sum, mean, max) and attention mechanisms under three multiset distance functions, derives Lipschitz upper bounds for set neural networks, and analyzes perturbation stability and generalization under distribution shifts.
tags:
  - "ICLR 2026"
  - "Lipschitz Continuity"
  - "Set Aggregation Functions"
  - "Multiset"
  - "Robustness"
  - "Generalization"
date: 2026-05-08
content_hash: 41b9e873fafdd0a5
---

# On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets

**Conference**: ICLR 2026  
**arXiv**: [2505.24403](https://arxiv.org/abs/2505.24403)  
**Code**: None  
**Area**: Others  
**Keywords**: Lipschitz Continuity, Set Aggregation Functions, Multiset, Robustness, Generalization

## TL;DR

This paper systematically investigates the Lipschitz continuity of three common set aggregation functions (sum, mean, max) and attention mechanisms under three multiset distance functions, derives Lipschitz upper bounds for set neural networks, and analyzes perturbation stability and generalization under distribution shifts.

## Background & Motivation

The Lipschitz constant of a neural network is closely related to its robustness and generalization ability. Prior work primarily estimated Lipschitz constants for MLPs and CNNs, but in many practical scenarios (e.g., point cloud processing, natural language processing), input data consists of sets or multisets of vectors. Models like DeepSets and PointNet typically use permutation-invariant aggregation functions (sum, mean, max) to process multiset inputs. However, research on the Lipschitz continuity and stability of these aggregation functions remains insufficient.

**Core Problem**: Under which multiset distance functions are common aggregation functions Lipschitz continuous? What are their Lipschitz constants? Do neural network models based on these aggregation functions maintain Lipschitz continuity?

## Method

### Overall Architecture

This paper addresses a fundamental yet overlooked question: when the input is a **multiset** (e.g., point clouds, sets of word vectors in a document) rather than a single vector, are the permutation-invariant aggregation functions that compress the set into a vector stable? Stability is measured using Lipschitz continuity—if $\text{Lip}(f)$ is small, minor changes in the input multiset only cause small changes in the output. The challenge lies in the fact that the "magnitude of input change" depends on the choice of distance used to measure the difference between multisets, and multiple definitions of multiset distance exist.

The paper expands along a 2D grid of "Distance $\times$ Aggregation Function": first, fixing three multiset distances (Earth Mover's Distance, Hausdorff distance, Matching distance), it determines whether three standard aggregation functions (sum/mean/max) and attention aggregation are Lipschitz continuous under each distance and identifies the constants; then, it extends these findings from single aggregation functions to complete set neural networks following the composite structure "$\text{MLP}_1 \to \text{agg} \to \text{MLP}_2$." Finally, it derives downstream corollaries for perturbation stability and generalization under distribution shift.

### Key Designs

**1. Pairing Aggregation Functions and Distances: Each function is "naturally" Lipschitz for only one distance**

The core finding is a clean diagonal correspondence between the three aggregation functions and the three distances: on multisets of any size, mean is only Lipschitz continuous with respect to EMD ($L=1$), sum only with respect to Matching distance ($L=1$), and max only with respect to Hausdorff distance ($L=\sqrt{d}$, where $d$ is element dimension). Deviating from these specific pairings makes the Lipschitz constants unbounded. Intuitively, EMD measures the average cost of "moving" one multiset to another, matching the averaging behavior of the mean; Matching distance accumulates element-wise differences, aligning with the additive behavior of sum; and Hausdorff distance focuses on the worst-case element, corresponding to the extreme value behavior of max. This pairing table is the foundation for all subsequent conclusions.

**2. Extension Under Equicardinal Multisets: Relaxed pairings when size is fixed**

In practical tasks, multiset size is often fixed (e.g., point clouds sampled to $M$ points). When all multisets have the same cardinality ($M$), a deterministic relationship exists between Matching distance and EMD ($d_M = M \cdot d_{\text{EMD}}$). Consequently, aggregation functions previously continuous under only one distance become Lipschitz continuous under more distances, though the constants scale with factors related to $M$. The most useful special case is max: under the equicardinal setting, it is Lipschitz continuous under all three distances. This result relaxes the "strict diagonal" of point 1 for common fixed-size scenarios, providing additional stability guarantees for applications like point clouds.

**3. Negative Conclusions for Attention Aggregation: Non-Lipschitz under all three distances**

Unlike standard aggregations, the attention aggregation function is **not** Lipschitz continuous with respect to any of the considered distances. This is proven by constructing counterexamples: one can always find a pair of multisets differing by only one element such that the change in attention output is arbitrarily amplified relative to their distance; this holds even if softmax attention is replaced by $\ell_2$-style attention. This explains why attention mechanisms are relatively fragile under adversarial perturbations and suggests the need for additional stabilization if used.

**4. Set Neural Network Lipschitz Bounds and Generalization Corollaries**

Actual models are composite functions: $\text{MLP}_1 \text{ (element-wise)} \to \text{agg} \to \text{MLP}_2$. Using the property that Lipschitz constants multiply under function composition yields the full network upper bound:

$$\text{Lip}(\text{NN}) \le \text{Lip}(\text{MLP}_2)\cdot \text{Lip}(\text{agg})\cdot \text{Lip}(\text{MLP}_1)$$

Explicit Lipschitz upper bounds are provided for NN_mean and NN_max. NN_sum, however, may no longer be Lipschitz continuous in general cases (especially with non-zero biases)—removing biases can restore continuity. This bound is not just a robustness metric; substituting it into the domain adaptation theory of Shen et al. (2018) provides an upper bound for target error under distribution shift:

$$\varepsilon_T(h) \le \varepsilon_S(h) + 2L\cdot W_1(\mu_S, \mu_T) + \lambda$$

where $L$ is the network's Lipschitz constant and $W_1$ is the Wasserstein distance between source distribution $\mu_S$ and target distribution $\mu_T$, using the appropriate paired distance (EMD for NN_mean, Hausdorff for NN_max) as the underlying metric.

## Key Experimental Results

### Main Results

| Dataset | Model | Perturbation Type | Accuracy Drop | Description |
|--------|------|----------|-----------|------|
| ModelNet40 | NN_mean | Element Addition | 2.0% (±1.3) | Robust to single element addition |
| ModelNet40 | NN_max | Element Addition | 20.1% (±1.8) | Sensitive to single element addition |
| Polarity | NN_mean | Random Noise | 13.6% (±7.1) | Relatively sensitive to distribution noise |
| Polarity | NN_max | Random Noise | 4.8% (±3.7) | Robust to distribution noise |

### Ablation Study

| Configuration | Correlation Coefficient (S→T / T→S) | Description |
|------|----------------------|------|
| NN_mean + EMD | r=0.92 / r=0.94 | Wasserstein distance highly correlates with generalization error |
| NN_max + Hausdorff | r=0.90 / r=0.90 | Strong correlation also observed |

### Key Findings

- Each aggregation function has a natural correspondence with exactly one distance function.
- Mean aggregation is more robust to "local" perturbations (element addition); max aggregation is more robust to "global" noise (small perturbations to all elements).
- Generalization error under distribution drift is highly correlated with Wasserstein distance (r > 0.90).
- Attention mechanisms are not Lipschitz continuous under any of the three distances.

## Highlights & Insights

- Establishes the first systematic correspondence of Lipschitz continuity between set aggregation functions and multiset distance functions.
- Theoretical results show that while sum aggregation is the most expressive, it may lead to non-Lipschitz models (in the presence of non-zero biases).
- Each aggregation function exhibits complementary robustness characteristics toward different types of perturbations, providing a theoretical basis for model selection.

## Limitations & Future Work

- Only considers permutation-invariant aggregation functions, excluding permutation-equivariant message-passing architectures.
- Lipschitz upper bounds may be loose (especially for the max function); tighter bounds could be explored.
- Experiments use simple three-layer networks and do not verify applicability to more complex architectures.
- The analysis of attention mechanisms is largely negative without proposing alternatives.

## Related Work & Insights

- Complements work by Chuang & Jegelka (2022) which studied Lipschitz continuity in GNNs.
- Inspires the design choice of "which aggregation function to select" based on desired stability guarantees in set learning.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of Lipschitz properties for set aggregations, though technical methods are standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Theory and experiments validate each other, though dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-organized with rigorous theorem statements.
- Value: ⭐⭐⭐⭐ Provides an important theoretical foundation for robustness analysis in set learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)
- [\[ICLR 2026\] Beyond Uniformity: Regularizing Implicit Neural Representations through a Lipschitz Lens](beyond_uniformity_regularizing_implicit_neural_representations_through_a_lipschi.md)
- [\[ICLR 2026\] A Brain-Inspired Gating Mechanism Unlocks Robust Computation in Spiking Neural Networks](a_brain-inspired_gating_mechanism_unlocks_robust_computation_in_spiking_neural_n.md)
- [\[ICLR 2026\] Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)
- [\[ICLR 2026\] Bayesian Influence Functions for Hessian-Free Data Attribution](bayesian_influence_functions_for_hessian-free_data_attribution.md)

</div>

<!-- RELATED:END -->
