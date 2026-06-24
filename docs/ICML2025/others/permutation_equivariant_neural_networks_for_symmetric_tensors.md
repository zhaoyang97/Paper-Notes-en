---
title: >-
  [Paper Note] Permutation Equivariant Neural Networks for Symmetric Tensors
description: >-
  [ICML 2025][permutation equivariance] This work presents the first study on permutation equivariant neural networks with symmetric tensors as inputs. It provides two complete characterizations of all linear permutation equivariant functions between symmetric power spaces, and experimentally demonstrates that this method significantly outperforms standard MLPs in terms of data efficiency and generalization capability.
tags:
  - "ICML 2025"
  - "permutation equivariance"
  - "symmetric tensors"
  - "representation theory"
  - "data efficiency"
  - "neural networks"
date: 2026-05-08
content_hash: adf0063aeb91458e
---

# Permutation Equivariant Neural Networks for Symmetric Tensors

**Conference**: ICML 2025  
**arXiv**: [2503.11276](https://arxiv.org/abs/2503.11276)  
**Code**: None  
**Area**: Equivariant Neural Networks / Representation Theory  
**Keywords**: permutation equivariance, symmetric tensors, representation theory, data efficiency, neural networks

## TL;DR
This work presents the first study on permutation equivariant neural networks with symmetric tensors as inputs. It provides two complete characterizations of all linear permutation equivariant functions between symmetric power spaces, and experimentally demonstrates that this method significantly outperforms standard MLPs in terms of data efficiency and generalization capability.

## Background & Motivation
**Background**: Permutation equivariance is an important inductive bias in deep learning. Methods such as DeepSets and Janossy Pooling have been widely used for set and graph data. Symmetric tensors naturally arise in statistics (moment tensors), machine learning (kernel methods), and graph theory (subgraph counting).

**Limitations of Prior Work**: Existing permutation equivariant models primarily take vector sets or graphs as inputs, leaving symmetric tensor inputs unexplored. Prior work on symmetric tensors mainly focuses on Euclidean groups (such as $SE(3)$ equivariance) rather than permutation groups.

**Key Challenge**: Symmetric tensors are crucial in physics, chemistry, and materials science, but there is a lack of neural network tools tailored to their permutation symmetries.

**Goal**: Fully characterize all linear permutation equivariant functions between the symmetric power spaces of $\mathbb{R}^n$ and construct neural networks accordingly.

**Key Insight**: Leverage Schur-Weyl duality and irreducible representations of the symmetric group from representation theory.

**Core Idea**: Construct parameter-efficient networks with strict equivariance guarantees by mathematically describing the complete parameter space of equivariant linear layers.

## Method

### Overall Architecture
Input: Symmetric tensor $T \in \text{Sym}^k(\mathbb{R}^n)$ ($k$-th symmetric power space of $\mathbb{R}^n$)  
Output: Symmetric tensor $T' \in \text{Sym}^{k'}(\mathbb{R}^n)$

### Key Designs

1. **First Characterization of Equivariant Linear Layers (Contraction-Symmetrization Based)**:

    - **Function**: Describe all linear equivariant mappings from $\text{Sym}^k(\mathbb{R}^n)$ to $\text{Sym}^{k'}(\mathbb{R}^n)$.
    - **Mechanism**: Any such mapping can be factorized into: (a) choosing pairs of indices to contract, (b) performing tensor contraction (trace), (c) taking the tensor product with identity tensors to increase the order, and (d) symmetrizing the result. This is equivalent to a linear combination of several basic operations.
    - **Design Motivation**: Provides a constructive understanding, directly showing how to parameterize the equivariant layers.

2. **Second Characterization of Equivariant Linear Layers (Irreducible Decomposition Based)**:

    - **Function**: Provide another complete description using Schur-Weyl duality.
    - **Mechanism**: The decomposition of symmetric power spaces is a direct sum of irreducible representations of $S_n$ (the symmetric group). Equivariant mappings are determined by the mappings between each irreducible component (Schur's Lemma guarantees these mappings are scalar multiplications).
    - **Design Motivation**: Provides a more elegant mathematical understanding and a potentially more efficient implementation.

3. **Equivariant Neural Network Architecture**:

    - **Function**: Stack equivariant linear layers + non-linear activations to construct deep networks.
    - **Mechanism**: Equivariant linear layers replace standard linear layers. The number of parameters depends only on the order of the tensors rather than the size of $n$, resulting in very few parameters. The network is defined as $f = \sigma \circ L_d \circ \cdots \circ \sigma \circ L_1$, where each $L_i$ is an equivariant linear layer.
    - **Design Motivation**: Compared to standard MLPs requiring $O(n^k \times n^{k'})$ parameters, the equivariant layer only requires $O(\text{poly}(k, k'))$ parameters, achieving extremely high data efficiency.

### Loss & Training
Standard supervised learning loss (MSE or cross-entropy) with the Adam optimizer. The core does not lie in the loss design, but rather in the mathematical structure of the network layers.

## Key Experimental Results

### Main Results

| Task | Metric | Equivariant Network | Standard MLP | Data Efficiency Gain |
|---|---|---|---|---|
| Moment Tensor Regression (n=10) | MSE | **0.003** | 0.145 | ~48x |
| Moment Tensor Regression (n=20) | MSE | **0.005** | 0.312 | ~62x |
| Subgraph Counting (n=15) | MAE | **0.02** | 0.18 | ~9x |
| Subgraph Counting (n=20) | MAE | **0.03** | 0.35 | ~12x |

### Ablation Study (Generalization to different sizes)

| Train n | Test n | Equivariant Network MSE | MLP MSE | Description |
|---|---|---|---|---|
| 10 | 10 | 0.003 | 0.145 | In-distribution |
| 10 | 15 | **0.008** | N/A (different dimensions) | Equivariant Net generalizes |
| 10 | 20 | **0.015** | N/A | Cross-scale generalization |
| 15 | 20 | **0.006** | N/A | Closer generalization |

### Key Findings
- The equivariant network significantly outperforms standard MLPs on both tasks while requiring an order of magnitude less training data.
- The equivariant network can generalize to tensor sizes unseen during training (as parameters do not depend on $n$), which is completely impossible for standard MLPs.
- The parameter count is extremely small (dozens vs. tens of thousands in MLPs), and training is also faster.

## Highlights & Insights
- Complete mathematical characterization: Two independent descriptions of equivariant mappings validate each other, ensuring completeness.
- High parameter efficiency: The number of parameters is independent of $n$ and only depends on the order of the tensors.
- Cross-scale generalization is a unique advantage: Models trained on $n=10$ can be directly applied to data with $n=20$.

## Limitations & Future Work
- Currently, only linear equivariant layers are discussed, and the guarantees of equivariance for the non-linear parts require further research.
- The computational complexity of high-order symmetric tensors may limit practical applications.
- Experiments are only validated on two tasks, and evaluations in more application scenarios remain to be improved.

## Related Work & Insights
- Connection to DeepSets (Zaheer et al.): Equivariance on symmetric tensors is a high-order generalization of DeepSets.
- Complementary to $SE(3)$ equivariant networks (e3nn, MACE, etc.): Focuses on different symmetry groups.
- Symmetric tensors are ubiquitous in quantum chemistry and materials science, suggesting broad potential applications.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to address the permutation equivariance problem on symmetric tensors.
- Experimental Thoroughness: ⭐⭐⭐ Only two tasks; could be more extensive.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous but potentially difficult for readers without a mathematical background.
- Value: ⭐⭐⭐⭐ Fills a theoretical gap, and the cross-scale generalization property is highly promising.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] The Price of Freedom: Exploring Expressivity and Runtime Tradeoffs in Equivariant Networks](the_price_of_freedom_exploring_expressivity_and_runtime_tradeoffs_in_equivariant.md)
- [\[ICML 2025\] GLGENN: A Novel Parameter-Light Equivariant Neural Networks Architecture Based on Clifford Geometric Algebras](glgenn_a_novel_parameter-light_equivariant_neural_networks_architecture_based_on.md)
- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](../../NeurIPS2025/others/on_universality_classes_of_equivariant_networks.md)
- [\[ICML 2026\] Identifiable Equivariant Networks are Layerwise Equivariant](../../ICML2026/others/identifiable_equivariant_networks_are_layerwise_equivariant.md)
- [\[ICML 2025\] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks](improved_exploration_in_gflownets_via_enhanced_epistemic_neural_networks.md)

</div>

<!-- RELATED:END -->
