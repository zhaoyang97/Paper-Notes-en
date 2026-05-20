---
title: >-
  [Paper Note] Topology and Geometry of the Learning Space of ReLU Networks: Connectivity and Size
description: >-
  [ICLR 2026][Model Compression][ReLU networks] From the perspectives of algebraic geometry and algebraic topology, this paper systematically investigates the connectivity and singularity of the parameter space of feedforw…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "ReLU networks"
  - "parameter space topology"
  - "connectivity"
  - "singularity"
  - "DAG architecture"
  - "differentiable pruning"
date: 2026-05-08
content_hash: 81228b13669b172f
---

# Topology and Geometry of the Learning Space of ReLU Networks: Connectivity and Size

**Conference**: ICLR 2026
**arXiv**: [2602.00693](https://arxiv.org/abs/2602.00693)  
**Code**: None  
**Area**: Model Theory / Neural Network Theory
**Keywords**: ReLU networks, parameter space topology, connectivity, singularity, DAG architecture, differentiable pruning

## TL;DR

From the perspectives of algebraic geometry and algebraic topology, this paper systematically investigates the connectivity and singularity of the parameter space of feedforward ReLU networks defined on general DAG architectures. It reveals the critical role of bottleneck nodes and balance conditions in determining the topological structure of the parameter space, and establishes a theoretical connection between singularities and differentiable pruning.

## Background & Motivation

### State of the Field

**Background**: Understanding the geometric and topological properties of the parameter space of ReLU networks is essential for analyzing and guiding training dynamics. During gradient-flow training, the homogeneity of the ReLU activation function constrains the parameter space to an algebraic variety. The structure of this variety directly affects:

### Limitations of Prior Work

**Limitations of Prior Work**: Optimization landscape: whether gradient descent can move freely among different solutions.

### Root Cause

**Key Challenge**: Model equivalence: which parameter configurations represent the same function.

### Starting Point

**Key Insight**: Network compression: when certain parameters can be safely removed.

Prior work has focused primarily on simple sequential architectures (i.e., standard multilayer perceptrons), leaving general DAG architectures largely unanalyzed. Understanding the parameter space topology of general DAG architectures is particularly important, as modern networks (e.g., skip connections in ResNets, dense connections in DenseNets) are inherently DAG-structured.

This paper addresses two core questions:

**Connectivity**: Is the parameter space connected? If not, what is the structure of its connected components?

**Singularity**: Where are the singular points in the parameter space, and how does their existence relate to the network architecture?

## Method

### Overall Architecture

The theoretical framework of this paper is built upon the following chain of core concepts:

1. **ReLU homogeneity → balance equations**: ReLU (and its generalizations such as Leaky ReLU) is positively homogeneous, i.e., $\text{ReLU}(\alpha x) = \alpha \cdot \text{ReLU}(x)$ holds for $\alpha > 0$. This implies that training under gradient flow produces "balance conditions"—conservation laws governing the weight norms of adjacent layers.
2. **Balance conditions → algebraic variety**: Balance conditions define a system of algebraic equations in the parameter space, restricting the parameters to the zero set of these equations (an algebraic variety).
3. **DAG topology → variety topology**: The connectivity and singularity of the algebraic variety are directly determined by the graph-theoretic properties of the underlying DAG.

### Key Designs

1. **Connectivity analysis**: A complete characterization of when the parameter space is connected.

    - Core result: The connectivity of the parameter space is determined by **bottleneck nodes** in the network DAG.
    - Bottleneck nodes: nodes that lie on every path from input to output. If a bottleneck node has width 1, the parameter space may fracture into disconnected components.
    - Intuition: A width-1 bottleneck means all information must pass through a scalar channel; the sign (positive/negative) of that scalar creates a barrier that cannot be crossed continuously.
    - Generalizes previously known results from sequential architectures to general DAGs.

2. **Balance conditions and their relation to subsets**: Fine-grained characterization of connectivity.

    - The authors introduce the notion of "balance conditions" associated with specific subsets of the network.
    - Balance conditions for different subsets correspond to different algebraic constraints, which together determine the fine topological structure of the parameter space.
    - This enables the analysis to be refined from global constraints to local ones, providing a powerful tool for understanding complex DAG architectures.

3. **Singularity analysis**: Points in the parameter space where gradients vanish or are discontinuous.

    - Core finding: Singularities are closely related to the **subnetwork topology** of the DAG.
    - When the weights of certain edges are zero (equivalent to removing the corresponding connections), the network degenerates into a subnetwork, producing singular points in the parameter space.
    - The structure of the singular point set can be characterized precisely in terms of subgraphs of the DAG.
    - These singularities are not mathematical pathologies—they correspond to valid "pruned" states of the network.

4. **Connection between singularities and differentiable pruning**: Theory guiding practice.

    - Differentiable pruning methods achieve network compression by learning weights that approach zero.
    - This paper proves that the process of approaching singular points along gradient-flow training corresponds mathematically and precisely to differentiable pruning.
    - The "reachability" of singular points indicates which pruned configurations can be attained through continuous training.
    - This provides a rigorous geometric foundation for pruning methods.

### Loss & Training

This paper is a theoretical study and does not involve the design of new loss functions. Numerical experiments serve only to validate theoretical predictions:
- Small-scale ReLU networks are used for visualization of training trajectories.
- The conservation of balance conditions during training is verified.
- The disconnectedness of the parameter space and the locations of singular points are demonstrated.

## Key Experimental Results

### Main Results

The paper is primarily a theoretical contribution; numerical experiments play a supporting and validating role.

| Experimental Setup | Key Conclusion |
|---|---|
| 2-layer ReLU, bottleneck width = 1 | Parameter space splits into 2 disconnected components; training cannot cross between them |
| 2-layer ReLU, bottleneck width ≥ 2 | Parameter space is connected |
| DAG with skip connections | Connectivity depends on whether skip connections bypass the bottleneck |
| Singular points across various DAG architectures | Singular point locations are consistent with theoretical predictions |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---|---|---|
| Sequential architecture | Connectivity determined by the minimum-width layer | Consistent with known results |
| DAG with parallel paths | Connected when no width-1 bottleneck exists | New result |
| Training trajectory with weights approaching zero | Converges near singular points | Validates reachability |

### Key Findings

1. **Necessary and sufficient condition for connectivity**: The parameter space is connected if and only if the DAG contains no strict bottleneck node of width 1 (after accounting for balance conditions).
2. **Precise correspondence between singularities and subnetworks**: Each stratum of singularities corresponds exactly to the topology of some induced subgraph of the DAG.
3. **Hierarchy of reachability**: Not all singular points are reachable via continuous gradient flow; reachability depends on the topology of the current connected component.
4. **Practical implications for training**: In a disconnected parameter space, different initializations may lead the model into different connected components, potentially of varying quality.

## Highlights & Insights

1. **A perfect marriage of mathematical depth and network theory**: Tools from algebraic geometry (algebraic varieties) and algebraic topology (connectivity, singularity) are introduced into neural network analysis, yielding a precise and elegant theory.
2. **Generalization from sequential to DAG architectures**: Beyond extending the scope of existing results, this work reveals that DAG topology—not merely layer width—plays a central role in determining the structure of the parameter space.
3. **A geometric perspective on pruning**: For the first time within a rigorous mathematical framework, singularities are connected to network pruning, providing a fundamentally new angle for understanding why pruning works.
4. **Conceptual clarity**: Although mathematically sophisticated, concepts such as bottleneck nodes, balance conditions, and singularity stratification align intuitively with practical network design considerations (e.g., avoiding information bottlenecks, the role of skip connections).

## Limitations & Future Work

1. **Restricted to ReLU-type activations**: The theory relies on positive homogeneity and does not apply to smooth activations such as GELU or Swish.
2. **Insufficient validation on large-scale networks**: Numerical verification is conducted only on small networks; the implications for practical architectures such as ResNet-50 or Transformers remain unexamined.
3. **Applicability of balance conditions to practical training**: Strict balance conditions hold only under exact gradient flow; practical techniques such as stochastic gradient descent, learning rate scheduling, and batch normalization may violate these conditions.
4. **Relationship to generalization not discussed**: Whether the topological properties of the parameter space (connectivity/singularity) affect model generalization is an important open question.
5. **Bias terms not addressed**: The analysis omits bias terms, which break homogeneity.

## Related Work & Insights

- **Loss landscape analysis**: This work has deep connections to research on connectivity in neural network loss landscapes (e.g., mode connectivity), but focuses on the parameter space itself rather than loss function values.
- **Neural network equivalence classes**: Complementary to work on permutation symmetry—the latter concerns functional equivalence, while this paper concerns equivalence of training dynamics.
- **Applications of algebraic geometry in ML**: Continues the tradition of bringing algebraic geometry tools into machine learning, as exemplified by Watanabe's singular learning theory.
- **Insights**: The results provide theoretical grounding for designing better initialization strategies (selecting the correct connected component) and structured pruning methods (leveraging singularities as guidance).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Systematic application of algebraic geometry/topology tools to ReLU networks on general DAG architectures.
- Experimental Thoroughness: ⭐⭐⭐ — Acceptable for a theoretical work, but large-scale validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ — Mathematically rigorous and clearly argued, though with a non-trivial barrier for readers without a mathematical background.
- Value: ⭐⭐⭐⭐ — An important foundational theoretical contribution; the connection to pruning carries practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Distilling and Adapting: A Topology-Aware Framework for Zero-Shot Interaction Prediction in Multiplex Biological Networks](distilling_and_adapting_a_topology-aware_framework_for_zero-shot_interaction_pre.md)
- [\[NeurIPS 2025\] Global Minimizers of ℓp-Regularized Objectives Yield the Sparsest ReLU Neural Networks](../../NeurIPS2025/model_compression/global_minimizers_of_ellp-regularized_objectives_yield_the_sparsest_relu_neural_.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](a_recovery_guarantee_for_sparse_neural_networks.md)
- [\[ICLR 2026\] Adaptive Width Neural Networks](adaptive_width_neural_networks.md)
- [\[ICLR 2026\] Boomerang Distillation Enables Zero-Shot Model Size Interpolation](boomerang_distillation_enables_zero-shot_model_size_interpolation.md)

</div>

<!-- RELATED:END -->
