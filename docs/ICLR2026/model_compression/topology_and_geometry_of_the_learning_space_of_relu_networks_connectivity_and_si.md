---
title: >-
  [Paper Note] Topology and Geometry of the Learning Space of ReLU Networks: Connectivity and Size
description: >-
  [ICLR 2026][Model Compression][ReLU networks] This work systematically investigates the connectivity and singularity of the parameter space for feedforward ReLU networks based on general Directed Acyclic Graph (DAG) architectures from the perspectives of algebraic geometry and algebraic topology. It reveals the critical roles of bottleneck nodes and balance conditions in determining the topology of the parameter space and establishes a theoretical link between singularities a…
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
content_hash: 81d6a9497c3ccd47
---

# Topology and Geometry of the Learning Space of ReLU Networks: Connectivity and Size

**Conference**: ICLR 2026  
**arXiv**: [2602.00693](https://arxiv.org/abs/2602.00693)  
**Code**: None  
**Area**: Model Theory / Neural Network Theory  
**Keywords**: ReLU networks, parameter space topology, connectivity, singularity, DAG architecture, differentiable pruning

## TL;DR

This work systematically investigates the connectivity and singularity of the parameter space for feedforward ReLU networks based on general Directed Acyclic Graph (DAG) architectures from the perspectives of algebraic geometry and algebraic topology. It reveals the critical roles of bottleneck nodes and balance conditions in determining the topology of the parameter space and establishes a theoretical link between singularities and differentiable pruning.

## Background & Motivation

**Background**: Understanding the geometric and topological properties of ReLU network parameter spaces is essential for analyzing and guiding training dynamics. Under gradient flow training, the positive homogeneity of ReLU restrictively confines reachable parameter configurations to an algebraic variety. The structure of this variety directly determines three aspects: the optimization landscape (whether gradient descent can move freely between different solutions), model equivalence (which parameter configurations represent the same function), and network compression (when certain parameters can be safely removed).

**Limitations of Prior Work**: Previous studies primarily targeted simple sequential architectures (i.e., standard multi-layer perceptrons), lacking systematic analysis of more general DAG architectures. This represents a significant gap, as modern networks—such as ResNets with skip connections and DenseNets with dense connections—are essentially DAG structures. Their parameter space topology cannot be directly covered by traditional "layer-by-layer" analysis.

**Goal**: The authors formalize the characterization of parameter space topology for general DAG architectures into two core problems: connectivity (Is the parameter space connected? If not, how are the connected components determined by the network structure?) and singularity (Where are the singular points, what is their relationship to the network structure, and can they be reached during training?).

## Method

### Overall Architecture

The paper translates the question of "what the parameter space of a ReLU network looks like" into a derivable conceptual chain. Since ReLU and its generalizations are positively homogeneous, gradient flow training exhibits rescaling symmetry (multiplying the input weights of a hidden neuron by $\alpha>0$ and its output weights by $\alpha^{-1}$ does not change the function). This symmetry yields conservation laws, keeping the "balance value" $c_v=\langle\!\langle\theta,\theta\rangle\!\rangle_v$ at each hidden neuron constant along the training trajectory. Collectively, these conservation laws form a set of quadratic equations that pin reachable parameters to their common zero set—termed the invariant set $H_G(c)$, which is an algebraic variety. The problem is thus transformed into studying the geometry of this variety: its connectivity is determined by bottleneck nodes in the DAG, and its singular points correspond to subnetworks degenerated from the DAG, which precisely characterize "pruned" states of the network. Through this chain, the authors reduce the topological characterization of any DAG architecture to the analysis of its underlying graph structure. Since these pruned states are generally unreachable during training, the paper introduces a regularization term to actively approach them.

### Key Designs

**1. Invariant Set: Pinning Training Dynamics to a Graph-Determined Algebraic Variety**

This is the starting point of the chain. ReLU and Leaky ReLU satisfy positive homogeneity $\sigma(\alpha x)=\alpha\,\sigma(x)$ for $\alpha>0$, leading to rescaling symmetry and conservation laws under gradient flow. The balance value $c_v$ of each hidden neuron $v$ (the difference between the squared norms of input and output weights, determined at initialization) remains constant. Summing these laws for all neurons yields a system of quadratic equations whose common solution set is the invariant set $H_G(c)$. Training trajectories are confined within this variety. The authors prove that any balance configuration $c$ corresponds to a non-empty $H_G(c)$ (feasible balance). Analysis thus shifts from "optimization in Euclidean space" to the geometry of this variety. By using the graph's incidence matrix, the conservation laws are formulated in a compact version applicable to any DAG.

**2. Bottleneck Nodes + Balance Constraints: Determining Parameter Space Connectivity**

This step addresses connectivity, which determines whether gradient descent can navigate between different solutions. The authors prove that connectivity is controlled by **bottleneck nodes** in the DAG: hidden neurons with an in-degree of 1 (in-bottleneck) or an out-degree of 1 (out-bottleneck) that serve as unique channels for information flow. Intuitively, a bottleneck forces information through a scalar channel whose sign cannot flip under gradient flow (as crossing zero is prohibited by the balance conditions). If the "supply and demand" (sum of balance values) of its pure ancestors or descendants becomes infeasible, the invariant set splits into disconnected components. Theorem 1 provides the necessary and sufficient condition: $H_G(c)$ is connected if and only if the balance constraints for every bottleneck node are feasible. A corollary is that architectures without bottleneck nodes are inherently connected. While standard MLPs are a special case, the novelty lies in extending this to arbitrary DAGs and using subset-specific balance constraints to characterize whether parallel paths or skip connections bypass a bottleneck and thus restore connectivity.

**3. Singular Points as Disconnected Subnetworks: Generally Unreachable During Training**

This step addresses singularity. Points on the variety where the Jacobian is not full rank are singular points. Theorem 2 states that singular points of $H_G(c)$ correspond to configurations where a set of neurons is "disconnected" (associated edge weights are simultaneously zero, equivalent to deleting those connections), causing the network to degenerate into a subnetwork. Thus, the stratification of singularities is characterized by the induced subgraphs of the DAG. A counterintuitive but crucial finding is that from a general initialization, standard gradient flow **cannot reach** these singular points (generically unreachable). These points represent the effective "pruned" states, but training will not naturally converge to them.

**4. Nuclear Norm Regularization: Turning "Training to Singularities" into Differentiable Pruning**

This addresses the reachability gap. The authors propose a nuclear norm-based regularization term to encourage convergence toward singular configurations, enabling differentiable, structure-agnostic pruning. Approaching a singularity effectively "learns away" the connections of a subnetwork, thereby compressing the model. Experiments also show that while L1 regularization does not explicitly target neuron sparsity, it empirically induces similar singular behavior, facilitating near-lossless pruning. Consequently, pruning shifts from relying on heuristic sparsity constraints to utilizing the geometric structure of the optimization space itself.

### Loss & Training

While primarily theoretical, the paper introduces a nuclear norm regularization term to push parameters toward singular points of the invariant set (disconnecting subnetworks) for differentiable pruning. L1 regularization is also reported to have similar empirical effects. Numerical experiments verify the theory by visualizing training trajectories in small ReLU networks, verifying the conservation of balance conditions, and checking if disconnected components and singular points align with the graph-theoretic characterizations.

## Key Experimental Results

### Main Results

As a theoretical contribution, numerical experiments serve as validation.

| Experimental Configuration | Key Conclusion |
|----------|----------|
| 2-layer ReLU, bottleneck width = 1 | Parameter space splits into 2 disconnected components; training cannot cross them |
| 2-layer ReLU, bottleneck width ≥ 2 | Parameter space is connected |
| DAG with skip connections | Connectivity depends on whether skip connections bypass the bottleneck |
| Singularities in various DAGs | Singular point locations align with theoretical predictions |

### Ablation Study

| Configuration | Key Metric | Description |
|------|----------|------|
| Sequential architecture | Connectivity determined by minimum width layer | Consistent with known results |
| DAG with parallel paths | Connected if no width-1 bottleneck exists | New result |
| Trajectories near zero weights | Convergence near singular points | Validates reachability via regularization |

### Key Findings

1.  **Necessary and Sufficient Conditions for Connectivity**: Parameter space connectivity is guaranteed if and only if no strict width-1 bottleneck nodes exist in the DAG (considering balance conditions).
2.  **Precise Correspondence Between Singularity and Subnetworks**: Every singular stratification corresponds exactly to the topology of an induced subgraph of the DAG.
3.  **Generic Unreachability of Singularities**: Standard gradient flow from general initialization does not reach these singular (pruned) points; active approximation via nuclear norm regularization is required.
4.  **Practical Impact on Training**: In disconnected parameter spaces, different initializations can lead the model into different (and potentially sub-optimal) connected components.

## Highlights & Insights

1.  **Symmetry between Mathematics and Network Theory**: The integration of algebraic geometry (varieties) and algebraic topology (connectivity, singularity) provides a precise and elegant theoretical framework for neural network analysis.
2.  **Generalization from Sequential to DAG**: Beyond extending existing results, it reveals that DAG topology (rather than just layer width) is the central factor in determining parameter space structure.
3.  **Geometric Perspective on Pruning**: This work is among the first to link singularities to network pruning within a rigorous mathematical framework, offering a new perspective on why pruning is effective.
4.  **Conceptual Clarity**: Notions like bottleneck nodes, balance conditions, and singular stratification align intuitively with network design practices, such as avoiding information bottlenecks and the utility of skip connections.

## Limitations & Future Work

1.  **Limited to ReLU-like Activations**: The theory heavily relies on positive homogeneity and does not apply to smooth activations like GELU or Swish.
2.  **Insufficient Validation on Large-scale Networks**: Numerical validation is limited to small networks; the guidance for architectures like ResNet-50 or Transformers remains untested.
3.  **Applicability of Balance Conditions in Practice**: Strict balance conditions hold under exact gradient flow. Techniques like SGD, learning rate scheduling, and Batch Normalization may break these conditions.
4.  **Link to Generalization**: Whether topological properties (connectivity/singularity) of the parameter space influence model generalization remains an important open question.
5.  **Exclusion of Bias Terms**: Biases are ignored in the analysis, though they break the homogeneity of the network.

## Related Work & Insights

*   **Loss Landscape Analysis**: This work shares deep connections with research on mode connectivity in loss landscapes but focuses on the parameter space itself rather than loss values.
*   **Neural Network Equivalence Classes**: Complements studies on permutation symmetry by discussing equivalence in training dynamics.
*   **Algebraic Geometry in ML**: Continues the tradition of applying algebraic tools to ML, such as Watanabe’s Singular Learning Theory.
*   **Insights**: Provides a theoretical basis for designing better initialization strategies (selecting the correct connected component) and structured pruning methods (using singularities as guides).

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ — Systematic application of algebraic geometry/topology to general DAG ReLU networks.
*   Experimental Thoroughness: ⭐⭐⭐ — Acceptable for a theoretical paper, though large-scale validation is lacking.
*   Writing Quality: ⭐⭐⭐⭐ — Mathematically rigorous and clear, though potentially challenging for readers without a mathematical background.
*   Value: ⭐⭐⭐⭐ — Significant fundamental theory contribution with practical implications for pruning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Distilling and Adapting: A Topology-Aware Framework for Zero-Shot Interaction Prediction in Multiplex Biological Networks](distilling_and_adapting_a_topology-aware_framework_for_zero-shot_interaction_pre.md)
- [\[NeurIPS 2025\] Global Minimizers of ℓp-Regularized Objectives Yield the Sparsest ReLU Neural Networks](../../NeurIPS2025/model_compression/global_minimizers_of_ellp-regularized_objectives_yield_the_sparsest_relu_neural_.md)
- [\[ICLR 2026\] MaskPro: Linear-Space Probabilistic Learning for Strict (N:M)-Sparsity on LLMs](maskpro_linear-space_probabilistic_learning_for_strict_nm-sparsity_on_llms.md)
- [\[ICLR 2026\] Navigating the Accuracy-Size Trade-Off with Flexible Model Merging](navigating_the_accuracy-size_trade-off_with_flexible_model_merging.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](a_recovery_guarantee_for_sparse_neural_networks.md)

</div>

<!-- RELATED:END -->
