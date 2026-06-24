---
title: >-
  [Paper Note] Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems
description: >-
  [ICML 2025][Physics & Scientific Computing][Hierarchical Transformer] The authors propose Erwin, a Transformer architecture based on a hierarchical ball tree structure. By restricting attention computation within fixed-size local spherical regions, Erwin achieves linear time complexity. Meanwhile, it captures multi-scale features through progressive coarsening/refinement and cross-ball interaction mechanisms, achieving SOTA performance in multiple domains including cosmology…
tags:
  - "ICML 2025"
  - "Physics & Scientific Computing"
  - "Hierarchical Transformer"
  - "Ball Tree"
  - "Linear Attention"
  - "Large-scale Physical Systems"
  - "Multi-scale Modeling"
date: 2026-05-08
content_hash: 10601f3099c7bc10
---

# Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems

**Conference**: ICML 2025  
**arXiv**: [2502.17019](https://arxiv.org/abs/2502.17019)  
**Code**: None  
**Area**: Human Understanding  
**Keywords**: Hierarchical Transformer, Ball Tree, Linear Attention, Large-scale Physical Systems, Multi-scale Modeling

## TL;DR

The authors propose Erwin, a Transformer architecture based on a hierarchical ball tree structure. By restricting attention computation within fixed-size local spherical regions, Erwin achieves linear time complexity. Meanwhile, it captures multi-scale features through progressive coarsening/refinement and cross-ball interaction mechanisms, achieving SOTA performance in multiple domains including cosmology, molecular dynamics, PDE solving, and particle fluid dynamics.

## Background & Motivation

Large-scale physical systems (e.g., molecular dynamics, weather forecasting, cosmology simulations) are often defined on irregular grids containing thousands to millions of nodes. Existing deep learning methods face several critical challenges:

**Quadratic Complexity Bottleneck**: Standard self-attention computes all pairwise interactions with $O(N^2)$ complexity, making the computational cost prohibitive when the number of nodes reaches tens of thousands.

**Scale Gap**: In computational chemistry, models are typically trained and validated on molecules with dozens of atoms. However, real-world molecular dynamics simulations often involve thousands of atoms. Models validated on small scales lack the necessary architectural components to scale up.

**Long-range Interactions and Multi-scale Coupling**: Physical systems exhibit long-range effects due to slowly decaying potential functions, as well as coupling across different scales. This requires simultaneously capturing both local fine-grained features and global characteristics.

**Irregular Geometry**: Point clouds and non-uniform grids cannot directly apply patch strategies designed for the image domain.

In computational multi-body physics, sub-quadratic tree-based algorithms (such as Barnes-Hut and the Fast Multipole Method) have long been developed. Their core intuition is that distant particles can be approximated via mean-field effects. However, these methods align poorly with GPU architectures, limiting their application in deep learning. Erwin aims to combine the efficiency of tree-based methods with the representational power of attention mechanisms.

## Method

### Overall Architecture

Erwin adopts an encoder-processor-decoder architecture, with the processing pipeline as follows:

1. **Input**: Point cloud $P = \{p_1, ..., p_n\} \subset \mathbb{R}^d$, where each point is associated with a feature vector $x \in \mathbb{R}^c$.
2. **Ball Tree Construction**: Recursively split the point cloud to construct a perfect binary tree $T = \{L_0, L_1, ..., L_m\}$.
3. **Encoder**: Perform ball attention at the finest granularity level.
4. **Processor**: Progressively coarsen the representation to higher levels to perform attention, and then restore the original resolution through refinement.
5. **Decoder**: Output node-level predictions.

The key innovation lies in the contiguous storage property of the ball tree, which allows any level of ball to be accessed via simple tensor reshape operations. This greatly simplifies the implementation and accommodates GPU parallel computing.

### Key Designs

#### 1. Ball Tree Partitioning

A ball tree is a hierarchical data structure that recursively partitions a set of points into equal-sized nested subsets:

- **Construction Method**: In each recursion, find the axis with the maximum spread (the coordinate axis with the largest max - min value) and split along the median.
- **Tree Completion**: Fill the tree with dummy nodes to construct a perfect binary tree with a total of $2^m$ nodes, where $m = \text{ceil}(\log_2(n))$.
- **Key Properties**:
    - Perfect binary tree structure.
    - Each ball at level $i$ contains exactly $2^i$ leaf nodes.
    - Balls at each level cover the entire point cloud.
    - **Contiguous Storage**: There exists a permutation $\pi$ such that points within the same ball have contiguous indices after permutation.

The contiguous storage property is fundamental to the efficient implementation of Erwin. Accessing balls at any level $i$ only requires selecting contiguous blocks of $2^i$ indices, which is equivalent to reshaping the leaf-level tensor.

Compared to octrees, the advantage of ball trees is that nodes at the same level correspond to regions of the same spatial scale, whereas an octree covers the entire space, meaning nodes at the same level can correspond to vastly different spatial scales.

#### 2. Ball Attention

At level $k$ of the ball tree, each ball $B \in L_k$ contains $2^k$ leaf nodes. Ball attention computes standard self-attention independently within each ball:

$$X'_B = \text{BAtt}(X_B) := \text{Att}(X_B W_q, X_B W_k, X_B W_v)$$

- Weights are shared across all balls.
- Complexity is reduced from $O(N^2)$ to $O(|B|^2 \cdot N/|B|)$, which becomes linear $O(N)$ when the ball size $|B|$ is fixed.
- The ball size $k$ represents a work balance between local accuracy and computational efficiency.

#### 3. Position Encoding

Two types of positional information injection are introduced:

**Relative Position Encoding (RPE)**:

$$X_B = X_B + (P_B - c_B) W_{\text{pos}}$$

This encodes the geometric structure within the ball by projecting the offset of leaf nodes relative to the ball centroid $c_B$ through a learnable matrix and adding it to the features.

**Distance Bias**:

$$\mathcal{B}_B = -\sigma^2 \|c_{B'} - c_{B''}\|_2, \quad B', B'' \in \text{leaves}_B$$

where $\sigma$ is a learnable parameter. This term decays rapidly with distance, reinforcing locality and mitigating boundary artifacts where distant points might be grouped into the same ball during tree construction.

#### 4. Cross-ball Connection

Inspired by the shifted window approach in Swin Transformer, a second ball tree $T_{\text{rot}}$ is constructed by rotating the point cloud, yielding a new leaf permutation $\pi_{\text{rot}}$. The cross-ball attention is computed as:

$$X'_B = \pi_{\text{rot}}^{-1}(\text{BAtt}(\pi_{\text{rot}}(X_B)))$$

Alternating between the original and rotated configurations in consecutive layers ensures that leaf nodes belonging to different balls can interact. This is a key innovation in generalizing the "shifted window" strategy from regular grids to irregular point clouds—rotating a point cloud is equivalent to translating windows on a regular grid.

Since the construction of a ball tree depends on the orientation of the coordinate axes (splitting along the median of the maximum spread axis), rotating the point cloud alters the split outcomes, naturally yielding a different partitioning.

#### 5. Tree Coarsening & Refinement

**Coarsening Operation**: Aggregate nodes from the current leaf level $k$ to a higher level $k+l$ ball center:

$$x_B = \left(\bigoplus_{B' \in \text{leaves}_B} [x_{B'}, c_{B'} - c_B]\right) W_c$$

where $\oplus$ represents concatenation at the leaf level, and $W_c \in \mathbb{R}^{C' \times 2^l(C+d)}$ is a learnable projection that projects features to a higher dimension to maintain representational capacity. After coarsening, $L_{k+l}$ becomes the new leaf level.

**Refinement Operation**: The inverse of coarsening, which distributes coarse-level features back to the original fine-grained nodes to restore resolution.

Through alternating coarsening, attention, and refinement, Erwin processes information at different scales, similar to a U-Net encoder-decoder structure, but built entirely on the ball tree hierarchy.

### Loss & Training

- Training is conducted in a standard supervised manner, with the loss function adapted to specific tasks (e.g., using MSE loss for forces/energies in molecular dynamics, or $L_2$ loss for fields in PDE solving).
- The ball tree is constructed deterministically prior to the forward pass (requiring no gradients).
- Dummy node features are masked out and do not participate in the softmax computation of the attention mechanism.
- The rotation angle is a hyperparameter and is fixed across different experiments.

## Key Experimental Results

### Main Results

Erwin was validated across four large-scale physical domains:

| Area | Dataset/Task | Metric | Erwin | Prev. SOTA | Remarks |
|------|-------------|------|-------|----------|------|
| Cosmology | N-body dark matter simulation | Displacement field error | **Best** | GNN-based | Captures long-range gravitational interactions |
| Molecular Dynamics | Force prediction in large molecular systems | Force MAE | **Best** | Equivariant GNN | Significantly faster inference speed |
| PDE Solving | CFD benchmark | Relative L2 error | **Best** | FNO / GNN-based | Handles irregular grids |
| Particle Fluids | Turbulent fluid dynamics | MSE | **Best** | MPNN / Cluster Att. | Multi-scale feature capture |

### Ablation Study

| Configuration | Key Metric Change | Explanation |
|------|-------------|------|
| W/o cross-ball connection | Significant degradation | Cross-ball interaction is crucial for global information propagation |
| W/o RPE | Degradation | Relative position encoding provides critical geometric information |
| W/o distance bias | Slight degradation | Distance bias enhances locality, playing a larger role in long-range tasks |
| Fixed single layer (no coarsening) | Significant degradation | Multi-scale hierarchy is necessary to capture global features |
| Ball size $|B|=8$ vs $16$ vs $32$ | Accuracy/efficiency trade-off | $|B|=16$ offers the best balance for most tasks |
| Octree instead of ball tree | Degradation | The equal-scale property of the ball tree is superior to that of the octree |

### Key Findings

1. **Verification of Linear Scalability**: As the number of nodes $N$ increases, both the runtime and memory footprint of Erwin scale linearly, whereas standard attention scales quadratically. Erwin is already significantly faster than full attention at $N=10\text{K}$, with an even more pronounced advantage at $N=100\text{K}$.
2. **Coarsening Levels and Receptive Field**: A global receptive field is achieved with $\log_2(N) - \log_2(|B|)$ coarsening layers, with each layer maintaining attention computation of a fixed size $|B|$.
3. **Cross-domain Generalization**: The same architecture achieves competitive performance across four highly distinct domains (cosmology, molecular dynamics, PDEs, and fluids) without massive hyperparameter tuning.
4. **Ball Tree vs. Other Partitioning Strategies**: Due to benefits such as nodes at the same level being associated with equal-scale regions, simple construction, and contiguous storage, ball tree partitioning outperforms octrees and clustering-based partitioning methods.

## Highlights & Insights

1. **Elegant Fusion of Computational Physics and Deep Learning**: Drawing inspiration from classical numerical methods like Barnes-Hut / FMM, using a ball tree to organize attention computations is a prime example of applying "old methods to new tools."
2. **Contiguous Storage $\rightarrow$ Partitioning via Reshape**: Leveraging the contiguous storage property of the ball tree, all hierarchical operations can be simplified to tensor reshaping, making the implementation minimal and highly GPU-friendly.
3. **Rotation as Shifted Windows**: Elegantly extending Swin's shifted window strategy to irregular point clouds, where rotating the coordinate axes alters the ball tree partitioning, is an ingenious geometric insight.
4. **True Linear Complexity**: Unlike many methods claiming "near-linear" complexity that rely on approximation or kernel tricks, Erwin guarantees exact linear complexity through structural constraints (fixed ball size).
5. **Multi-scale Information Flow**: The cascaded structure of coarsening-attention-refinement naturally creates a multi-scale flow of information, employing exact attention rather than approximations at each scale.

## Limitations & Future Work

1. **Loss of Rotational Invariance**: Ball tree construction depends on the orientation of the coordinate axes (splitting along the maximum spread direction), which inherently breaks rotational invariance. Although the cross-ball rotation mechanism partially addresses this, it remains a limitation for tasks requiring strict equivariance (e.g., molecular property prediction).
2. **Limitations of Fixed Ball Size**: The ball size $|B|$ is a global hyperparameter that cannot adapt to local density variations. Dense regions might require smaller balls to capture fine-grained interactions, while sparse regions waste computation.
3. **Overhead of Dummy Nodes**: Padding the number of nodes to $2^m$ can introduce a substantial number of dummy nodes (approaching $50\%$ in the worst case), wasting computation and memory.
4. **Tree Reconstruction in Dynamic Systems**: For temporal physical simulations, the ball tree must be reconstructed at each time step. Although tree construction is $O(N \log N)$, it still introduces additional overhead.
5. **Information Loss in Coarsening**: Concatenation followed by linear projection may not fully preserve all sub-node information. More powerful aggregation strategies, such as attention-based pooling, could improve performance.

## Related Work & Insights

- **Swin Transformer** (Liu et al., 2021): The shifted window strategy is the direct inspiration for the cross-ball connection.
- **PointTransformer v3** (Wu et al., 2024): Serializes point clouds using space-filling curves for patching, though such curves may disrupt spatial locality.
- **OctFormer** (Wang, 2023): Serializes point clouds based on octree traversal, but octree convolutional calculations suffer from high computational overhead.
- **Fast Multipole Method** (Carrier et al., 1988): A classic $O(N)$ algorithm for multi-body problems, which shares the same hierarchical spirit as Erwin.
- **Cluster Attention** (Janny et al., 2023; Alkin et al., 2024): Attention methods based on clustering, which introduce an information bottleneck in the clustering step.

Potential research directions: Combining Erwin's ball tree attention with equivariant networks, exploring adaptive ball size strategies, and applying the coarsening-refinement paradigm to autoregressive sequence prediction tasks.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Novelty | 4.5 | Original integration of classical computational physics methods with Transformers |
| Theoretical Depth | 4.0 | Clear formal definitions and rigorous complexity analysis |
| Experimental Thoroughness | 4.5 | Extensive ablation studies with validation across four highly distinct physical domains |
| Value | 4.0 | Clean implementation (via reshaping), but the lack of rotational invariance limits some application scenarios |
| Writing Quality | 4.5 | Clear illustrations with a complete logical chain from methodology and experiments |
| **Overall Score** | **4.3** | High-quality work that introduces classical numerical methods into modern architecture design |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Rethink the Role of Deep Learning towards Large-scale Quantum Systems](rethink_the_role_of_deep_learning_towards_large-scale_quantum_systems.md)
- [\[ICML 2025\] Causal-PIK: Causality-based Physical Reasoning with a Physics-Informed Kernel](causal-pik_causality-based_physical_reasoning_with_a_physics-informed_kernel.md)
- [\[ICML 2025\] L2D: Large Language Models to Diffusion Finetuning](large_language_models_to_diffusion_finetuning.md)
- [\[ICML 2025\] Liger: Linearizing Large Language Models to Gated Recurrent Structures](liger_linearizing_large_language_models_to_gated_recurrent_structures.md)
- [\[NeurIPS 2025\] TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE](../../NeurIPS2025/physics/titan_a_trajectory-informed_technique_for_adaptive_parameter_freezing_in_large-s.md)

</div>

<!-- RELATED:END -->
