---
title: >-
  [Paper Note] A Direct Approach to Viewing Graph Solvability
description: >-
  [ECCV 2024][3D Vision][Viewing graph] This paper proposes a more direct reformulation of the viewing graph solvability problem than prior works, introduces new concepts to understand the solvability of real-world SfM graphs, and presents more efficient algorithms for detecting and decomposing unsolvable scenarios.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Viewing graph"
  - "solvability"
  - "fundamental matrix"
  - "Structure from Motion"
  - "multiview geometry"
date: 2026-05-08
content_hash: 99e04b1f5471ca37
---

# A Direct Approach to Viewing Graph Solvability

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Others  
**Keywords**: Viewing graph, solvability, fundamental matrix, Structure from Motion, multiview geometry

## TL;DR

This paper proposes a more direct reformulation of the viewing graph solvability problem than prior works, introduces new concepts to understand the solvability of real-world SfM graphs, and presents more efficient algorithms for detecting and decomposing unsolvable scenarios.

## Background & Motivation

**Background**: Structure from Motion (SfM) is a foundational technology in 3D reconstruction, with its core being the recovery of camera parameters and 3D scene structure from multiple images. In SfM, the viewing graph is an important mathematical tool to describe the geometric relationships between cameras: nodes in the graph represent cameras, and edges represent the fundamental matrix relationships between pairs of cameras. The solvability of a viewing graph is a fundamental theoretical question—it answers whether, given a set of fundamental matrix constraints, one can uniquely (up to projective equivalence) determine the projection matrices of all cameras.

**Limitations of Prior Work**: Previous studies on viewing graph solvability relied on indirect algebraic methods, analyzing multilinear constraints (such as trifocal tensor constraints) to determine solvability. Although theoretically correct, these methods suffer from the following issues: (1) the formulation is overly indirect, non-intuitive, and difficult to relate to the graph structure; (2) detection algorithms are inefficient, especially on large-scale viewing graphs; (3) for unsolvable cases, there is a lack of effective decomposition methods to extract solvable subgraphs.

**Key Challenge**: The development of viewing graph solvability theory lags behind the practical application of SfM. Although unsolvable viewing graph configurations are encountered in practical SfM pipelines, efficient theoretical tools to detect and handle these cases are lacking. The indirect formulations create a gap between theory and practice.

**Goal**: (1) Propose a more direct and intuitive formulation for viewing graph solvability; (2) introduce new theoretical concepts to help understand the structure of real-world SfM graphs; (3) design more efficient algorithms for solvability detection and decomposition of unsolvable subgraphs.

**Key Insight**: The authors start from a fundamental formula—the fundamental matrix $F_{ij}$ explicitly relates two cameras $P_i$ and $P_j$ via $F_{ij} = [e_{ij}]_\times P_j P_i^+$ (where $e_{ij}$ is the epipole and $P_i^+$ is the pseudoinverse). This direct formula transforms the solvability problem into solving a system of equations, which is more intuitive than indirect derivations via multilinear constraints.

**Core Idea**: Starting from the formula directly relating camera pairs via the fundamental matrix, establish more direct solvability conditions and introduce new concepts to analyze and process unsolvable viewing graph configurations.

## Method

### Overall Architecture

This paper is a theoretically oriented work, and its "method" is mainly reflected in three aspects: (1) a new mathematical formulation—providing direct necessary and sufficient conditions for viewing graph solvability; (2) the introduction of new concepts—proposing an intermediate concept between fully solvable and fully unsolvable that helps characterize the structure of practical SfM graphs; (3) efficient algorithms—providing faster algorithms for detecting solvability and extracting solvable components. The overall pipeline is: given a viewing graph (with edges labeled with fundamental matrices), determine whether the graph is solvable, and if it is unsolvable, decompose it into maximal solvable subgraphs.

### Key Designs

1. **Direct Formulation**:

    - **Function**: Establish direct mathematical conditions for viewing graph solvability.
    - **Mechanism**: The starting point is the direct relationship between the fundamental matrix and the camera matrices $F_{ij} = [e_{ij}]_\times P_j P_i^+$. This equation yields a constraint equation for each edge. The authors ensemble the constraint equations of all edges to form a large-scale system of linear equations. Solvability is equivalent to the dimension of the solution space of this system being exactly equal to the degrees of freedom of the projective transformation group (i.e., the 15-dimensional $PGL(4)$ degrees of freedom). By analyzing the null space dimension of this linear system, solvability can be directly determined without relying on indirect derivations via multilinear constraints.
    - **Design Motivation**: Compared to previous methods that determine solvability via indirect constraints such as trifocal tensors, the direct formulation is easier to understand—solvability simply boils down to "whether there are enough equations to uniquely determine the cameras." This perspective is more transparent and easier to extend to new scenarios.

2. **New Intermediate Concepts**:

    - **Function**: Describe the viewing graph structures that lie between fully solvable and fully unsolvable.
    - **Mechanism**: The authors introduce a new graph-theoretic concept that characterizes scenarios where some cameras in a viewing graph can be uniquely determined while others cannot. Specifically, although certain subgraphs cannot independently determine all cameras, the internal cameras can be uniquely determined if boundary conditions are given (i.e., some cameras are known). This hierarchical solvability structure is highly common in practical SfM—for instance, in a corridor scene, the front and back segments might be individually solvable, but a weak connection in the middle makes the overall graph unsolvable.
    - **Design Motivation**: Practical SfM graphs are often neither fully solvable nor fully unsolvable, but rather possess complex hierarchical structures. The new concept provides a more fine-grained analytical tool, helping SfM systems understand which parts are reliable and which parts require additional constraints.

3. **Efficient Detection and Decomposition Algorithms**:

    - **Function**: Detect solvability and extract solvable subgraphs faster than existing methods.
    - **Mechanism**: Based on the direct formulation, solvability detection reduces to computing the null space dimension of a large-scale sparse system of linear equations. Exploiting the sparse structure of the viewing graph and the special algebraic properties of the fundamental matrix, the computational complexity is reduced from the high-degree polynomials of previous methods. For unsolvable cases, the algorithm analyzes the null space structure to identify which edge constraints are redundant and which subgraphs form unsolvable "bottlenecks," and then performs graph decomposition using a block-triangularization-like strategy based on connectivity structure to obtain the maximal solvable components.
    - **Design Motivation**: Practical SfM systems need to run efficiently on large-scale graphs. Previously, due to indirect formulations, the algorithmic complexity was high, making them impractical for large-scale scenes (thousands of images). The matrix structure resulting from the direct formulation allows the use of more efficient linear algebra algorithms.

### Theoretical Analysis

- Proved that the direct formulation is equivalent to existing definitions of solvability.
- Analyzed the relationship between the new concept and classical solvability, and posed an open problem.
- Provided several concrete examples of unsolvable viewing graphs and their solvable subgraph decompositions.

## Key Experimental Results

### Main Results

Since this is a theoretical work, the experiments mainly include: (1) validating the correctness of the solvability criterion on synthetic viewing graphs; (2) analyzing the solvability structure of viewing graphs on practical SfM datasets.

| Scene | Graph Scale (Nodes/Edges) | Solvability | Decomposition Result | Algorithm Runtime |
|------|-----------------|--------|----------|----------|
| Synthetic Small Graph | <20 nodes | Validated Correct | Consistent with theory | <1s |
| Medium-scale SfM | 100-500 nodes | Partially solvable | Extracted 2-3 solvable components | Seconds |
| Large-scale SfM | 1000+ nodes | Complex structure | Hierarchical decomposition | Faster than prior methods |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Direct vs. Indirect Formulation | Consistent results | Theoretical equivalence is validated |
| New Algorithm vs. Old Algorithm | Faster speed | Significant advantage on large-scale graphs |
| Noisy vs. Noise-free | Stable under noise | Practical fundamental matrices contain estimation errors |

### Key Findings

- The direct formulation is not only theoretically equivalent but is also easier to implement and more efficient in practice.
- Partially unsolvable cases frequently occur in real SfM datasets, and the new concept helps identify these weak connections.
- The improved efficiency of the algorithm makes real-time solvability detection in practical large-scale SfM pipelines feasible.
- The proposed open problem—the precise relationship between the new concept and solvability—points out a direction for future research.

## Highlights & Insights

- **Solid Theoretical Contribution**: The direct formulation makes solvability theory more transparent and operational, lowering the barrier to theoretical understanding.
- **Practical Value in New Concepts**: Provides a more fine-grained tool for analyzing the structure of practical SfM graphs.
- **Practical Significance in Efficiency Gains**: Enables the integration of solvability analysis into real-world SfM pipelines.
- **Posing of Open Problems**: Leaves valuable research directions for multiview geometry theory.

## Limitations & Future Work

- The practical application validation of this theoretical work is not sufficiently comprehensive, wanting integrated testing with a complete SfM pipeline.
- Numerical stability under noisy conditions requires deeper analysis.
- Only the solvability of projective reconstruction is discussed, without extension to affine or Euclidean reconstruction.
- The solvability analysis of fundamental matrices does not consider the case of essential matrices (calibrated cameras).
- Although the open problem is valuable, no solution or directional conjecture is provided.

## Related Work & Insights

- **vs. Trager et al. (2015)**: Previous solvability analyses were based on multilinear constraints (trifocal and quadrifocal tensors), which were indirect in formulation and high in algorithmic complexity. This paper drastically simplifies the analytical framework through the direct formula.
- **vs. Hartley & Zisserman's Classic Textbook**: The textbook introduces the relationship between the fundamental matrix and camera matrices but does not systematically apply it to solvability analysis. This paper fills this gap.
- **vs. Graph-Theoretic Methods**: Prior works analyzed solvability purely from a graph-theoretic perspective (e.g., requiring a minimum degree of the graph) but ignored the specific structure of algebraic constraints. This paper couples graph structure with algebraic constraints.

## Rating

- Novelty: ⭐⭐⭐⭐ The perspective of the direct formulation is novel and valuable, and the introduction of the new concept adds theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐ While there is some experimental validation matching its nature as a theoretical work, the validation of practical applications could be more comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations, clear problem statement, and the inclusion of open problems increases its impact.
- Value: ⭐⭐⭐ Contributes to multiview geometry theory, but the direct scope of application is narrow, primarily impacting theoretical research in the SfM domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Solvability of the Viewing Graph Under the Affine Camera Model](../../CVPR2026/3d_vision/solvability_of_the_viewing_graph_under_the_affine_camera_model.md)
- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [\[ECCV 2024\] Heterogeneous Graph Learning for Scene Graph Prediction in 3D Point Clouds](heterogeneous_graph_learning_for_scene_graph_prediction_in_3d_point_clouds.md)
- [\[ECCV 2024\] GAURA: Generalizable Approach for Unified Restoration and Rendering of Arbitrary Views](gaura_generalizable_approach_for_unified_restoration_and_rendering_of_arbitrary_.md)
- [\[ECCV 2024\] Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation](mesh2nerf_direct_mesh_supervision_for_neural_radiance_field_representation_and_g.md)

</div>

<!-- RELATED:END -->
