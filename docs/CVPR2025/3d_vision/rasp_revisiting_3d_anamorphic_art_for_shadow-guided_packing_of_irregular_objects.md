---
title: >-
  [Paper Note] RASP: Revisiting 3D Anamorphic Art for Shadow-Guided Packing of Irregular Objects
description: >-
  [CVPR 2025][3D Vision][Irregular Object Packing] Inspired by 3D anamorphic art, RASP utilizes a differentiable rendering framework guided by multi-view shadow/silhouette images to optimize the arrangement of irregular 3D objects inside a container. Concurrently, it introduces SDF-based collision and extrusion handling strategies to achieve high-occupancy packing, part assembly, and multi-view art creation.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Irregular Object Packing"
  - "3D Anamorphic Art"
  - "Differentiable Rendering"
  - "Shadow-Guided Optimization"
  - "SDF Collision Detection"
date: 2026-05-08
content_hash: 75f6f02e0b8aa19c
---

# RASP: Revisiting 3D Anamorphic Art for Shadow-Guided Packing of Irregular Objects

**Conference**: CVPR 2025  
**arXiv**: [2504.02465](https://arxiv.org/abs/2504.02465)  
**Code**: None (Project Page available)  
**Area**: 3D Vision  
**Keywords**: Irregular Object Packing, 3D Anamorphic Art, Differentiable Rendering, Shadow-Guided Optimization, SDF Collision Detection

## TL;DR

Inspired by 3D anamorphic art, RASP utilizes a differentiable rendering framework guided by multi-view shadow/silhouette images to optimize the arrangement of irregular 3D objects inside a container. Concurrently, it introduces SDF-based collision and extrusion handling strategies to achieve high-occupancy packing, part assembly, and multi-view art creation.

## Background & Motivation

**Background**: 3D irregular object packing is a classic NP-hard combinatorial optimization problem with widespread applications in logistics, manufacturing, 3D printing, and robotics. Existing methods are primarily categorized into heuristic approaches (such as DBLF and heightmap minimization), learning-based methods (such as reinforcement learning for online packing), and physics simulation-based methods.

**Limitations of Prior Work**: (1) A large portion of existing methods only process simple geometries (e.g., cuboids, spheres, ellipsoids) or require convex/concave polyhedra, failing to handle truly arbitrary 3D shapes. (2) Collision detection presents a major bottleneck—face-to-face detection is computationally prohibitive, while bounding sphere/box approximations lead to over-penalization. (3) Voxel-based methods suffer from high memory requirements and discretization errors. (4) Multi-stage optimization methods, such as Ma et al., require slow, iterative operations like object swapping and scaling, introducing massive time overhead.

**Key Challenge**: How to efficiently find dense packing arrangements and avoid collisions without being constrained by the geometric complexity of the objects?

**Goal**: (1) Design an end-to-end optimization framework capable of achieving 3D packing guided solely by 2D shadows/silhouettes. (2) Propose an efficient and accurate collision detection scheme.

**Key Insight**: The authors observe that the core concept of 3D anamorphic art—where arrangements of objects project meaningful silhouettes from specific viewpoints—is naturally linked to the packing problem: a fully packed container should project as a solid rectangle from all dominant views. Therefore, the packing problem can be formulated as an optimization task that matches the multi-view silhouettes of the object set to target silhouettes.

**Core Idea**: Formulate the multi-view shadow-matching problem as gradient-based optimization using differentiable rendering. The framework learns the rigid transformation (rotation + translation) of each object, integrated with SDF-based collision and extrusion losses, to achieve a one-stop solution for irregular object packing.

## Method

### Overall Architecture

Input: A set of $N$ arbitrarily shaped triangular mesh objects $\mathcal{S}$, an arbitrarily shaped container $\mathcal{C}$, and $K$ target silhouette images with their corresponding camera parameters. Output: The optimal rotation (represented by quaternions) and translation parameters for each object. The pipeline consists of: initializing object positions $\rightarrow$ generating multi-view silhouettes via differentiable rendering $\rightarrow$ computing the silhouette matching loss + SDF collision loss + container extrusion loss $\rightarrow$ iteratively optimizing for 1000 steps using the Adam optimizer. The entire process takes approximately 15-17 minutes (for 106 objects, 80,000 mesh vertices).

### Key Designs

1. **Shadow-guided Optimization**:

    - **Function**: Drives objects to cluster inside the container by matching rendered silhouettes with target silhouette images.
    - **Mechanism**: For the packing scenario, the target silhouettes are the projected appearances of the container itself from various views (solid projections). A differentiable renderer is used to generate multi-view silhouettes $\hat{I}_k$ of the current object arrangement, and the MSE loss is calculated against the target $I_k$: $\mathcal{L}_{sil} = \frac{1}{MNK}\sum_k\sum_i \|I_k(i) - \hat{I}_k(i)\|_2^2$. Quaternions are used to represent rotation to avoid gimbal lock.
    - **Design Motivation**: Pure 2D shadow guidance transforms complex 3D collision and arrangement problems into simple image-matching tasks, which naturally supports containers of arbitrary shapes.

2. **SDF-based Intersection Loss**:

    - **Function**: Detects and penalizes overlapping between objects.
    - **Mechanism**: Pre-computes the SDF for each object, evaluated on fixed query points inside the container. When an object undergoes cooperative rigid transformation, instead of recomputing the SDF, inverse transformation (linear warping) is applied to the query points. This identifies points located inside multiple objects simultaneously (SDF < 0), and their absolute SDF values are summed to evaluate the collision degree $D_{is}(\mathbf{p}) = \sum_{\{O_i | \tilde{S}_{O_i}(\mathbf{p}) < 0\}} -\tilde{S}_{O_i}(\mathbf{p})$. The total interpenetration loss is the sum of collision degrees over all query points. The key advantage is using actual SDF values instead of simple binary counting, providing smoother gradient guidance.
    - **Design Motivation**: Bounding sphere/box methods over-penalize asymmetric objects (spheres intersect even when actual meshes do not), whereas face-by-face collision detection is too slow. The SDF scheme achieves an elegant trade-off between accuracy and efficiency.

3. **Container Extrusion Loss**:

    - **Function**: Prevents objects from breaking through the container boundaries.
    - **Mechanism**: Utilizes the container's SDF $S_\mathcal{C}$ to penalize each vertex $\mathbf{v}$ of each object when it lies outside the container ($S_\mathcal{C}(\mathbf{v}) > 0$): $\mathcal{L}_{ext} = \sum_i\sum_{\mathbf{v}\in V_i} \max(-\epsilon, S_\mathcal{C}(\mathbf{v}))$, where $\epsilon = 0.01$ serves as a buffer.
    - **Design Motivation**: This loss acts as a regularizer, preventing objects from being pushed too far away to avoid collisions. Introducing the extrusion loss leads to tighter arrangements and better silhouette matching.

### Loss & Training

The total loss is: $\mathcal{L}_{total} = \mathcal{L}_{sil} + \mathcal{L}_{is} + \lambda \mathcal{L}_{ext}$, where $\lambda = 0.001$. The optimizer is Adam, with the learning rate decaying from $1e{-2}$ to $1e{-4}$ over 1000 iterations. Objects are randomly initialized within the container. The initial number of objects is estimated based on the ratio between the container volume and the average object volume, and then incrementally increased until reaching the maximum capacity of the container, $N_{max}$.

## Key Experimental Results

### Main Results

Packing experiments are conducted across four categories on the IR-BPP dataset:

| Dataset Category | Number of Objects | Packing Density $\rho$ | Description |
|-----------|--------|-----------|------|
| General | 78 | ~45% | General irregular objects |
| Kitchen | 106 | ~45% | Kitchen utensil objects |
| ABC | 85 | ~45% | CAD models |
| Block Out | 92 | ~45% | Toy blocks objects |

Comparison with existing methods:

| Method | Average Packing Density | Average Optimization Time |
|------|------------|------------|
| Ma et al. | 34% | 40.55 min |
| Zhao et al. (RL Online) | 51.9% | - |
| RASP (Ours) | 45% | ~15 min |

### Ablation Study

| Configuration | Collision Metric | Silhouette Matching Quality | Description |
|------|---------|-------------|------|
| $\mathcal{L}_{sil}$ Only | Collisions exist | Has gaps | Silhouettes match but objects intersect |
| $\mathcal{L}_{sil} + \mathcal{L}_{is}$ | Zero collision | Average | No intersection but large inter-object spacing |
| $\mathcal{L}_{sil} + \mathcal{L}_{is} + \mathcal{L}_{ext}$ | Zero collision | Optimal | Tightly arranged objects, complete model |

### Key Findings

- The SDF collision loss is a core component—removing it results in object intersection, as the silhouette loss alone cannot perceive 3D overlaps.
- Although the extrusion loss has a very small weight ($\lambda=0.001$), it plays a crucial regularizing role, making the object packing layout more compact.
- The framework is equally effective for arbitrary container shapes, such as toruses and squirrel-like shapes, demonstrating great generalization.
- When the number of objects is less than the maximum capacity, objects tend to "float" in the middle of the container rather than settle at the bottom; this can be mitigated by adjusting silhouette widths.

## Highlights & Insights

- **Shadow as Information**: Casts 3D packing ingeniously as a 2D image-matching problem, dramatically reducing problem complexity. This approach can be transferred to any scenario that requires inverse 3D arrangement from projections, such as CAD design and robotic assembly planning.
- **Clever Use of SDF**: Instead of performing collision detection directly at the mesh level, the method uses SDF values as a "soft" collision metric. This guarantees accuracy while providing smooth gradients—a technique that can be adapted to other differentiable physical simulation scenarios.
- **One Framework, Multiple Uses**: The same framework can perform three tasks—packing, part assembly, and multi-view art creation—without modifications, demonstrating the generality of approaching 3D problems from a projection/shadow-based perspective.

## Limitations & Future Work

- Physical dynamics (such as gravity) are not considered, meaning the packing results are not guaranteed to be physically stable.
- Performance degrades on assembly tasks involving more than 2-3 parts, especially with symmetric or identical parts.
- As an offline optimization method, it is incapable of handling online packing scenarios (where objects arrive in real-time).
- The packing density (45%) is still lower than that of physics-based reinforcement learning methods (51.9%), though the optimization time is shorter.
- Potential improvements: introducing a physics engine for post-processing to ensure physical stability; combining with RL methods for hybrid optimization.

## Related Work & Insights

- **vs Ma et al.**: They perform multi-stage combinatorial optimization (swapping + scaling + putting back), whereas RASP is single-stage and end-to-end—being 2.7x faster although with a slightly lower packing density.
- **vs Zhao et al.**: They use physics simulation + RL for online packing, processing one object at a time. While their packing density is higher, they rely heavily on the arrival sequence of the objects, whereas RASP performs global optimization on all objects simultaneously, making it more suitable for offline planning.
- **vs Shadow Art (Mitra & Pauly)**: Shadow Art uses shadows for 3D reconstruction, whereas RASP utilizes shadows for 3D spatial arrangement, demonstrating the multi-faceted value of shadows/projections in 3D understanding.

## Rating

- Novelty: ⭐⭐⭐⭐ The cross-disciplinary concept of bridging anamorphic art with 3D packing is highly novel, but the core remains a standard combination of differentiable rendering and SDF.
- Experimental Thoroughness: ⭐⭐⭐⭐ Results are shown across multiple datasets and tasks, with quantitative comparisons and ablation studies, though it lacks more diverse quantitative metrics.
- Writing Quality: ⭐⭐⭐⭐ The motivation narrative is engaging, and the analogy between art and engineering is clearly articulated.
- Value: ⭐⭐⭐⭐ Offers a brand new perspective on packing, though practical industrial applications may still require physical constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] UnCommon Objects in 3D](uncommon_objects_in_3d.md)
- [\[CVPR 2025\] BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis](bfanet_revisiting_3d_semantic_segmentation_with_boundary_feature_analysis.md)
- [\[CVPR 2025\] PICO: Reconstructing 3D People In Contact with Objects](pico_reconstructing_3d_people_in_contact_with_objects.md)
- [\[CVPR 2025\] Gen3DEval: Using vLLMs for Automatic Evaluation of Generated 3D Objects](gen3deval_using_vllms_for_automatic_evaluation_of_generated_3d_objects.md)
- [\[CVPR 2025\] Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects](instant3dit_multiview_inpainting_for_fast_editing_of_3d_objects.md)

</div>

<!-- RELATED:END -->
